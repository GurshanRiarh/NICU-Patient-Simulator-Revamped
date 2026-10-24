#!/usr/bin/env python
"""Train the single NeoFlow model on an NVIDIA GPU.

Example
-------
python scripts/train.py --epochs 40 --batch-size 256 --out models/neosim_flow.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neosim.config import DEFAULT_MODEL, RESULTS_DIR
from neosim.data.dataset import build_datasets, load_norm, make_sampler
from neosim.models.flow import EMA, NeoFlow, NeoFlowConfig, flow_matching_loss, save_checkpoint


def make_context(x1: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Random prefix context for inpainting-style training."""
    b, seq_len, _ = x1.shape
    context_mask = torch.zeros_like(mask)
    mode = torch.rand(b, device=x1.device)
    for i in range(b):
        if mode[i] < 0.50:
            continue
        ctx_len = 30 if mode[i] < 0.80 else 60
        context_mask[i, :ctx_len] = mask[i, :ctx_len]
    context = x1 * context_mask
    return context, context_mask


def evaluate(model: NeoFlow, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            x1 = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            context, context_mask = make_context(x1, mask)
            loss, _ = flow_matching_loss(
                model,
                x1,
                mask,
                batch["static"].to(device),
                batch["events"].to(device),
                batch["time"].to(device),
                context,
                context_mask,
                cond_dropout=0.0,
            )
            total += float(loss) * len(x1)
            count += len(x1)
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU training requested, but CUDA is not available.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    print(
        f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
    )

    norm = load_norm()
    train_set, val_set, weights = build_datasets(norm, augment=True)
    print(f"windows: train={len(train_set)} val={len(val_set)}")
    sampler = make_sampler(weights)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    cfg = NeoFlowConfig(
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
    )
    model = NeoFlow(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NeoFlow parameters: {n_params / 1e6:.2f} M")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, decay=0.999)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 does not need scaling

    history = {"train": [], "val": [], "lr": [], "step": []}
    best_val = float("inf")
    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x1 = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            events = batch["events"].to(device, non_blocking=True)
            static = batch["static"].to(device, non_blocking=True)
            time_feats = batch["time"].to(device, non_blocking=True)
            context, context_mask = make_context(x1, mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = flow_matching_loss(
                    model,
                    x1,
                    mask,
                    static,
                    events,
                    time_feats,
                    context,
                    context_mask,
                    cond_dropout=cfg.cond_dropout,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            global_step += 1
            running += float(loss.detach()) * len(x1)
            seen += len(x1)
            if global_step % 20 == 0:
                pbar.set_postfix(loss=f"{running / seen:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if args.max_steps and global_step >= args.max_steps:
                break
        train_loss = running / max(seen, 1)
        val_loss = evaluate(model, val_loader, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["step"].append(global_step)
        print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} ({time.time() - t0:.0f}s)")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                args.out,
                model,
                ema,
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "norm": norm,
                    "gpu": torch.cuda.get_device_name(0),
                    "n_params": n_params,
                },
            )
            print(f"  saved best checkpoint -> {args.out}")
        if args.max_steps and global_step >= args.max_steps:
            break

    history["best_val"] = best_val
    history["params"] = n_params
    history["training_seconds"] = time.time() - t0
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"done in {time.time() - t0:.0f}s; best val={best_val:.4f}")


if __name__ == "__main__":
    main()
