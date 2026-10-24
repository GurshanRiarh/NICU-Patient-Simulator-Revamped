"""PyTorch datasets for the processed vital-sign windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from neosim.config import PROCESSED_DIR, SIGNALS


class VitalsWindowDataset(Dataset):
    """One npz split of 120-minute windows with masks and clinical events."""

    def __init__(self, path: str | Path, norm: dict, augment: bool = False):
        data = np.load(path)
        self.x = data["x"].astype(np.float32)
        self.mask = data["mask"].astype(np.float32)
        self.events = data["events"].astype(np.float32)
        self.static = data["static"].astype(np.float32)
        self.time = data["time"].astype(np.float32)
        self.patient = data["patient"]
        self.start_min = data["start_min"]
        self.condition = data["condition"]
        self.augment = augment

        self.channel_mean = np.array([norm["channels"][s]["mean"] for s in SIGNALS], dtype=np.float32)
        self.channel_std = np.array([norm["channels"][s]["std"] for s in SIGNALS], dtype=np.float32)
        self.static_mean = np.asarray(norm["static"]["mean"], dtype=np.float32)
        self.static_std = np.asarray(norm["static"]["std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.x[idx].copy()
        mask = self.mask[idx].copy()
        x = (x - self.channel_mean) / self.channel_std
        x = np.where(mask > 0, x, 0.0)
        static = (self.static[idx] - self.static_mean) / self.static_std
        if self.augment:
            # Small amplitude jitter keeps the model from memorising exact
            # monitor quantisation while preserving event morphology.
            jitter = np.random.normal(0, 0.015, size=x.shape).astype(np.float32)
            x = x + jitter * mask
            scale = np.random.uniform(0.98, 1.02, size=(1, x.shape[1])).astype(np.float32)
            x = x * scale
        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "events": torch.from_numpy(self.events[idx]),
            "static": torch.from_numpy(static.astype(np.float32)),
            "time": torch.from_numpy(self.time[idx]),
            "patient": torch.tensor(int(self.patient[idx]), dtype=torch.long),
            "condition": torch.tensor(int(self.condition[idx]), dtype=torch.long),
        }


def load_norm(path: str | Path = PROCESSED_DIR / "norm_stats.json") -> dict:
    return json.loads(Path(path).read_text())


def build_datasets(norm: dict, augment: bool = True) -> tuple[ConcatDataset, ConcatDataset, list[float]]:
    """Combine PreMo and PICS with dataset-balanced sampling weights."""
    train_paths = [PROCESSED_DIR / "premo_train.npz", PROCESSED_DIR / "pics_train.npz"]
    val_paths = [PROCESSED_DIR / "premo_val.npz", PROCESSED_DIR / "pics_val.npz"]
    train_parts = [VitalsWindowDataset(p, norm, augment=augment) for p in train_paths if p.exists()]
    val_parts = [VitalsWindowDataset(p, norm, augment=False) for p in val_paths if p.exists()]
    if not train_parts:
        raise FileNotFoundError("run scripts/preprocess.py first")
    train = ConcatDataset(train_parts)
    val = ConcatDataset(val_parts)
    # PreMo is ~99% of the windows; upweight PICS so the single model actually
    # learns respiratory dynamics.  25% PICS / 75% PreMo works well in practice.
    weights: list[float] = []
    total = len(train)
    for part_index, part in enumerate(train_parts):
        share = 0.25 if part_index == len(train_parts) - 1 else 0.75
        weights.extend([share / len(part)] * len(part))
    weights = [w * total for w in weights]
    return train, val, weights


def make_sampler(weights: list[float]) -> WeightedRandomSampler:
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
