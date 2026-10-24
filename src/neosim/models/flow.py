"""NeoFlow: one conditional rectified-flow transformer for all vital signs.

The model jointly models heart rate, respiratory rate and SpO2.  A single model
replaces the four per-signal GANs used by the original simulator, and it is
conditioned on patient context, clinical events and (optionally) a known prefix
of the trajectory so it can be used as a rolling simulator.

Key ideas
---------
* Rectified flow / conditional flow matching: learn a velocity field that
  transports N(0, I) to the data distribution.  Sampling is a short ODE
  integration (20-50 Euler steps), which is fast enough for interactive use.
* Masked channels: PreMo provides HR/SpO2, PICS provides HR/RR.  The loss and
  the conditioning mask let one model learn all channels from heterogeneous
  data.
* Inpainting-style context: observed prefix values are fed as clean context;
  the model generates the remainder.  This gives smooth rolling simulations.
* Classifier-free guidance: static/event conditions are randomly dropped during
  training so the sampling routine can trade realism against condition strength.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class NeoFlowConfig:
    n_channels: int = 3
    n_events: int = 16
    n_static: int = 13
    n_time: int = 1
    d_model: int = 192
    n_layers: int = 6
    n_heads: int = 6
    dropout: float = 0.1
    max_seq_len: int = 256
    cond_dropout: float = 0.10


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding for a scalar flow time in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(timestep_embedding(t, self.net[0].in_features))


class AdaLNBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero conditioning."""

    def __init__(self, cfg: NeoFlowConfig):
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 4, d),
            nn.Dropout(cfg.dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class NeoFlow(nn.Module):
    """Conditional rectified-flow transformer."""

    def __init__(self, cfg: NeoFlowConfig | None = None):
        super().__init__()
        self.cfg = cfg or NeoFlowConfig()
        c = self.cfg
        token_dim = c.n_channels * 3 + c.n_events + c.n_time
        self.token_proj = nn.Linear(token_dim, c.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, c.max_seq_len, c.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.static_proj = nn.Sequential(
            nn.Linear(c.n_static, c.d_model),
            nn.SiLU(),
            nn.Linear(c.d_model, c.d_model),
        )
        self.time_proj = TimestepMLP(c.d_model)
        self.blocks = nn.ModuleList([AdaLNBlock(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.d_model, elementwise_affine=False)
        self.out_proj = nn.Sequential(
            nn.Linear(c.d_model, c.d_model),
            nn.GELU(),
            nn.Linear(c.d_model, c.n_channels),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        static: torch.Tensor,
        events: torch.Tensor,
        time_feats: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the flow velocity.

        Shapes: x_t [B,T,C], t [B], static [B,F], events [B,T,K],
        time_feats [B,T,1], mask [B,T,C], context/context_mask [B,T,C].
        """
        b, seq_len, _ = x_t.shape
        if context is None:
            context = torch.zeros_like(x_t)
        if context_mask is None:
            context_mask = torch.zeros_like(x_t)
        ctx = torch.where(context_mask.bool(), context, x_t)
        token = torch.cat([ctx, context_mask, mask, events, time_feats], dim=-1)
        h = self.token_proj(token)
        h = h + self.pos_emb[:, :seq_len]
        cond = self.time_proj(t) + self.static_proj(static)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.final_norm(h))


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.shadow[key].copy_(value.detach())

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow


def flow_matching_loss(
    model: NeoFlow,
    x1: torch.Tensor,
    mask: torch.Tensor,
    static: torch.Tensor,
    events: torch.Tensor,
    time_feats: torch.Tensor,
    context: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    cond_dropout: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rectified-flow MSE loss with channel and context masking."""
    b, seq_len, _ = x1.shape
    device = x1.device
    x0 = torch.randn_like(x1)
    t = torch.rand(b, device=device)
    x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    if context is None:
        context = torch.zeros_like(x1)
    if context_mask is None:
        context_mask = torch.zeros_like(x1)
    if cond_dropout:
        drop = (torch.rand(b, 1, device=device) < cond_dropout).float()
        static = static * (1 - drop)
        events = events * (1 - drop[:, :, None])
    v = model(x_t, t, static, events, time_feats, mask, context, context_mask)
    target = x1 - x0
    weight = mask * (1 - context_mask)
    denom = weight.sum().clamp_min(1.0)
    loss = (((v - target) ** 2) * weight).sum() / denom
    with torch.no_grad():
        per_channel = (((v - target) ** 2) * weight).sum(dim=(0, 1)) / weight.sum(dim=(0, 1)).clamp_min(1.0)
    return loss, {f"loss_ch{i}": float(v) for i, v in enumerate(per_channel.detach().cpu())}


@torch.no_grad()
def sample(
    model: NeoFlow,
    static: torch.Tensor,
    events: torch.Tensor,
    time_feats: torch.Tensor,
    mask: torch.Tensor,
    context: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    steps: int = 40,
    cfg_scale: float = 1.4,
    seed: int | None = None,
    noise_scale: float = 0.9,
    noise_phi: float = 0.5,
) -> torch.Tensor:
    """Generate trajectories with Euler integration and classifier-free guidance.

    ``noise_phi`` applies a stationary AR(1) correlation to the initial noise so
    the generated traces retain the minute-to-minute autocorrelation seen in
    neonatal monitoring; ``noise_scale`` calibrates its amplitude.
    """
    device = static.device
    b, seq_len, c = static.shape[0], time_feats.shape[1], model.cfg.n_channels
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    eps = torch.randn((b, seq_len, c), device=device, generator=generator)
    x = torch.empty_like(eps)
    previous = torch.zeros((b, c), device=device)
    innovation = math.sqrt(max(1.0 - noise_phi**2, 0.0))
    for t in range(seq_len):
        previous = noise_phi * previous + innovation * eps[:, t]
        x[:, t] = previous
    x = x * noise_scale
    if context is None:
        context = torch.zeros_like(x)
    if context_mask is None:
        context_mask = torch.zeros_like(x)
    null_static = torch.zeros_like(static)
    null_events = torch.zeros_like(events)
    for step in range(steps):
        t = torch.full((b,), step / steps, device=device)
        v_cond = model(x, t, static, events, time_feats, mask, context, context_mask)
        if cfg_scale != 1.0:
            v_uncond = model(x, t, null_static, null_events, time_feats, mask, context, context_mask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        x = x + v / steps
        if context_mask is not None:
            x = torch.where(context_mask.bool(), context, x)
    return x


def save_checkpoint(path, model: NeoFlow, ema: EMA | None, config: dict) -> None:
    payload = {
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "model_config": asdict(model.cfg),
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu") -> tuple[NeoFlow, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = NeoFlow(NeoFlowConfig(**payload["model_config"]))
    state = payload.get("ema") or payload["model"]
    model.load_state_dict(state)
    model.eval()
    return model, payload.get("config", {})
