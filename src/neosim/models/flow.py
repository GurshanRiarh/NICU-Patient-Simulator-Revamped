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


def _masked_moments(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denom = weight.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (x * weight).sum(dim=(0, 1)) / denom
    var = (((x - mean) ** 2) * weight).sum(dim=(0, 1)) / denom
    return mean, torch.sqrt(var.clamp_min(1e-6))


def _acf_stats(
    x: torch.Tensor, mask: torch.Tensor, lags: tuple[int, ...] = (1, 5, 15)
) -> torch.Tensor | None:
    stats: list[torch.Tensor] = []
    for channel in range(x.shape[-1]):
        full = (mask[:, :, channel] > 0).all(dim=1)
        if not bool(full.any()):
            continue
        values = x[full, :, channel].float()
        values = values - values.mean(dim=1, keepdim=True)
        denom = (values * values).sum(dim=1).clamp_min(1e-6)
        for lag in lags:
            if values.shape[1] <= lag:
                continue
            stats.append(((values[:, :-lag] * values[:, lag:]).sum(dim=1) / denom).mean())
    if not stats:
        return None
    return torch.stack(stats)


def _correlation_stats(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    stats: list[torch.Tensor] = []
    for a, b in ((0, 2), (0, 1)):
        full = (mask[:, :, a] > 0).all(dim=1) & (mask[:, :, b] > 0).all(dim=1)
        if not bool(full.any()):
            continue
        xa = x[full, :, a].float()
        xb = x[full, :, b].float()
        xa = xa - xa.mean(dim=1, keepdim=True)
        xb = xb - xb.mean(dim=1, keepdim=True)
        denom = torch.sqrt((xa * xa).sum(dim=1) * (xb * xb).sum(dim=1)).clamp_min(1e-6)
        stats.append(((xa * xb).sum(dim=1) / denom).mean())
    if not stats:
        return None
    return torch.stack(stats)


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
    aux_moment: float = 0.0,
    aux_acf: float = 0.0,
    aux_corr: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rectified-flow MSE loss with channel, context and auxiliary structure losses.

    The auxiliary terms act on the one-step estimate ``x1_hat = x_t + (1-t)v``.
    They are deliberately computed only for ``t > 0.5``, where that estimate is
    close enough to the clean sample to give a stable moment, autocorrelation
    and cross-channel correlation target.
    """
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
    metrics: dict[str, float] = {}
    with torch.no_grad():
        per_channel = (((v - target) ** 2) * weight).sum(dim=(0, 1)) / weight.sum(dim=(0, 1)).clamp_min(1.0)
        metrics.update({f"loss_ch{i}": float(value) for i, value in enumerate(per_channel.detach().cpu())})

    if aux_moment or aux_acf or aux_corr:
        x1_hat = x_t + (1 - t[:, None, None]) * v
        high_t = (t > 0.5).float()[:, None, None]
        aux_weight = weight * high_t
        if aux_moment:
            mean_hat, std_hat = _masked_moments(x1_hat.float(), aux_weight)
            mean_true, std_true = _masked_moments(x1.float(), aux_weight)
            moment_loss = ((mean_hat - mean_true) ** 2 + (std_hat - std_true) ** 2).mean()
            loss = loss + aux_moment * moment_loss
            metrics["aux_moment"] = float(moment_loss.detach().cpu())
        if aux_acf:
            acf_mask = mask * high_t
            acf_hat = _acf_stats(x1_hat.float(), acf_mask, lags=(1, 5, 15))
            acf_true = _acf_stats(x1.float(), acf_mask, lags=(1, 5, 15))
            if acf_hat is not None and acf_true is not None:
                acf_loss = ((acf_hat - acf_true) ** 2).mean()
                loss = loss + aux_acf * acf_loss
                metrics["aux_acf"] = float(acf_loss.detach().cpu())
        if aux_corr:
            corr_mask = mask * high_t
            corr_hat = _correlation_stats(x1_hat.float(), corr_mask)
            corr_true = _correlation_stats(x1.float(), corr_mask)
            if corr_hat is not None and corr_true is not None:
                corr_loss = ((corr_hat - corr_true) ** 2).mean()
                loss = loss + aux_corr * corr_loss
                metrics["aux_corr"] = float(corr_loss.detach().cpu())
    return loss, metrics


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
    noise_phi_channels: tuple[float, float, float] | None = None,
    noise_scale_channels: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Generate trajectories with Euler integration and classifier-free guidance.

    ``noise_phi`` applies a stationary AR(1) correlation to the initial noise so
    the generated traces retain the minute-to-minute autocorrelation seen in
    neonatal monitoring; ``noise_scale`` calibrates its amplitude.  The optional
    per-channel tuples override the scalar values, which is useful because HR,
    RR and SpO2 need different amounts of temporal noise.
    """
    device = static.device
    b, seq_len, c = static.shape[0], time_feats.shape[1], model.cfg.n_channels
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    eps = torch.randn((b, seq_len, c), device=device, generator=generator)
    x = torch.empty_like(eps)
    if noise_phi_channels is None:
        phi = torch.full((1, c), noise_phi, device=device)
    else:
        phi = torch.tensor(noise_phi_channels, device=device).view(1, c)
    if noise_scale_channels is None:
        scale = torch.full((1, c), noise_scale, device=device)
    else:
        scale = torch.tensor(noise_scale_channels, device=device).view(1, c)
    previous = torch.zeros((b, c), device=device)
    innovation = torch.sqrt((1.0 - phi**2).clamp_min(0.0))
    for t in range(seq_len):
        previous = phi * previous + innovation * eps[:, t]
        x[:, t] = previous
    x = x * scale
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
