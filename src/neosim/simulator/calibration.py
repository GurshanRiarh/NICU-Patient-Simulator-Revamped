"""Online patient-specific calibration for generated vital-sign windows.

The calibration uses only a patient's prior real windows and the synthetic
windows generated for the same prior context.  It fits a monotonic quantile map
per channel and can optionally match the mean and standard deviation of the most
recent prior window.  No current or future observations are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(slots=True)
class ChannelCalibrator:
    source_quantiles: np.ndarray
    target_quantiles: np.ndarray
    target_mean: float | None = None
    target_std: float | None = None

    def apply(self, values: np.ndarray, jitter: float = 0.02, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        jittered = values + rng.normal(0.0, jitter, size=values.shape)
        out = np.interp(jittered, self.source_quantiles, self.target_quantiles)
        if self.target_mean is not None and self.target_std is not None:
            mean = float(np.mean(out))
            std = float(np.std(out))
            if std > 1e-6:
                out = self.target_mean + (out - mean) * (self.target_std / std)
        return out.astype(np.float32)


class OnlinePatientCalibrator:
    """Per-patient calibration fitted on a warm-up of prior windows."""

    def __init__(
        self,
        channels: tuple[int, ...],
        jitter: float = 0.02,
        quantiles: int = 2001,
    ):
        self.channels = channels
        self.jitter = jitter
        self.quantiles = quantiles
        self.maps: dict[int, ChannelCalibrator] = {}

    def fit(
        self,
        real_history: np.ndarray,
        synthetic_history: np.ndarray,
        mask: np.ndarray | None = None,
        stats_window: int = 0,
        seed: int = 0,
    ) -> OnlinePatientCalibrator:
        if real_history.ndim != 3 or synthetic_history.ndim != 3:
            raise ValueError("history arrays must have shape [windows, time, channels]")
        if real_history.shape != synthetic_history.shape:
            raise ValueError("real and synthetic histories must have the same shape")
        if mask is None:
            mask = np.ones_like(real_history, dtype=bool)
        for channel in self.channels:
            channel_mask = mask[:, :, channel] > 0
            real_values = real_history[:, :, channel][channel_mask]
            synthetic_values = synthetic_history[:, :, channel][channel_mask]
            if len(real_values) < 2 or len(synthetic_values) < 2:
                continue
            rng = np.random.default_rng(seed + channel)
            source = np.sort(synthetic_values + rng.normal(0.0, self.jitter, size=synthetic_values.shape))
            grid = np.linspace(0.0, 1.0, self.quantiles)
            source_quantiles = np.quantile(source, grid)
            target_quantiles = np.quantile(real_values, grid)
            source_quantiles = np.maximum.accumulate(
                source_quantiles + np.arange(len(source_quantiles)) * 1e-9
            )
            calibrator = ChannelCalibrator(source_quantiles, target_quantiles)
            if stats_window > 0:
                recent_mask = mask[-1, -stats_window:, channel] > 0
                recent = real_history[-1, -stats_window:, channel][recent_mask]
                if len(recent) > 1:
                    calibrator.target_mean = float(np.mean(recent))
                    calibrator.target_std = float(np.std(recent))
            self.maps[channel] = calibrator
        return self

    def apply(self, window: np.ndarray, mask: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
        if window.ndim != 2:
            raise ValueError("window must have shape [time, channels]")
        out = window.copy()
        if mask is None:
            mask = np.ones_like(window, dtype=bool)
        for channel, calibrator in self.maps.items():
            valid = mask[:, channel] > 0
            if valid.any():
                out[valid, channel] = calibrator.apply(
                    out[valid, channel], jitter=self.jitter, seed=seed + channel
                )
        return out.astype(np.float32)


def _acf1(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denom)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _adjust_cross_correlation(
    window: np.ndarray,
    channel_a: int,
    channel_b: int,
    source_corr: float,
    target_corr: float,
) -> np.ndarray:
    a = window[:, channel_a]
    b = window[:, channel_b]
    za = (a - np.mean(a)) / max(float(np.std(a)), 1e-6)
    zb = (b - np.mean(b)) / max(float(np.std(b)), 1e-6)
    source = float(np.clip(source_corr, -0.95, 0.95))
    target = float(np.clip(target_corr, -0.95, 0.95))
    residual = (zb - source * za) / np.sqrt(max(1.0 - source**2, 1e-6))
    zb_new = target * za + np.sqrt(max(1.0 - target**2, 1e-6)) * residual
    out = window.copy()
    out[:, channel_b] = float(np.mean(b)) + zb_new * float(np.std(b))
    return out


class DynamicPatientCalibrator:
    """Online calibration with temporal noise and cross-channel coupling.

    The dynamic layer is fitted on prior real/synthetic pairs and applied before
    the monotonic quantile map.  It makes the synthetic autocorrelation and
    cross-channel correlation closer to the patient's recent history.
    """

    def __init__(
        self,
        channels: tuple[int, ...],
        cross_channels: tuple[int, int] | None = None,
        cross_factor: float = 1.0,
        acf_noise_scale: float = 0.0,
        rank_channels: tuple[int, ...] = (),
        rank_stats_scale: float = 1.0,
        jitter: float = 0.02,
        quantiles: int = 2001,
    ):
        self.base = OnlinePatientCalibrator(channels, jitter=jitter, quantiles=quantiles)
        self.channels = channels
        self.cross_channels = cross_channels
        self.cross_factor = cross_factor
        self.acf_noise_scale = acf_noise_scale
        self.rank_channels = rank_channels
        self.rank_stats_scale = rank_stats_scale
        self.source_corr = 0.0
        self.target_corr = 0.0
        self.relative_noise: dict[int, float] = {}
        self.rank_targets: dict[int, np.ndarray] = {}
        self.rank_stats: dict[int, tuple[float, float]] = {}

    def fit(
        self,
        real_history: np.ndarray,
        synthetic_history: np.ndarray,
        mask: np.ndarray | None = None,
        stats_window: int = 0,
        seed: int = 0,
    ) -> DynamicPatientCalibrator:
        if mask is None:
            mask = np.ones_like(real_history, dtype=bool)
        work = synthetic_history.copy()
        if self.cross_channels is not None:
            channel_a, channel_b = self.cross_channels
            self.source_corr = _correlation(work[:, :, channel_a].ravel(), work[:, :, channel_b].ravel())
            target = _correlation(real_history[:, :, channel_a].ravel(), real_history[:, :, channel_b].ravel())
            self.target_corr = float(np.clip(target * self.cross_factor, -0.95, 0.95))
            for index in range(len(work)):
                work[index] = _adjust_cross_correlation(
                    work[index],
                    channel_a,
                    channel_b,
                    self.source_corr,
                    self.target_corr,
                )
        if self.acf_noise_scale > 0:
            rng = np.random.default_rng(seed + 50)
            for channel in self.channels:
                source_acf = float(
                    np.mean([_acf1(work[index, :, channel]) for index in range(len(work))])
                )
                target_acf = float(
                    np.mean([_acf1(real_history[index, :, channel]) for index in range(len(real_history))])
                )
                ratio = max(source_acf / max(target_acf, 1e-6) - 1.0, 0.0)
                self.relative_noise[channel] = float(np.sqrt(ratio) * self.acf_noise_scale)
                sigma = float(np.std(work[:, :, channel])) * self.relative_noise[channel]
                work[:, :, channel] += rng.normal(0.0, sigma, size=work[:, :, channel].shape)
        for channel in self.rank_channels:
            channel_mask = mask[:, :, channel] > 0
            self.rank_targets[channel] = real_history[:, :, channel][channel_mask].astype(np.float32)
            recent_mask = mask[-1, :, channel] > 0
            recent = real_history[-1, :, channel][recent_mask]
            if len(recent) > 1:
                self.rank_stats[channel] = (float(np.mean(recent)), float(np.std(recent)))
        self.base.fit(real_history, work, mask=mask, stats_window=stats_window, seed=seed)
        return self

    def apply(self, window: np.ndarray, mask: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
        out = window.copy()
        if self.cross_channels is not None:
            channel_a, channel_b = self.cross_channels
            out = _adjust_cross_correlation(
                out, channel_a, channel_b, self.source_corr, self.target_corr
            )
        if self.acf_noise_scale > 0:
            rng = np.random.default_rng(seed + 100)
            for channel in self.channels:
                sigma = float(np.std(out[:, channel])) * self.relative_noise.get(channel, 0.0)
                out[:, channel] += rng.normal(0.0, sigma, size=out.shape[0])
        out = self.base.apply(out, mask=mask, seed=seed)
        for channel, target in self.rank_targets.items():
            if len(target) < 2:
                continue
            valid = mask[:, channel] > 0 if mask is not None else np.ones(len(out), dtype=bool)
            if not valid.any():
                continue
            values = out[valid, channel]
            ranks = rankdata(values, method="average") / (len(values) + 1.0)
            mapped = np.quantile(target, ranks)
            if channel in self.rank_stats:
                target_mean, target_std = self.rank_stats[channel]
                current_mean = float(np.mean(mapped))
                current_std = float(np.std(mapped))
                if current_std > 1e-6:
                    blend_mean = current_mean + self.rank_stats_scale * (target_mean - current_mean)
                    blend_std = current_std + self.rank_stats_scale * (target_std - current_std)
                    mapped = blend_mean + (mapped - current_mean) * (blend_std / current_std)
            out[valid, channel] = mapped
        return out
