"""Online patient-specific calibration for generated vital-sign windows.

The calibration uses only a patient's prior real windows and the synthetic
windows generated for the same prior context.  It fits a monotonic quantile map
per channel and can optionally match the mean and standard deviation of the most
recent prior window.  No current or future observations are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
