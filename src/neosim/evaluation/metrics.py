"""Evaluation metrics for synthetic neonatal vital-sign windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHANNELS = ("hr", "rr", "spo2")


@dataclass(slots=True)
class WindowSet:
    """A collection of fixed-length windows and their channel masks."""

    x: np.ndarray  # [N, T, C]
    mask: np.ndarray  # [N, T, C]

    def values(self, channel: int) -> np.ndarray:
        x = self.x[:, :, channel]
        m = self.mask[:, :, channel] > 0
        return x[m].astype(np.float64)

    def windows(self, channel: int, min_coverage: float = 0.9) -> list[np.ndarray]:
        out = []
        for i in range(len(self.x)):
            m = self.mask[i, :, channel] > 0
            if m.mean() < min_coverage:
                continue
            values = self.x[i, :, channel].astype(np.float64)
            if np.isfinite(values).all():
                out.append(values)
        return out

    def subset(self, indices: np.ndarray) -> WindowSet:
        return WindowSet(self.x[indices], self.mask[indices])


def distribution_metrics(real: np.ndarray, synth: np.ndarray) -> dict[str, float]:
    real = real[np.isfinite(real)]
    synth = synth[np.isfinite(synth)]
    ks = ks_2samp(real, synth)
    return {
        "real_mean": float(real.mean()),
        "synth_mean": float(synth.mean()),
        "mean_diff": float(abs(real.mean() - synth.mean())),
        "real_std": float(real.std()),
        "synth_std": float(synth.std()),
        "std_diff": float(abs(real.std() - synth.std())),
        "real_q01": float(np.quantile(real, 0.01)),
        "synth_q01": float(np.quantile(synth, 0.01)),
        "real_q99": float(np.quantile(real, 0.99)),
        "synth_q99": float(np.quantile(synth, 0.99)),
        "ks": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "wasserstein": float(wasserstein_distance(real, synth)),
    }


def _masked_values(window_set: WindowSet, channel: int, indices: np.ndarray) -> np.ndarray:
    values = window_set.x[indices, :, channel]
    mask = window_set.mask[indices, :, channel] > 0
    return values[mask].astype(np.float64)


def null_ks(window_set: WindowSet, channel: int, n_splits: int = 200, seed: int = 0) -> dict[str, float]:
    """KS between random halves of the same real cohort.

    This is the finite-sample noise floor: a synthetic KS at or below this value
    is indistinguishable from comparing two real halves.
    """
    n = len(window_set.x)
    if n < 4:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_splits": 0.0}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        half = n // 2
        a = _masked_values(window_set, channel, perm[:half])
        b = _masked_values(window_set, channel, perm[half:])
        if len(a) > 1 and len(b) > 1:
            stats.append(float(ks_2samp(a, b).statistic))
    if not stats:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_splits": 0.0}
    return {
        "mean": float(np.mean(stats)),
        "ci95_low": float(np.quantile(stats, 0.025)),
        "ci95_high": float(np.quantile(stats, 0.975)),
        "n_splits": float(len(stats)),
    }


def bootstrap_ks(
    real: WindowSet,
    synth: WindowSet,
    channel: int,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Window-level bootstrap confidence interval for the observed KS."""
    if len(real.x) < 2 or len(synth.x) < 2:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_boot": 0.0}
    rng = np.random.default_rng(seed + 1000)
    stats = []
    for _ in range(n_boot):
        ri = rng.integers(0, len(real.x), len(real.x))
        si = rng.integers(0, len(synth.x), len(synth.x))
        a = _masked_values(real, channel, ri)
        b = _masked_values(synth, channel, si)
        if len(a) > 1 and len(b) > 1:
            stats.append(float(ks_2samp(a, b).statistic))
    if not stats:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_boot": 0.0}
    return {
        "mean": float(np.mean(stats)),
        "ci95_low": float(np.quantile(stats, 0.025)),
        "ci95_high": float(np.quantile(stats, 0.975)),
        "n_boot": float(len(stats)),
    }


def _acf(window: np.ndarray, lag: int) -> float:
    x = window - window.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def autocorrelation(windows: list[np.ndarray], lags: tuple[int, ...] = (1, 5, 15)) -> dict[str, float]:
    if not windows:
        return {f"acf_{lag}": float("nan") for lag in lags}
    return {f"acf_{lag}": float(np.mean([_acf(w, lag) for w in windows if len(w) > lag])) for lag in lags}


def _cross_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 10) -> tuple[float, int]:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 1e-12:
        return 0.0, 0
    best, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            value = float(np.dot(a[-lag:], b[:lag]) / denom)
        elif lag > 0:
            value = float(np.dot(a[:-lag], b[lag:]) / denom)
        else:
            value = float(np.dot(a, b) / denom)
        if abs(value) > abs(best):
            best, best_lag = value, lag
    return best, best_lag


def cross_correlation(real: WindowSet, synth: WindowSet, channel_a: int, channel_b: int) -> dict[str, float]:
    """Maximum cross-correlation and its lag for pairs present in both sets."""
    values = []
    for i in range(min(len(real.x), 300)):
        ma = real.mask[i, :, channel_a] > 0
        mb = real.mask[i, :, channel_b] > 0
        if ma.mean() < 0.95 or mb.mean() < 0.95:
            continue
        values.append(_cross_corr(real.x[i, :, channel_a], real.x[i, :, channel_b]))
    synth_values = [
        _cross_corr(row[:, channel_a], row[:, channel_b]) for row in synth.x[: min(len(synth.x), 300)]
    ]
    real_corr = float(np.mean([v[0] for v in values])) if values else float("nan")
    synth_corr = float(np.mean([v[0] for v in synth_values])) if synth_values else float("nan")
    real_lag = float(np.mean([v[1] for v in values])) if values else float("nan")
    synth_lag = float(np.mean([v[1] for v in synth_values])) if synth_values else float("nan")
    return {
        "real_peak_corr": real_corr,
        "synth_peak_corr": synth_corr,
        "real_peak_lag": real_lag,
        "synth_peak_lag": synth_lag,
    }


def event_rates(window_set: WindowSet, channel: int, threshold: float, side: str) -> dict[str, float]:
    m = window_set.mask[:, :, channel] > 0
    x = window_set.x[:, :, channel]
    if side == "below":
        hit = (x < threshold) & m
    else:
        hit = (x > threshold) & m
    minutes = float(m.sum())
    fraction = float(hit.sum() / max(minutes, 1))
    per_hour = fraction * 60.0
    return {"fraction": fraction, "minutes_per_hour": per_hour}


def _window_features(window_set: WindowSet, channels: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for i in range(len(window_set.x)):
        feats = []
        valid = True
        for c in channels:
            m = window_set.mask[i, :, c] > 0
            if m.mean() < 0.9:
                valid = False
                break
            x = window_set.x[i, m, c]
            if len(x) < 10:
                valid = False
                break
            feats.extend(
                [
                    float(np.mean(x)),
                    float(np.std(x)),
                    float(np.min(x)),
                    float(np.max(x)),
                    float(np.quantile(x, 0.05)),
                    float(np.quantile(x, 0.95)),
                    float(np.mean(np.diff(x))),
                    _acf(x, min(5, len(x) - 1)),
                ]
            )
        if valid:
            rows.append(feats)
            labels.append(i)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels)


def discriminative_auc(
    real: WindowSet, synth: WindowSet, channels: tuple[int, ...] = (0, 2)
) -> dict[str, float]:
    """AUC of a classifier trying to separate real from synthetic windows.

    Values near 0.5 indicate that the synthetic windows are indistinguishable
    from real windows under simple summary statistics.
    """
    xr, _ = _window_features(real, channels)
    xs, _ = _window_features(synth, channels)
    n = min(len(xr), len(xs), 500)
    if n < 20:
        return {"auc": float("nan"), "n": float(n)}
    x = np.concatenate([xr[:n], xs[:n]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    prob = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    return {"auc": float(roc_auc_score(y, prob)), "n": float(n)}


def _normalised_matrix(
    window_set: WindowSet, channels: tuple[int, ...], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    rows = []
    for i in range(len(window_set.x)):
        if all((window_set.mask[i, :, c] > 0).mean() >= 0.98 for c in channels):
            rows.append(((window_set.x[i][:, channels] - mean) / std).ravel())
    return np.asarray(rows, dtype=np.float64)


def distance_to_closest(
    real: WindowSet, synth: WindowSet, channels: tuple[int, ...] = (0, 2)
) -> dict[str, float]:
    """Privacy-style nearest-neighbour distance in normalised window space."""
    real_mean = np.array([real.values(c).mean() for c in channels])
    real_std = np.array([real.values(c).std() + 1e-6 for c in channels])
    r = _normalised_matrix(real, channels, real_mean, real_std)
    s = _normalised_matrix(synth, channels, real_mean, real_std)
    n = min(len(r), len(s), 300)
    if n < 20:
        return {"dcr_mean": float("nan"), "real_dcr_mean": float("nan"), "ratio": float("nan")}
    r, s = r[:n], s[:n]
    dist = np.sqrt(((s[:, None, :] - r[None, :, :]) ** 2).mean(axis=2))
    synth_min = np.min(dist, axis=1)
    np.fill_diagonal(dist, np.inf)
    real_min = np.min(dist, axis=1)
    return {
        "dcr_mean": float(synth_min.mean()),
        "real_dcr_mean": float(real_min.mean()),
        "dcr_p05": float(np.quantile(synth_min, 0.05)),
        "ratio": float(synth_min.mean() / max(real_min.mean(), 1e-6)),
    }


def evaluate_pair(
    real: WindowSet,
    synth: WindowSet,
    real_name: str,
    available_channels: tuple[int, ...],
) -> dict:
    out: dict = {"real": real_name, "n_real": len(real.x), "n_synth": len(synth.x)}
    for c in available_channels:
        name = CHANNELS[c]
        rv = real.values(c)
        sv = synth.values(c)
        distribution = distribution_metrics(rv, sv)
        distribution["ks_bootstrap"] = bootstrap_ks(real, synth, c)
        out[name] = {
            "distribution": distribution,
            "autocorrelation": {
                "real": autocorrelation(real.windows(c)),
                "synth": autocorrelation(synth.windows(c)),
            },
        }
    if 0 in available_channels and 2 in available_channels:
        out["hr_spo2_cross_correlation"] = cross_correlation(real, synth, 0, 2)
    if 0 in available_channels and 1 in available_channels:
        out["hr_rr_cross_correlation"] = cross_correlation(real, synth, 0, 1)
    if 0 in available_channels:
        out["bradycardia"] = {
            "real": event_rates(real, 0, 100.0, "below"),
            "synth": event_rates(synth, 0, 100.0, "below"),
        }
    if 2 in available_channels:
        out["desaturation"] = {
            "real": event_rates(real, 2, 85.0, "below"),
            "synth": event_rates(synth, 2, 85.0, "below"),
        }
    out["null_ks"] = {
        CHANNELS[c]: null_ks(real, c) for c in available_channels
    }
    out["discriminative_auc"] = discriminative_auc(real, synth, available_channels)
    out["distance_to_closest"] = distance_to_closest(
        real, synth, tuple(c for c in (0, 2) if c in available_channels) or available_channels
    )
    return out
