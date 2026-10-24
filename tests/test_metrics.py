import numpy as np

from neosim.evaluation.metrics import WindowSet, evaluate_pair


def _set(offset: float, n: int = 30) -> WindowSet:
    rng = np.random.default_rng(0)
    x = np.zeros((n, 120, 3), dtype=np.float32)
    x[:, :, 0] = 150 + offset + rng.normal(0, 4, (n, 120))
    x[:, :, 1] = 45 + offset + rng.normal(0, 2, (n, 120))
    x[:, :, 2] = 95 - offset + rng.normal(0, 1, (n, 120))
    return WindowSet(x, np.ones_like(x))


def test_evaluate_pair_returns_expected_sections():
    report = evaluate_pair(_set(0.0), _set(2.0), "real", (0, 1, 2))
    assert report["n_real"] == 30
    assert "distribution" in report["hr"]
    assert 0 <= report["discriminative_auc"]["auc"] <= 1
    assert np.isfinite(report["distance_to_closest"]["dcr_mean"])
    assert "real" in report["bradycardia"]
