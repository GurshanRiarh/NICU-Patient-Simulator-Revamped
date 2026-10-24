import numpy as np
import pytest

from neosim.config import SIGNALS


@pytest.fixture()
def dummy_norm() -> dict:
    return {
        "channels": {
            name: {
                "mean": {"hr": 150.0, "rr": 45.0, "spo2": 95.0}[name],
                "std": {"hr": 15.0, "rr": 7.0, "spo2": 4.0}[name],
            }
            for name in SIGNALS
        },
        "static": {
            "mean": [30.0, 1.2, 2.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
            "std": [3.0, 0.4, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3],
        },
        "n_windows": 1,
    }


@pytest.fixture()
def synthetic_windows() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.zeros((4, 120, 3), dtype=np.float32)
    mask = np.ones_like(x)
    x[:, :, 0] = 150 + rng.normal(0, 5, (4, 120))
    x[:, :, 1] = 45 + rng.normal(0, 2, (4, 120))
    x[:, :, 2] = 95 + rng.normal(0, 1, (4, 120))
    return x, mask
