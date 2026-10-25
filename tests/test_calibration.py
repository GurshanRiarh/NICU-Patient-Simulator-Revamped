import numpy as np

from neosim.simulator.calibration import DynamicPatientCalibrator, OnlinePatientCalibrator


def test_quantile_calibration_matches_target_moments():
    rng = np.random.default_rng(0)
    target = rng.normal(100.0, 10.0, size=(1, 4000, 1)).astype(np.float32)
    source = rng.normal(80.0, 15.0, size=(1, 4000, 1)).astype(np.float32)
    mask = np.ones_like(source, dtype=bool)
    calibrator = OnlinePatientCalibrator((0,)).fit(target, source, mask=mask, seed=0)
    out = calibrator.apply(source[0], mask=mask[0], seed=0)
    assert abs(float(np.mean(out)) - 100.0) < 0.5
    assert abs(float(np.std(out)) - 10.0) < 0.5


def test_recent_window_stats_are_applied():
    rng = np.random.default_rng(1)
    source = rng.normal(80.0, 5.0, size=(3, 120, 1)).astype(np.float32)
    target = rng.normal(100.0, 10.0, size=(3, 120, 1)).astype(np.float32)
    target[-1, -30:] = 120.0
    mask = np.ones_like(source, dtype=bool)
    calibrator = OnlinePatientCalibrator((0,)).fit(
        target, source, mask=mask, stats_window=30, seed=0
    )
    out = calibrator.apply(source[0], mask=mask[0], seed=0)
    assert abs(float(np.mean(out)) - 120.0) < 1.0
    assert abs(float(np.std(out)) - float(np.std(target[-1, -30:]))) < 1.0


def test_calibration_only_changes_selected_channels():
    rng = np.random.default_rng(2)
    source = rng.normal(80.0, 5.0, size=(2, 120, 3)).astype(np.float32)
    target = rng.normal(100.0, 10.0, size=(2, 120, 3)).astype(np.float32)
    mask = np.ones_like(source, dtype=bool)
    calibrator = OnlinePatientCalibrator((0,)).fit(target, source, mask=mask, seed=0)
    window = source[0].copy()
    out = calibrator.apply(window, mask=mask[0], seed=0)
    assert np.any(out[:, 0] != window[:, 0])
    assert np.allclose(out[:, 1:], window[:, 1:])


def test_dynamic_calibration_adds_cross_channel_coupling():
    rng = np.random.default_rng(4)
    real_base = rng.normal(0.0, 1.0, size=(3, 120))
    target = np.stack([real_base, -real_base + rng.normal(0, 0.2, size=real_base.shape)], axis=-1)
    target = (target * np.array([10.0, 5.0]) + np.array([100.0, 40.0])).astype(np.float32)
    source = rng.normal(0.0, 1.0, size=(3, 120, 2)).astype(np.float32)
    source = (source * np.array([10.0, 5.0]) + np.array([100.0, 40.0])).astype(np.float32)
    mask = np.ones_like(source, dtype=bool)
    calibrator = DynamicPatientCalibrator(
        (0, 1),
        cross_channels=(0, 1),
        cross_factor=1.0,
        acf_noise_scale=0.0,
    ).fit(target, source, mask=mask, seed=0)
    out = calibrator.apply(source[0], mask=mask[0], seed=0)
    corr = np.corrcoef(out[:, 0], out[:, 1])[0, 1]
    assert corr < -0.8


def test_rank_channel_matches_target_range():
    rng = np.random.default_rng(5)
    source = rng.normal(45.0, 6.0, size=(3, 120, 2)).astype(np.float32)
    target = source.copy()
    target[:, :, 1] = rng.uniform(30.0, 50.0, size=(3, 120)).astype(np.float32)
    mask = np.ones_like(source, dtype=bool)
    calibrator = DynamicPatientCalibrator(
        (0, 1),
        rank_channels=(1,),
        acf_noise_scale=0.0,
    ).fit(target, source, mask=mask, seed=0)
    out = calibrator.apply(source[0], mask=mask[0], seed=0)
    assert out[:, 1].min() >= float(target[:, :, 1].min()) - 2.0
    assert out[:, 1].max() <= float(target[:, :, 1].max()) + 2.0
