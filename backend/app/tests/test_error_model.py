"""Unit tests for ErrorModel and training pipeline (Phase P)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app import config
from app.analysis.error_model import ErrorModel, train_error_model
from app.analysis.features import FEATURE_NAMES


def test_error_model_train_and_monotonicity(tmp_path: Path):
    # Create synthetic dataset of 500 samples
    rng = np.random.RandomState(42)
    n = 600

    # is_error depends strongly on clock_ratio (lower clock ratio -> much higher error rate)
    clock_ratio = rng.uniform(0.01, 1.0, size=n)
    error_prob = np.clip(1.0 - clock_ratio, 0.05, 0.95)
    y = rng.binomial(1, error_prob)

    data = {feat: rng.randn(n) for feat in FEATURE_NAMES}
    data["clock_ratio"] = clock_ratio
    data["is_error"] = y

    df = pd.DataFrame(data)
    train_df = df.iloc[:400]
    val_df = df.iloc[400:]

    model = train_error_model(
        X_train=train_df[FEATURE_NAMES],
        y_train=train_df["is_error"],
        X_val=val_df[FEATURE_NAMES],
        y_val=val_df["is_error"],
    )

    # 1. Predictions must be between 0 and 1
    preds = model.predict(val_df)
    assert len(preds) == len(val_df)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)

    # 2. Test isotonic monotonicity: as raw probabilities increase,
    # calibrated probabilities from IsotonicRegression must be monotonic non-decreasing.
    test_raw_inputs = np.linspace(0.0, 1.0, 50)
    calibrated_curve = model.iso.predict(test_raw_inputs)
    diffs = np.diff(calibrated_curve)
    assert np.all(diffs >= -1e-7), "Isotonic calibration must be monotonic non-decreasing"

    # 3. Serialization round-trip
    save_path = tmp_path / "error_model.joblib"
    model.save(save_path)
    loaded = ErrorModel.load(save_path)

    preds_loaded = loaded.predict(val_df)
    np.testing.assert_allclose(preds, preds_loaded, atol=1e-6)


def test_error_model_fallback_behavior():
    # If invalid input is provided, predict returns fallback rate (0.08)
    rng = np.random.RandomState(0)
    X = rng.randn(100, len(FEATURE_NAMES))
    y = rng.binomial(1, 0.2, size=100)
    model = train_error_model(X, y, X, y)

    # Invalid input (None or corrupted structure)
    fallback_preds = model.predict(None)
    assert len(fallback_preds) == 1
    assert fallback_preds[0] == config.ERROR_MODEL_FALLBACK_RATE
