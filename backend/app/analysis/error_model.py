"""Error probability model training and inference using HistGradientBoosting + IsotonicRegression (Phase P)."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from app import config
from app.analysis.features import FEATURE_NAMES


class ErrorModel:
    """Error prediction model wrapping HistGradientBoostingClassifier and IsotonicRegression."""

    def __init__(self, clf: HistGradientBoostingClassifier, iso: IsotonicRegression):
        self.clf = clf
        self.iso = iso

    def predict(self, features_input: Any) -> np.ndarray:
        """Predict calibrated error probability (p_err) for input features.

        Accepts pandas DataFrame, dict, list of dicts, or numpy 2D array.
        Returns 1D array of float probabilities clipped to [0.0, 1.0].
        Falls back to config.ERROR_MODEL_FALLBACK_RATE on failure.
        """
        fallback_rate = float(getattr(config, "ERROR_MODEL_FALLBACK_RATE", 0.08))

        try:
            if isinstance(features_input, dict):
                X = np.array([[features_input.get(f, 0.0) for f in FEATURE_NAMES]], dtype=float)
            elif isinstance(features_input, pd.DataFrame):
                # Ensure correct columns and order
                cols = [c for c in FEATURE_NAMES if c in features_input.columns]
                if len(cols) == len(FEATURE_NAMES):
                    X = features_input[FEATURE_NAMES].to_numpy(dtype=float)
                else:
                    arr = []
                    for _, row in features_input.iterrows():
                        arr.append([row.get(f, 0.0) for f in FEATURE_NAMES])
                    X = np.array(arr, dtype=float)
            elif isinstance(features_input, list):
                if len(features_input) == 0:
                    return np.array([], dtype=float)
                if isinstance(features_input[0], dict):
                    X = np.array(
                        [[d.get(f, 0.0) for f in FEATURE_NAMES] for d in features_input],
                        dtype=float,
                    )
                else:
                    X = np.array(features_input, dtype=float)
            else:
                X = np.array(features_input, dtype=float)

            if X.ndim == 1:
                X = X.reshape(1, -1)

            raw_probs = self.clf.predict_proba(X)[:, 1]
            calibrated = self.iso.predict(raw_probs)
            calibrated = np.clip(np.nan_to_num(calibrated, nan=fallback_rate), 0.0, 1.0)
            return calibrated.astype(float)
        except Exception:
            n_samples = len(features_input) if hasattr(features_input, "__len__") else 1
            return np.full(n_samples, fallback_rate, dtype=float)

    def save(self, path: Path | str) -> None:
        """Serialize model components to disk using joblib."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "iso": self.iso, "features": FEATURE_NAMES}, save_path)

    @classmethod
    def load(cls, path: Path | str) -> ErrorModel:
        """Load serialized model from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"ErrorModel file not found at: {load_path}")
        payload = joblib.load(load_path)
        return cls(clf=payload["clf"], iso=payload["iso"])


def train_error_model(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
) -> ErrorModel:
    """Train HistGradientBoostingClassifier and fit IsotonicRegression on validation set.

    X_train, X_val can be DataFrames or numpy arrays.
    y_train, y_val are binary 0/1 error labels.
    """
    if isinstance(X_train, pd.DataFrame):
        X_tr = X_train[FEATURE_NAMES].to_numpy(dtype=float)
    else:
        X_tr = np.array(X_train, dtype=float)

    if isinstance(X_val, pd.DataFrame):
        X_v = X_val[FEATURE_NAMES].to_numpy(dtype=float)
    else:
        X_v = np.array(X_val, dtype=float)

    y_tr = np.array(y_train, dtype=int)
    y_v = np.array(y_val, dtype=int)

    clf = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=200,
        learning_rate=0.05,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)

    val_raw_probs = clf.predict_proba(X_v)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_raw_probs, y_v)

    return ErrorModel(clf=clf, iso=iso)
