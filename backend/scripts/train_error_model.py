"""Train error probability model from analyzed game positions (Phase P)."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from app import config
from app.analysis.error_model import ErrorModel, train_error_model
from app.analysis.features import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Train ErrorModel on analyzed games.")
    parser.add_argument("username", type=str, nargs="?", default="the_king_crusher", help="Player username")
    parser.add_argument("--out", type=str, default=None, help="Output path for error_model.joblib")

    args = parser.parse_args()

    data_root = Path(config.DATA_ROOT)
    player_dir = data_root / "players" / args.username

    train_path = player_dir / "analysis_train.parquet"
    val_path = player_dir / "analysis_val.parquet"

    if not train_path.exists():
        test_path = player_dir / "analysis_test.parquet"
        if test_path.exists():
            print(f"analysis_train.parquet not found. Using {test_path} for demo training...")
            df = pq.read_table(test_path).to_pandas()
            split_idx = int(len(df) * 0.8)
            df_train = df.iloc[:split_idx]
            df_val = df.iloc[split_idx:]
        else:
            raise FileNotFoundError(f"Analyzed data not found at {train_path} or {test_path}")
    else:
        print(f"Loading train data from {train_path}...")
        df_train = pq.read_table(train_path).to_pandas()
        if val_path.exists():
            print(f"Loading val data from {val_path}...")
            df_val = pq.read_table(val_path).to_pandas()
        else:
            split_idx = int(len(df_train) * 0.8)
            df_val = df_train.iloc[split_idx:]
            df_train = df_train.iloc[:split_idx]

    print(f"Training set: {len(df_train)} rows | Validation set: {len(df_val)} rows")

    X_train = df_train[FEATURE_NAMES]
    y_train = df_train["is_error"]
    X_val = df_val[FEATURE_NAMES]
    y_val = df_val["is_error"]

    print("Fitting HistGradientBoostingClassifier and IsotonicRegression...")
    model = train_error_model(X_train, y_train, X_val, y_val)

    # Calculate validation metrics
    X_val_np = X_val.to_numpy(dtype=float)
    y_val_np = y_val.to_numpy(dtype=int)
    val_preds_binary = model.clf.predict(X_val_np)
    val_acc = accuracy_score(y_val_np, val_preds_binary)

    # Compute top-5 feature importances using permutation importance
    print("Computing feature importances on validation set...")
    perm_res = permutation_importance(model.clf, X_val_np, y_val_np, n_repeats=5, random_state=42)
    sorted_idx = np.argsort(perm_res.importances_mean)[::-1]

    out_path = Path(args.out) if args.out is not None else (player_dir / "models" / "v1" / "error_model.joblib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"ErrorModel successfully saved to {out_path}")

    # Predicted p_err distribution
    val_p_err = model.predict(X_val_np)
    frac_gt_10 = float(np.mean(val_p_err > 0.10))
    mean_p_err = float(np.mean(val_p_err))

    print("\n--- Training Results ---")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.1f}%)")
    print(f"Predicted p_err Distribution (Validation):")
    print(f"  Mean: {mean_p_err:.4f}")
    print(f"  Median: {float(np.median(val_p_err)):.4f}")
    print(f"  Min / Max: {float(np.min(val_p_err)):.4f} / {float(np.max(val_p_err)):.4f}")
    print(f"  Fraction > 0.10: {frac_gt_10 * 100:.1f}%")
    print("Top-5 Feature Importances:")
    for rank, idx in enumerate(sorted_idx[:5], 1):
        feat = FEATURE_NAMES[idx]
        imp = perm_res.importances_mean[idx]
        print(f"  {rank}. {feat:20s}: {imp:.4f}")


if __name__ == "__main__":
    main()
