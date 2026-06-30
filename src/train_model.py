# =============================================================================
# train_model.py
# Responsibility: Chronological train/val/test split, train Linear Regression,
#                 save model as models/linear_regression.pkl
# =============================================================================
#
# Split (PRD Section - Phase 5):
#   Training   : 2010 – 2022
#   Validation : 2023
#   Testing    : 2024 onwards
#
# IMPORTANT: Never randomly shuffle stock market data (time series).
# =============================================================================

import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader        import load_data
from preprocessing      import preprocess
from feature_engineering import engineer_features, get_feature_list


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/linear_regression.pkl"
SCALER_PATH = "models/scaler.pkl"

TRAIN_START = "2010-01-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"

FEATURES = get_feature_list()
TARGET   = "Next_Close"


def split_data(df: pd.DataFrame):
    """
    Perform chronological train / validation / test split.
    Never shuffle — this is time-series data.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    train = df[(df["Date"] >= TRAIN_START) & (df["Date"] <= TRAIN_END)]
    val   = df[(df["Date"] >= VAL_START)   & (df["Date"] <= VAL_END)]
    test  = df[ df["Date"] >= TEST_START]

    print(f"[TrainModel] Train set  : {len(train):,} rows  ({TRAIN_START} → {TRAIN_END})")
    print(f"[TrainModel] Val set    : {len(val):,}  rows  ({VAL_START} → {VAL_END})")
    print(f"[TrainModel] Test set   : {len(test):,}  rows  ({TEST_START} → present)")

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val   = val[FEATURES]
    y_val   = val[TARGET]

    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train(df: pd.DataFrame):
    """
    Full training pipeline:
        1. Split data chronologically
        2. Scale features
        3. Fit Linear Regression
        4. Save model and scaler to disk

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame.

    Returns:
        model    : Trained LinearRegression model
        scaler   : Fitted StandardScaler
        splits   : (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    print("\n[TrainModel] Starting training pipeline...")

    # ── 1. Split ──────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # ── 2. Scale features ─────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    print(f"[TrainModel] Features scaled with StandardScaler.")

    # ── 3. Train Linear Regression ────────────────────────────────────────────
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print(f"[TrainModel] Model trained. Intercept: {model.intercept_:.4f}")

    # ── 4. Save model and scaler ──────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[TrainModel] ✅ Model  saved → {MODEL_PATH}")
    print(f"[TrainModel] ✅ Scaler saved → {SCALER_PATH}\n")

    splits = (X_train_scaled, X_val_scaled, X_test_scaled,
              y_train, y_val, y_test)

    return model, scaler, splits


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw    = load_data()
    clean  = preprocess(raw)
    feat   = engineer_features(clean)

    model, scaler, splits = train(feat)

    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test = splits

    # Quick sanity check — score on each split
    print(f"Train R²      : {model.score(X_train_s, y_train):.4f}")
    print(f"Validation R² : {model.score(X_val_s,   y_val):.4f}")
    print(f"Test R²       : {model.score(X_test_s,  y_test):.4f}")