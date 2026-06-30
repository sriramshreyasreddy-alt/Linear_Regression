# =============================================================================
# predict.py
# Responsibility: Load trained model + scaler, run inference,
#                 predict next-day closing price
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import get_feature_list


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/linear_regression.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES    = get_feature_list()


def load_model():
    """
    Load the trained LinearRegression model and StandardScaler from disk.

    Returns:
        model  : Trained LinearRegression
        scaler : Fitted StandardScaler
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Run train_model.py first to generate the model."
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at '{SCALER_PATH}'.\n"
            "Run train_model.py first to generate the scaler."
        )

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f"[Predict] ✅ Model loaded  from: {MODEL_PATH}")
    print(f"[Predict] ✅ Scaler loaded from: {SCALER_PATH}")

    return model, scaler


def predict_next_close(row: pd.Series, model, scaler) -> float:
    """
    Predict the next trading day's closing price for a single row.

    Args:
        row    (pd.Series) : One row of feature-engineered data.
        model              : Trained LinearRegression model.
        scaler             : Fitted StandardScaler.

    Returns:
        float: Predicted next-day closing price (USD).
    """

    # Extract feature values in the correct order
    X = row[FEATURES].values.reshape(1, -1)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    return round(float(prediction), 4)


def predict_from_latest(df: pd.DataFrame) -> dict:
    """
    Use the most recent row in the feature-engineered dataset to predict
    the next trading day's closing price.

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame (output of engineer_features).

    Returns:
        dict with:
            - date        : The date of the most recent row (last known trading day)
            - last_close  : The actual closing price on that day
            - prediction  : Predicted closing price for the next trading day
            - change      : Predicted change from last close
            - change_pct  : Predicted change as a percentage
    """

    model, scaler = load_model()

    # Most recent row
    latest_row  = df.iloc[-1]
    latest_date = latest_row["Date"]
    last_close  = latest_row["Close"]

    prediction  = predict_next_close(latest_row, model, scaler)
    change      = round(prediction - last_close, 4)
    change_pct  = round((change / last_close) * 100, 4)

    result = {
        "date"       : str(latest_date.date()),
        "last_close" : round(float(last_close), 4),
        "prediction" : prediction,
        "change"     : change,
        "change_pct" : change_pct,
    }

    print(f"\n[Predict] Last known trading day : {result['date']}")
    print(f"[Predict] Last closing price     : ${result['last_close']:.2f}")
    print(f"[Predict] Predicted next close   : ${result['prediction']:.2f}")
    print(f"[Predict] Expected change        : ${result['change']:.2f}  ({result['change_pct']:.2f}%)")

    return result


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions across the entire dataset (useful for evaluation plots).

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame.

    Returns:
        pd.DataFrame with columns: Date, Actual, Predicted
    """

    model, scaler = load_model()

    X        = df[FEATURES].values
    X_scaled = scaler.transform(X)
    preds    = model.predict(X_scaled)

    result_df = pd.DataFrame({
        "Date"      : df["Date"].values,
        "Actual"    : df["Next_Close"].values,
        "Predicted" : preds,
    })

    print(f"[Predict] Batch prediction complete. {len(result_df):,} rows predicted.")

    return result_df


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader         import load_data
    from preprocessing       import preprocess
    from feature_engineering import engineer_features

    raw    = load_data()
    clean  = preprocess(raw)
    feat   = engineer_features(clean)

    # ── Single prediction (latest row) ────────────────────────────────
    result = predict_from_latest(feat)

    # ── Batch prediction ──────────────────────────────────────────────
    batch = predict_batch(feat)
    print(f"\nBatch predictions (last 5 rows):")
    print(batch.tail())