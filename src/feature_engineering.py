# =============================================================================
# feature_engineering.py
# Responsibility: Generate all predictive features from the cleaned dataset
# =============================================================================
#
# Features engineered (as per PRD Section 7):
#   ✔ Daily Return          → (Close - Open) / Open
#   ✔ Daily Range           → High - Low
#   ✔ Moving Average 5      → MA5
#   ✔ Moving Average 10     → MA10
#   ✔ Moving Average 20     → MA20
#   ✔ Rolling Volatility    → Rolling Std (20-day)
#   ✔ Momentum              → Close - Close.shift(5)
#   ✔ Previous Close        → Lag 1
#   ✔ Previous Volume       → Lag 1
#   ✔ Previous Return       → Lag 1
#
# Target Variable (PRD Section 6):
#   ✔ Next_Close            → Close.shift(-1)
# =============================================================================

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all features and the target variable.

    Args:
        df (pd.DataFrame): Cleaned DataFrame from preprocessing.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame with target column,
                      NaN rows (from rolling/lag ops) dropped.
    """

    print("[FeatureEngineering] Generating features...")

    df = df.copy()

    # ── 1. Daily Return ───────────────────────────────────────────────────────
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]

    # ── 2. Daily Range ────────────────────────────────────────────────────────
    df["Daily_Range"] = df["High"] - df["Low"]

    # ── 3. Moving Averages ────────────────────────────────────────────────────
    df["MA5"]  = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()

    # ── 4. Rolling Volatility (20-day std of Close) ───────────────────────────
    df["Rolling_Volatility"] = df["Close"].rolling(window=20).std()

    # ── 5. Momentum (Close vs 5 days ago) ────────────────────────────────────
    df["Momentum"] = df["Close"] - df["Close"].shift(5)

    # ── 6. Lag Features ───────────────────────────────────────────────────────
    df["Prev_Close"]  = df["Close"].shift(1)
    df["Prev_Volume"] = df["Volume"].shift(1)
    df["Prev_Return"] = df["Daily_Return"].shift(1)

    # ── 7. Target Variable: Next Day Closing Price ────────────────────────────
    df["Next_Close"] = df["Close"].shift(-1)

    # ── 8. Drop NaN rows created by rolling/shift operations ─────────────────
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"[FeatureEngineering] Rows dropped (NaN from rolling/lag) : {before - after}")
    print(f"[FeatureEngineering] Final dataset shape                  : {df.shape}")
    print("[FeatureEngineering] ✅ Feature engineering complete.\n")

    return df


def get_feature_list() -> list:
    """
    Returns the list of feature column names used for model training.
    Keeps this in one place so train_model.py and app.py stay in sync.
    """
    return [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Daily_Return",
        "Daily_Range",
        "MA5",
        "MA10",
        "MA20",
        "Rolling_Volatility",
        "Momentum",
        "Prev_Close",
        "Prev_Volume",
        "Prev_Return",
    ]


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess

    raw   = load_data()
    clean = preprocess(raw)
    feat  = engineer_features(clean)

    print("Feature columns:")
    for col in feat.columns:
        print(f"  {col}")

    print(f"\nFirst row:\n{feat.iloc[0]}")


    