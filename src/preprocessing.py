# =============================================================================
# preprocessing.py
# Responsibility: Remove missing values, duplicates, fix data types,
#                 normalize date format
# =============================================================================

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw AAPL DataFrame.

    Steps:
        1. Drop rows with missing values
        2. Drop duplicate rows
        3. Ensure correct data types
        4. Normalize date format (set as index)
        5. Reset index for a clean sequential integer index

    Args:
        df (pd.DataFrame): Raw DataFrame from data_loader.

    Returns:
        pd.DataFrame: Fully cleaned DataFrame.
    """

    print("[Preprocessing] Starting preprocessing...")
    print(f"[Preprocessing] Input shape  : {df.shape}")

    # ── 1. Drop missing values ────────────────────────────────────────────────
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"[Preprocessing] Rows dropped (NaN)        : {before - after}")

    # ── 2. Drop duplicate rows ────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[Preprocessing] Rows dropped (duplicates) : {before - after}")

    # ── 3. Enforce correct data types ─────────────────────────────────────────
    df["Date"]      = pd.to_datetime(df["Date"])
    df["Open"]      = df["Open"].astype(float)
    df["High"]      = df["High"].astype(float)
    df["Low"]       = df["Low"].astype(float)
    df["Close"]     = df["Close"].astype(float)
    df["Adj Close"] = df["Adj Close"].astype(float)
    df["Volume"]    = df["Volume"].astype(int)

    # ── 4. Normalize date format ──────────────────────────────────────────────
    # Keep Date as a proper datetime column (not string)
    df["Date"] = df["Date"].dt.normalize()   # strips any time component

    # ── 5. Reset index ────────────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    print(f"[Preprocessing] Output shape : {df.shape}")
    print("[Preprocessing] ✅ Preprocessing complete.\n")

    return df


def get_summary(df: pd.DataFrame) -> None:
    """Print a quick statistical summary of the cleaned dataset."""
    print("\n===== Dataset Summary =====")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {df.columns.tolist()}")
    print(f"\nDate range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDescriptive stats:\n{df.describe()}")


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_data

    raw = load_data()
    clean = preprocess(raw)
    get_summary(clean)
    print(clean.head())