# =============================================================================
# data_loader.py
# Responsibility: Load CSV, parse dates, filter AAPL, sort by date
# =============================================================================

import pandas as pd
import os


def load_data(filepath: str = "data/NASDAQ100_Historical_Data.csv") -> pd.DataFrame:
    """
    Load the NASDAQ-100 historical dataset, filter for Apple (AAPL),
    parse dates, and sort chronologically.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Clean DataFrame containing only AAPL rows,
                      sorted by date ascending.
    """

    # ── 1. Check file exists ──────────────────────────────────────────────────
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Make sure 'NASDAQ100_Historical_Data.csv' is inside the data/ folder."
        )

    print(f"[DataLoader] Loading dataset from: {filepath}")

    # ── 2. Load CSV ───────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"[DataLoader] Total rows loaded : {len(df):,}")
    print(f"[DataLoader] Columns           : {df.columns.tolist()}")

    # ── 3. Filter Apple (AAPL) ────────────────────────────────────────────────
    df = df[df["Ticker"] == "AAPL"].copy()
    print(f"[DataLoader] AAPL rows         : {len(df):,}")

    # ── 4. Parse dates ────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])

    # ── 5. Sort chronologically ───────────────────────────────────────────────
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"[DataLoader] Date range        : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print("[DataLoader] ✅ Data loaded successfully.\n")

    return df


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.dtypes)