# =============================================================================
# evaluation.py
# Responsibility: Compute MAE, RMSE, R² and generate evaluation plots
# =============================================================================
#
# Metrics  : MAE, RMSE, R² Score
# Plots    : Actual vs Predicted, Residual Plot, Prediction Error Distribution
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label: str = "Set") -> dict:
    """
    Compute MAE, RMSE, and R² Score.

    Args:
        y_true : Actual target values.
        y_pred : Predicted target values.
        label  : Label for printing (e.g. 'Validation', 'Test').

    Returns:
        dict with keys: mae, rmse, r2
    """

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n===== {label} Metrics =====")
    print(f"  MAE  (Mean Absolute Error)       : ${mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error)   : ${rmse:.4f}")
    print(f"  R²   (Coefficient of Det.)       : {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_all_splits(model, X_train, X_val, X_test,
                        y_train, y_val, y_test) -> dict:
    """
    Evaluate model on train, validation, and test splits.

    Returns:
        dict with keys: train, val, test — each containing metrics dict
                        and predictions array.
    """

    results = {}

    for label, X, y in [
        ("Train",      X_train, y_train),
        ("Validation", X_val,   y_val),
        ("Test",       X_test,  y_test),
    ]:
        y_pred   = model.predict(X)
        metrics  = compute_metrics(y, y_pred, label=label)
        results[label.lower()] = {"metrics": metrics, "y_pred": y_pred, "y_true": y}

    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred, label: str = "Test",
                             save_path: str = None):
    """
    Line chart of Actual vs Predicted closing prices.
    """

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(range(len(y_true)), list(y_true), label="Actual",    color="#1f77b4", linewidth=1.2)
    ax.plot(range(len(y_pred)), y_pred,        label="Predicted", color="#ff7f0e", linewidth=1.2, linestyle="--")

    ax.set_title(f"Actual vs Predicted — {label} Set", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Closing Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[Evaluation] Plot saved → {save_path}")

    plt.show()
    plt.close()


def plot_residuals(y_true, y_pred, label: str = "Test",
                   save_path: str = None):
    """
    Residual plot: residuals vs predicted values + distribution histogram.
    """

    residuals = np.array(y_true) - np.array(y_pred)

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # ── Residuals vs Predicted ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.4, color="#2ca02c", s=12)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax1.set_title(f"Residuals vs Predicted — {label} Set", fontweight="bold")
    ax1.set_xlabel("Predicted Price (USD)")
    ax1.set_ylabel("Residual (USD)")
    ax1.grid(True, alpha=0.3)

    # ── Residual Distribution ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=40, color="#9467bd", edgecolor="white", alpha=0.8)
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax2.set_title(f"Residual Distribution — {label} Set", fontweight="bold")
    ax2.set_xlabel("Residual (USD)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[Evaluation] Plot saved → {save_path}")

    plt.show()
    plt.close()


def run_full_evaluation(model, X_train, X_val, X_test,
                        y_train, y_val, y_test,
                        save_plots: bool = True) -> dict:
    """
    Convenience function: compute all metrics + generate all plots.

    Args:
        model       : Trained model.
        X_train/val/test : Scaled feature arrays.
        y_train/val/test : Target arrays.
        save_plots  : If True, saves plots to visuals/ folder.

    Returns:
        dict of all metrics and predictions per split.
    """

    results = evaluate_all_splits(model, X_train, X_val, X_test,
                                  y_train, y_val, y_test)

    # ── Generate plots for Test set ────────────────────────────────────
    test_pred = results["test"]["y_pred"]
    test_true = results["test"]["y_true"]

    avp_path = "visuals/actual_vs_predicted.png" if save_plots else None
    res_path = "visuals/residual_plot.png"        if save_plots else None

    plot_actual_vs_predicted(test_true, test_pred, label="Test", save_path=avp_path)
    plot_residuals(test_true, test_pred,           label="Test", save_path=res_path)

    return results


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import joblib
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from data_loader         import load_data
    from preprocessing       import preprocess
    from feature_engineering import engineer_features
    from train_model         import train

    raw    = load_data()
    clean  = preprocess(raw)
    feat   = engineer_features(clean)

    model, scaler, splits = train(feat)
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test = splits

    results = run_full_evaluation(model,
                                  X_train_s, X_val_s, X_test_s,
                                  y_train,   y_val,   y_test,
                                  save_plots=True)