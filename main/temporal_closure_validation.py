"""
Temporal closure validation for downscaled daily TWSA.

Reviewers asked for an explicit test that the *daily* reconstruction is
internally and physically consistent, not only a comparison of monthly model
output against monthly GRACE. This module implements the requested
"temporal closure test":

    predicted daily TWSA  --(monthly mean)-->  monthly TWSA*
    monthly TWSA*  vs  original observed monthly GRACE

Because the daily target used for training is a linear interpolation of the
monthly GRACE series, the daily predictions are driven by *independent daily
predictors* (precipitation, ET, SMS, runoff, GWSA). Re-aggregating those daily
predictions back to monthly and comparing them against the ORIGINAL (non
interpolated, gap-carrying) monthly GRACE observations therefore checks whether
the downscaling preserves the monthly mass that GRACE actually measured.

Key design choices for rigour:
- The comparison is restricted to months GRACE actually observed (`TWS_JPL.xlsx`
  lists only observed months), so we never score the model against interpolated
  gap-fill values.
- Closure metrics carry moving-block bootstrap confidence intervals
  (see `stats_utils.py`), so the closure quality is reported with uncertainty.

Expected result (documented for the response): since the daily series is tied
to the monthly GRACE signal, a faithful model recovers the monthly observations
with high R2 / low RMSE and near-zero bias; residuals concentrate at the
monsoon onset/withdrawal where sub-monthly dynamics depart most from a linear
month-to-month path.
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from scipy import stats

from utils import calculate_metrics, EvaluationMetrics
from stats_utils import block_bootstrap_metric_cis, format_ci, load_prediction_csv
from plot_style import DPI, R2, pretty_model, SB_BLUE, SB_ORANGE, BAR

MONTH_MAP = {m: i + 1 for i, m in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"])}


# =============================================================================
# Reference monthly GRACE (original, observed-only, non-interpolated)
# =============================================================================

def load_observed_monthly_grace(
    tws_file: str = "../Data/TWS_JPL.xlsx",
) -> pd.DataFrame:
    """
    Load the ORIGINAL observed monthly GRACE TWSA (before interpolation).

    `TWS_JPL.xlsx` contains only months GRACE actually observed (data gaps are
    simply absent), which is exactly the reference we want for closure.

    Returns
    -------
    pd.DataFrame with columns ['Date', 'Month_Start', 'GRACE_monthly'].
    """
    tws = pd.read_excel(tws_file)
    tws = tws.copy()
    tws["Month_Num"] = tws["Month"].map(MONTH_MAP)
    tws["Date"] = pd.to_datetime(dict(year=tws["Year"], month=tws["Month_Num"], day=1))
    tws = tws.sort_values("Date").drop_duplicates(subset="Date")
    out = tws[["Date", "TWS"]].rename(columns={"TWS": "GRACE_monthly"})
    out["Month_Start"] = out["Date"].values.astype("datetime64[M]")
    return out.reset_index(drop=True)


# =============================================================================
# Core closure computation
# =============================================================================

def aggregate_daily_to_monthly(
    dates: np.ndarray,
    predicted_daily: np.ndarray,
    min_days: int = 10,
) -> pd.DataFrame:
    """
    Re-aggregate a daily predicted TWSA series to monthly means.

    Parameters
    ----------
    dates : array-like of datetime
    predicted_daily : array-like of float
    min_days : int
        Minimum number of daily samples required to accept a month's mean
        (guards against months represented by only a few days).

    Returns
    -------
    pd.DataFrame ['Month_Start', 'Predicted_monthly', 'n_days'].
    """
    df = pd.DataFrame({
        "Date": pd.to_datetime(pd.Series(dates)),
        "pred": np.asarray(predicted_daily, float).ravel(),
    })
    df["Month_Start"] = df["Date"].values.astype("datetime64[M]")
    g = df.groupby("Month_Start")["pred"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "Predicted_monthly", "count": "n_days"})
    g = g[g["n_days"] >= min_days].reset_index(drop=True)
    return g


def compute_temporal_closure(
    dates: np.ndarray,
    predicted_daily: np.ndarray,
    monthly_reference: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 20,
    min_days: int = 10,
) -> Dict[str, object]:
    """
    Full temporal closure test for one daily prediction series.

    Aggregates the daily predictions to monthly, merges with the observed
    monthly GRACE (observed months only) and computes closure metrics with
    bootstrap confidence intervals plus Pearson correlation.

    Returns a dict with the merged frame, metrics, CIs and correlation.
    """
    monthly_pred = aggregate_daily_to_monthly(dates, predicted_daily, min_days=min_days)
    merged = pd.merge(
        monthly_reference[["Month_Start", "GRACE_monthly"]],
        monthly_pred[["Month_Start", "Predicted_monthly", "n_days"]],
        on="Month_Start", how="inner",
    ).sort_values("Month_Start").reset_index(drop=True)

    if len(merged) < 8:
        return {"merged": merged, "n_months": len(merged), "metrics": None,
                "cis": None, "pearson_r": np.nan, "pearson_p": np.nan}

    y_obs = merged["GRACE_monthly"].values
    y_pred = merged["Predicted_monthly"].values

    metrics = calculate_metrics(y_obs, y_pred)
    cis = block_bootstrap_metric_cis(y_obs, y_pred, n_boot=n_boot, seed=seed)
    r, p = stats.pearsonr(y_obs, y_pred)

    merged["closure_residual"] = merged["GRACE_monthly"] - merged["Predicted_monthly"]
    return {
        "merged": merged,
        "n_months": len(merged),
        "metrics": metrics,
        "cis": cis,
        "pearson_r": float(r),
        "pearson_p": float(p),
    }


# =============================================================================
# Plots
# =============================================================================

def plot_closure(
    result: Dict[str, object],
    model_name: str,
    output_dir: str,
    prefix: str = "temporal_closure",
) -> Optional[str]:
    """Time series + scatter + residual panel for one model's closure test."""
    merged = result["merged"]
    if merged is None or len(merged) < 8:
        print(f"[{model_name}] Not enough overlapping months for closure plot.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe = model_name.replace("+", "_").replace(" ", "_")
    disp = pretty_model(model_name)
    metrics: EvaluationMetrics = result["metrics"]
    r = result["pearson_r"]

    fig = plt.figure(figsize=(15, 5))

    # 1. Time series
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(merged["Month_Start"], merged["GRACE_monthly"], "o-", ms=3,
             label="Original monthly GRACE", color=SB_BLUE, alpha=0.9)
    ax1.plot(merged["Month_Start"], merged["Predicted_monthly"], "s--", ms=3,
             label="Daily prediction re-aggregated", color=SB_ORANGE, alpha=0.9)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("TWSA (mm)")
    ax1.set_title(f"Temporal closure: monthly re-aggregation\n[{disp}]")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Scatter with 1:1
    ax2 = fig.add_subplot(1, 3, 2)
    lo = min(merged["GRACE_monthly"].min(), merged["Predicted_monthly"].min())
    hi = max(merged["GRACE_monthly"].max(), merged["Predicted_monthly"].max())
    ax2.scatter(merged["GRACE_monthly"], merged["Predicted_monthly"], s=14, alpha=0.6, color=SB_BLUE)
    ax2.plot([lo, hi], [lo, hi], "--", color="#555555", label="1:1")
    ax2.set_xlabel("Original monthly GRACE (mm)")
    ax2.set_ylabel("Re-aggregated daily prediction (mm)")
    ax2.set_title(
        f"{R2}={metrics.r2:.3f}  RMSE={metrics.rmse:.2f} mm\n"
        f"r={r:.3f}  PBIAS={metrics.pbias:.1f}%  (n={result['n_months']} months)"
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Closure residuals
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(merged["Month_Start"], merged["closure_residual"], color="gray", alpha=0.8)
    ax3.axhline(0, color="red", ls="--")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Closure residual (observed - predicted, mm)")
    ax3.set_title("Monthly closure residuals")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_{safe}.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_closure_metric_summary(
    summary: pd.DataFrame, output_dir: str, prefix: str = "temporal_closure"
) -> Optional[str]:
    """Bar chart of closure R2 and RMSE (with CI whiskers) across models."""
    if summary.empty:
        return None
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    labels = [pretty_model(m) for m in summary["Model"].values]
    x = np.arange(len(labels))

    r2 = summary["R2"].values
    r2_lo = summary["R2_lower"].values
    r2_hi = summary["R2_upper"].values
    axes[0].bar(x, r2, color=BAR, alpha=0.95,
                yerr=[r2 - r2_lo, r2_hi - r2], capsize=4, ecolor="#0b0b0b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel(f"Closure {R2}")
    axes[0].set_title(f"Temporal closure {R2} (95% block-bootstrap CI)")
    axes[0].grid(True, alpha=0.3, axis="y")

    rmse = summary["RMSE"].values
    rmse_lo = summary["RMSE_lower"].values
    rmse_hi = summary["RMSE_upper"].values
    axes[1].bar(x, rmse, color=BAR, alpha=0.95,
                yerr=[rmse - rmse_lo, rmse_hi - rmse], capsize=4, ecolor="#0b0b0b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("Closure RMSE (mm)")
    axes[1].set_title("Temporal closure RMSE (95% block-bootstrap CI)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Temporal closure test: daily predictions re-aggregated to monthly "
                 "vs original monthly GRACE", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_summary.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


# =============================================================================
# Batch driver over a holdout directory
# =============================================================================

def run_temporal_closure_validation(
    predictions_dir: str = "../Results/figures/temporal_holdout",
    tws_file: str = "../Data/TWS_JPL.xlsx",
    output_dir: Optional[str] = None,
    pattern: str = "*_full_predictions_*.csv",
    use_split: Optional[str] = None,
    n_boot: int = 2000,
    seed: int = 20,
    min_days: int = 10,
) -> pd.DataFrame:
    """
    Run the temporal closure test for every model prediction CSV in a directory.

    Parameters
    ----------
    predictions_dir : str
        Directory holding `*_full_predictions_<model>.csv`.
    tws_file : str
        Original monthly GRACE file (observed months only).
    output_dir : str, optional
        Where to write figures / summary (defaults to
        `<predictions_dir>/temporal_closure`).
    use_split : {'Train','Test',None}
        Restrict the daily series before aggregation. None uses all days (the
        most complete monthly coverage; recommended for closure).
    n_boot, seed, min_days : see `compute_temporal_closure`.

    Returns
    -------
    pd.DataFrame
        One row per model with closure metrics, CIs and correlation. Also
        written to `<output_dir>/compute_temporal_closure_summary.csv`.
    """
    if output_dir is None:
        output_dir = os.path.join(predictions_dir, "temporal_closure")
    os.makedirs(output_dir, exist_ok=True)

    monthly_ref = load_observed_monthly_grace(tws_file)
    files = sorted(glob.glob(os.path.join(predictions_dir, pattern)))
    if not files:
        print(f"No prediction files matching {pattern} in {predictions_dir}")
        return pd.DataFrame()

    rows = []
    for f in files:
        base = os.path.basename(f)
        # Extract model name from '<prefix>_full_predictions_<model>.csv'
        model = base.split("full_predictions_")[-1].replace(".csv", "")
        model = model.replace("_", "+") if "Attention" in model else model
        df = load_prediction_csv(f, split=use_split)
        if df.empty:
            continue

        result = compute_temporal_closure(
            df["Date"].values, df["Predicted_TWSA"].values,
            monthly_ref, n_boot=n_boot, seed=seed, min_days=min_days,
        )
        if result["metrics"] is None:
            print(f"[{model}] insufficient overlap; skipped.")
            continue

        plot_closure(result, model, output_dir)

        m: EvaluationMetrics = result["metrics"]
        cis = result["cis"]
        rows.append({
            "Model": model,
            "n_months": result["n_months"],
            "MAE": m.mae, "MAE_lower": cis["MAE"]["lower"], "MAE_upper": cis["MAE"]["upper"],
            "RMSE": m.rmse, "RMSE_lower": cis["RMSE"]["lower"], "RMSE_upper": cis["RMSE"]["upper"],
            "R2": m.r2, "R2_lower": cis["R2"]["lower"], "R2_upper": cis["R2"]["upper"],
            "NSE": m.nse, "NSE_lower": cis["NSE"]["lower"], "NSE_upper": cis["NSE"]["upper"],
            "PBIAS": m.pbias, "PBIAS_lower": cis["PBIAS"]["lower"], "PBIAS_upper": cis["PBIAS"]["upper"],
            "Pearson_r": result["pearson_r"], "Pearson_p": result["pearson_p"],
        })

        print(f"[{model}] closure over {result['n_months']} months: "
              f"R2={m.r2:.3f} {format_ci(cis)['R2']}, "
              f"RMSE={m.rmse:.2f} mm, r={result['pearson_r']:.3f}, "
              f"PBIAS={m.pbias:.1f}%")

    summary = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    if not summary.empty:
        summary.to_csv(os.path.join(output_dir, "temporal_closure_summary.csv"), index=False)
        plot_closure_metric_summary(summary, output_dir)
        print(f"\nSaved closure summary and figures to {output_dir}")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Temporal closure validation of daily TWSA reconstructions."
    )
    parser.add_argument("--predictions-dir", default="../Results/figures/temporal_holdout")
    parser.add_argument("--tws-file", default="../Data/TWS_JPL.xlsx")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default=None, choices=[None, "Train", "Test"],
                        help="Restrict daily series before aggregation (default: all days).")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()

    summary = run_temporal_closure_validation(
        predictions_dir=args.predictions_dir,
        tws_file=args.tws_file,
        output_dir=args.output_dir,
        use_split=args.split,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    if not summary.empty:
        print("\n===== Temporal closure summary =====")
        cols = ["Model", "n_months", "R2", "RMSE", "PBIAS", "Pearson_r"]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
