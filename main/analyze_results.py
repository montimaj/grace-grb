"""
Post-hoc statistical analysis of downscaling results.

This driver turns the already-saved predictions into the inferential outputs
reviewers requested, WITHOUT retraining:

  * per-model test metrics with moving-block bootstrap 95% confidence intervals
    (MAE, RMSE, R2, NSE, PBIAS);
  * a pairwise Diebold-Mariano p-value matrix across models;
  * a significance-ranked table testing whether the best model is *significantly*
    better than the others.

It also provides a leakage diagnostic that DOES train models, contrasting a
naive shuffled KFold with contiguous-block and purged+embargoed CV to quantify
the optimism that temporal autocorrelation injects into the random split.

Usage
-----
    # Confidence intervals + model-comparison significance (no training)
    python analyze_results.py --holdout-dir ../Results/figures/temporal_holdout

    # Leakage diagnostic (trains a few models across CV schemes)
    python analyze_results.py --leakage --models randomforest xgboost
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from models import get_model, set_seed
from utils import (
    load_and_preprocess_data, create_lagged_features, prepare_features_target,
)
from stats_utils import (
    block_bootstrap_metric_cis, metric_ci_dataframe,
    model_comparison_matrix, rank_models_with_significance,
    pairwise_dm_comparison_table,
    paired_bootstrap_metric_diff, compare_cv_leakage, load_prediction_dir,
)
from plot_style import DPI, R2, R2_BOLD, pretty_model, pretty_scheme, BAR, BAR_3


def load_holdout_predictions(
    holdout_dir: str, split: str = "Test", pattern: str = "*_full_predictions_*.csv"
) -> Dict[str, pd.DataFrame]:
    """{model: test-split DataFrame} from a holdout directory (see stats_utils)."""
    return load_prediction_dir(holdout_dir, pattern=pattern, split=split)


# =============================================================================
# 1. Metrics with confidence intervals
# =============================================================================

def _consistent_truth(preds):
    """
    Ground truth shared by every model, with a check that it really is shared.

    These functions previously took Actual_TWSA from whichever prediction CSV
    sorted first. If two models were run against different data versions the
    mismatch would pass silently and every comparison would be against the wrong
    reference.
    """
    import numpy as _np
    ref = None
    for name, d in preds.items():
        y = d["Actual_TWSA"].to_numpy()
        if ref is None:
            ref, ref_name = y, name
        elif not _np.allclose(ref, y, equal_nan=True):
            raise ValueError(
                f"ground truth disagrees between {ref_name} and {name}; the "
                f"prediction CSVs come from different runs - regenerate them.")
    return ref


def metrics_with_cis(
    preds: Dict[str, pd.DataFrame], n_boot: int = 2000, seed: int = 20
) -> pd.DataFrame:
    """Per-model test metrics with block-bootstrap 95% CIs, tidy long form."""
    frames = []
    for model, df in preds.items():
        cis = block_bootstrap_metric_cis(
            df["Actual_TWSA"].values, df["Predicted_TWSA"].values,
            n_boot=n_boot, seed=seed,
        )
        frames.append(metric_ci_dataframe(cis, model_name=model))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[["Model", "Metric", "point", "lower", "upper", "std", "block_size", "n_boot"]]


def plot_metrics_with_cis(
    ci_long: pd.DataFrame, output_dir: str, prefix: str = ""
) -> Optional[str]:
    """Bar charts of R2, NSE, RMSE, MAE with 95% CI whiskers across models."""
    if ci_long.empty:
        return None
    os.makedirs(output_dir, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    panel_metrics = ["R2", "NSE", "RMSE", "MAE"]
    metric_label = {"R2": R2, "NSE": "NSE", "RMSE": "RMSE", "MAE": "MAE"}
    # Bold-weight labels for the (bold) panel titles, so R2 does not render thin.
    metric_label_bold = {"R2": R2_BOLD, "NSE": "NSE", "RMSE": "RMSE", "MAE": "MAE"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, metric in zip(axes, panel_metrics):
        sub = ci_long[ci_long["Metric"] == metric]
        labels = [pretty_model(m) for m in sub["Model"].values]
        pts = sub["point"].values
        lo = sub["lower"].values
        hi = sub["upper"].values
        x = np.arange(len(labels))
        ax.bar(x, pts, color=BAR, alpha=0.9,
               yerr=[pts - lo, hi - pts], capsize=4, ecolor="#0b0b0b")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        unit = " (mm)" if metric in ("RMSE", "MAE") else ""
        ax.set_ylabel(f"{metric_label[metric]}{unit}")
        ax.set_title(f"{metric_label_bold[metric]} with 95% block-bootstrap CI", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Test-set performance with bootstrap confidence intervals",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{pfx}metrics_with_ci.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


# =============================================================================
# 2. Statistical comparison among models
# =============================================================================

def model_significance(
    preds: Dict[str, pd.DataFrame], loss: str = "squared", alpha: float = 0.05
) -> Dict[str, pd.DataFrame]:
    """
    DM pairwise p-value matrix and significance-ranked table on the common
    overlapping test dates (aligned across models).
    """
    # Align all models on their common test dates (random holdout differs per model
    # only if seeds differ; here splits are shared, but we intersect to be safe).
    common = None
    for df in preds.values():
        s = set(df["Date"])
        common = s if common is None else (common & s)
    common = sorted(common)
    aligned = {}
    y_true = None
    for model, df in preds.items():
        d = df[df["Date"].isin(common)].sort_values("Date")
        aligned[model] = d["Predicted_TWSA"].values
        if y_true is None:
            y_true = d["Actual_TWSA"].values

    dm_matrix = model_comparison_matrix(y_true, aligned, loss=loss)
    ranked = rank_models_with_significance(y_true, aligned, loss=loss, alpha=alpha)
    comparison = pairwise_dm_comparison_table(y_true, aligned, loss=loss, alpha=alpha)
    return {"dm_matrix": dm_matrix, "ranked": ranked,
            "comparison": comparison, "n_common": len(common)}


def best_vs_rest_rmse_diff(
    preds: Dict[str, pd.DataFrame], n_boot: int = 2000, seed: int = 20
) -> pd.DataFrame:
    """
    Paired block-bootstrap CI/p-value for the RMSE difference between the best
    (lowest-RMSE) model and every other model. Supports 'model X is best' claims.
    """
    from utils import calculate_metrics
    common = None
    for df in preds.values():
        s = set(df["Date"])
        common = s if common is None else (common & s)
    common = sorted(common)
    aligned = {}
    y_true = None
    for model, df in preds.items():
        d = df[df["Date"].isin(common)].sort_values("Date")
        aligned[model] = d["Predicted_TWSA"].values
        if y_true is None:
            y_true = d["Actual_TWSA"].values

    rmse = {m: calculate_metrics(y_true, p).rmse for m, p in aligned.items()}
    best = min(rmse, key=rmse.get)
    rows = []
    for m in sorted(aligned, key=lambda k: rmse[k]):
        if m == best:
            continue
        res = paired_bootstrap_metric_diff(
            y_true, aligned[best], aligned[m], metric="RMSE",
            n_boot=n_boot, seed=seed,
        )
        rows.append({
            "Best_model": best, "Other_model": m,
            "RMSE_best": rmse[best], "RMSE_other": rmse[m],
            "RMSE_diff(best-other)": res["diff"],
            "CI_lower": res["lower"], "CI_upper": res["upper"],
            "p_value": res["p_value"],
            "significant_at_0.05": bool(np.isfinite(res["p_value"]) and res["p_value"] < 0.05),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 3. Leakage diagnostic (trains models)
# =============================================================================

def run_leakage_diagnostic(
    predictor_file: str = "../Data/All_Data.xlsx",
    tws_file: str = "../Data/TWS_GRACE_GEE.csv",
    models: Optional[List[str]] = None,
    predictors: Optional[List[str]] = None,
    lags: int = 7,
    n_splits: int = 5,
    embargo: int = 7,
    seed: int = 20,
    output_dir: str = "../Results/figures/leakage_diagnostic",
) -> pd.DataFrame:
    """
    Contrast shuffled-random, blocked and purged+embargoed CV to quantify the
    optimism of the random split under temporal autocorrelation.

    `embargo` defaults to the lag window so purged folds cannot share
    lag-feature information across the train/test boundary.
    """
    os.makedirs(output_dir, exist_ok=True)
    if predictors is None:
        predictors = ["SMS", "ET", "rainfall", "runoff", "GWSA"]
    if models is None:
        models = ["randomforest", "xgboost"]

    set_seed(seed)
    merged = load_and_preprocess_data(predictor_file, tws_file, predictors)
    lagged = create_lagged_features(merged.copy(), predictors, lags=lags)
    X, y, feature_names = prepare_features_target(lagged, predictors, "TWS", lags)
    print(f"Leakage diagnostic on X={X.shape}, embargo={embargo}, lags={lags}")

    all_frames = []
    for model_name in models:
        print(f"\n--- {model_name}: random vs blocked vs purged CV ---")
        table = compare_cv_leakage(
            X, y,
            model_factory=lambda mn=model_name: get_model(mn, seed=seed),
            n_splits=n_splits, embargo=embargo, seed=seed,
        )
        table.insert(0, "Model", model_name)
        all_frames.append(table)
        # Print the R2 optimism for a quick read.
        piv = table[table["Metric"] == "R2"].set_index("Scheme")["Mean"]
        if {"random_shuffled", "purged_embargo"}.issubset(piv.index):
            gap = piv["random_shuffled"] - piv["purged_embargo"]
            print(f"  R2: random={piv['random_shuffled']:.3f}  "
                  f"blocked={piv.get('blocked', float('nan')):.3f}  "
                  f"purged={piv['purged_embargo']:.3f}  "
                  f"(optimism = {gap:.3f})")

    out = pd.concat(all_frames, ignore_index=True)
    out.to_csv(os.path.join(output_dir, "cv_leakage_comparison.csv"), index=False)

    # Figure: R-squared by scheme per model.
    r2 = out[out["Metric"] == "R2"]
    models_u = out["Model"].unique()
    schemes = ["random_shuffled", "blocked", "purged_embargo"]
    colors = {"random_shuffled": BAR_3[0], "blocked": BAR_3[1], "purged_embargo": BAR_3[2]}
    width = 0.25
    x = np.arange(len(models_u))

    # Two panels: R2 (correlation, partly survives) and NSE (efficiency, collapses).
    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(models_u) + 7, 5), sharex=True)
    panel_specs = [("R2", f"Cross-validated {R2}"), ("NSE", "Cross-validated NSE")]
    for ax, (metric, ylabel) in zip(axes, panel_specs):
        sub = out[out["Metric"] == metric]
        for i, sch in enumerate(schemes):
            means = [sub[(sub["Model"] == m) & (sub["Scheme"] == sch)]["Mean"].values[0]
                     if not sub[(sub["Model"] == m) & (sub["Scheme"] == sch)].empty else np.nan
                     for m in models_u]
            stds = [sub[(sub["Model"] == m) & (sub["Scheme"] == sch)]["Std"].values[0]
                    if not sub[(sub["Model"] == m) & (sub["Scheme"] == sch)].empty else 0
                    for m in models_u]
            ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
                   label=pretty_scheme(sch), color=colors[sch], alpha=0.9, ecolor="#0b0b0b")
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_model(m) for m in models_u])
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="#0b0b0b", lw=0.8)
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_title("Correlation partly survives", fontsize=11)
    axes[1].set_title("Efficiency collapses (NSE < 0)", fontsize=11)
    axes[0].legend(title="Cross-validation scheme", fontsize=9)
    fig.suptitle("Random-split optimism from temporal autocorrelation: shuffled vs blocked vs purged + embargo CV",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_leakage_comparison.png"), dpi=DPI)
    plt.close()
    print(f"\nSaved leakage diagnostic to {output_dir}")
    return out


# =============================================================================
# Driver
# =============================================================================

def analyze_holdout(
    holdout_dir: str, split: str = "Test", n_boot: int = 2000, seed: int = 20,
    output_dir: Optional[str] = None,
) -> None:
    """Full prediction-based analysis for one holdout directory."""
    if output_dir is None:
        output_dir = os.path.join(holdout_dir, "statistics")
    os.makedirs(output_dir, exist_ok=True)

    preds = load_holdout_predictions(holdout_dir, split=split)
    if not preds:
        print(f"No predictions found in {holdout_dir}")
        return
    print(f"Loaded {len(preds)} models from {holdout_dir} (split={split})")

    ci_long = metrics_with_cis(preds, n_boot=n_boot, seed=seed)
    ci_long.to_csv(os.path.join(output_dir, "metrics_with_ci.csv"), index=False)
    plot_metrics_with_cis(ci_long, output_dir)

    sig = model_significance(preds)
    sig["dm_matrix"].to_csv(os.path.join(output_dir, "dm_pairwise_pvalues.csv"))
    sig["ranked"].to_csv(os.path.join(output_dir, "model_ranking_significance.csv"), index=False)
    sig["comparison"].to_csv(os.path.join(output_dir, "dm_comparison_table.csv"), index=False)

    diff = best_vs_rest_rmse_diff(preds, n_boot=n_boot, seed=seed)
    diff.to_csv(os.path.join(output_dir, "best_vs_rest_rmse_diff.csv"), index=False)

    print("\n== Test metrics (point [95% CI]) ==")
    for model in preds:
        sub = ci_long[ci_long["Model"] == model]
        parts = []
        for _, r in sub.iterrows():
            parts.append(f"{r['Metric']}={r['point']:.3f}[{r['lower']:.3f},{r['upper']:.3f}]")
        print(f"  {model:18s} " + "  ".join(parts))

    print(f"\n== Model ranking with DM significance (n_common={sig['n_common']}) ==")
    print(sig["ranked"].to_string(index=False))
    print("\n== Best-vs-rest RMSE difference (paired block bootstrap) ==")
    print(diff.to_string(index=False))
    print(f"\nSaved statistics to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Post-hoc statistics for TWS downscaling results.")
    parser.add_argument("--holdout-dir", default="../Results/figures/temporal_holdout")
    parser.add_argument("--split", default="Test", choices=["Train", "Test"])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--leakage", action="store_true", help="Run the CV leakage diagnostic (trains models).")
    parser.add_argument("--models", nargs="+", default=["randomforest", "xgboost"])
    args = parser.parse_args()

    if args.leakage:
        run_leakage_diagnostic(models=args.models, seed=args.seed)
    else:
        analyze_holdout(args.holdout_dir, split=args.split, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
