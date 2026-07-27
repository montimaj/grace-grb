"""
Statistical rigor utilities for TWS downscaling analysis.

This module supplies the inferential machinery requested by reviewers:

1. Uncertainty on performance metrics
   - `block_bootstrap_metric_cis`: moving-block bootstrap confidence intervals
     for MAE, RMSE, R2, NSE and PBIAS. The moving-block scheme preserves the
     short-range temporal autocorrelation of hydrological series, so the
     intervals are not artificially narrow (as i.i.d. bootstrap would be).

2. Statistical comparison among models
   - `diebold_mariano`: Diebold-Mariano test (with the Harvey-Leybourne-Newbold
     small-sample correction and a Newey-West HAC variance) for whether two
     models have significantly different predictive accuracy.
   - `paired_bootstrap_metric_diff`: paired block-bootstrap confidence interval
     and two-sided p-value for the difference in any metric between two models.
   - `wilcoxon_abs_error`: non-parametric paired test on absolute errors.
   - `model_comparison_matrix`: pairwise DM p-value matrix across all models.

3. Leakage-aware cross-validation
   - `blocked_kfold_indices` / `purged_kfold_indices`: contiguous-block and
     purged+embargoed CV splitters that break the train/test adjacency that
     makes i.i.d. random splitting leak information in autocorrelated series.
   - `compare_cv_leakage`: runs naive random KFold, blocked KFold and
     purged+embargoed KFold for a model and reports the optimism gap.

4. Reproducible split reporting
   - `describe_split` / `repeated_cv`: exact sample sizes, date ranges and
     multi-seed repeated cross-validation summaries.

All routines are deterministic given a seed and depend only on
numpy / scipy / pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from scipy import stats

# Reuse the exact metric definitions used everywhere else in the project.
from utils import calculate_metrics, EvaluationMetrics

METRIC_NAMES = ["MAE", "RMSE", "R2", "NSE", "PBIAS"]


# =============================================================================
# Metric helpers
# =============================================================================

def _metric_vector(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return the five project metrics as a plain dict."""
    m = calculate_metrics(y_true, y_pred)
    return m.to_dict()


def estimate_block_length(x: np.ndarray, max_lag: Optional[int] = None) -> int:
    """
    Data-driven moving-block length from the integrated autocorrelation.

    Uses the first zero-crossing of the autocorrelation function to gauge the
    de-correlation time, then rounds up. Falls back to the n**(1/3) rule when
    the series is short. The block length caps at n // 2.

    Parameters
    ----------
    x : np.ndarray
        1-D series (typically the residuals or the target).
    max_lag : int, optional
        Maximum lag to inspect (default: min(n // 2, 400)).

    Returns
    -------
    int
        Block length (>= 1).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 8:
        return 1
    if max_lag is None:
        max_lag = int(min(n // 2, 400))
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return max(1, int(round(n ** (1.0 / 3.0))))
    # Autocorrelation until first zero crossing.
    first_zero = None
    for lag in range(1, max_lag + 1):
        r = np.sum(x[:-lag] * x[lag:]) / denom
        if r <= 0:
            first_zero = lag
            break
    if first_zero is None:
        first_zero = max_lag
    # De-correlation time ~ first zero crossing; block a bit longer than that.
    block = int(np.ceil(first_zero))
    block = max(block, int(round(n ** (1.0 / 3.0))))
    return int(min(max(block, 1), n // 2))


# =============================================================================
# 1. Moving-block bootstrap confidence intervals for metrics
# =============================================================================

def _moving_block_resample_indices(
    n: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """One moving-block bootstrap resample of indices of length n."""
    block_size = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n]


def block_bootstrap_metric_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    block_size: Optional[int] = None,
    ci: float = 0.95,
    seed: int = 20,
) -> Dict[str, Dict[str, float]]:
    """
    Moving-block bootstrap confidence intervals for all project metrics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Observed and predicted values, temporally ordered.
    n_boot : int
        Number of bootstrap resamples.
    block_size : int, optional
        Moving-block length. If None, estimated from the autocorrelation of the
        residuals via `estimate_block_length`.
    ci : float
        Central coverage (0.95 -> 2.5th/97.5th percentiles).
    seed : int
        RNG seed.

    Returns
    -------
    Dict[str, Dict[str, float]]
        {metric: {'point','mean','lower','upper','std','block_size','n_boot'}}
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = y_true.size
    if n != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length.")

    if block_size is None:
        block_size = estimate_block_length(y_true - y_pred)

    rng = np.random.default_rng(seed)
    point = _metric_vector(y_true, y_pred)

    boot = {k: np.empty(n_boot, dtype=float) for k in METRIC_NAMES}
    for b in range(n_boot):
        idx = _moving_block_resample_indices(n, block_size, rng)
        mv = _metric_vector(y_true[idx], y_pred[idx])
        for k in METRIC_NAMES:
            boot[k][b] = mv[k]

    alpha = (1.0 - ci) / 2.0
    out: Dict[str, Dict[str, float]] = {}
    for k in METRIC_NAMES:
        vals = boot[k]
        vals = vals[np.isfinite(vals)]
        out[k] = {
            "point": float(point[k]),
            "mean": float(np.mean(vals)) if vals.size else np.nan,
            "lower": float(np.quantile(vals, alpha)) if vals.size else np.nan,
            "upper": float(np.quantile(vals, 1 - alpha)) if vals.size else np.nan,
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            "block_size": int(block_size),
            "n_boot": int(n_boot),
        }
    return out


def metric_ci_dataframe(
    cis: Dict[str, Dict[str, float]], model_name: str = ""
) -> pd.DataFrame:
    """Tidy a `block_bootstrap_metric_cis` result into a DataFrame."""
    rows = []
    for metric, d in cis.items():
        row = {"Model": model_name, "Metric": metric}
        row.update(d)
        rows.append(row)
    return pd.DataFrame(rows)


def format_ci(cis: Dict[str, Dict[str, float]], decimals: int = 3) -> Dict[str, str]:
    """Human-readable 'point [lo, hi]' strings per metric."""
    out = {}
    for k, d in cis.items():
        out[k] = (
            f"{d['point']:.{decimals}f} "
            f"[{d['lower']:.{decimals}f}, {d['upper']:.{decimals}f}]"
        )
    return out


# =============================================================================
# 2. Statistical comparison among models
# =============================================================================

def _loss(errors: np.ndarray, kind: str) -> np.ndarray:
    if kind == "squared":
        return errors ** 2
    if kind == "absolute":
        return np.abs(errors)
    raise ValueError("loss must be 'squared' or 'absolute'")


def diebold_mariano(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    h: int = 1,
    loss: str = "squared",
    hac_lag: Optional[int] = None,
) -> Dict[str, float]:
    """
    Diebold-Mariano test for equal predictive accuracy of two models.

    Implements the Harvey-Leybourne-Newbold (1997) small-sample correction and
    a Newey-West HAC estimate of the long-run variance of the loss differential,
    which is important here because the loss series is strongly autocorrelated.

    H0: the two models have equal expected loss.
    A negative statistic favours model 1 (lower loss); positive favours model 2.

    Parameters
    ----------
    y_true : np.ndarray
        Observed values.
    pred1, pred2 : np.ndarray
        Competing predictions.
    h : int
        Forecast horizon (>=1). Sets the minimum HAC lag (h-1).
    loss : {'squared','absolute'}
        Loss function on the errors.
    hac_lag : int, optional
        Newey-West truncation lag. Defaults to max(h-1, floor(n**(1/3))).

    Returns
    -------
    dict
        {'dm_stat','p_value','mean_loss_diff','favored','n','hac_lag'}
    """
    y_true = np.asarray(y_true, float).ravel()
    e1 = y_true - np.asarray(pred1, float).ravel()
    e2 = y_true - np.asarray(pred2, float).ravel()
    d = _loss(e1, loss) - _loss(e2, loss)
    n = d.size
    if n < 8:
        return {
            "dm_stat": np.nan, "p_value": np.nan,
            "mean_loss_diff": float(np.mean(d)) if n else np.nan,
            "favored": "n/a", "n": int(n), "hac_lag": 0,
        }

    if hac_lag is None:
        hac_lag = int(max(h - 1, int(np.floor(n ** (1.0 / 3.0)))))
    dbar = float(np.mean(d))
    dd = d - dbar

    # Newey-West long-run variance of the mean.
    gamma0 = np.sum(dd * dd) / n
    var = gamma0
    for k in range(1, hac_lag + 1):
        if k >= n:
            break
        w = 1.0 - k / (hac_lag + 1.0)
        gamma_k = np.sum(dd[k:] * dd[:-k]) / n
        var += 2.0 * w * gamma_k
    if var <= 0:
        var = gamma0 if gamma0 > 0 else np.nan

    dm = dbar / np.sqrt(var / n)

    # Harvey-Leybourne-Newbold small-sample correction.
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_star = dm * corr
    p_value = 2.0 * stats.t.cdf(-abs(dm_star), df=n - 1)

    favored = "model1" if dbar > 0 else ("model2" if dbar < 0 else "tie")
    return {
        "dm_stat": float(dm_star),
        "p_value": float(p_value),
        "mean_loss_diff": dbar,
        "favored": favored,
        "n": int(n),
        "hac_lag": int(hac_lag),
    }


def wilcoxon_abs_error(
    y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray
) -> Dict[str, float]:
    """Wilcoxon signed-rank test on paired absolute errors (non-parametric)."""
    y_true = np.asarray(y_true, float).ravel()
    a1 = np.abs(y_true - np.asarray(pred1, float).ravel())
    a2 = np.abs(y_true - np.asarray(pred2, float).ravel())
    try:
        stat, p = stats.wilcoxon(a1, a2, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = np.nan, np.nan
    return {
        "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
        "p_value": float(p) if np.isfinite(p) else np.nan,
        "median_abs_err_diff": float(np.median(a1 - a2)),
    }


def paired_bootstrap_metric_diff(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    metric: str = "RMSE",
    n_boot: int = 2000,
    block_size: Optional[int] = None,
    ci: float = 0.95,
    seed: int = 20,
) -> Dict[str, float]:
    """
    Paired moving-block bootstrap for the difference (model1 - model2) in a
    metric. Uses common resample indices so the pairing (and cancellation of
    shared sampling noise) is preserved.

    Returns point difference, CI and a two-sided bootstrap p-value.
    """
    if metric not in METRIC_NAMES:
        raise ValueError(f"metric must be one of {METRIC_NAMES}")
    y_true = np.asarray(y_true, float).ravel()
    p1 = np.asarray(pred1, float).ravel()
    p2 = np.asarray(pred2, float).ravel()
    n = y_true.size
    if block_size is None:
        block_size = estimate_block_length(y_true - 0.5 * (p1 + p2))

    rng = np.random.default_rng(seed)
    point_diff = _metric_vector(y_true, p1)[metric] - _metric_vector(y_true, p2)[metric]

    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _moving_block_resample_indices(n, block_size, rng)
        m1 = _metric_vector(y_true[idx], p1[idx])[metric]
        m2 = _metric_vector(y_true[idx], p2[idx])[metric]
        diffs[b] = m1 - m2
    diffs = diffs[np.isfinite(diffs)]

    alpha = (1.0 - ci) / 2.0
    # Two-sided bootstrap p-value: proportion on the other side of zero, x2.
    if diffs.size:
        frac_below = np.mean(diffs < 0)
        frac_above = np.mean(diffs > 0)
        p_value = float(min(1.0, 2.0 * min(frac_below, frac_above)))
    else:
        p_value = np.nan
    return {
        "metric": metric,
        "diff": float(point_diff),
        "lower": float(np.quantile(diffs, alpha)) if diffs.size else np.nan,
        "upper": float(np.quantile(diffs, 1 - alpha)) if diffs.size else np.nan,
        "p_value": p_value,
        "block_size": int(block_size),
        "n_boot": int(n_boot),
    }


def model_comparison_matrix(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    loss: str = "squared",
    h: int = 1,
) -> pd.DataFrame:
    """
    Pairwise Diebold-Mariano p-value matrix across models.

    Cell (i, j) is the two-sided DM p-value for H0: model_i and model_j have
    equal accuracy. The lower-triangle also encodes which model is favoured.
    """
    names = list(predictions.keys())
    mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                mat.loc[a, b] = np.nan
                continue
            res = diebold_mariano(y_true, predictions[a], predictions[b], h=h, loss=loss)
            mat.loc[a, b] = res["p_value"]
    return mat


def rank_models_with_significance(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    reference: Optional[str] = None,
    loss: str = "squared",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Rank models by RMSE and test each against a reference (default: best RMSE)
    with the Diebold-Mariano test. Answers 'is the best model *significantly*
    better than the others?'.
    """
    names = list(predictions.keys())
    rmse = {n: calculate_metrics(y_true, predictions[n]).rmse for n in names}
    order = sorted(names, key=lambda n: rmse[n])
    if reference is None:
        reference = order[0]

    rows = []
    for n in order:
        if n == reference:
            rows.append({
                "Model": n, "RMSE": rmse[n], "vs_reference": reference,
                "dm_stat": 0.0, "p_value": np.nan, "significant": False,
            })
            continue
        dm = diebold_mariano(y_true, predictions[reference], predictions[n], loss=loss)
        rows.append({
            "Model": n, "RMSE": rmse[n], "vs_reference": reference,
            "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
            "significant": bool(np.isfinite(dm["p_value"]) and dm["p_value"] < alpha),
        })
    return pd.DataFrame(rows)


def pairwise_dm_comparison_table(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    loss: str = "squared",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Full pairwise Diebold-Mariano comparison table for a set of models.

    For every unordered pair the model with the lower RMSE is taken as the
    reference, and the table reports the RMSE difference (reference - other,
    always <= 0) and the DM p-value / significance. This is the tidy,
    reproducible source for "model A is (not) significantly better than B"
    statements, e.g. showing both ensembles against the rest.

    Returns rows sorted so the strongest reference (lowest RMSE) comes first.
    """
    names = list(predictions.keys())
    rmse = {n: calculate_metrics(y_true, predictions[n]).rmse for n in names}
    order = sorted(names, key=lambda n: rmse[n])
    rows = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            # a has the lower (or equal) RMSE -> reference.
            dm = diebold_mariano(y_true, predictions[a], predictions[b], loss=loss)
            rows.append({
                "Reference": a, "Other": b,
                "RMSE_reference": rmse[a], "RMSE_other": rmse[b],
                "dRMSE": rmse[a] - rmse[b],
                "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
                "significant": bool(np.isfinite(dm["p_value"]) and dm["p_value"] < alpha),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 3. Leakage-aware cross-validation
# =============================================================================

def blocked_kfold_indices(n: int, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Contiguous-block K-fold: each test fold is a single contiguous time block,
    with all remaining samples used for training. Unlike shuffled KFold this
    keeps test blocks temporally coherent, reducing train/test adjacency.
    """
    folds = np.array_split(np.arange(n), n_splits)
    splits = []
    for k in range(n_splits):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        splits.append((np.sort(train_idx), np.sort(test_idx)))
    return splits


def purged_kfold_indices(
    n: int, n_splits: int = 5, embargo: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Purged + embargoed contiguous K-fold (Lopez de Prado style).

    Training samples within `embargo` steps on either side of the test block are
    dropped ('purged'), so lagged-feature overlap cannot leak the test target
    into training. This is the honest analogue of the random split for
    autocorrelated, lag-featured series.
    """
    folds = np.array_split(np.arange(n), n_splits)
    splits = []
    for k in range(n_splits):
        test_idx = folds[k]
        lo, hi = test_idx[0], test_idx[-1]
        purge_lo, purge_hi = lo - embargo, hi + embargo
        train_idx = np.array(
            [i for i in range(n) if (i < purge_lo or i > purge_hi)], dtype=int
        )
        splits.append((train_idx, np.sort(test_idx)))
    return splits


def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], object],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, List[float]]:
    """Fit/predict over the given splits; return fold-wise metrics."""
    out = {k: [] for k in METRIC_NAMES}
    for train_idx, test_idx in splits:
        model = model_factory()
        model.fit(X[train_idx], y[train_idx], verbose=False)
        pred = model.predict(X[test_idx])
        mv = _metric_vector(y[test_idx], pred)
        for k in METRIC_NAMES:
            out[k].append(mv[k])
    return out


def compare_cv_leakage(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], object],
    n_splits: int = 5,
    embargo: int = 7,
    seed: int = 20,
) -> pd.DataFrame:
    """
    Quantify random-split leakage by running three CV schemes on one model:

    - `random_shuffled` : sklearn-style shuffled KFold (optimistic; leaks).
    - `blocked`         : contiguous-block KFold.
    - `purged_embargo`  : contiguous-block KFold with an embargo purge.

    A large drop from `random_shuffled` to `purged_embargo` is the size of the
    optimism induced by temporal autocorrelation. `embargo` should be >= the lag
    window used for feature construction.

    Returns a tidy DataFrame of mean +/- std per scheme and metric.
    """
    from sklearn.model_selection import KFold

    n = len(y)
    rng_kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    schemes = {
        "random_shuffled": list(rng_kf.split(np.arange(n))),
        "blocked": blocked_kfold_indices(n, n_splits),
        "purged_embargo": purged_kfold_indices(n, n_splits, embargo=embargo),
    }

    rows = []
    for scheme, splits in schemes.items():
        fold_metrics = _run_cv(X, y, model_factory, splits)
        for metric, vals in fold_metrics.items():
            vals = np.asarray(vals, float)
            rows.append({
                "Scheme": scheme,
                "Metric": metric,
                "Mean": float(np.mean(vals)),
                "Std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "n_splits": n_splits,
                "embargo": embargo if scheme == "purged_embargo" else 0,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 4. Reproducible split reporting
# =============================================================================

@dataclass
class SplitReport:
    """Exact composition of one train/test split."""
    name: str
    n_total: int
    n_train: int
    n_test: int
    test_fraction: float
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_features: int

    def to_dict(self) -> Dict:
        return asdict(self)


def describe_split(
    name: str,
    dates_train: np.ndarray,
    dates_test: np.ndarray,
    n_features: int,
) -> SplitReport:
    """Build a `SplitReport` documenting exact sizes and date ranges."""
    dtr = pd.to_datetime(pd.Series(dates_train)).sort_values()
    dte = pd.to_datetime(pd.Series(dates_test)).sort_values()
    n_tr, n_te = len(dtr), len(dte)
    total = n_tr + n_te
    fmt = "%Y-%m-%d"
    return SplitReport(
        name=name,
        n_total=total,
        n_train=n_tr,
        n_test=n_te,
        test_fraction=round(n_te / total, 4) if total else np.nan,
        train_start=dtr.iloc[0].strftime(fmt) if n_tr else "",
        train_end=dtr.iloc[-1].strftime(fmt) if n_tr else "",
        test_start=dte.iloc[0].strftime(fmt) if n_te else "",
        test_end=dte.iloc[-1].strftime(fmt) if n_te else "",
        n_features=int(n_features),
    )


def repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[int], object],
    cv_kind: str = "blocked",
    n_splits: int = 5,
    n_repeats: int = 5,
    seeds: Optional[Sequence[int]] = None,
    embargo: int = 7,
) -> pd.DataFrame:
    """
    Repeat cross-validation over multiple seeds and summarise stability.

    `model_factory` takes a seed and returns a fresh model. For `blocked` and
    `purged_embargo` the fold geometry is fixed, so repeats vary only the model
    seed (initialisation / bootstrap); for `random_shuffled` the fold geometry
    also varies with the seed.

    Returns per-metric mean, std and 95% range across all folds x repeats.
    """
    from sklearn.model_selection import KFold

    if seeds is None:
        seeds = [20 + 10 * i for i in range(n_repeats)]
    n = len(y)

    all_vals = {k: [] for k in METRIC_NAMES}
    for s in seeds:
        if cv_kind == "random_shuffled":
            splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=s).split(np.arange(n)))
        elif cv_kind == "blocked":
            splits = blocked_kfold_indices(n, n_splits)
        elif cv_kind == "purged_embargo":
            splits = purged_kfold_indices(n, n_splits, embargo=embargo)
        else:
            raise ValueError("cv_kind must be 'random_shuffled', 'blocked' or 'purged_embargo'")

        for train_idx, test_idx in splits:
            model = model_factory(s)
            model.fit(X[train_idx], y[train_idx], verbose=False)
            pred = model.predict(X[test_idx])
            mv = _metric_vector(y[test_idx], pred)
            for k in METRIC_NAMES:
                all_vals[k].append(mv[k])

    rows = []
    for k, vals in all_vals.items():
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        rows.append({
            "Metric": k,
            "Mean": float(np.mean(vals)),
            "Std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "P2.5": float(np.quantile(vals, 0.025)),
            "P97.5": float(np.quantile(vals, 0.975)),
            "n_estimates": int(vals.size),
            "cv_kind": cv_kind,
            "n_splits": n_splits,
            "n_repeats": len(seeds),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Convenience: build a full CI + comparison report from prediction CSVs
# =============================================================================

def load_prediction_csv(path: str, split: Optional[str] = "Test") -> pd.DataFrame:
    """
    Load a `*_full_predictions_*.csv` (Date, Actual_TWSA, Predicted_TWSA,
    Residual, Split). If `split` is given, filter to that split.
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    if split is not None and "Split" in df.columns:
        df = df[df["Split"] == split].copy()
    return df.sort_values("Date").reset_index(drop=True)


def emit_holdout_statistics(
    y_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: str,
    prefix: str = "",
    dates_train: Optional[np.ndarray] = None,
    dates_test: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
    n_boot: int = 2000,
    seed: int = 20,
) -> None:
    """
    Write the reviewer-requested statistical outputs for a completed holdout.

    Produces, in `output_dir`:
      - `<prefix>_split_report.csv`     : exact sizes / date ranges (if dates given)
      - `<prefix>_metrics_with_ci.csv`  : per-model metrics with 95% bootstrap CI
      - `<prefix>_dm_pairwise_pvalues.csv` : Diebold-Mariano p-value matrix
      - `<prefix>_model_ranking_significance.csv` : best-model significance table

    Intended to be called at the end of each `run_*_holdout_analysis` with the
    test actuals and each model's test predictions already in scope.
    """
    import os as _os
    _os.makedirs(output_dir, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    # Split report.
    if dates_train is not None and dates_test is not None and n_features is not None:
        rep = describe_split(prefix or "holdout", dates_train, dates_test, n_features)
        pd.DataFrame([rep.to_dict()]).to_csv(
            _os.path.join(output_dir, f"{pfx}split_report.csv"), index=False)

    # Metrics with CI (one row per model x metric).
    frames = []
    for model, pred in predictions.items():
        cis = block_bootstrap_metric_cis(y_test, pred, n_boot=n_boot, seed=seed)
        frames.append(metric_ci_dataframe(cis, model_name=model))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            _os.path.join(output_dir, f"{pfx}metrics_with_ci.csv"), index=False)

    # Model-comparison significance (needs >= 2 models).
    if len(predictions) >= 2:
        dm = model_comparison_matrix(y_test, predictions)
        dm.to_csv(_os.path.join(output_dir, f"{pfx}dm_pairwise_pvalues.csv"))
        ranked = rank_models_with_significance(y_test, predictions)
        ranked.to_csv(
            _os.path.join(output_dir, f"{pfx}model_ranking_significance.csv"), index=False)
        comparison = pairwise_dm_comparison_table(y_test, predictions)
        comparison.to_csv(
            _os.path.join(output_dir, f"{pfx}dm_comparison_table.csv"), index=False)


if __name__ == "__main__":
    # Minimal self-test on synthetic data.
    rng = np.random.default_rng(0)
    n = 500
    t = np.linspace(0, 20, n)
    truth = np.sin(t) + 0.05 * t
    good = truth + rng.normal(0, 0.2, n)
    poor = truth + rng.normal(0, 0.6, n)

    print("Block length estimate:", estimate_block_length(truth))
    cis = block_bootstrap_metric_cis(truth, good, n_boot=500, seed=1)
    print("Good model CIs:", format_ci(cis))
    dm = diebold_mariano(truth, good, poor)
    print("DM good vs poor:", dm)
    print(rank_models_with_significance(truth, {"good": good, "poor": poor}))
