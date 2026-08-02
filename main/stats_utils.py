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

5. Trend significance
   - `mann_kendall` / `sens_slope`: rank-based trend test with the tie-corrected
     variance, and the Theil-Sen slope that belongs with it.
   - `seasonal_mann_kendall`: Hirsch-Slack seasonal variant, kept as a
     cross-check rather than as the main test (its docstring says why).
   - `mann_kendall_field`: the same test at every pixel of a (time, lat, lon)
     field, with a correction for serial dependence and a Benjamini-Hochberg
     pass over the field. A trend map without this is a picture of a slope with
     no statement about whether the slope is distinguishable from noise.
   - `benjamini_hochberg`: FDR control, with the caveat that the pixels of this
     product are nowhere near independent tests.

All routines are deterministic given a seed and depend only on
numpy / scipy / pandas.
"""

from __future__ import annotations

import os
import glob

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from scipy import stats

# Reuse the exact metric definitions used everywhere else in the project.
from utils import calculate_metrics

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


def cluster_bootstrap_metric_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 20,
) -> Dict[str, Dict[str, float]]:
    """
    Confidence intervals for SPATIALLY grouped predictions, resampling GROUPS.

    `block_bootstrap_metric_cis` below resamples contiguous blocks of a time
    series, which is right for a basin-mean series and wrong here. The gridded
    leave-one-mascon-out predictions are not a time series: they are ~83,000
    mascon-months drawn from only ~19 independent spatial units, and every pixel
    of a mascon-month shares one GRACE value. Resampling rows treats those as
    independent -- the same mistake, in space, that the moving-block bootstrap
    exists to avoid in time.

    How much it matters depends on how correlated the error is WITHIN a mascon.
    If residuals were i.i.d. the two agree (measured: identical widths). Under
    the mascon-level bias that leave-one-mascon-out actually produces -- a whole
    held-out mascon offset together -- the cluster interval came out ~1.3x wider
    in a matched synthetic test. Not an order of magnitude, but in the direction
    that matters and free to compute.

    So the resampling unit is the MASCON. Each replicate draws n_groups mascons
    with replacement and pools every sample belonging to them, which propagates
    the real constraint: skill is estimated from ~19 independent things, not from
    83,000.

    Expect wide intervals. That is the point -- they are what ~19 independent
    observations actually support, and a narrow interval here would be an
    artefact of counting pixels as evidence.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if not (y_true.size == y_pred.size == groups.size):
        raise ValueError('y_true, y_pred and groups must be the same length.')

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred, groups = y_true[ok], y_pred[ok], groups[ok]
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    rng = np.random.default_rng(seed)
    point = _metric_vector(y_true, y_pred)
    boot = {k: np.full(n_boot, np.nan) for k in METRIC_NAMES}
    for b in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        mv = _metric_vector(y_true[idx], y_pred[idx])
        for k in METRIC_NAMES:
            boot[k][b] = mv[k]

    alpha = (1.0 - ci) / 2.0
    out: Dict[str, Dict[str, float]] = {}
    for k in METRIC_NAMES:
        vals = boot[k][np.isfinite(boot[k])]
        out[k] = {
            'point': float(point[k]),
            'mean': float(np.mean(vals)) if vals.size else np.nan,
            'lower': float(np.quantile(vals, alpha)) if vals.size else np.nan,
            'upper': float(np.quantile(vals, 1 - alpha)) if vals.size else np.nan,
            'std': float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            'n_groups': int(len(uniq)),
            'n_boot': int(n_boot),
            'resampling_unit': 'group (mascon), not row',
        }
    return out


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

    # dbar = mean(loss1 - loss2). Positive means model1 has the LARGER loss, so
    # model2 is favoured. The mapping was inverted, reversing every reported
    # "which model wins" verdict while leaving the p-values correct.
    favored = "model2" if dbar > 0 else ("model1" if dbar < 0 else "tie")
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
# 5. Trend significance: Mann-Kendall, Sen's slope, serial dependence
# =============================================================================

MK_AUTOCORR_METHODS = ("hamed_rao", "tfpw", "none")

# Two-sided level at which a lag autocorrelation counts as distinguishable from
# white noise. It decides which lags enter the Hamed-Rao sum and whether
# Yue-Pilon prewhitens at all; it is NOT the level of the trend test.
ACF_ALPHA = 0.05


def _as_columns(x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Flatten a (n_time, ...) array to (n_time, n_series), keeping the map shape."""
    x = np.asarray(x, dtype="float64")
    if x.ndim == 1:
        return x[:, None], ()
    return x.reshape(x.shape[0], -1), x.shape[1:]


def _tie_term(x: np.ndarray) -> np.ndarray:
    """
    Sum of t(t-1)(2t+5) over tied groups, per column of an (n, m) array.

    This is the term subtracted inside the Mann-Kendall variance. Exact ties are
    rare in a floating-point field, but a pixel that is constant over the record
    would otherwise be handed the full no-ties variance and a p-value, when it
    carries no information about trend at all. Costs one sort.
    """
    n, m = x.shape
    s = np.sort(x, axis=0)
    new = np.ones((n, m), dtype=bool)
    np.not_equal(s[1:], s[:-1], out=new[1:])
    gid = np.cumsum(new, axis=0, dtype=np.int64) - 1
    # One bincount over the whole chunk: column j owns bins [n*j, n*j + n).
    flat = gid + n * np.arange(m, dtype=np.int64)[None, :]
    counts = np.bincount(flat.ravel(), minlength=n * m).reshape(m, n).astype("float64")
    return np.sum(counts * (counts - 1.0) * (2.0 * counts + 5.0), axis=1)


def _mk_variance(x: np.ndarray) -> np.ndarray:
    """Tie-corrected Var(S) per column of an (n, m) array."""
    n = x.shape[0]
    return (n * (n - 1.0) * (2.0 * n + 5.0) - _tie_term(x)) / 18.0


def _sign_sum(x: np.ndarray) -> np.ndarray:
    """Mann-Kendall S per column, by summing sign(x_j - x_i) over all lags."""
    s = np.zeros(x.shape[1], dtype="float64")
    for k in range(1, x.shape[0]):
        s += np.sign(x[k:] - x[:-k]).sum(axis=0)
    return s


def _pairwise_slopes(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    All n(n-1)/2 pairwise slopes (x_j - x_i)/(t_j - t_i) per column.

    Materialised in full because the median needs it. The caller is responsible
    for chunking the columns: at n = 312 this is 48,516 rows, so a chunk of 512
    columns is ~200 MB.
    """
    n, m = x.shape
    out = np.empty((n * (n - 1) // 2, m), dtype="float64")
    pos = 0
    for k in range(1, n):
        cnt = n - k
        np.divide(x[k:] - x[:-k], (t[k:] - t[:-k])[:, None], out=out[pos:pos + cnt])
        pos += cnt
    return out


def _sens_and_s(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sen's slope and Mann-Kendall S per column, from one pass over the pairs.

    S is the sign sum of the same pairwise differences the slopes are built
    from, and the time spacing is positive, so sign(slope) == sign(x_j - x_i)
    and the two statistics come out of a single (large) array.
    """
    slopes = _pairwise_slopes(x, t)
    return np.median(slopes, axis=0), np.sign(slopes).sum(axis=0)


def _mk_z(s: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Continuity-corrected normal score for S."""
    good = np.isfinite(var) & (var > 0)
    sd = np.sqrt(np.where(good, var, np.nan))
    z = np.where(s > 0, (s - 1.0) / sd, np.where(s < 0, (s + 1.0) / sd, 0.0))
    return np.where(good, z, np.nan)


def _acf_all_lags(y: np.ndarray) -> np.ndarray:
    """
    Autocorrelation at lags 0..n-1 for every column, via FFT.

    The direct estimator is O(n^2) per column; over 9,538 columns that is the
    single most expensive step in the Hamed-Rao path, and the FFT makes it
    negligible. Uses the biased (divide-by-n) form, which is what Hamed and Rao
    assume.
    """
    n = y.shape[0]
    y = y - y.mean(axis=0, keepdims=True)
    nfft = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(y, n=nfft, axis=0)
    ac = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[:n]
    c0 = ac[0].copy()
    c0[c0 <= 0] = np.nan
    return ac / c0


def _hamed_rao_factor(resid: np.ndarray, alpha: float = ACF_ALPHA) -> np.ndarray:
    """
    Hamed-Rao (1998) variance inflation factor n/n*, per column.

    Ranks the detrended series, keeps the lag autocorrelations that clear the
    Anderson white-noise bound at `alpha`, and inflates Var(S) by

        n/n* = 1 + 2/(n(n-1)(n-2)) * sum_i (n-i)(n-i-1)(n-i-2) r_i

    Floored at 1. The factor can drop below 1 when negative autocorrelations
    dominate the sum, which would SHARPEN the test; the correction is here to
    guard against the false positives that positive persistence creates, and
    sharpening a trend test on the strength of a noisy negative autocorrelation
    estimate is not a trade this analysis wants to make. The floor is a choice,
    not a result, and it makes the test conservative rather than exact.

    It is blind to seasonality. The seasonal autocorrelation alternates sign
    (positive at lag 12, negative at lag 6) and very nearly cancels inside the
    weighted sum: with a strong annual cycle left in the series this function
    returned a median factor of 1.00, against 4.11 on the same series once the
    calendar-month median had been removed (`trend_selftest()`, 400 AR(1)
    series, rho = 0.8, n = 312, seed 20). Deseasonalise first; this will not do
    it for you.
    """
    n, m = resid.shape
    if n < 6:
        return np.ones(m)
    ranks = stats.rankdata(resid, axis=0)
    r = _acf_all_lags(ranks)[1:]
    lags = np.arange(1, n, dtype="float64")
    zc = float(stats.norm.ppf(1.0 - alpha / 2.0))
    denom = n - lags
    half = zc * np.sqrt(np.maximum(n - lags - 1.0, 0.0))
    lo = (-1.0 - half) / denom
    hi = (-1.0 + half) / denom
    sig = ((r < lo[:, None]) | (r > hi[:, None])) & np.isfinite(r)
    # Weights vanish at i = n-2 and i = n-1, so the tail lags cannot contribute
    # however wild their (few-sample) autocorrelation estimates are.
    w = denom * (denom - 1.0) * (denom - 2.0)
    contrib = np.where(sig, r, 0.0) * w[:, None]
    factor = 1.0 + (2.0 / (n * (n - 1.0) * (n - 2.0))) * contrib.sum(axis=0)
    return np.maximum(np.where(np.isfinite(factor), factor, 1.0), 1.0)


def _tfpw_series(
    x: np.ndarray, t: np.ndarray, slope: np.ndarray, alpha: float = ACF_ALPHA
) -> np.ndarray:
    """
    Yue-Pilon (2002) trend-free prewhitening: returns the (n-1, m) series to test.

    Detrend with Sen's slope, remove the lag-1 AR component from the residual,
    then add the trend back so the slope being tested is the one that was there
    before prewhitening (the point of the "trend-free" variant: prewhitening a
    trending series removes part of the trend along with the persistence).

    Where the lag-1 autocorrelation does not clear the white-noise bound the
    coefficient is set to zero, which leaves that column as the original series
    with its first sample dropped. Doing it that way rather than skipping the
    column keeps the chunk rectangular; the cost is one lost sample on columns
    that did not need prewhitening.

    NOT THE DEFAULT, AND IT MEASURED WORSE THAN DOING NOTHING. On 400 trendless
    AR(1) series (rho = 0.8, n = 312, annual cycle removed) it rejected H0 at
    p < 0.05 in 65.8% of them, against 49.3% for uncorrected Mann-Kendall and
    18.8% for Hamed-Rao (`trend_selftest()`, seed 20). The mechanism is the
    trend-free step itself: the Sen
    slope added back is estimated from a persistent series and is therefore
    noisy, while the prewhitened residual it is added to has had its variance
    cut, so a slope that is pure sampling noise ends up dominating the series
    the test then sees. Bayazit and Onoz (2007) and Serinaldi and Kilsby (2016)
    report the same behaviour. Kept because it is the other standard option and
    a reviewer may ask for it; do not make it the default without repeating that
    calibration.
    """
    n, m = x.shape
    tt = (t - t[0])[:, None]
    detr = x - slope[None, :] * tt
    d = detr - detr.mean(axis=0, keepdims=True)
    den = np.sum(d * d, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = np.where(den > 0, np.sum(d[1:] * d[:-1], axis=0) / den, 0.0)
    zc = float(stats.norm.ppf(1.0 - alpha / 2.0))
    lo = (-1.0 - zc * np.sqrt(n - 2.0)) / (n - 1.0)
    hi = (-1.0 + zc * np.sqrt(n - 2.0)) / (n - 1.0)
    r1 = np.where((r1 < lo) | (r1 > hi), r1, 0.0)
    return detr[1:] - r1[None, :] * detr[:-1] + slope[None, :] * tt[1:]


def _deseasonalise_columns(x: np.ndarray, month_of_year: np.ndarray) -> np.ndarray:
    """
    Subtract the calendar-month median from each column.

    The median rather than the mean keeps the procedure distribution-free, which
    is the reason Mann-Kendall was chosen over ordinary least squares here.

    This only leaves the trend alone when the record spans a whole number of
    years. Otherwise the over-represented months are pulled towards their own
    climatology and a spurious trend appears; `mann_kendall_field` checks and
    says so.
    """
    out = x.copy()
    for m in np.unique(month_of_year):
        sel = month_of_year == m
        out[sel] -= np.median(x[sel], axis=0, keepdims=True)
    return out


def benjamini_hochberg(
    p: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, float]:
    """
    Benjamini-Hochberg step-up FDR control. Returns (reject, p-threshold).

    Controls the expected proportion of false discoveries among the rejected
    tests at `alpha`. Valid under independence and under positive regression
    dependence; a spatial field of trend p-values is positively dependent, so BH
    (rather than the far more conservative Benjamini-Yekutieli) is the right
    member of the family.

    What it does NOT fix, and this matters more here than the multiplicity does:
    BH assumes the m tests are m tests. In this product the pixels are not
    independent experiments -- GRACE resolves ~20 mascons over the basin and the
    fine structure is interpolated, so the number of independent trends being
    tested is nearer 20 than 9,538. BH applied at pixel level therefore still
    reads as if far more evidence were present than there is.

    Non-finite p-values are passed through as non-rejections.
    """
    p = np.asarray(p, dtype="float64")
    reject = np.zeros(p.shape, dtype=bool)
    ok = np.isfinite(p)
    pv = p[ok]
    if pv.size == 0:
        return reject, np.nan
    order = np.argsort(pv)
    ps = pv[order]
    m = ps.size
    passed = ps <= alpha * np.arange(1, m + 1) / m
    if not passed.any():
        return reject, 0.0
    kmax = int(np.flatnonzero(passed).max())
    flags = np.zeros(m, dtype=bool)
    flags[order[:kmax + 1]] = True
    reject[ok] = flags
    return reject, float(ps[kmax])


def mann_kendall(
    x: np.ndarray,
    t: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Mann-Kendall trend test with the tie-corrected variance, plus Sen's slope.

    Tests H0: the series has no monotonic trend. S is the sign sum over all
    pairs, its variance is

        Var(S) = [n(n-1)(2n+5) - sum_ties t(t-1)(2t+5)] / 18

    and the normal score carries the usual +-1 continuity correction. Sen's
    slope, the median of the pairwise slopes, is the estimator that belongs with
    this test: both are rank-based and neither assumes a distribution.

    Plain MK assumes the observations are INDEPENDENT. On a monthly hydrological
    series they are not, and this function makes no attempt to fix that -- it is
    the uncorrected baseline. Use `mann_kendall_field` (or pass the series
    through `_deseasonalise_columns` and inflate the variance yourself) when the
    answer has to survive review.

    Parameters
    ----------
    x : 1-D series.
    t : times in years, same length as `x`. Defaults to 0, 1, 2, ... so the
        slope is then per sample, not per year.
    alpha : two-sided level for the `significant` flag.

    Returns
    -------
    dict with S, var_s, z, p_value, tau, sens_slope, the Sen confidence limits
    and n.
    """
    x = np.asarray(x, dtype="float64").ravel()
    ok = np.isfinite(x)
    x = x[ok]
    n = x.size
    if n < 4:
        return {"n": int(n), "S": np.nan, "var_s": np.nan, "z": np.nan,
                "p_value": np.nan, "tau": np.nan, "sens_slope": np.nan,
                "sens_lower": np.nan, "sens_upper": np.nan, "significant": False}
    if t is None:
        t = np.arange(n, dtype="float64")
    else:
        t = np.asarray(t, dtype="float64").ravel()[ok]

    col = x[:, None]
    slopes = _pairwise_slopes(col, t)[:, 0]
    s = float(np.sign(slopes).sum())
    var = float(_mk_variance(col)[0])
    z = float(_mk_z(np.array([s]), np.array([var]))[0])
    p = float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan

    # Non-parametric Sen interval: the two order statistics of the ranked
    # pairwise slopes that sit +-z_{1-a/2} sqrt(Var(S)) either side of the
    # centre. Same variance as the test, so the interval and the p-value agree.
    npair = slopes.size
    zc = float(stats.norm.ppf(1.0 - alpha / 2.0))
    c = zc * np.sqrt(var) if np.isfinite(var) and var > 0 else np.nan
    if np.isfinite(c):
        srt = np.sort(slopes)
        lo_i = int(np.clip(np.floor((npair - c) / 2.0), 0, npair - 1))
        hi_i = int(np.clip(np.ceil((npair + c) / 2.0), 0, npair - 1))
        lo, hi = float(srt[lo_i]), float(srt[hi_i])
    else:
        lo = hi = np.nan

    return {
        "n": int(n),
        "S": s,
        "var_s": var,
        "z": z,
        "p_value": p,
        # tau-a: S over the total pair count. Identical to tau-b when there are
        # no ties, which is the normal case for a floating-point series.
        "tau": s / (0.5 * n * (n - 1.0)),
        "sens_slope": float(np.median(slopes)),
        "sens_lower": lo,
        "sens_upper": hi,
        "significant": bool(np.isfinite(p) and p < alpha),
    }


def sens_slope(x: np.ndarray, t: Optional[np.ndarray] = None) -> float:
    """Theil-Sen slope: the median of all pairwise slopes. Units are x per t."""
    x = np.asarray(x, dtype="float64").ravel()
    ok = np.isfinite(x)
    x = x[ok]
    if x.size < 2:
        return np.nan
    t = (np.arange(x.size, dtype="float64") if t is None
         else np.asarray(t, dtype="float64").ravel()[ok])
    return float(np.median(_pairwise_slopes(x[:, None], t)[:, 0]))


def seasonal_mann_kendall(
    x: np.ndarray,
    month_of_year: np.ndarray,
    t: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Hirsch-Slack seasonal Mann-Kendall for one series. Provided as a CROSS-CHECK.

    Splits the record into calendar-month sub-series, sums S and Var(S) over
    them and tests the total, so only like-with-like pairs enter S and the
    annual cycle cannot contribute to the statistic at all.

    It is NOT what `mann_kendall_field` uses, and the reason is measured rather
    than argued. On 400 trendless AR(1) series (rho = 0.8, n = 312, 5-sd annual
    cycle) this test rejected the null in 46.5% of them at a nominal 5%, against
    49.3% for plain Mann-Kendall and 18.8% for the deseasonalise + Hamed-Rao
    path the field uses (`trend_selftest()`, seed 20). Removing the annual cycle
    from the STATISTIC, which is all a seasonal test does, buys almost nothing
    when the problem is persistence.

    Why it fails that way: the variance here is the sum of the per-season
    variances, i.e. it assumes the twelve sub-series are mutually independent.
    Terrestrial water storage carries from month to month, so they are not, and
    the variance comes out too small. Hirsch and Slack (1984) give a covariance
    correction, but it needs 66 cross-season covariances estimated from 26
    values each on this record, which is not enough to estimate them with. The
    test also discards every cross-month pair, which on 312 months is a real
    loss of power for a benefit that subtracting the calendar-month median
    delivers while keeping all the pairs.

    Kept so that comparison can be re-run, and because it is the test a reviewer
    familiar with monthly streamflow records will expect to see named.
    """
    x = np.asarray(x, dtype="float64").ravel()
    moy = np.asarray(month_of_year).ravel()
    t = (np.arange(x.size, dtype="float64") if t is None
         else np.asarray(t, dtype="float64").ravel())
    s_tot, var_tot, slopes = 0.0, 0.0, []
    for m in np.unique(moy):
        sel = (moy == m) & np.isfinite(x)
        if sel.sum() < 4:
            continue
        col = x[sel][:, None]
        sl = _pairwise_slopes(col, t[sel])[:, 0]
        s_tot += float(np.sign(sl).sum())
        var_tot += float(_mk_variance(col)[0])
        slopes.append(sl)
    if not slopes or var_tot <= 0:
        return {"S": np.nan, "var_s": np.nan, "z": np.nan, "p_value": np.nan,
                "sens_slope": np.nan, "n_seasons": 0, "significant": False}
    z = float(_mk_z(np.array([s_tot]), np.array([var_tot]))[0])
    p = float(2.0 * stats.norm.sf(abs(z)))
    return {
        "S": s_tot,
        "var_s": var_tot,
        "z": z,
        "p_value": p,
        # Median over the pooled within-season slopes, per Hirsch and Slack.
        "sens_slope": float(np.median(np.concatenate(slopes))),
        "n_seasons": int(len(slopes)),
        "significant": bool(p < alpha),
        "variance_assumption": "seasons independent (not true here; see docstring)",
    }


@dataclass
class TrendField:
    """
    Per-pixel trend test, every array shaped like one map of the input field.

    `slope` is Sen's; `ols_slope` is the least-squares slope the figure used
    before this test existed, kept so the two can be differenced. Pixels with
    any non-finite month are not tested and come back NaN / False.
    """
    slope: np.ndarray
    ols_slope: np.ndarray
    p_value: np.ndarray
    z: np.ndarray
    tau: np.ndarray
    variance_factor: np.ndarray
    significant: np.ndarray
    significant_fdr: np.ndarray
    valid: np.ndarray
    method: str
    deseasonalised: bool
    whole_years: bool
    alpha: float
    fdr_alpha: float
    fdr_p_threshold: float
    n_time: int
    n_tested: int
    n_significant: int
    n_significant_fdr: int
    seconds: float

    def summary(self) -> Dict[str, object]:
        """Scalar fields only, for stamping into a netCDF header or a log line."""
        return {k: v for k, v in vars(self).items()
                if not isinstance(v, np.ndarray)}


def mann_kendall_field(
    field: np.ndarray,
    times,
    method: str = "hamed_rao",
    deseasonalise: bool = True,
    alpha: float = 0.05,
    fdr_alpha: float = 0.05,
    chunk: int = 512,
) -> TrendField:
    """
    Mann-Kendall + Sen's slope at every pixel of a (n_time, lat, lon) field.

    What this does, in order, per pixel:

    1. Subtract the calendar-month median (`deseasonalise=True`). A monthly TWSA
       series is dominated by the annual cycle; leaving it in inflates the lag-12
       autocorrelation enormously and makes every downstream correction fight a
       signal that is not the trend.
    2. Sen's slope from the median of all n(n-1)/2 pairwise slopes, and S from
       the signs of those same slopes.
    3. Correct for the serial dependence that remains.
    4. Normal score, two-sided p-value, and a Benjamini-Hochberg FDR pass over
       the whole field.

    Which autocorrelation correction, and why
    -----------------------------------------
    Default is `hamed_rao`: the Hamed-Rao (1998) variance inflation factor,
    which rescales Var(S) using EVERY lag autocorrelation that clears a
    white-noise bound. The alternative, Yue-Pilon trend-free prewhitening, is
    implemented as `method='tfpw'` but is not the default because it measured
    worse than doing nothing.

    How often each rejects a TRUE null, on 400 simulated series with a 5-sd
    annual cycle at the record length of this product (n = 312), nominal 5%.
    Reproduce with `trend_selftest()`, seed 20:

        noise           none    hamed_rao   tfpw    seasonal_mk
        white           5.5%      4.3%      6.0%      6.0%
        AR(1) 0.8      49.3%     18.8%     65.8%     46.5%
        AR(1) 0.8, BH  41.5%      5.8%     61.8%     39.5%

    Read those numbers before believing any trend map. What they say:

    * Plain Mann-Kendall on a persistent monthly series rejects the null half
      the time when there is nothing there. This is the failure the layer exists
      to prevent, and it is large.
    * Hamed-Rao cuts it to 18.8%, and Benjamini-Hochberg on top brings it to
      5.8%. That is the only combination in the table that lands anywhere near
      nominal, which is why it is the default -- but 5.8% is the rate over 400
      INDEPENDENT columns, and the pixels of this product are not independent
      (see multiple comparisons below), so it is an upper bound on how well the
      layer behaves, not a description of the real field.
    * Uncorrected Hamed-Rao alone is still liberal at 18.8%. A per-pixel p-value
      from this function is best read as an ordering of pixels by strength of
      evidence, not as a calibrated error rate.
    * The correction costs power: with a real trend of 0.05 sd/yr the post-FDR
      detection rate fell from 74.8% (uncorrected) to 42.8%.
    * TFPW inflates the error rate rather than controlling it, for the reason set
      out in `_tfpw_series`.

    What NONE of this fixes:

    * The SPATIAL dependence, which is the larger problem here -- see multiple
      comparisons below.
    * Months with no GRACE observation. Those are reconstruction, not
      measurement (the product's `grace_observed` flag), and they enter the test
      as though they were data.
    * A trend that is not monotonic. Mann-Kendall tests monotonicity; a storage
      series that fell and then recovered can return a small S and a large
      p-value while having moved a long way.

    Why not a seasonal Mann-Kendall
    -------------------------------
    Considered and rejected as the field test; `seasonal_mann_kendall` carries
    the argument and a 1-D implementation to cross-check against. Short version:
    the Hirsch-Slack variance assumes the twelve sub-series are independent,
    which for storage they plainly are not, and the table above shows what that
    costs -- 46.5% false positives on persistent trendless series, barely better
    than doing nothing at all. Blocking the annual cycle is not the hard part of
    this problem; the persistence is, and a seasonal test does not address it.
    Deseasonalising by the calendar-month median blocks the cycle just as well
    while keeping all 48,516 pairs, and Hamed-Rao then handles the rest.

    Multiple comparisons
    --------------------
    A correction IS applied: `significant_fdr` is Benjamini-Hochberg at
    `fdr_alpha`. Without it, testing 9,538 in-basin pixels at p < 0.05 would be
    expected to return about 477 significant pixels from nothing at all
    (9538 x 0.05). Both masks are returned so the difference is visible.

    The honest caveat is that FDR is not the binding constraint. GRACE resolves
    ~20 independent mascons over this basin and the trend is inherited from them
    by mass conservation, so the sub-mascon detail of any trend map is
    interpolated rather than observed and the pixels are nothing like 9,538
    independent tests. A p-value per pixel describes the series in that pixel; it
    is not evidence that the trend differs from its neighbour's, and the count of
    significant pixels is not a sample size.

    Cost
    ----
    Measured (synthetic 312 x 9,538 float64, single process, Darwin arm64,
    `trend_selftest(timing=True)`): 9.3 s for the default `hamed_rao` path,
    11.1 s for `tfpw`, 9.2 s for `none`; spread over three repeats was under
    0.3 s. The correction is therefore nearly free -- the bill is the
    pairwise-slope pass over 462 million pairs, which Sen's slope needs in full
    because it takes their median. Memory is bounded by `chunk`: 512 columns
    x 48,516 pairs is ~200 MB.

    Parameters
    ----------
    field : (n_time, ...) array. Any trailing shape; results come back in it.
    times : DatetimeIndex or anything `pd.to_datetime` accepts, length n_time.
    method : 'hamed_rao', 'tfpw' or 'none' (uncorrected -- diagnostic only).
    deseasonalise : subtract the calendar-month median first.
    alpha : two-sided level for the uncorrected mask.
    fdr_alpha : level for the Benjamini-Hochberg mask.
    chunk : columns processed at once; trades memory for nothing else.
    """
    import time as _time

    t_start = _time.perf_counter()
    if method not in MK_AUTOCORR_METHODS:
        raise ValueError(f"method must be one of {MK_AUTOCORR_METHODS}")

    times = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
    arr, map_shape = _as_columns(field)
    n, m = arr.shape
    if n != len(times):
        raise ValueError("field's first axis and times must have the same length.")

    t_years = (times - times[0]).days.to_numpy().astype("float64") / 365.25
    moy = np.asarray(times.month)
    whole_years = bool(n % 12 == 0)
    if deseasonalise and not whole_years:
        # Not fatal, but it biases the slope, so it must not pass silently.
        print(f"  warning: {n} months is not a whole number of years; the "
              f"calendar-month median removal will leave a small spurious trend.")

    slope = np.full(m, np.nan)
    ols = np.full(m, np.nan)
    zs = np.full(m, np.nan)
    pvals = np.full(m, np.nan)
    taus = np.full(m, np.nan)
    factor = np.full(m, np.nan)

    valid = np.isfinite(arr).all(axis=0)
    cols = np.flatnonzero(valid)

    if cols.size:
        # OLS slope on the raw series, for comparison with Sen's on the same
        # pixels. Cheap, and the difference between them is a useful read on how
        # much a few extreme months were driving the old figure.
        design = np.vstack([t_years, np.ones_like(t_years)]).T
        coef, *_ = np.linalg.lstsq(design, arr[:, cols], rcond=None)
        ols[cols] = coef[0]

    for start in range(0, cols.size, chunk):
        idx = cols[start:start + chunk]
        x = arr[:, idx]
        if deseasonalise:
            x = _deseasonalise_columns(x, moy)

        sen, s_raw = _sens_and_s(x, t_years)
        slope[idx] = sen
        # tau always describes the series as it stands, even when the p-value
        # comes from a prewhitened version of it; a rank correlation of a
        # filtered series is not a statement about the data.
        taus[idx] = s_raw / (0.5 * n * (n - 1.0))

        if method == "tfpw":
            z_series = _tfpw_series(x, t_years, sen)
            s_use = _sign_sum(z_series)
            var = _mk_variance(z_series)
            fac = np.ones(idx.size)
        else:
            s_use = s_raw
            var = _mk_variance(x)
            if method == "hamed_rao":
                detrended = x - sen[None, :] * (t_years - t_years[0])[:, None]
                fac = _hamed_rao_factor(detrended)
                var = var * fac
            else:
                fac = np.ones(idx.size)

        factor[idx] = fac
        zi = _mk_z(s_use, var)
        zs[idx] = zi
        pvals[idx] = 2.0 * stats.norm.sf(np.abs(zi))

    sig = np.isfinite(pvals) & (pvals < alpha)
    sig_fdr, p_thresh = benjamini_hochberg(pvals, alpha=fdr_alpha)

    def shape_back(a):
        return a.reshape(map_shape) if map_shape else a[0]

    return TrendField(
        slope=shape_back(slope),
        ols_slope=shape_back(ols),
        p_value=shape_back(pvals),
        z=shape_back(zs),
        tau=shape_back(taus),
        variance_factor=shape_back(factor),
        significant=shape_back(sig),
        significant_fdr=shape_back(sig_fdr),
        valid=shape_back(valid),
        method=method,
        deseasonalised=bool(deseasonalise),
        whole_years=whole_years,
        alpha=float(alpha),
        fdr_alpha=float(fdr_alpha),
        fdr_p_threshold=float(p_thresh),
        n_time=int(n),
        n_tested=int(cols.size),
        n_significant=int(sig.sum()),
        n_significant_fdr=int(sig_fdr.sum()),
        seconds=float(_time.perf_counter() - t_start),
    )


def trend_selftest(
    n_series: int = 400,
    n_months: int = 312,
    rho: float = 0.8,
    seed: int = 20,
    timing: bool = False,
) -> pd.DataFrame:
    """
    Regenerate the calibration numbers quoted in `mann_kendall_field`'s docstring.

    Simulates trendless AR(1) series with an annual cycle at the record length
    of the product (312 months), runs all three autocorrelation settings, and
    reports how often each rejects a null that is true. Then repeats with a
    real trend to show what the correction costs in power.

    This exists because the choice of `hamed_rao` over `tfpw` is an empirical
    claim about this record length and this amount of persistence, and a claim
    like that should be re-runnable rather than cited.

    `timing=True` additionally times a full 312 x 9,538 field, which is the
    in-basin cell count of the 0.1 degree product.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range('1999-01-01', periods=n_months, freq='MS')
    t_years = (months - months[0]).days.to_numpy() / 365.25
    cycle = 5.0 * np.sin(2.0 * np.pi * np.arange(n_months) / 12.0)

    def simulate(m: int, r: float, trend: float) -> np.ndarray:
        """AR(1) noise + a 5-sd annual cycle + an optional linear trend."""
        e = rng.normal(0.0, 1.0, (n_months, m))
        if r == 0.0:
            y = e
        else:
            y = np.empty_like(e)
            y[0] = e[0] / np.sqrt(1.0 - r ** 2)
            for i in range(1, n_months):
                y[i] = r * y[i - 1] + e[i]
        return y + cycle[:, None] + trend * t_years[:, None]

    def row(case: str, method: str, res: TrendField) -> Dict[str, object]:
        return {
            'case': case,
            'method': method,
            'reject_uncorrected_pct': 100.0 * res.n_significant / res.n_tested,
            'reject_after_fdr_pct': 100.0 * res.n_significant_fdr / res.n_tested,
            'median_variance_factor': float(np.nanmedian(res.variance_factor)),
            'median_sens_slope': float(np.nanmedian(res.slope)),
            'seconds': res.seconds,
        }

    cases = [
        (f'white noise, no trend (nominal {100 * 0.05:.0f}%)', 0.0, 0.0),
        (f'AR(1) rho={rho}, no trend', rho, 0.0),
        (f'AR(1) rho={rho}, trend 0.05 sd/yr', rho, 0.05),
    ]
    moy = np.asarray(months.month)
    rows = []
    for label, r_case, trend in cases:
        f = simulate(n_series, r_case, trend)
        for meth in MK_AUTOCORR_METHODS:
            rows.append(row(label, meth,
                            mann_kendall_field(f, months, method=meth,
                                               chunk=n_series)))
        # Same series with the annual cycle left in, to show that the Hamed-Rao
        # factor does not see seasonality.
        rows.append(row(label, 'hamed_rao (cycle left in)',
                        mann_kendall_field(f, months, method='hamed_rao',
                                           deseasonalise=False, chunk=n_series)))
        # Hirsch-Slack on the same columns, looped because it is a 1-D test.
        # Here so that the reason it was rejected as the field test is measured
        # on this record length rather than asserted from the literature.
        p_seasonal = np.array([
            seasonal_mann_kendall(f[:, j], moy, t_years)['p_value']
            for j in range(f.shape[1])])
        rows.append({
            'case': label,
            'method': 'seasonal_mk (Hirsch-Slack)',
            'reject_uncorrected_pct': 100.0 * float(np.mean(p_seasonal < 0.05)),
            'reject_after_fdr_pct': 100.0 * benjamini_hochberg(p_seasonal)[0].mean(),
            'median_variance_factor': 1.0,
            'median_sens_slope': np.nan,
            'seconds': np.nan,
        })

    if timing:
        big = simulate(9538, rho, 0.05)
        for meth in MK_AUTOCORR_METHODS:
            rows.append(row('timing 312 x 9538', meth,
                            mann_kendall_field(big, months, method=meth)))
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


def model_name_from_path(path: str) -> str:
    """
    Model name from a `*_full_predictions_<Model>.csv` filename.

    The `_`->`+` substitution is GUARDED on "Attention": applied unconditionally
    it would also rewrite names like `Random_Forest` into `Random+Forest`. Two
    call sites previously carried different versions of this, one guarded and
    one not.
    """
    base = os.path.basename(path)
    model = base.split("full_predictions_")[-1].replace(".csv", "")
    return model.replace("_", "+") if "Attention" in model else model


def load_prediction_dir(
    directory: str,
    pattern: str = "*_full_predictions_*.csv",
    split: Optional[str] = "Test",
    add_time_parts: bool = False,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Load every prediction CSV in a holdout directory, keyed by model name.

    Single loader for all callers. `analyze_results` and `generate_monthly_maps`
    each had their own copy, which had already drifted apart in how they derive
    the model name.

    Parameters
    ----------
    split : "Test" to score held-out performance, or None to keep the full
        train+test reconstruction (what the map figures need).
    add_time_parts : also derive `Month` and `Year` columns.
    """
    out: Dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        df = load_prediction_csv(path, split=split)
        if df.empty:
            continue
        if add_time_parts:
            df["Month"] = df["Date"].dt.month
            df["Year"] = df["Date"].dt.year
        name = model_name_from_path(path)
        out[name] = df
        if verbose:
            print(f"Loaded predictions for {name}: {len(df)} samples")
    return out


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

    Rows are sorted by date first. The moving-block bootstrap and the
    Diebold-Mariano HAC variance both assume the array IS a time series; on the
    random holdout the rows arrive shuffled, which destroys the autocorrelation
    those methods exist to account for and yields intervals several times too
    narrow.
    """
    import os as _os
    _os.makedirs(output_dir, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    # Restore chronological order before any bootstrap or DM call. The random
    # holdout hands these arrays over shuffled; a moving-block bootstrap on a
    # shuffled series resamples independent points, so the blocks carry no
    # autocorrelation and the intervals come out several times too narrow.
    if dates_test is not None:
        _order = np.argsort(pd.to_datetime(pd.Series(np.asarray(dates_test))).values)
        if not np.array_equal(_order, np.arange(len(_order))):
            y_test = np.asarray(y_test)[_order]
            predictions = {k: np.asarray(v)[_order] for k, v in predictions.items()}
            dates_test = np.asarray(dates_test)[_order]

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

    print("\nMann-Kendall on the same series:", mann_kendall(truth, t))
    # Defaults, not a faster reduced run: these are the numbers quoted in
    # `mann_kendall_field`'s docstring, and they should match it line for line.
    print("\nTrend-test calibration (this is what picks hamed_rao over tfpw):")
    print(trend_selftest().to_string(index=False))
