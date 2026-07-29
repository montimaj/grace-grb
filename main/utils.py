"""
Utility Functions for TWS Downscaling Analysis
Data loading, preprocessing, evaluation metrics, visualization utilities, and SHAP analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Tuple, List, Dict, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataclasses import dataclass
from plot_style import (
    pretty_model, clean_feature, clean_features, R2, R2_BOLD,
    BAR, BAR_2,
)

# Optional SHAP import
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Install with: pip install shap")


# JPL mascon lwe_thickness is centimetres; this project reports millimetres.
TWS_CM_TO_MM = 10.0


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mae: float
    rmse: float
    r2: float
    nse: float  # Nash-Sutcliffe Efficiency
    pbias: float  # Percent Bias
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'MAE': self.mae,
            'RMSE': self.rmse,
            'R2': self.r2,
            'NSE': self.nse,
            'PBIAS': self.pbias
        }
    
    def __str__(self) -> str:
        return (f"MAE: {self.mae:.4f}, RMSE: {self.rmse:.4f}, "
                f"R²: {self.r2:.4f}, NSE: {self.nse:.4f}, PBIAS: {self.pbias:.2f}%")

MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Keyed on lower-case names. TWS_JPL.xlsx contains 'January ' (trailing space,
# 20 rows) and 'july' (lowercase); an exact-match map turned all 21 into NaT and
# they were silently dropped -- 8.9% of the GRACE record, of which 20 were
# Januaries spanning 2003-2024. Callers must .strip().lower() before lookup.
MONTH_MAP = {m.lower(): i + 1 for i, m in enumerate(MONTH_NAMES)}


MAX_GAP_DAYS = 62


def grace_segments(obs_dates: pd.DatetimeIndex,
                   daily: pd.DatetimeIndex) -> pd.DataFrame:
    """
    For each daily date, the observed GRACE months bracketing it.

    GRACE observes monthly and has gaps, so the daily target is interpolated.
    Two kinds of interpolation get conflated if you do not look at the span:

      * BETWEEN ADJACENT OBSERVED MONTHS (~30 days) -- this is the method. The
        daily target is an interpolation by construction, it is disclosed, and
        the temporal closure test exists to validate it.
      * ACROSS A GAP -- the GRACE/GRACE-FO transition is ~11 missing months,
        which becomes a 334-day straight line anchored by nothing while the real
        signal completed a full monsoon cycle. Training on that teaches the model
        that storage is linear when precipitation says otherwise.

    Returns a frame indexed like `daily` with:
      prev, next   the bracketing observed month starts
      span_days    distance between them (NaN outside the observed record)
      segment      integer id of the interval between two observations

    Used for two things that must agree:
      1. bounding interpolation (`load_and_preprocess_data`)
      2. grouping the random split, so no test day shares a segment with a
         training day (`holdout_random`). Interpolated within a segment, a test
         day is an exact linear blend of its neighbours -- reconstructible from
         the training targets alone at R2 = 0.999997, with no predictors used.
    """
    obs = pd.Series(np.arange(len(obs_dates)), index=pd.DatetimeIndex(obs_dates))
    daily = pd.DatetimeIndex(daily)
    prev_i = obs.reindex(daily, method='ffill')
    next_i = obs.reindex(daily, method='bfill')
    prev_d = pd.Series(pd.NaT, index=daily, dtype='datetime64[ns]')
    next_d = pd.Series(pd.NaT, index=daily, dtype='datetime64[ns]')
    ok_p, ok_n = prev_i.notna(), next_i.notna()
    prev_d[ok_p] = obs.index[prev_i[ok_p].astype(int)]
    next_d[ok_n] = obs.index[next_i[ok_n].astype(int)]
    return pd.DataFrame({
        'prev': prev_d, 'next': next_d,
        'span_days': (next_d - prev_d).dt.days,
        'segment': prev_i,
    }, index=daily)


def read_grace_monthly(tws_file: str) -> pd.DataFrame:
    """
    Read the observed monthly GRACE TWSA series, in MILLIMETRES.

    Single source of truth for GRACE ingestion, used by both
    `load_and_preprocess_data` and `temporal_closure_validation`. Those two
    previously carried independent copies of the month map and the unit
    handling, which is how the closure test kept the month bug after `utils`
    was fixed -- and would then have compared millimetre predictions against
    centimetre observations.

    Two accepted sources
    --------------------
    PREFERRED  `Data/TWS_GRACE_GEE.csv` -- the area-weighted basin mean computed
        by `export_basin_grace.py` directly from NASA/GRACE/MASS_GRIDS_V04/
        MASCON_CRI. Reproducible from code, already in millimetres, and the same
        series the 0.1 degree downscaling uses, so both halves of the project
        rest on one GRACE source.

    LEGACY  `Data/TWS_JPL.xlsx` -- unrecorded provenance, malformed month names,
        native centimetres despite being labelled millimetres, and correlating
        only r = 0.94 with a proper basin mean of the product it claims to be
        (difference std 56.8 mm against a 164 mm signal). Supported so earlier
        results can be reproduced; not recommended.

    Returns
    -------
    pd.DataFrame with columns ['Date', 'TWS'], sorted, one row per observed
    month. Months GRACE did not observe are absent rather than interpolated.
    """
    if str(tws_file).lower().endswith('.csv'):
        tws = pd.read_csv(tws_file, parse_dates=['Date'])
    else:
        tws = pd.read_excel(tws_file)
        tws['Month_Num'] = (tws['Month'].astype(str)
                            .str.strip().str.lower().map(MONTH_MAP))
        unparsed = tws['Month_Num'].isna().sum()
        if unparsed:
            raise ValueError(
                f'{unparsed} rows in {tws_file} have an unrecognised Month '
                f'value: {sorted(tws.loc[tws.Month_Num.isna(), "Month"].unique())}')
        tws['Date'] = pd.to_datetime(
            dict(year=tws.Year, month=tws.Month_Num, day=1))
        # Native centimetres -> millimetres, so the label and the number agree.
        tws['TWS'] = tws['TWS'] * TWS_CM_TO_MM

    return (tws[['Date', 'TWS']]
            .sort_values('Date').drop_duplicates(subset='Date')
            .reset_index(drop=True))


def load_and_preprocess_data(
    predictor_file: str,
    tws_file: str,
    predictors: List[str] = None,
    fill_missing: bool = True,
    max_gap_days: int = MAX_GAP_DAYS,
) -> pd.DataFrame:
    """
    Load predictors and the GRACE target, and merge them on date.

    `max_gap_days` bounds how far apart two observed GRACE months may be for the
    days between them to be kept. The default (62 days) allows one missing month
    and rejects longer gaps, notably the ~11-month GRACE/GRACE-FO transition.
    A `grace_segment` column is returned so downstream splits can group on it.

    NOTE on "daily" results: GRACE observes TWSA MONTHLY. The daily target is a
    linear interpolation of the monthly series, so it contains no independent
    daily observations and direct daily validation is impossible from this
    dataset. Daily-scale consistency is assessed instead by the temporal closure
    test (`temporal_closure_validation.py`), which re-aggregates the daily
    predictions to monthly and compares them against the ORIGINAL observations.
    """
    if predictors is None:
        predictors = ['SMS', 'ET', 'rainfall', 'runoff', 'GWSA']

    all_data = pd.read_excel(predictor_file)

    # Recover GWSA from runoff rather than losing the tail of the record.
    #
    # `GWSA` in All_Data.xlsx is exactly `runoff - mean(runoff)` (verified by
    # linear-identity test, max residual 1.8e-15) -- it contains no groundwater
    # data despite the name. Its 366 trailing NaNs are an artefact of how that
    # spreadsheet was assembled, not a limit of the underlying data: `runoff`
    # (ERA5-Land surface runoff) has complete coverage. Recomputing restores the
    # whole 2024 tail from real observations, and puts the column's definition
    # in code instead of baked invisibly into a spreadsheet.
    if 'runoff' in all_data.columns and 'GWSA' in all_data.columns:
        missing = all_data['GWSA'].isna().sum()
        all_data['GWSA'] = all_data['runoff'] - all_data['runoff'].mean()
        if missing:
            print(f'  GWSA: recomputed from runoff, recovering {missing} rows')

    # Fill INTERIOR gaps only, and never fabricate outside the observed span.
    #
    # The previous `.bfill().ffill()` filled in both directions without limit.
    # That was not gap-filling: these columns have ZERO interior gaps, so bfill
    # only ever extended a flat constant across the leading edge -- 396 days for
    # SMS and ET, whose source (GLDAS V022 CLSM) simply does not exist before
    # 2003-02-01. A constant predictor carries no information, yet still
    # generates training rows and can land in the random-holdout test set.
    #
    # `limit_area='inside'` interpolates only between real observations, so
    # genuine interior gaps are filled while the edges stay NaN and are dropped.
    if fill_missing:
        for col in ['SMS', 'ET', 'GWSA']:
            if col in all_data.columns:
                before = all_data[col].isna().sum()
                all_data[col] = all_data[col].interpolate(
                    method='linear', limit_area='inside')
                after = all_data[col].isna().sum()
                if before - after:
                    print(f'  {col}: interpolated {before - after} interior gaps')

    present = [c for c in ['SMS', 'ET', 'GWSA'] if c in all_data.columns]
    if present:
        n0 = len(all_data)
        all_data = all_data.dropna(subset=present)
        if len(all_data) < n0:
            print(f'  dropped {n0 - len(all_data)} rows outside the predictor '
                  f'record (not fabricated by back-fill)')

    tws = read_grace_monthly(tws_file).set_index('Date')
    tws_daily = (tws['TWS'].resample('D')
                 .interpolate(method='linear').reset_index())

    # Drop days whose bracketing observations are too far apart. Note this is
    # NOT `interpolate(limit=...)`: pandas fills the FIRST `limit` NaNs of a run
    # and leaves the rest, so a 334-day gap would keep 62 fabricated days plus a
    # hole. Rejecting on the bracketing span removes whole gaps instead.
    seg = grace_segments(tws.index, pd.DatetimeIndex(tws_daily['Date']))
    keep = seg['span_days'].to_numpy() <= max_gap_days
    n_drop = int((~keep).sum())
    if n_drop:
        print(f'  dropped {n_drop} interpolated days spanning GRACE gaps '
              f'longer than {max_gap_days} days (not observed, not fabricated)')
    tws_daily = tws_daily[keep]
    tws_daily['grace_segment'] = seg['segment'].to_numpy()[keep]

    return pd.merge(all_data, tws_daily, on='Date', how='inner')


def create_lagged_features(
    df: pd.DataFrame,
    cols: List[str],
    lags: int = 7
) -> pd.DataFrame:
    """
    Create lagged features for time series modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : List[str]
        Columns to create lags for
    lags : int
        Number of lag periods
    
    Returns
    -------
    pd.DataFrame
        Dataframe with lagged features
    """
    df = df.copy()
    for col in cols:
        for lag in range(1, lags + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df.dropna()


def prepare_features_target(
    df: pd.DataFrame,
    predictors: List[str],
    target_col: str = 'TWS',
    lags: int = 7
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and target vector.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with lagged features
    predictors : List[str]
        List of predictor names (without lag suffix)
    target_col : str
        Target column name
    lags : int
        Number of lag periods
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Feature matrix X, target vector y, and feature names
    """
    feature_names = [f'{var}_lag{i}' for var in predictors for i in range(1, lags + 1)]
    X = df[feature_names].values
    y = df[target_col].values
    
    return X, y, feature_names


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationMetrics:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    EvaluationMetrics
        Container with all metrics
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # R2 as the SQUARED PEARSON CORRELATION COEFFICIENT. This measures the
    # strength of linear association between observed and predicted values and
    # is insensitive to bias and scale. It is deliberately kept distinct from
    # NSE: sklearn's r2_score is algebraically identical to NSE
    # (both equal 1 - SSres/SStot), so reporting r2_score and NSE side by side
    # would duplicate a single number. Pearson r-squared and NSE together
    # separate "does the model track the shape of the signal?" (R2) from
    # "does it also get the magnitude/bias right?" (NSE).
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        r2 = np.nan
    else:
        r = np.corrcoef(y_true, y_pred)[0, 1]
        r2 = float(r ** 2)

    # Nash-Sutcliffe Efficiency (algebraically equal to sklearn's r2_score).
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = float(1 - (np.sum((y_true - y_pred) ** 2) / denom)) if denom != 0 else np.nan

    # Percent Bias. Guarded: for a mean-zero target (an anomaly about its own
    # mean, as used by the gridded downscaling) the denominator collapses and
    # the ratio explodes to meaningless magnitudes. Report NaN instead.
    if np.abs(np.sum(y_true)) > 0.01 * np.sum(np.abs(y_true)):
        pbias = float(100 * np.sum(y_pred - y_true) / np.sum(y_true))
    else:
        pbias = np.nan

    return EvaluationMetrics(mae=mae, rmse=rmse, r2=r2, nse=nse, pbias=pbias)


def plot_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str,
    prefix: str = ""
) -> None:
    """
    Create and save prediction visualization plots.
    
    Parameters
    ----------
    dates : np.ndarray
        Date array
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Name of the model
    output_dir : str
        Output directory for plots
    prefix : str
        Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    
    if prefix:
        prefix = f"{prefix}_"
    
    # 1. Temporal Plot
    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_true, label='Actual TWSA', alpha=0.8)
    plt.plot(dates, y_pred, label='Predicted TWSA', linestyle='--', alpha=0.8)
    plt.xlabel("Date")
    plt.ylabel("TWSA (mm)")
    plt.title(f"Actual vs Predicted TWSA [{pretty_model(model_name)}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}temporal_plot_{safe_name}.png", dpi=600)
    plt.close()
    
    # 2. Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='1:1 Line')
    plt.xlabel("Actual TWSA (mm)")
    plt.ylabel("Predicted TWSA (mm)")
    plt.title(f"Scatter Plot [{pretty_model(model_name)}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}scatter_plot_{safe_name}.png", dpi=600)
    plt.close()
    
    # 3. Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(14, 5))
    plt.plot(dates, residuals, color='gray', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Residuals Plot [{pretty_model(model_name)}]")
    plt.xlabel("Date")
    plt.ylabel("Residual (mm)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}residuals_plot_{safe_name}.png", dpi=600)
    plt.close()
    
    # 4. Residual Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label='Zero')
    plt.axvline(np.mean(residuals), color='blue', linestyle='--', label=f'Mean: {np.mean(residuals):.2f} mm')
    plt.xlabel("Residual (mm)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution [{pretty_model(model_name)}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}residual_hist_{safe_name}.png", dpi=600)
    plt.close()


def plot_training_loss(
    losses: List[float],
    model_name: str,
    output_dir: str,
    prefix: str = ""
) -> None:
    """Plot training loss curve for neural network models."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    
    if prefix:
        prefix = f"{prefix}_"
    
    plt.figure(figsize=(8, 4))
    plt.plot(losses, marker='o', markersize=3)
    plt.title(f"Training Loss Curve [{pretty_model(model_name)}]")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}training_loss_{safe_name}.png", dpi=600)
    plt.close()


def plot_feature_importance(
    importance: Dict[str, float],
    model_name: str,
    output_dir: str,
    prefix: str = "",
    top_n: int = 20
) -> None:
    """Plot feature importance for tree-based models."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    
    if prefix:
        prefix = f"{prefix}_"
    
    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_items = sorted_items[:top_n]
    
    names, values = zip(*sorted_items)
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(names)), values, align='center')
    plt.yticks(range(len(names)), clean_features(names))
    plt.xlabel("Importance")
    plt.title(f"Feature Importance [{pretty_model(model_name)}]")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}feature_importance_{safe_name}.png", dpi=600)
    plt.close()


def plot_model_comparison(
    results: Dict[str, EvaluationMetrics],
    output_dir: str,
    prefix: str = ""
) -> None:
    """
    Create comparison plots for multiple models.
    
    Parameters
    ----------
    results : Dict[str, EvaluationMetrics]
        Dictionary mapping model names to their metrics
    output_dir : str
        Output directory
    prefix : str
        Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if prefix:
        prefix = f"{prefix}_"
    
    model_names = list(results.keys())
    metrics_dict = {name: m.to_dict() for name, m in results.items()}
    
    # Create DataFrame for plotting
    df = pd.DataFrame(metrics_dict).T
    
    # Plot each metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['MAE', 'RMSE', 'R2', 'NSE', 'PBIAS']
    # Each panel is a separate metric, so the per-metric hue carries no
    # information: a single consistent colour (seaborn blue) is used for all bars.
    colors = [BAR] * len(metrics)
    
    disp_names = [pretty_model(m) for m in model_names]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i]
        values = df[metric].values
        bars = ax.bar(disp_names, values, color=color, alpha=0.8)
        ax.set_title(R2_BOLD if metric == 'R2' else metric, fontsize=12, fontweight='bold')
        ax.set_xticklabels(disp_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Remove extra subplot
    axes[-1].set_visible(False)
    
    plt.suptitle("Model Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}model_comparison.png", dpi=600)
    plt.close()
    
    # Save metrics to CSV
    df.to_csv(f"{output_dir}/{prefix}metrics_comparison.csv")


def plot_train_test_comparison(
    train_metrics: Dict[str, EvaluationMetrics],
    test_metrics: Dict[str, EvaluationMetrics],
    output_dir: str,
    prefix: str = ""
) -> Dict[str, Dict[str, float]]:
    """
    Create comparison plots for training vs test performance to visualize overfitting.
    
    This function generates visualizations that help identify overfitting by comparing
    training and test metrics side-by-side. Large gaps between train and test performance
    indicate overfitting.
    
    Parameters
    ----------
    train_metrics : Dict[str, EvaluationMetrics]
        Dictionary mapping model names to training metrics
    test_metrics : Dict[str, EvaluationMetrics]
        Dictionary mapping model names to test metrics
    output_dir : str
        Output directory for figures and CSV files
    prefix : str
        Prefix for output filenames
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        Overfitting analysis results with gaps for each metric
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if prefix:
        prefix = f"{prefix}_"
    
    model_names = list(test_metrics.keys())
    
    # Convert to DataFrames
    train_df = pd.DataFrame({name: m.to_dict() for name, m in train_metrics.items()}).T
    test_df = pd.DataFrame({name: m.to_dict() for name, m in test_metrics.items()}).T
    
    # Calculate overfitting indicators (train - test gaps)
    overfitting_analysis = {}
    for model in model_names:
        train_m = train_metrics[model]
        test_m = test_metrics[model]
        
        # For error metrics (MAE, RMSE): train < test indicates overfitting
        # For performance metrics (R2, NSE): train > test indicates overfitting
        overfitting_analysis[model] = {
            'MAE_gap': test_m.mae - train_m.mae,  # Positive = overfitting
            'RMSE_gap': test_m.rmse - train_m.rmse,  # Positive = overfitting
            'R2_gap': train_m.r2 - test_m.r2,  # Positive = overfitting
            'NSE_gap': train_m.nse - test_m.nse,  # Positive = overfitting
            'PBIAS_gap': abs(test_m.pbias) - abs(train_m.pbias),  # Positive = overfitting
            'Train_R2': train_m.r2,
            'Test_R2': test_m.r2,
            'Train_RMSE': train_m.rmse,
            'Test_RMSE': test_m.rmse
        }
    
    # =============================================
    # Plot 1: Side-by-side bar comparison (main metrics)
    # =============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [('R2', 'R² Score', True), ('NSE', 'NSE', True), 
                       ('MAE', 'MAE (mm)', False), ('RMSE', 'RMSE (mm)', False)]
    x = np.arange(len(model_names))
    width = 0.35
    
    for idx, (metric, label, higher_better) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        train_vals = train_df[metric].values
        test_vals = test_df[metric].values
        
        bars1 = ax.bar(x - width/2, train_vals, width, label='Train', color=BAR_2[0], alpha=0.95)
        bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color=BAR_2[1], alpha=0.95)

        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} - Train vs Test', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_model(m) for m in model_names], rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

        # Highlight overfitting (shade the column where the train-test gap is large)
        shaded = False
        for i, model in enumerate(model_names):
            gap = overfitting_analysis[model][f'{metric}_gap']
            if (higher_better and gap > 0.05) or (not higher_better and gap > 0.5):
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='red')
                shaded = True

        # Legend: Train/Test plus (where drawn) an explanation of the red shading.
        handles, labels_ = ax.get_legend_handles_labels()
        if shaded:
            handles.append(Patch(facecolor='red', alpha=0.1,
                                 label='Overfitting flag (large train–test gap)'))
        ax.legend(handles=handles, loc='best', fontsize=8)
    
    plt.suptitle('Training vs Test Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}train_test_comparison.png", dpi=600)
    plt.close()
    
    # =============================================
    # Plot 2: Overfitting Gap Analysis
    # =============================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² gap (how much better is train vs test)
    ax1 = axes[0]
    r2_gaps = [overfitting_analysis[m]['R2_gap'] for m in model_names]
    colors = BAR  # single-colour bars; the dashed threshold lines convey severity
    bars = ax1.bar([pretty_model(m) for m in model_names], r2_gaps, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=0.05, color='orange', linestyle='--', linewidth=1, label='Moderate overfitting (0.05)')
    ax1.axhline(y=0.1, color='red', linestyle='--', linewidth=1, label='Severe overfitting (0.1)')
    ax1.set_ylabel('R² Gap (Train - Test)', fontsize=11)
    ax1.set_title('Overfitting Indicator: R² Gap', fontsize=12, fontweight='bold')
    ax1.set_xticklabels([pretty_model(m) for m in model_names], rotation=45, ha='right')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, r2_gaps):
        ypos = bar.get_height() if bar.get_height() >= 0 else bar.get_height() - 0.02
        ax1.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.3f}', 
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # RMSE gap
    ax2 = axes[1]
    rmse_gaps = [overfitting_analysis[m]['RMSE_gap'] for m in model_names]
    colors = BAR  # single-colour bars; the dashed threshold lines convey severity
    bars = ax2.bar([pretty_model(m) for m in model_names], rmse_gaps, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=1, color='orange', linestyle='--', linewidth=1, label='Moderate overfitting (1 mm)')
    ax2.axhline(y=2, color='red', linestyle='--', linewidth=1, label='Severe overfitting (2 mm)')
    ax2.set_ylabel('RMSE Gap (Test - Train) [mm]', fontsize=11)
    ax2.set_title('Overfitting Indicator: RMSE Gap', fontsize=12, fontweight='bold')
    ax2.set_xticklabels([pretty_model(m) for m in model_names], rotation=45, ha='right')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, rmse_gaps):
        ypos = bar.get_height() if bar.get_height() >= 0 else bar.get_height() - 0.2
        ax2.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.2f}', 
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    plt.suptitle('Overfitting Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}overfitting_analysis.png", dpi=600)
    plt.close()
    
    # =============================================
    # Plot 3: Scatter plot of Train vs Test R²
    # =============================================
    fig, ax = plt.subplots(figsize=(8, 8))
    
    train_r2 = [overfitting_analysis[m]['Train_R2'] for m in model_names]
    test_r2 = [overfitting_analysis[m]['Test_R2'] for m in model_names]
    
    # Plot 1:1 line
    min_val, max_val = min(min(train_r2), min(test_r2)) - 0.05, max(max(train_r2), max(test_r2)) + 0.05
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='No Overfitting (1:1)')
    ax.fill_between([min_val, max_val], [min_val - 0.05, max_val - 0.05], [min_val, max_val],
                    alpha=0.2, color='orange', label='Slight Overfitting Zone')
    ax.fill_between([min_val, max_val], [min_val - 0.15, max_val - 0.15], [min_val - 0.05, max_val - 0.05],
                    alpha=0.2, color='red', label='Severe Overfitting Zone')
    
    # Plot points with different colors for model types
    neural_models = ['BiLSTM+Attention', 'BiLSTM', 'LSTM']
    tree_models = ['XGBoost', 'LightGBM', 'RandomForest']
    
    for i, model in enumerate(model_names):
        if model in neural_models:
            marker, color = 'o', BAR_2[1]
        else:
            marker, color = 's', BAR_2[0]
        ax.scatter(train_r2[i], test_r2[i], s=150, marker=marker, c=color, 
                  edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(pretty_model(model), (train_r2[i], test_r2[i]), textcoords="offset points",
                   xytext=(5, 5), fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Training R²', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Train vs Test R² (Overfitting Visualization)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}train_test_scatter.png", dpi=600)
    plt.close()
    
    # =============================================
    # Plot 4: Comprehensive metrics heatmap
    # =============================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create combined DataFrame
    combined_data = []
    for model in model_names:
        train_m = train_metrics[model]
        test_m = test_metrics[model]
        combined_data.append({
            'Model': model,
            'Train_R2': train_m.r2,
            'Test_R2': test_m.r2,
            'Train_NSE': train_m.nse,
            'Test_NSE': test_m.nse,
            'Train_RMSE': train_m.rmse,
            'Test_RMSE': test_m.rmse,
            'Train_MAE': train_m.mae,
            'Test_MAE': test_m.mae,
            'R2_Gap': overfitting_analysis[model]['R2_gap'],
            'RMSE_Gap': overfitting_analysis[model]['RMSE_gap']
        })
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.set_index('Model', inplace=True)
    
    # Normalize for heatmap visualization
    heatmap_cols = ['Train_R2', 'Test_R2', 'Train_NSE', 'Test_NSE']
    heatmap_data = combined_df[heatmap_cols]
    
    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    heatmap_labels = [c.replace('_R2', f' {R2}').replace('_', ' ') for c in heatmap_cols]
    ax.set_xticks(np.arange(len(heatmap_cols)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(heatmap_labels, rotation=45, ha='right')
    ax.set_yticklabels([pretty_model(m) for m in model_names])
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(heatmap_cols)):
            val = heatmap_data.values[i, j]
            text_color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                   fontsize=10, color=text_color, fontweight='bold')
    
    plt.colorbar(im, label='Score (0-1)')
    ax.set_title('Performance Heatmap: Train vs Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}performance_heatmap.png", dpi=600)
    plt.close()
    
    # =============================================
    # Save comprehensive results to CSV
    # =============================================
    combined_df.to_csv(f"{output_dir}/{prefix}train_test_metrics.csv")
    
    # Save overfitting analysis
    overfit_df = pd.DataFrame(overfitting_analysis).T
    overfit_df.index.name = 'Model'
    overfit_df.to_csv(f"{output_dir}/{prefix}overfitting_metrics.csv")
    
    # =============================================
    # Print overfitting summary
    # =============================================
    print("\n" + "="*70)
    print("OVERFITTING ANALYSIS SUMMARY")
    print("="*70)
    print("\nR² Gap = Train R² - Test R² (Higher positive values indicate more overfitting)")
    print("RMSE Gap = Test RMSE - Train RMSE (Higher positive values indicate more overfitting)")
    print("\nInterpretation Guidelines:")
    print("  • R² Gap < 0.05: Good generalization (minimal overfitting)")
    print("  • R² Gap 0.05-0.10: Moderate overfitting")
    print("  • R² Gap > 0.10: Severe overfitting - model memorizing training data")
    print("-"*70)
    print(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'R² Gap':>10} {'Status':>15}")
    print("-"*70)
    
    for model in model_names:
        train_r2 = overfitting_analysis[model]['Train_R2']
        test_r2 = overfitting_analysis[model]['Test_R2']
        gap = overfitting_analysis[model]['R2_gap']
        
        if gap > 0.10:
            status = "⚠️  SEVERE"
        elif gap > 0.05:
            status = "⚡ MODERATE"
        else:
            status = "✓ GOOD"
        
        print(f"{model:<20} {train_r2:>10.4f} {test_r2:>10.4f} {gap:>10.4f} {status:>15}")
    
    print("-"*70)
    print("\nRecommendations:")
    
    # Identify overfitting models
    overfit_models = [m for m in model_names if overfitting_analysis[m]['R2_gap'] > 0.1]
    moderate_models = [m for m in model_names if 0.05 < overfitting_analysis[m]['R2_gap'] <= 0.1]
    good_models = [m for m in model_names if overfitting_analysis[m]['R2_gap'] <= 0.05]
    
    if overfit_models:
        print(f"  • Models with SEVERE overfitting: {', '.join(overfit_models)}")
        print("    Consider: More regularization, fewer parameters, more training data, or early stopping")
    if moderate_models:
        print(f"  • Models with MODERATE overfitting: {', '.join(moderate_models)}")
        print("    Consider: Slight regularization increase or cross-validation")
    if good_models:
        print(f"  • Models with GOOD generalization: {', '.join(good_models)}")
    
    print("="*70 + "\n")
    
    return overfitting_analysis


def save_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str,
    prefix: str = ""
) -> str:
    """Save predictions to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    
    if prefix:
        prefix = f"{prefix}_"
    
    pred_df = pd.DataFrame({
        'Date': dates,
        'Actual_TWSA': y_true.flatten(),
        'Predicted_TWSA': y_pred.flatten(),
        'Residual': y_true.flatten() - y_pred.flatten()
    })
    
    filepath = f"{output_dir}/{prefix}predictions_{safe_name}.csv"
    pred_df.to_csv(filepath, index=False)
    
    return filepath


def save_full_predictions(
    dates_train: np.ndarray,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    dates_test: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    model_name: str,
    output_dir: str,
    prefix: str = ""
) -> str:
    """
    Save full predictions (train + test) to CSV file.
    
    Parameters
    ----------
    dates_train : np.ndarray
        Training dates
    y_train : np.ndarray
        Actual training values
    y_train_pred : np.ndarray
        Predicted training values
    dates_test : np.ndarray
        Test dates
    y_test : np.ndarray
        Actual test values
    y_test_pred : np.ndarray
        Predicted test values
    model_name : str
        Name of the model
    output_dir : str
        Output directory
    prefix : str
        Prefix for output filename
    
    Returns
    -------
    str
        Path to the saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    
    if prefix:
        prefix = f"{prefix}_"
    
    # Create train DataFrame
    train_df = pd.DataFrame({
        'Date': dates_train,
        'Actual_TWSA': y_train.flatten(),
        'Predicted_TWSA': y_train_pred.flatten(),
        'Residual': y_train.flatten() - y_train_pred.flatten(),
        'Split': 'Train'
    })
    
    # Create test DataFrame
    test_df = pd.DataFrame({
        'Date': dates_test,
        'Actual_TWSA': y_test.flatten(),
        'Predicted_TWSA': y_test_pred.flatten(),
        'Residual': y_test.flatten() - y_test_pred.flatten(),
        'Split': 'Test'
    })
    
    # Combine and sort by date
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df = full_df.sort_values('Date').reset_index(drop=True)
    
    filepath = f"{output_dir}/{prefix}full_predictions_{safe_name}.csv"
    full_df.to_csv(filepath, index=False)
    
    print(f"Full predictions saved to: {filepath}")
    return filepath


def create_summary_report(
    results: Dict[str, Dict],
    output_dir: str,
    analysis_type: str,
    prefix: str = ""
) -> str:
    """
    Create a summary report of all model results.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        Results dictionary containing metrics for each model
    output_dir : str
        Output directory
    analysis_type : str
        Type of analysis (e.g., 'Random Holdout', 'Temporal Holdout')
    prefix : str
        Prefix for output filename
    
    Returns
    -------
    str
        Path to the saved report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if prefix:
        prefix = f"{prefix}_"
    
    report_lines = [
        f"{'='*60}",
        f"TWS DOWNSCALING - {analysis_type.upper()} ANALYSIS REPORT",
        f"{'='*60}",
        ""
    ]
    
    for model_name, result in results.items():
        report_lines.extend([
            f"\n{'-'*40}",
            f"Model: {model_name}",
            f"{'-'*40}",
        ])
        
        if 'metrics' in result:
            metrics = result['metrics']
            report_lines.append(f"  MAE:   {metrics.mae:.4f}")
            report_lines.append(f"  RMSE:  {metrics.rmse:.4f}")
            report_lines.append(f"  R²:    {metrics.r2:.4f}")
            report_lines.append(f"  NSE:   {metrics.nse:.4f}")
            report_lines.append(f"  PBIAS: {metrics.pbias:.2f}%")
        
        if 'train_time' in result:
            report_lines.append(f"  Training Time: {result['train_time']:.2f}s")
    
    report_lines.extend([
        "",
        f"{'='*60}",
        "Report generated successfully",
        f"{'='*60}"
    ])
    
    report_text = '\n'.join(report_lines)
    filepath = f"{output_dir}/{prefix}summary_report.txt"
    
    with open(filepath, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    
    return filepath


# =============================================================================
# SHAP Analysis Functions
# =============================================================================

def compute_shap_values(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_type: str = 'auto',
    max_samples: int = 500,
    seed: int = 20
) -> Optional[Dict[str, Any]]:
    """
    Compute SHAP values for model interpretation.
    
    Parameters
    ----------
    model : Any
        Trained model (wrapper or underlying model)
    X_train : np.ndarray
        Training data for background (tree models) or reference (kernel)
    X_test : np.ndarray
        Test data to explain
    feature_names : List[str]
        Feature names for labeling
    model_type : str
        Model type: 'tree', 'neural', 'kernel', or 'auto'
    max_samples : int
        Maximum samples for background/explanation
    seed : int
        Random seed
    
    Returns
    -------
    Dict containing SHAP values and explainer, or None if SHAP unavailable
    """
    if not HAS_SHAP:
        print("SHAP not available. Skipping SHAP analysis.")
        return None
    
    np.random.seed(seed)
    
    # Subsample if necessary
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_background = X_train[idx]
    else:
        X_background = X_train
    
    if len(X_test) > max_samples:
        idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_explain = X_test[idx]
    else:
        X_explain = X_test
    
    # Get underlying model if wrapped
    underlying_model = model
    if hasattr(model, 'model'):
        underlying_model = model.model
    
    # Auto-detect model type
    if model_type == 'auto':
        model_name = type(underlying_model).__name__.lower()
        if any(t in model_name for t in ['xgb', 'lgb', 'lightgbm', 'randomforest', 'forest', 'tree', 'gradient']):
            model_type = 'tree'
        elif any(t in model_name for t in ['lstm', 'bilstm', 'neural', 'net', 'torch']):
            model_type = 'neural'
        else:
            model_type = 'kernel'
    
    try:
        if model_type == 'tree':
            # TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(underlying_model)
            shap_values = explainer.shap_values(X_explain)
            
        elif model_type == 'neural':
            # For neural networks, use DeepExplainer or KernelExplainer
            # First check if model has predict function for KernelExplainer
            if hasattr(model, 'predict'):
                predict_fn = model.predict
            else:
                predict_fn = underlying_model.predict
            
            # Use KernelExplainer for neural networks (more compatible)
            explainer = shap.KernelExplainer(predict_fn, X_background)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
            
        else:  # kernel
            # KernelExplainer for any model
            if hasattr(model, 'predict'):
                predict_fn = model.predict
            else:
                predict_fn = underlying_model.predict
            
            explainer = shap.KernelExplainer(predict_fn, X_background)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
        
        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'X_explain': X_explain,
            'feature_names': feature_names,
            'expected_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
        }
        
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_shap_summary(
    shap_result: Dict[str, Any],
    model_name: str,
    output_dir: str,
    prefix: str = "",
    max_display: int = 20
) -> None:
    """
    Create SHAP summary plot (beeswarm plot).
    
    Parameters
    ----------
    shap_result : Dict
        Result from compute_shap_values
    model_name : str
        Model name for labeling
    output_dir : str
        Output directory
    prefix : str
        Filename prefix
    max_display : int
        Maximum features to display
    """
    if shap_result is None or not HAS_SHAP:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    
    if prefix:
        prefix = f"{prefix}_"
    
    shap_values = shap_result['shap_values']
    X_explain = shap_result['X_explain']
    feature_names = shap_result['feature_names']
    
    # Handle multi-output case
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values[1]
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=clean_features(feature_names),
        max_display=max_display,
        show=False
    )
    plt.title(f"SHAP Summary Plot [{pretty_model(model_name)}]")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}shap_summary_{safe_name}.png", dpi=600, bbox_inches='tight')
    plt.close()


def plot_shap_bar(
    shap_result: Dict[str, Any],
    model_name: str,
    output_dir: str,
    prefix: str = "",
    max_display: int = 20
) -> None:
    """
    Create SHAP bar plot showing mean absolute SHAP values.
    
    Parameters
    ----------
    shap_result : Dict
        Result from compute_shap_values
    model_name : str
        Model name for labeling
    output_dir : str
        Output directory
    prefix : str
        Filename prefix
    max_display : int
        Maximum features to display
    """
    if shap_result is None or not HAS_SHAP:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    
    if prefix:
        prefix = f"{prefix}_"
    
    shap_values = shap_result['shap_values']
    feature_names = shap_result['feature_names']
    
    # Handle multi-output case
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values[1]
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_values = mean_abs_shap[sorted_idx]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_names)), sorted_values[::-1], align='center')
    plt.yticks(range(len(sorted_names)), [clean_feature(n) for n in sorted_names[::-1]])
    plt.xlabel("Mean |SHAP Value|")
    plt.title(f"SHAP Feature Importance [{pretty_model(model_name)}]")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}shap_bar_{safe_name}.png", dpi=600, bbox_inches='tight')
    plt.close()


def plot_shap_dependence(
    shap_result: Dict[str, Any],
    model_name: str,
    output_dir: str,
    prefix: str = "",
    top_n_features: int = 4
) -> None:
    """
    Create SHAP dependence plots for top features.
    
    Parameters
    ----------
    shap_result : Dict
        Result from compute_shap_values
    model_name : str
        Model name for labeling
    output_dir : str
        Output directory
    prefix : str
        Filename prefix
    top_n_features : int
        Number of top features to plot
    """
    if shap_result is None or not HAS_SHAP:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    
    if prefix:
        prefix = f"{prefix}_"
    
    shap_values = shap_result['shap_values']
    X_explain = shap_result['X_explain']
    feature_names = shap_result['feature_names']
    
    # Handle multi-output case
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values[1]
    
    # Get top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n_features]
    
    n_cols = 2
    n_rows = (top_n_features + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if top_n_features > 1 else [axes]
    
    for i, idx in enumerate(top_indices):
        ax = axes[i]
        shap.dependence_plot(
            idx,
            shap_values,
            X_explain,
            feature_names=clean_features(feature_names),
            ax=ax,
            show=False
        )
        ax.set_title(f"{clean_feature(feature_names[idx])}")
    
    # Hide unused subplots
    for i in range(len(top_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"SHAP Dependence Plots [{pretty_model(model_name)}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}shap_dependence_{safe_name}.png", dpi=600, bbox_inches='tight')
    plt.close()


def plot_shap_waterfall(
    shap_result: Dict[str, Any],
    model_name: str,
    output_dir: str,
    prefix: str = "",
    sample_idx: int = 0
) -> None:
    """
    Create SHAP waterfall plot for a single prediction.
    
    Parameters
    ----------
    shap_result : Dict
        Result from compute_shap_values
    model_name : str
        Model name for labeling
    output_dir : str
        Output directory
    prefix : str
        Filename prefix
    sample_idx : int
        Index of sample to explain
    """
    if shap_result is None or not HAS_SHAP:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    
    if prefix:
        prefix = f"{prefix}_"
    
    shap_values = shap_result['shap_values']
    X_explain = shap_result['X_explain']
    feature_names = shap_result['feature_names']
    expected_value = shap_result['expected_value']
    
    # Handle multi-output case
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values[1]
    
    if expected_value is None:
        expected_value = 0
    elif isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0] if len(expected_value) > 0 else 0
    
    # Create Explanation object for waterfall
    try:
        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=expected_value,
            data=X_explain[sample_idx],
            feature_names=clean_features(feature_names)
        )
        
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(f"SHAP Waterfall Plot (Sample {sample_idx}) [{pretty_model(model_name)}]")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}shap_waterfall_{safe_name}.png", dpi=600, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not create waterfall plot: {e}")


def run_shap_analysis(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str,
    output_dir: str,
    prefix: str = "",
    model_type: str = 'auto',
    max_samples: int = 500,
    seed: int = 20
) -> Optional[Dict[str, Any]]:
    """
    Run complete SHAP analysis and generate all plots.
    
    Parameters
    ----------
    model : Any
        Trained model
    X_train : np.ndarray
        Training data
    X_test : np.ndarray
        Test data
    feature_names : List[str]
        Feature names
    model_name : str
        Model name
    output_dir : str
        Output directory
    prefix : str
        Filename prefix
    model_type : str
        Model type for SHAP
    max_samples : int
        Max samples for computation
    seed : int
        Random seed
    
    Returns
    -------
    Dict with SHAP results or None
    """
    if not HAS_SHAP:
        print(f"[{pretty_model(model_name)}] SHAP analysis skipped - SHAP not installed")
        return None
    
    print(f"\n[{pretty_model(model_name)}] Running SHAP analysis...")
    
    # Compute SHAP values
    shap_result = compute_shap_values(
        model, X_train, X_test, feature_names,
        model_type=model_type, max_samples=max_samples, seed=seed
    )
    
    if shap_result is None:
        print(f"[{pretty_model(model_name)}] SHAP computation failed")
        return None
    
    # Generate plots
    print(f"[{pretty_model(model_name)}] Generating SHAP summary plot...")
    plot_shap_summary(shap_result, model_name, output_dir, prefix)
    
    print(f"[{pretty_model(model_name)}] Generating SHAP bar plot...")
    plot_shap_bar(shap_result, model_name, output_dir, prefix)
    
    print(f"[{pretty_model(model_name)}] Generating SHAP dependence plots...")
    plot_shap_dependence(shap_result, model_name, output_dir, prefix)
    
    print(f"[{pretty_model(model_name)}] Generating SHAP waterfall plot...")
    plot_shap_waterfall(shap_result, model_name, output_dir, prefix)
    
    # Save SHAP values to CSV
    shap_values = shap_result['shap_values']
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values[1]
    
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    if prefix:
        prefix = f"{prefix}_"
    
    # Mean absolute SHAP values
    mean_shap = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0),
        'Mean_SHAP': shap_values.mean(axis=0),
        'Std_SHAP': shap_values.std(axis=0)
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    mean_shap.to_csv(f"{output_dir}/{prefix}shap_importance_{safe_name}.csv", index=False)
    
    print(f"[{pretty_model(model_name)}] SHAP analysis complete!")
    
    return shap_result
