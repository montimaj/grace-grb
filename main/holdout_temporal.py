"""
Temporal Holdout Analysis for TWS Downscaling
Performs time-based train-test split maintaining temporal order.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit

from models import (
    get_model, get_all_models, set_seed,
    BiLSTMAttentionWrapper, BiLSTMWrapper, LSTMWrapper,
    XGBoostWrapper, LightGBMWrapper, RandomForestWrapper
)
from utils import (
    load_and_preprocess_data, create_lagged_features, prepare_features_target,
    calculate_metrics, plot_predictions, plot_training_loss,
    plot_feature_importance, plot_model_comparison, plot_train_test_comparison,
    save_predictions, save_full_predictions, create_summary_report, 
    EvaluationMetrics, run_shap_analysis
)


def temporal_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    test_size: float = 0.2,
    validation_size: float = 0.0
) -> Dict[str, np.ndarray]:
    """
    Perform temporal (chronological) train-test split.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    dates : np.ndarray
        Date array
    test_size : float
        Proportion of data for testing (from end)
    validation_size : float
        Proportion of data for validation (before test)
    
    Returns
    -------
    Dict containing train/val/test splits
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * validation_size)
    n_train = n_samples - n_test - n_val
    
    result = {
        'X_train': X[:n_train],
        'y_train': y[:n_train],
        'dates_train': dates[:n_train],
    }
    
    if validation_size > 0:
        result['X_val'] = X[n_train:n_train + n_val]
        result['y_val'] = y[n_train:n_train + n_val]
        result['dates_val'] = dates[n_train:n_train + n_val]
    
    result['X_test'] = X[-n_test:]
    result['y_test'] = y[-n_test:]
    result['dates_test'] = dates[-n_test:]
    
    return result


def temporal_split_by_year(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    test_years: List[int] = None,
    train_end_year: int = None
) -> Dict[str, np.ndarray]:
    """
    Split data by specific years.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    dates : np.ndarray
        Date array
    test_years : List[int]
        Years to use for testing
    train_end_year : int
        Last year for training data
    
    Returns
    -------
    Dict containing train/test splits
    """
    dates_pd = pd.to_datetime(dates)
    years = dates_pd.year
    
    if test_years is not None:
        train_mask = ~years.isin(test_years)
        test_mask = years.isin(test_years)
    elif train_end_year is not None:
        train_mask = years <= train_end_year
        test_mask = years > train_end_year
    else:
        raise ValueError("Either test_years or train_end_year must be specified")
    
    return {
        'X_train': X[train_mask],
        'y_train': y[train_mask],
        'dates_train': dates[train_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'dates_test': dates[test_mask],
    }


def temporal_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_splits: int = 5,
    seed: int = 20,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform time series cross-validation (walk-forward validation).
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    model_name : str
        Name of the model to use
    n_splits : int
        Number of CV folds
    seed : int
        Random seed
    **model_kwargs
        Additional arguments for model
    
    Returns
    -------
    Dict with fold-wise metrics
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = {
        'MAE': [], 'RMSE': [], 'R2': [], 'NSE': [], 'PBIAS': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        print(f"  Train: indices {train_idx[0]}-{train_idx[-1]} ({len(train_idx)} samples)")
        print(f"  Test:  indices {val_idx[0]}-{val_idx[-1]} ({len(val_idx)} samples)")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = get_model(model_name, seed=seed, **model_kwargs)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)
        
        metrics = calculate_metrics(y_val, y_pred)
        
        for key in fold_metrics:
            fold_metrics[key].append(getattr(metrics, key.lower()))
        
        print(f"  {metrics}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Time Series CV Summary for {model_name}")
    print(f"{'='*50}")
    for metric_name, values in fold_metrics.items():
        print(f"  {metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return fold_metrics


def plot_temporal_split(
    dates_train: np.ndarray,
    y_train: np.ndarray,
    dates_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    prefix: str = ""
) -> None:
    """Visualize the temporal split of data."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    if prefix:
        prefix = f"{prefix}_"
    
    plt.figure(figsize=(14, 5))
    plt.plot(dates_train, y_train, label='Training Data', alpha=0.8)
    plt.plot(dates_test, y_test, label='Test Data', alpha=0.8, color='orange')
    plt.axvline(dates_test[0], color='red', linestyle='--', label='Train/Test Split')
    plt.xlabel("Date")
    plt.ylabel("TWSA (mm)")
    plt.title("Temporal Data Split Visualization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}temporal_split.png", dpi=600)
    plt.close()


def run_temporal_holdout_analysis(
    predictor_file: str,
    tws_file: str,
    output_dir: str = "figures/temporal_holdout",
    predictors: List[str] = None,
    lags: int = 7,
    test_size: float = 0.2,
    test_years: List[int] = None,
    train_end_year: int = None,
    seed: int = 20,
    models_to_run: List[str] = None,
    tuned_params: dict = None,
    verbose: bool = True,
    run_cv: bool = False,
    cv_splits: int = 5,
    run_shap: bool = True,
    shap_max_samples: int = 500
) -> Dict[str, Dict]:
    """
    Run temporal holdout analysis for all specified models.
    
    Parameters
    ----------
    predictor_file : str
        Path to predictor data file
    tws_file : str
        Path to TWS data file
    output_dir : str
        Output directory for results
    predictors : List[str]
        List of predictor variables
    lags : int
        Number of lag periods
    test_size : float
        Proportion for test set (if test_years and train_end_year are None)
    test_years : List[int]
        Specific years for testing
    train_end_year : int
        Last year for training
    seed : int
        Random seed
    models_to_run : List[str]
        List of model names to evaluate (None = all)
    verbose : bool
        Print progress
    run_cv : bool
        Whether to run time series cross-validation
    cv_splits : int
        Number of CV splits
    run_shap : bool
        Whether to run SHAP analysis
    shap_max_samples : int
        Maximum samples for SHAP computation
    
    Returns
    -------
    Dict containing results for each model
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    if predictors is None:
        predictors = ['SMS', 'ET', 'rainfall', 'runoff', 'GWSA']
    
    if models_to_run is None:
        models_to_run = ['bilstm_attention', 'bilstm', 'lstm', 'xgboost', 'lightgbm', 'randomforest']
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    merged = load_and_preprocess_data(predictor_file, tws_file, predictors)
    lagged = create_lagged_features(merged.copy(), predictors, lags=lags)
    X, y, feature_names = prepare_features_target(lagged, predictors, 'TWS', lags)
    dates = lagged['Date'].values
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Date range: {pd.to_datetime(dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')}")
    
    # Temporal split
    if test_years is not None or train_end_year is not None:
        split_data = temporal_split_by_year(X, y, dates, test_years, train_end_year)
        split_type = f"Year-based (test years: {test_years or f'>{train_end_year}'})"
    else:
        split_data = temporal_holdout_split(X, y, dates, test_size=test_size)
        split_type = f"Proportion-based (test size: {test_size})"
    
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    dates_train = split_data['dates_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    dates_test = split_data['dates_test']
    
    print(f"Split type: {split_type}")
    print(f"Training samples: {len(X_train)} ({pd.to_datetime(dates_train[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(dates_train[-1]).strftime('%Y-%m-%d')})")
    print(f"Test samples: {len(X_test)} ({pd.to_datetime(dates_test[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(dates_test[-1]).strftime('%Y-%m-%d')})")
    
    # Plot temporal split
    plot_temporal_split(dates_train, y_train, dates_test, y_test, output_dir, prefix="temporal")
    
    results = {}
    metrics_dict = {}
    cv_results = {}
    
    for model_name in models_to_run:
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            model = get_model(model_name, seed=seed, **((tuned_params or {}).get(model_name, {})))
            model.fit(X_train, y_train, verbose=verbose)
            
            train_time = time.time() - start_time
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_metrics = calculate_metrics(y_train, y_train_pred)
            test_metrics = calculate_metrics(y_test, y_test_pred)
            
            print(f"\nTraining Metrics: {train_metrics}")
            print(f"Test Metrics: {test_metrics}")
            
            # Save results
            results[model.name] = {
                'model': model,
                'train_metrics': train_metrics,
                'metrics': test_metrics,
                'train_time': train_time,
                'y_test_pred': y_test_pred,
                'dates_test': dates_test
            }
            
            metrics_dict[model.name] = test_metrics
            
            # Plot predictions on test set (temporal order preserved)
            plot_predictions(
                dates_test,
                y_test,
                y_test_pred,
                model.name,
                output_dir,
                prefix="temporal_test"
            )
            
            # Also plot full timeline with predictions
            y_full_pred = model.predict(X)
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(14, 5))
            plt.plot(dates, y, label='Actual TWSA', alpha=0.8)
            plt.plot(dates, y_full_pred, label='Predicted TWSA', linestyle='--', alpha=0.8)
            plt.axvline(dates_test[0], color='red', linestyle='--', linewidth=2, label='Train/Test Split')
            plt.fill_between(dates, y.min(), y.max(), 
                           where=[d >= dates_test[0] for d in dates],
                           alpha=0.1, color='red', label='Test Period')
            from plot_style import pretty_model
            plt.xlabel("Date")
            plt.ylabel("TWSA (mm)")
            plt.title(f"Full Timeline Prediction [{pretty_model(model.name)}]")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            safe_name = model.name.replace('+', '_').replace(' ', '_')
            plt.savefig(f"{output_dir}/temporal_full_timeline_{safe_name}.png", dpi=600)
            plt.close()
            
            # Plot training loss for neural networks
            if hasattr(model, 'train_losses') and model.train_losses:
                plot_training_loss(model.train_losses, model.name, output_dir, prefix="temporal")
            
            # Plot feature importance for tree models
            importance = model.get_feature_importance(feature_names)
            if importance:
                plot_feature_importance(importance, model.name, output_dir, prefix="temporal")
            
            # Save predictions (test only)
            save_predictions(dates_test, y_test, y_test_pred, model.name, output_dir, prefix="temporal_test")
            
            # Save full predictions (train + test)
            save_full_predictions(
                dates_train, y_train, y_train_pred,
                dates_test, y_test, y_test_pred,
                model.name, output_dir, prefix="temporal"
            )
            
            # Run SHAP analysis
            if run_shap:
                shap_result = run_shap_analysis(
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=feature_names,
                    model_name=model.name,
                    output_dir=output_dir,
                    prefix="temporal",
                    max_samples=shap_max_samples,
                    seed=seed
                )
                if shap_result is not None:
                    results[model.name]['shap_result'] = shap_result
            
            # Run time series CV if requested
            if run_cv:
                print(f"\nRunning Time Series Cross-Validation for {model.name}...")
                cv_result = temporal_cv(X, y, model_name, n_splits=cv_splits, seed=seed)
                cv_results[model.name] = cv_result
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Model comparison
    if len(metrics_dict) > 1:
        plot_model_comparison(metrics_dict, output_dir, prefix="temporal")
    
    # Train vs Test comparison (overfitting analysis)
    train_metrics_dict = {name: res['train_metrics'] for name, res in results.items() if 'train_metrics' in res}
    if len(train_metrics_dict) > 1 and len(metrics_dict) > 1:
        plot_train_test_comparison(train_metrics_dict, metrics_dict, output_dir, prefix="temporal")
    
    # Create summary report
    create_summary_report(results, output_dir, "Temporal Holdout", prefix="temporal")
    
    # Save CV results if available
    if cv_results:
        cv_summary = {}
        for model_name, cv_metrics in cv_results.items():
            cv_summary[model_name] = {
                metric: f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                for metric, values in cv_metrics.items()
            }
        cv_df = pd.DataFrame(cv_summary).T
        cv_df.to_csv(f"{output_dir}/temporal_cv_results.csv")

    # Statistical outputs: bootstrap CIs, model-comparison significance, split report.
    from stats_utils import emit_holdout_statistics
    test_preds = {name: res['y_test_pred'] for name, res in results.items() if 'y_test_pred' in res}
    if test_preds:
        emit_holdout_statistics(
            y_test, test_preds, output_dir, prefix="temporal",
            dates_train=dates_train, dates_test=dates_test,
            n_features=len(feature_names), seed=seed,
        )

    # Temporal closure test: re-aggregate predicted daily TWSA to monthly and
    # compare against the ORIGINAL observed monthly GRACE (reviewer request for
    # explicit daily-reconstruction validation via temporal self-consistency).
    try:
        from temporal_closure_validation import run_temporal_closure_validation
        run_temporal_closure_validation(
            predictions_dir=output_dir, tws_file=tws_file, seed=seed,
        )
    except Exception as e:
        print(f"Temporal closure validation skipped: {e}")

    return results


def main():
    """Main function for temporal holdout analysis."""
    # Configuration
    predictor_file = "All_Data.xlsx"
    tws_file = "TWS_JPL.xlsx"
    output_dir = "figures/temporal_holdout"
    
    # Check if data files exist
    if not os.path.exists(predictor_file):
        print(f"Error: {predictor_file} not found!")
        print("Please ensure the data files are in the current directory.")
        return
    
    if not os.path.exists(tws_file):
        print(f"Error: {tws_file} not found!")
        print("Please ensure the data files are in the current directory.")
        return
    
    # Run analysis with temporal split
    # Option 1: Use last 20% of data for testing
    results = run_temporal_holdout_analysis(
        predictor_file=predictor_file,
        tws_file=tws_file,
        output_dir=output_dir,
        test_size=0.2,  # Last 20% of timeline for testing
        seed=20,
        models_to_run=['bilstm_attention', 'bilstm', 'lstm', 'xgboost', 'lightgbm', 'randomforest'],
        run_cv=True,
        cv_splits=5
    )
    
    # Option 2 (alternative): Use specific years for testing
    # Uncomment to use year-based split:
    # results = run_temporal_holdout_analysis(
    #     predictor_file=predictor_file,
    #     tws_file=tws_file,
    #     output_dir=f"{output_dir}_by_year",
    #     train_end_year=2020,  # Train on data up to 2020, test on 2021+
    #     seed=20,
    #     models_to_run=['bilstm_attention', 'bilstm', 'lstm', 'xgboost', 'lightgbm', 'randomforest'],
    # )
    
    print(f"\n{'='*50}")
    print("Temporal Holdout Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
