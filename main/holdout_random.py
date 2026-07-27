"""
Random Holdout Analysis for TWS Downscaling
Performs random train-test split for model evaluation.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, KFold

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


def random_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    test_size: float = 0.2,
    seed: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform random train-test split.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    dates : np.ndarray
        Date array
    test_size : float
        Proportion of data for testing
    seed : int
        Random seed
    
    Returns
    -------
    Tuple containing train/test splits for X, y, and dates
    """
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    dates_train, dates_test = dates[train_idx], dates[test_idx]
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test


def random_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_splits: int = 5,
    seed: int = 20,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform K-Fold cross-validation with random splits.
    
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
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_metrics = {
        'MAE': [], 'RMSE': [], 'R2': [], 'NSE': [], 'PBIAS': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
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
    print(f"Cross-Validation Summary for {model_name}")
    print(f"{'='*50}")
    for metric_name, values in fold_metrics.items():
        print(f"  {metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return fold_metrics


def run_random_holdout_analysis(
    predictor_file: str,
    tws_file: str,
    output_dir: str = "figures/random_holdout",
    predictors: List[str] = None,
    lags: int = 7,
    test_size: float = 0.2,
    seed: int = 20,
    models_to_run: List[str] = None,
    tuned_params: dict = None,
    verbose: bool = True,
    run_shap: bool = True,
    shap_max_samples: int = 500
) -> Dict[str, Dict]:
    """
    Run random holdout analysis for all specified models.
    
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
        Proportion for test set
    seed : int
        Random seed
    models_to_run : List[str]
        List of model names to evaluate (None = all)
    verbose : bool
        Print progress
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
    
    # Random split
    X_train, X_test, y_train, y_test, dates_train, dates_test = random_holdout_split(
        X, y, dates, test_size=test_size, seed=seed
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    results = {}
    metrics_dict = {}
    
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
            
            # Plot predictions on test set
            # Sort by date for better visualization
            sort_idx = np.argsort(dates_test)
            plot_predictions(
                dates_test[sort_idx],
                y_test[sort_idx],
                y_test_pred[sort_idx],
                model.name,
                output_dir,
                prefix="random_test"
            )
            
            # Plot training loss for neural networks
            if hasattr(model, 'train_losses') and model.train_losses:
                plot_training_loss(model.train_losses, model.name, output_dir, prefix="random")
            
            # Plot feature importance for tree models
            importance = model.get_feature_importance(feature_names)
            if importance:
                plot_feature_importance(importance, model.name, output_dir, prefix="random")
            
            # Save predictions
            save_predictions(dates_test, y_test, y_test_pred, model.name, output_dir, prefix="random_test")
            
            # Save full predictions (train + test)
            save_full_predictions(
                dates_train, y_train, y_train_pred,
                dates_test, y_test, y_test_pred,
                model.name, output_dir, prefix="random"
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
                    prefix="random",
                    max_samples=shap_max_samples,
                    seed=seed
                )
                if shap_result is not None:
                    results[model.name]['shap_result'] = shap_result
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Model comparison
    if len(metrics_dict) > 1:
        plot_model_comparison(metrics_dict, output_dir, prefix="random")
    
    # Train vs Test comparison (overfitting analysis)
    train_metrics_dict = {name: res['train_metrics'] for name, res in results.items() if 'train_metrics' in res}
    if len(train_metrics_dict) > 1 and len(metrics_dict) > 1:
        plot_train_test_comparison(train_metrics_dict, metrics_dict, output_dir, prefix="random")
    
    # Create summary report
    create_summary_report(results, output_dir, "Random Holdout", prefix="random")

    # Statistical outputs: bootstrap CIs, model-comparison significance, split report
    # (reviewer requests: uncertainty on metrics + statistical model comparison).
    from stats_utils import emit_holdout_statistics
    test_preds = {name: res['y_test_pred'] for name, res in results.items() if 'y_test_pred' in res}
    if test_preds:
        emit_holdout_statistics(
            y_test, test_preds, output_dir, prefix="random",
            dates_train=dates_train, dates_test=dates_test,
            n_features=len(feature_names), seed=seed,
        )

    return results


def main():
    """Main function for random holdout analysis."""
    # Configuration
    predictor_file = "All_Data.xlsx"
    tws_file = "TWS_JPL.xlsx"
    output_dir = "figures/random_holdout"
    
    # Check if data files exist
    if not os.path.exists(predictor_file):
        print(f"Error: {predictor_file} not found!")
        print("Please ensure the data files are in the current directory.")
        return
    
    if not os.path.exists(tws_file):
        print(f"Error: {tws_file} not found!")
        print("Please ensure the data files are in the current directory.")
        return
    
    # Run analysis
    results = run_random_holdout_analysis(
        predictor_file=predictor_file,
        tws_file=tws_file,
        output_dir=output_dir,
        test_size=0.2,
        seed=20,
        models_to_run=['bilstm_attention', 'bilstm', 'lstm', 'xgboost', 'lightgbm', 'randomforest']
    )
    
    print(f"\n{'='*50}")
    print("Random Holdout Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
