"""
Spatial Holdout Analysis for TWS Downscaling
Performs location-based train-test split for evaluating spatial generalization.
Note: This requires spatially-distributed data with latitude and longitude columns, which is not available in the current dataset.
Hence, this script creates synthetic spatial data by replicating temporal data across multiple locations with added noise.
"""

import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.cluster import KMeans

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


def load_spatial_data(
    data_file: str,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    date_col: str = 'Date'
) -> pd.DataFrame:
    """
    Load data with spatial coordinates.
    
    Parameters
    ----------
    data_file : str
        Path to data file (CSV or Excel)
    lat_col : str
        Column name for latitude
    lon_col : str
        Column name for longitude
    date_col : str
        Column name for date
    
    Returns
    -------
    pd.DataFrame
        Loaded data with spatial information
    """
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.xlsx') or data_file.endswith('.xls'):
        df = pd.read_excel(data_file)
    else:
        raise ValueError(f"Unsupported file format: {data_file}")
    
    # Ensure coordinates exist
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns {lat_col} and {lon_col} must exist in the data")
    
    # Parse dates if needed
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    
    return df


def create_spatial_groups(
    coords: np.ndarray,
    method: str = 'kmeans',
    n_groups: int = 5,
    seed: int = 20
) -> np.ndarray:
    """
    Create spatial groups for cross-validation.
    
    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n_samples, 2) with [lat, lon] coordinates
    method : str
        Grouping method: 'kmeans', 'grid', 'region'
    n_groups : int
        Number of groups to create
    seed : int
        Random seed
    
    Returns
    -------
    np.ndarray
        Group labels for each sample
    """
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_groups, random_state=seed, n_init=10)
        groups = kmeans.fit_predict(coords)
    
    elif method == 'grid':
        # Create grid-based groups
        lat_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), int(np.sqrt(n_groups)) + 1)
        lon_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), int(np.sqrt(n_groups)) + 1)
        
        lat_groups = np.digitize(coords[:, 0], lat_bins) - 1
        lon_groups = np.digitize(coords[:, 1], lon_bins) - 1
        
        groups = lat_groups * len(lon_bins) + lon_groups
        # Remap to consecutive integers
        unique_groups = np.unique(groups)
        group_map = {g: i for i, g in enumerate(unique_groups)}
        groups = np.array([group_map[g] for g in groups])
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'grid'")
    
    return groups


def spatial_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    groups: np.ndarray,
    test_groups: List[int] = None,
    test_fraction: float = 0.2,
    seed: int = 20
) -> Dict[str, np.ndarray]:
    """
    Perform spatial holdout split based on location groups.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    dates : np.ndarray
        Date array
    groups : np.ndarray
        Spatial group labels
    test_groups : List[int]
        Specific groups to use for testing
    test_fraction : float
        Fraction of groups for testing (if test_groups is None)
    seed : int
        Random seed
    
    Returns
    -------
    Dict containing train/test splits with group information
    """
    unique_groups = np.unique(groups)
    
    if test_groups is None:
        np.random.seed(seed)
        n_test_groups = max(1, int(len(unique_groups) * test_fraction))
        test_groups = np.random.choice(unique_groups, n_test_groups, replace=False)
    
    train_mask = ~np.isin(groups, test_groups)
    test_mask = np.isin(groups, test_groups)
    
    return {
        'X_train': X[train_mask],
        'y_train': y[train_mask],
        'dates_train': dates[train_mask],
        'groups_train': groups[train_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'dates_test': dates[test_mask],
        'groups_test': groups[test_mask],
        'test_groups': test_groups,
        'train_groups': unique_groups[~np.isin(unique_groups, test_groups)]
    }


def spatial_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    seed: int = 20,
    leave_one_out: bool = False,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform spatial cross-validation (leave-one-region-out or grouped k-fold).
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    groups : np.ndarray
        Spatial group labels
    model_name : str
        Name of the model to use
    seed : int
        Random seed
    leave_one_out : bool
        Use leave-one-group-out instead of GroupKFold
    **model_kwargs
        Additional arguments for model
    
    Returns
    -------
    Dict with fold-wise metrics
    """
    if leave_one_out:
        cv = LeaveOneGroupOut()
    else:
        n_groups = len(np.unique(groups))
        cv = GroupKFold(n_splits=min(n_groups, 5))
    
    fold_metrics = {
        'MAE': [], 'RMSE': [], 'R2': [], 'NSE': [], 'PBIAS': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        test_groups_fold = np.unique(groups[val_idx])
        print(f"\n--- Fold {fold + 1} ---")
        print(f"  Test groups: {test_groups_fold}")
        print(f"  Train samples: {len(train_idx)}, Test samples: {len(val_idx)}")
        
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
    print(f"Spatial CV Summary for {model_name}")
    print(f"{'='*50}")
    for metric_name, values in fold_metrics.items():
        print(f"  {metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return fold_metrics


def plot_spatial_groups(
    coords: np.ndarray,
    groups: np.ndarray,
    test_groups: np.ndarray = None,
    output_dir: str = "figures",
    prefix: str = ""
) -> None:
    """Visualize spatial groups and train/test split."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    if prefix:
        prefix = f"{prefix}_"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: All groups
    ax1 = axes[0]
    scatter = ax1.scatter(coords[:, 1], coords[:, 0], c=groups, cmap='tab10', alpha=0.6, s=10)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Spatial Groups")
    plt.colorbar(scatter, ax=ax1, label='Group')
    
    # Plot 2: Train/Test split
    ax2 = axes[1]
    if test_groups is not None:
        train_mask = ~np.isin(groups, test_groups)
        test_mask = np.isin(groups, test_groups)
        ax2.scatter(coords[train_mask, 1], coords[train_mask, 0], 
                   c='blue', alpha=0.5, s=10, label='Training')
        ax2.scatter(coords[test_mask, 1], coords[test_mask, 0], 
                   c='red', alpha=0.5, s=10, label='Testing')
        ax2.legend()
    else:
        ax2.scatter(coords[:, 1], coords[:, 0], c=groups, cmap='tab10', alpha=0.6, s=10)
    
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Train/Test Spatial Split")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}spatial_groups.png", dpi=600)
    plt.close()


def run_spatial_holdout_analysis(
    data_file: str,
    output_dir: str = "figures/spatial_holdout",
    predictors: List[str] = None,
    target_col: str = 'TWS',
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    date_col: str = 'Date',
    lags: int = 7,
    n_spatial_groups: int = 5,
    grouping_method: str = 'kmeans',
    test_groups: List[int] = None,
    test_fraction: float = 0.2,
    seed: int = 20,
    models_to_run: List[str] = None,
    verbose: bool = True,
    run_cv: bool = True,
    run_shap: bool = True,
    shap_max_samples: int = 500
) -> Dict[str, Dict]:
    """
    Run spatial holdout analysis for all specified models.
    
    Parameters
    ----------
    data_file : str
        Path to data file with spatial coordinates
    output_dir : str
        Output directory for results
    predictors : List[str]
        List of predictor variables
    target_col : str
        Target column name
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    date_col : str
        Date column name
    lags : int
        Number of lag periods
    n_spatial_groups : int
        Number of spatial groups to create
    grouping_method : str
        Method for creating spatial groups
    test_groups : List[int]
        Specific groups for testing
    test_fraction : float
        Fraction of groups for testing
    seed : int
        Random seed
    models_to_run : List[str]
        List of model names to evaluate
    verbose : bool
        Print progress
    run_cv : bool
        Whether to run spatial cross-validation
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

    # --------------------------------------------------------------------- #
    # IMPORTANT CAVEAT (documented for reviewers).
    # This "spatial" holdout runs on SYNTHETIC data: the single basin-averaged
    # time series is replicated across a handful of fabricated point locations
    # with small distance-scaled noise. Because train and test locations share
    # near-identical replicates of the SAME series, the split does not measure
    # genuine spatial transferability, and near-perfect scores (e.g. R2 ~ 0.99)
    # are an artefact of that replication (information leakage by construction).
    # Real spatial generalisation requires gridded (per-pixel) GRACE + gridded
    # predictors, which are not part of this basin-averaged dataset. Treat the
    # results below as a noise-sensitivity demonstration only.
    # --------------------------------------------------------------------- #
    warn = ("SYNTHETIC SPATIAL HOLDOUT: replicated basin series + noise; "
            "scores reflect sensitivity to added noise, NOT spatial transferability.")
    print("\n" + "!" * 78 + f"\n{warn}\n" + "!" * 78)
    with open(os.path.join(output_dir, "SYNTHETIC_DATA_DISCLAIMER.txt"), "w") as fh:
        fh.write(warn + "\n")

    if predictors is None:
        predictors = ['SMS', 'ET', 'rainfall', 'runoff', 'GWSA']

    if models_to_run is None:
        models_to_run = ['bilstm_attention', 'bilstm', 'lstm', 'xgboost', 'lightgbm', 'randomforest']

    # Load data
    print("Loading spatial data...")
    df = load_spatial_data(data_file, lat_col, lon_col, date_col)
    
    # Create lagged features
    print("Creating lagged features...")
    lagged = create_lagged_features(df.copy(), predictors, lags=lags)
    
    # Prepare features and target
    feature_names = [f'{var}_lag{i}' for var in predictors for i in range(1, lags + 1)]
    X = lagged[feature_names].values
    y = lagged[target_col].values
    dates = lagged[date_col].values
    coords = lagged[[lat_col, lon_col]].values
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Coordinate range: Lat [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
          f"Lon [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    
    # Create spatial groups
    print(f"Creating {n_spatial_groups} spatial groups using {grouping_method}...")
    groups = create_spatial_groups(coords, method=grouping_method, n_groups=n_spatial_groups, seed=seed)
    unique_groups = np.unique(groups)
    
    print(f"Groups created: {unique_groups}")
    for g in unique_groups:
        print(f"  Group {g}: {np.sum(groups == g)} samples")
    
    # Spatial split
    split_data = spatial_holdout_split(X, y, dates, groups, test_groups, test_fraction, seed)
    
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    dates_train = split_data['dates_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    dates_test = split_data['dates_test']
    actual_test_groups = split_data['test_groups']
    
    print(f"\nSpatial Split:")
    print(f"  Training samples: {len(X_train)} (groups: {split_data['train_groups']})")
    print(f"  Test samples: {len(X_test)} (groups: {actual_test_groups})")
    
    # Plot spatial groups
    plot_spatial_groups(coords, groups, actual_test_groups, output_dir, prefix="spatial")
    
    results = {}
    metrics_dict = {}
    cv_results = {}
    
    for model_name in models_to_run:
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            model = get_model(model_name, seed=seed)
            model.fit(X_train, y_train, verbose=verbose)
            
            train_time = time.time() - start_time
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_metrics = calculate_metrics(y_train, y_train_pred)
            test_metrics = calculate_metrics(y_test, y_test_pred)
            
            print(f"\nTraining Metrics: {train_metrics}")
            print(f"Test Metrics (held-out regions): {test_metrics}")
            
            # Save results
            results[model.name] = {
                'model': model,
                'train_metrics': train_metrics,
                'metrics': test_metrics,
                'train_time': train_time,
                'y_test_pred': y_test_pred,
                'dates_test': dates_test,
                'test_groups': actual_test_groups
            }
            
            metrics_dict[model.name] = test_metrics
            
            # Sort by date for visualization
            sort_idx = np.argsort(dates_test)
            plot_predictions(
                dates_test[sort_idx],
                y_test[sort_idx],
                y_test_pred[sort_idx],
                model.name,
                output_dir,
                prefix="spatial_test"
            )
            
            # Plot training loss for neural networks
            if hasattr(model, 'train_losses') and model.train_losses:
                plot_training_loss(model.train_losses, model.name, output_dir, prefix="spatial")
            
            # Plot feature importance for tree models
            importance = model.get_feature_importance(feature_names)
            if importance:
                plot_feature_importance(importance, model.name, output_dir, prefix="spatial")
            
            # Save predictions
            save_predictions(dates_test, y_test, y_test_pred, model.name, output_dir, prefix="spatial_test")
            
            # Save full predictions (train + test)
            save_full_predictions(
                dates_train, y_train, y_train_pred,
                dates_test, y_test, y_test_pred,
                model.name, output_dir, prefix="spatial"
            )
            
            # Plot predictions by test group
            import matplotlib.pyplot as plt
            groups_test = split_data['groups_test']
            
            fig, axes = plt.subplots(1, len(actual_test_groups), figsize=(6 * len(actual_test_groups), 5))
            if len(actual_test_groups) == 1:
                axes = [axes]
            
            for i, tg in enumerate(actual_test_groups):
                mask = groups_test == tg
                ax = axes[i]
                ax.scatter(y_test[mask], y_test_pred[mask], alpha=0.5, s=10)
                ax.plot([y_test[mask].min(), y_test[mask].max()],
                       [y_test[mask].min(), y_test[mask].max()], 'r--')
                group_metrics = calculate_metrics(y_test[mask], y_test_pred[mask])
                ax.set_title(f"Group {tg} (R²={group_metrics.r2:.3f})")
                ax.set_xlabel("Actual TWSA (mm)")
                ax.set_ylabel("Predicted TWSA (mm)")
                ax.grid(True, alpha=0.3)
            
            from plot_style import pretty_model
            plt.suptitle(f"Predictions by Spatial Group [{pretty_model(model.name)}]")
            plt.tight_layout()
            safe_name = model.name.replace('+', '_').replace(' ', '_')
            plt.savefig(f"{output_dir}/spatial_group_scatter_{safe_name}.png", dpi=600)
            plt.close()
            
            # Run SHAP analysis
            if run_shap:
                shap_result = run_shap_analysis(
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=feature_names,
                    model_name=model.name,
                    output_dir=output_dir,
                    prefix="spatial",
                    max_samples=shap_max_samples,
                    seed=seed
                )
                if shap_result is not None:
                    results[model.name]['shap_result'] = shap_result
            
            # Run spatial CV if requested
            if run_cv:
                print(f"\nRunning Spatial Cross-Validation for {model.name}...")
                cv_result = spatial_cv(X, y, groups, model_name, seed=seed)
                cv_results[model.name] = cv_result
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Model comparison
    if len(metrics_dict) > 1:
        plot_model_comparison(metrics_dict, output_dir, prefix="spatial")
    
    # Train vs Test comparison (overfitting analysis)
    train_metrics_dict = {name: res['train_metrics'] for name, res in results.items() if 'train_metrics' in res}
    if len(train_metrics_dict) > 1 and len(metrics_dict) > 1:
        plot_train_test_comparison(train_metrics_dict, metrics_dict, output_dir, prefix="spatial")
    
    # Create summary report
    create_summary_report(results, output_dir, "Spatial Holdout", prefix="spatial")
    
    # Save CV results if available
    if cv_results:
        cv_summary = {}
        for model_name, cv_metrics in cv_results.items():
            cv_summary[model_name] = {
                metric: f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                for metric, values in cv_metrics.items()
            }
        cv_df = pd.DataFrame(cv_summary).T
        cv_df.to_csv(f"{output_dir}/spatial_cv_results.csv")

    # Statistical outputs (note: on synthetic replicated data; see disclaimer).
    from stats_utils import emit_holdout_statistics
    test_preds = {name: res['y_test_pred'] for name, res in results.items() if 'y_test_pred' in res}
    if test_preds:
        emit_holdout_statistics(
            y_test, test_preds, output_dir, prefix="spatial",
            dates_train=dates_train, dates_test=dates_test,
            n_features=len(feature_names), seed=seed,
        )

    return results


def create_synthetic_spatial_data(
    predictor_file: str,
    tws_file: str,
    output_file: str,
    n_locations: int = 10,
    lat_range: Tuple[float, float] = (22.0, 32.0),
    lon_range: Tuple[float, float] = (73.0, 90.0),
    seed: int = 20
) -> str:
    """
    Create synthetic spatial data by replicating temporal data across locations.
    
    This is useful when you have temporal data but want to test spatial holdout methods.
    The data will be identical across locations but with added spatial noise.
    
    Parameters
    ----------
    predictor_file : str
        Path to predictor data
    tws_file : str
        Path to TWS data
    output_file : str
        Output file path
    n_locations : int
        Number of synthetic locations
    lat_range : Tuple[float, float]
        Latitude range
    lon_range : Tuple[float, float]
        Longitude range
    seed : int
        Random seed
    
    Returns
    -------
    str
        Path to created file
    """
    np.random.seed(seed)
    
    # Load and preprocess original data
    merged = load_and_preprocess_data(predictor_file, tws_file)
    
    # Generate random locations
    lats = np.random.uniform(lat_range[0], lat_range[1], n_locations)
    lons = np.random.uniform(lon_range[0], lon_range[1], n_locations)
    
    # Create replicated data
    dfs = []
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        df_loc = merged.copy()
        df_loc['lat'] = lat
        df_loc['lon'] = lon
        df_loc['location_id'] = i
        
        # Add spatial noise to target (proportional to distance from center)
        center_lat, center_lon = np.mean(lat_range), np.mean(lon_range)
        dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
        noise_scale = dist * 0.1  # Scale noise by distance
        df_loc['TWS'] = df_loc['TWS'] + np.random.normal(0, noise_scale, len(df_loc))
        
        dfs.append(df_loc)
    
    result = pd.concat(dfs, ignore_index=True)
    result.to_csv(output_file, index=False)
    
    print(f"Created synthetic spatial data with {n_locations} locations")
    print(f"Total samples: {len(result)}")
    print(f"Saved to: {output_file}")
    
    return output_file


def main():
    """Main function for spatial holdout analysis."""
    # Configuration
    predictor_file = "All_Data.xlsx"
    tws_file = "TWS_JPL.xlsx"
    output_dir = "figures/spatial_holdout"
    
    # Check if data files exist
    if not os.path.exists(predictor_file):
        print(f"Error: {predictor_file} not found!")
        print("Please ensure the data files are in the current directory.")
        return
    
    if not os.path.exists(tws_file):
        print(f"Error: {tws_file} not found!")
        print("Please ensure the data files are in the current directory.")
        return
    
    # Create synthetic spatial data for demonstration
    # In real use, you would have actual spatially-distributed data
    spatial_data_file = "spatial_data_synthetic.csv"
    
    if not os.path.exists(spatial_data_file):
        print("Creating synthetic spatial data for demonstration...")
        create_synthetic_spatial_data(
            predictor_file=predictor_file,
            tws_file=tws_file,
            output_file=spatial_data_file,
            n_locations=10,
            lat_range=(22.0, 32.0),  # Ganga Basin approximate range
            lon_range=(73.0, 90.0),
            seed=20
        )
    
    # Run spatial holdout analysis
    results = run_spatial_holdout_analysis(
        data_file=spatial_data_file,
        output_dir=output_dir,
        predictors=['SMS', 'ET', 'rainfall', 'runoff', 'GWSA'],
        target_col='TWS',
        lat_col='lat',
        lon_col='lon',
        date_col='Date',
        lags=7,
        n_spatial_groups=5,
        grouping_method='kmeans',
        test_fraction=0.2,
        seed=20,
        models_to_run=['bilstm_attention', 'bilstm', 'lstm', 'xgboost', 'lightgbm', 'randomforest'],
        run_cv=True
    )
    
    print(f"\n{'='*50}")
    print("Spatial Holdout Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
