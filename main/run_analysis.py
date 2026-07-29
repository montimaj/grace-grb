"""
TWS Downscaling Analysis - Main Entry Point
Run different holdout analyses with multiple ML models.

Models available:
- BiLSTM+Attention (bilstm_attention)
- BiLSTM (bilstm)
- LSTM (lstm)
- XGBoost (xgboost)
- LightGBM (lightgbm)
- RandomForest (randomforest)

Holdout Types:
- Random: Random train-test split
- Temporal: Time-based split (chronological)
- Spatial: RETIRED (was synthetic and leaked); use downscale_model.py
  for the real leave-one-mascon-out holdout on gridded data
"""

import argparse
import os
import sys


def run_analysis(
    analysis_type: str,
    models: list = None,
    output_dir: str = None,
    seed: int = 20,
    **kwargs
):
    """
    Run the specified holdout analysis.
    
    Parameters
    ----------
    analysis_type : str
        Type of analysis: 'random', 'temporal', or 'all'
        ('spatial' is retired - see downscale_model.py)
    models : list
        List of model names to run
    output_dir : str
        Base output directory
    seed : int
        Random seed
    **kwargs
        Additional arguments passed to analysis functions
    """
    if models is None:
        models = ['bilstm_attention', 'bilstm', 'lstm', 'xgboost', 'lightgbm', 'randomforest']
    
    predictor_file = kwargs.get('predictor_file', 'All_Data.xlsx')
    tws_file = kwargs.get('tws_file', '../Data/TWS_GRACE_GEE.csv')
    
    results = {}
    
    if analysis_type in ['random', 'all']:
        print("\n" + "="*60)
        print("RUNNING RANDOM HOLDOUT ANALYSIS")
        print("="*60)
        
        from holdout_random import run_random_holdout_analysis
        
        out_dir = output_dir or "../Results/figures/random_holdout"
        results['random'] = run_random_holdout_analysis(
            predictor_file=predictor_file,
            tws_file=tws_file,
            output_dir=out_dir,
            models_to_run=models,
            tuned_params=kwargs.get('tuned_params'),
            seed=seed,
            test_size=kwargs.get('test_size', 0.2)
        )
    
    if analysis_type in ['temporal', 'all']:
        print("\n" + "="*60)
        print("RUNNING TEMPORAL HOLDOUT ANALYSIS")
        print("="*60)
        
        from holdout_temporal import run_temporal_holdout_analysis
        
        out_dir = output_dir or "../Results/figures/temporal_holdout"
        results['temporal'] = run_temporal_holdout_analysis(
            predictor_file=predictor_file,
            tws_file=tws_file,
            output_dir=out_dir,
            models_to_run=models,
            tuned_params=kwargs.get('tuned_params'),
            seed=seed,
            test_size=kwargs.get('test_size', 0.2),
            run_cv=kwargs.get('run_cv', True)
        )
    
    if analysis_type == 'spatial':
        raise SystemExit(
            "The synthetic spatial holdout has been RETIRED.\n\n"
            "It replicated the single basin-mean series across fabricated "
            "locations with added noise, so train and test shared near-identical\n"
            "copies and it leaked by construction (Random Forest R2 ~ 0.99). It "
            "measured noise sensitivity, not spatial transferability.\n\n"
            "It is superseded by a genuine spatial holdout on real gridded data:\n"
            "    python downscale_model.py --skip-product\n"
            "which runs leave-one-mascon-out cross-validation with a one-mascon "
            "neighbour buffer over the 19 independent GRACE mascons covering the\n"
            "basin. On the same quantity that test gives R2 = 0.05 (raw) and "
            "R2 = 0.64 on the component the predictors can actually explain.\n\n"
            "To reproduce the superseded figures for the earlier manuscript, call "
            "holdout_spatial.run_spatial_holdout_analysis() directly."
        )

    return results


def compare_holdout_methods(results: dict, output_dir: str = "../Results/figures/comparison"):
    """
    Compare results across different holdout methods.
    
    Parameters
    ----------
    results : dict
        Dictionary with results from each holdout analysis
    output_dir : str
        Output directory for comparison plots
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from plot_style import pretty_model, R2, BAR_3

    os.makedirs(output_dir, exist_ok=True)
    
    # Collect metrics
    comparison_data = []
    
    for holdout_type, holdout_results in results.items():
        for model_name, model_results in holdout_results.items():
            if 'metrics' in model_results:
                metrics = model_results['metrics']
                comparison_data.append({
                    'Holdout': holdout_type.capitalize(),
                    'Model': model_name,
                    'MAE': metrics.mae,
                    'RMSE': metrics.rmse,
                    'R2': metrics.r2,
                    'NSE': metrics.nse,
                    'PBIAS': metrics.pbias
                })
    
    if not comparison_data:
        print("No comparison data available")
        return
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(f"{output_dir}/holdout_comparison.csv", index=False)
    
    # Create comparison plots
    metrics = ['MAE', 'RMSE', 'R2', 'NSE']
    holdout_types = df['Holdout'].unique()
    models = df['Model'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = [BAR_3[i % len(BAR_3)] for i in range(len(holdout_types))]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        x = range(len(models))
        width = 0.25
        
        for i, holdout in enumerate(holdout_types):
            subset = df[df['Holdout'] == holdout]
            values = [subset[subset['Model'] == m][metric].values[0] 
                     if len(subset[subset['Model'] == m]) > 0 else 0 
                     for m in models]
            ax.bar([xi + i * width for xi in x], values, width, 
                  label=holdout, color=colors[i], alpha=0.8)
        
        metric_label = R2 if metric == 'R2' else metric
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} Comparison')
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels([pretty_model(m) for m in models], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Performance Across Holdout Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/holdout_comparison.png", dpi=600)
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("HOLDOUT METHOD COMPARISON SUMMARY")
    print("="*60)
    
    pivot = df.pivot_table(
        index='Model', 
        columns='Holdout', 
        values=['R2', 'RMSE'],
        aggfunc='mean'
    )
    print(pivot.to_string())
    
    return df


def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(
        description='TWS Downscaling Analysis with Multiple ML Models and Holdout Strategies'
    )
    
    parser.add_argument(
        '--analysis', '-a',
        type=str,
        choices=['random', 'temporal', 'spatial', 'all'],
        default='all',
        help="Type of holdout analysis to run (default: all). 'spatial' is retired "
             "and exits with a pointer to the real leave-one-mascon-out test."
    )
    
    parser.add_argument(
        '--models', '-m',
        type=str,
        nargs='+',
        default=None,
        help='Models to run (default: all available models)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: figures/<analysis_type>_holdout)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=20,
        help='Random seed (default: 20)'
    )
    
    parser.add_argument(
        '--test-size', '-t',
        type=float,
        default=0.2,
        help='Test set size fraction (default: 0.2)'
    )
    
    parser.add_argument(
        '--predictor-file',
        type=str,
        default='../Data/All_Data.xlsx',
        help='Path to predictor data file'
    )
    
    parser.add_argument(
        '--tws-file',
        type=str,
        default='../Data/TWS_GRACE_GEE.csv',
        help='Path to TWS data file'
    )
    
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Skip cross-validation'
    )

    parser.add_argument(
        '--tuned-params',
        type=str,
        default=None,
        help='Path to Optuna best-params JSON (from tune_hyperparameters.py); '
             'when given, models use the tuned hyperparameters'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Generate comparison plots across holdout methods'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.predictor_file):
        print(f"Error: Predictor file not found: {args.predictor_file}")
        sys.exit(1)
    
    if not os.path.exists(args.tws_file):
        print(f"Error: TWS file not found: {args.tws_file}")
        sys.exit(1)
    
    # Load tuned hyperparameters if requested
    tuned = None
    if args.tuned_params:
        from tune_hyperparameters import load_tuned_params
        tuned = load_tuned_params(args.tuned_params)
        print(f"Loaded tuned hyperparameters from {args.tuned_params} "
              f"for models: {sorted(k for k in tuned if '+' in k or k[0].isupper())}")

    # Run analysis
    results = run_analysis(
        analysis_type=args.analysis,
        models=args.models,
        output_dir=args.output,
        seed=args.seed,
        predictor_file=args.predictor_file,
        tws_file=args.tws_file,
        test_size=args.test_size,
        run_cv=not args.no_cv,
        tuned_params=tuned
    )
    
    # Generate comparison if requested
    if args.compare and args.analysis == 'all':
        compare_holdout_methods(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
