"""
Generate Monthly and Seasonal TWS Maps for Ganga Basin
Creates choropleth maps showing monthly and seasonal average TWS predictions
for each model using temporal holdout results.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from plot_style import pretty_model


# Season definitions
SEASONS = {
    'Pre-Monsoon (Apr-Jun)': [4, 5, 6],
    'Monsoon (Jul-Sep)': [7, 8, 9],
    'Post-Monsoon (Oct-Nov)': [10, 11],
    'Winter (Dec-Mar)': [12, 1, 2, 3]
}

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Honest labelling: the dataset is a single basin-averaged (spatially integrated)
# time series, so every map fills the whole basin polygon with one value. These
# are basin-scale summaries, NOT per-pixel maps. Genuine per-pixel maps would
# require gridded GRACE + gridded predictors, which this dataset does not contain.
BASIN_SCALE_NOTE = ("Basin-averaged (spatially integrated) value shown as a uniform "
                    "basin fill - this is a basin-scale summary, not a per-pixel map.")


def load_predictions(predictions_dir: str) -> Dict[str, pd.DataFrame]:
    """
    All temporal-holdout prediction CSVs, keyed by model name.

    Keeps BOTH splits (`split=None`): these figures show the full reconstruction,
    not held-out performance. Delegates to `stats_utils.load_prediction_dir`, the
    single loader.
    """
    from stats_utils import load_prediction_dir

    return load_prediction_dir(
        predictions_dir,
        pattern="temporal_full_predictions_*.csv",
        split=None,
        add_time_parts=True,
        verbose=True,
    )


def load_basin_shapefile(shapefile_path: str) -> gpd.GeoDataFrame:
    """
    Load the Ganga Basin shapefile.
    
    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing basin geometry
    """
    # Delegates to the single basin reader, which guarantees EPSG:4326. This
    # function previously called gpd.read_file without .to_crs and worked only
    # because the shapefile happens to already be in that CRS.
    from gridded_config import load_basin
    gdf = load_basin()
    print(f"Loaded shapefile with {len(gdf)} features")
    print(f"CRS: {gdf.crs}")
    return gdf


def calculate_monthly_averages(predictions: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate monthly average TWS for each model.
    
    Parameters
    ----------
    predictions : Dict[str, pd.DataFrame]
        Dictionary of model predictions
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Monthly averages for each model with standard deviations
    """
    monthly_avgs = {}
    
    for model_name, df in predictions.items():
        # Calculate monthly averages and standard deviations across all years
        monthly = df.groupby('Month').agg({
            'Actual_TWSA': ['mean', 'std', 'count'],
            'Predicted_TWSA': ['mean', 'std'],
            'Residual': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        monthly.columns = ['Month', 
                          'Actual_TWSA', 'Std_Actual', 'Count',
                          'Predicted_TWSA', 'Std_Predicted',
                          'Residual', 'Std_Residual']
        
        monthly_avgs[model_name] = monthly
    
    return monthly_avgs


def calculate_seasonal_averages(predictions: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate seasonal average TWS for each model.
    
    Parameters
    ----------
    predictions : Dict[str, pd.DataFrame]
        Dictionary of model predictions
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Seasonal averages for each model
    """
    seasonal_avgs = {}
    
    for model_name, df in predictions.items():
        season_data = []
        
        for season_name, months in SEASONS.items():
            season_df = df[df['Month'].isin(months)]
            
            if len(season_df) > 0:
                season_data.append({
                    'Season': season_name,
                    'Actual_TWSA': season_df['Actual_TWSA'].mean(),
                    'Predicted_TWSA': season_df['Predicted_TWSA'].mean(),
                    'Residual': season_df['Residual'].mean(),
                    'Std_Actual': season_df['Actual_TWSA'].std(),
                    'Std_Predicted': season_df['Predicted_TWSA'].std(),
                    'Count': len(season_df)
                })
        
        seasonal_avgs[model_name] = pd.DataFrame(season_data)
    
    return seasonal_avgs


def get_colormap_and_norm(values: np.ndarray, symmetric: bool = True) -> Tuple:
    """
    Create a diverging colormap centered at zero for TWS anomalies.
    
    Parameters
    ----------
    values : np.ndarray
        Array of TWS values
    symmetric : bool
        Whether to make colormap symmetric around zero
    
    Returns
    -------
    Tuple
        Colormap and normalization
    """
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    
    if symmetric:
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs
    
    # Use diverging colormap (brown for negative, blue for positive)
    cmap = plt.cm.RdYlBu
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    return cmap, norm


def plot_monthly_maps(
    monthly_avgs: Dict[str, pd.DataFrame],
    basin_gdf: gpd.GeoDataFrame,
    output_dir: str,
    data_type: str = 'Predicted_TWSA'
) -> None:
    """
    Create monthly maps for each model showing basin-averaged TWS.
    
    Parameters
    ----------
    monthly_avgs : Dict[str, pd.DataFrame]
        Monthly averages for each model
    basin_gdf : gpd.GeoDataFrame
        Basin geometry
    output_dir : str
        Output directory for figures
    data_type : str
        Column to plot ('Predicted_TWSA', 'Actual_TWSA', or 'Residual')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, monthly in monthly_avgs.items():
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        # Get all values for consistent colormap
        all_values = monthly[data_type].values
        cmap, norm = get_colormap_and_norm(all_values)
        
        for i, month in enumerate(range(1, 13)):
            ax = axes[i]
            
            # Get value for this month
            month_data = monthly[monthly['Month'] == month]
            if len(month_data) > 0:
                value = month_data[data_type].values[0]
            else:
                value = np.nan
            
            # Create a copy of the GeoDataFrame with the value
            gdf_plot = basin_gdf.copy()
            gdf_plot['TWS'] = value
            
            # Plot the basin
            gdf_plot.plot(
                column='TWS',
                ax=ax,
                cmap=cmap,
                norm=norm,
                edgecolor='black',
                linewidth=0.5
            )
            
            ax.set_title(f"{MONTH_NAMES[i]}\n{value:.1f} mm", fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # Title (placed first, at the very top)
        title_map = {
            'Predicted_TWSA': 'Predicted',
            'Actual_TWSA': 'Actual',
            'Residual': 'Residual (Actual - Predicted)'
        }
        fig.suptitle(f"Monthly Average {title_map.get(data_type, data_type)} TWS - {pretty_model(model_name)}", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Add colorbar at top using manual axes positioning [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.15, 0.93, 0.7, 0.02])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('TWS Anomaly (mm)', fontsize=12)
        
        safe_name = model_name.replace('+', '_').replace(' ', '_')
        plt.savefig(f"{output_dir}/monthly_maps_{data_type.lower()}_{safe_name}.png", 
                   dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"Saved monthly maps for {model_name} ({data_type})")


def plot_seasonal_maps(
    seasonal_avgs: Dict[str, pd.DataFrame],
    basin_gdf: gpd.GeoDataFrame,
    output_dir: str,
    data_type: str = 'Predicted_TWSA'
) -> None:
    """
    Create seasonal maps for each model showing basin-averaged TWS.
    
    Parameters
    ----------
    seasonal_avgs : Dict[str, pd.DataFrame]
        Seasonal averages for each model
    basin_gdf : gpd.GeoDataFrame
        Basin geometry
    output_dir : str
        Output directory for figures
    data_type : str
        Column to plot ('Predicted_TWSA', 'Actual_TWSA', or 'Residual')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, seasonal in seasonal_avgs.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Get all values for consistent colormap
        all_values = seasonal[data_type].values
        cmap, norm = get_colormap_and_norm(all_values)
        
        season_order = list(SEASONS.keys())
        
        for i, season in enumerate(season_order):
            ax = axes[i]
            
            # Get value for this season
            season_data = seasonal[seasonal['Season'] == season]
            if len(season_data) > 0:
                value = season_data[data_type].values[0]
                std = season_data[f'Std_{data_type.split("_")[0]}' if 'Std_' + data_type.split("_")[0] in season_data.columns else 'Std_Predicted'].values[0] if 'Std_Predicted' in season_data.columns else 0
            else:
                value = np.nan
                std = 0
            
            # Create a copy of the GeoDataFrame with the value
            gdf_plot = basin_gdf.copy()
            gdf_plot['TWS'] = value
            
            # Plot the basin
            gdf_plot.plot(
                column='TWS',
                ax=ax,
                cmap=cmap,
                norm=norm,
                edgecolor='black',
                linewidth=1
            )
            
            ax.set_title(f"{season}\n{value:.1f} ± {std:.1f} mm", fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # Title (placed first, at the very top)
        title_map = {
            'Predicted_TWSA': 'Predicted',
            'Actual_TWSA': 'Actual',
            'Residual': 'Residual (Actual - Predicted)'
        }
        fig.suptitle(f"Seasonal Average {title_map.get(data_type, data_type)} TWS - {pretty_model(model_name)}", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Add colorbar at top using manual axes positioning [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.15, 0.91, 0.7, 0.025])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('TWS Anomaly (mm)', fontsize=12)
        
        safe_name = model_name.replace('+', '_').replace(' ', '_')
        plt.savefig(f"{output_dir}/seasonal_maps_{data_type.lower()}_{safe_name}.png", 
                   dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"Saved seasonal maps for {model_name} ({data_type})")


def plot_model_comparison_maps(
    monthly_avgs: Dict[str, pd.DataFrame],
    basin_gdf: gpd.GeoDataFrame,
    output_dir: str
) -> None:
    """
    Create comparison maps showing all models for each month.
    
    Parameters
    ----------
    monthly_avgs : Dict[str, pd.DataFrame]
        Monthly averages for each model
    basin_gdf : gpd.GeoDataFrame
        Basin geometry
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(monthly_avgs.keys())
    n_models = len(model_names)
    
    # Get global min/max for consistent colormap
    all_values = []
    for monthly in monthly_avgs.values():
        all_values.extend(monthly['Predicted_TWSA'].values)
    cmap, norm = get_colormap_and_norm(np.array(all_values))
    
    for month in range(1, 13):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, model_name in enumerate(model_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            monthly = monthly_avgs[model_name]
            
            month_data = monthly[monthly['Month'] == month]
            if len(month_data) > 0:
                value = month_data['Predicted_TWSA'].values[0]
            else:
                value = np.nan
            
            gdf_plot = basin_gdf.copy()
            gdf_plot['TWS'] = value
            
            gdf_plot.plot(
                column='TWS',
                ax=ax,
                cmap=cmap,
                norm=norm,
                edgecolor='black',
                linewidth=0.5
            )
            
            ax.set_title(f"{pretty_model(model_name)}\n{value:.1f} mm", fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide extra axes
        for j in range(n_models, len(axes)):
            axes[j].set_visible(False)
        
        # Title (placed first, at the very top)
        fig.suptitle(f"Model Comparison - {MONTH_NAMES[month-1]}", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Add colorbar at top using manual axes positioning [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.15, 0.93, 0.7, 0.02])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('TWS Anomaly (mm)', fontsize=12)
        
        plt.savefig(f"{output_dir}/model_comparison_{MONTH_NAMES[month-1].lower()}.png", 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    print("Saved model comparison maps for all months")


def plot_seasonal_comparison_maps(
    seasonal_avgs: Dict[str, pd.DataFrame],
    basin_gdf: gpd.GeoDataFrame,
    output_dir: str
) -> None:
    """
    Create comparison maps showing all models for each season.
    
    Parameters
    ----------
    seasonal_avgs : Dict[str, pd.DataFrame]
        Seasonal averages for each model
    basin_gdf : gpd.GeoDataFrame
        Basin geometry
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(seasonal_avgs.keys())
    n_models = len(model_names)
    
    # Get global min/max for consistent colormap
    all_values = []
    for seasonal in seasonal_avgs.values():
        all_values.extend(seasonal['Predicted_TWSA'].values)
    cmap, norm = get_colormap_and_norm(np.array(all_values))
    
    for season_name in SEASONS.keys():
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, model_name in enumerate(model_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            seasonal = seasonal_avgs[model_name]
            
            season_data = seasonal[seasonal['Season'] == season_name]
            if len(season_data) > 0:
                value = season_data['Predicted_TWSA'].values[0]
            else:
                value = np.nan
            
            gdf_plot = basin_gdf.copy()
            gdf_plot['TWS'] = value
            
            gdf_plot.plot(
                column='TWS',
                ax=ax,
                cmap=cmap,
                norm=norm,
                edgecolor='black',
                linewidth=0.5
            )
            
            ax.set_title(f"{pretty_model(model_name)}\n{value:.1f} mm", fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide extra axes
        for j in range(n_models, len(axes)):
            axes[j].set_visible(False)
        
        # Create safe filename
        season_short = season_name.split('(')[0].strip().replace('-', '_').replace(' ', '_')
        
        # Title (placed first, at the very top)
        fig.suptitle(f"Model Comparison - {season_name}", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Add colorbar at top using manual axes positioning [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.15, 0.93, 0.7, 0.02])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('TWS Anomaly (mm)', fontsize=12)
        
        plt.savefig(f"{output_dir}/model_comparison_{season_short.lower()}.png", 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    print("Saved model comparison maps for all seasons")


def plot_annual_cycle(
    monthly_avgs: Dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Create line plot showing annual cycle of TWS for all models.
    
    Parameters
    ----------
    monthly_avgs : Dict[str, pd.DataFrame]
        Monthly averages for each model
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(monthly_avgs)))
    
    # Plot predicted TWS
    ax1 = axes[0]
    for i, (model_name, monthly) in enumerate(monthly_avgs.items()):
        monthly_sorted = monthly.sort_values('Month')
        ax1.plot(monthly_sorted['Month'], monthly_sorted['Predicted_TWSA'], 
                'o-', label=pretty_model(model_name), color=colors[i], linewidth=2, markersize=6)
    
    # Add actual TWS (should be same for all models)
    first_model = list(monthly_avgs.keys())[0]
    actual = monthly_avgs[first_model].sort_values('Month')
    ax1.plot(actual['Month'], actual['Actual_TWSA'], 
            'k--', label='Actual', linewidth=2.5, markersize=8)
    
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('TWS Anomaly (mm)', fontsize=12)
    ax1.set_title('Annual Cycle of TWS Predictions', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(MONTH_NAMES, rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)

    # Add seasonal shading
    season_colors = {'Pre-Monsoon (Apr-Jun)': 'yellow', 'Monsoon (Jul-Sep)': 'green',
                    'Post-Monsoon (Oct-Nov)': 'orange', 'Winter (Dec-Mar)': 'lightblue'}
    for season, months in SEASONS.items():
        for m in months:
            ax1.axvspan(m - 0.5, m + 0.5, alpha=0.1,
                       color=season_colors.get(season, 'gray'))
    # Legend: model lines plus a labelled swatch for each seasonal band, so the
    # background shading is explained rather than left as unlabelled colour.
    season_handles = [Patch(facecolor=c, alpha=0.1, label=s)
                      for s, c in season_colors.items()]
    line_handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=line_handles + season_handles, loc='best', fontsize=8, ncol=2)
    
    # Plot residuals
    ax2 = axes[1]
    for i, (model_name, monthly) in enumerate(monthly_avgs.items()):
        monthly_sorted = monthly.sort_values('Month')
        ax2.plot(monthly_sorted['Month'], monthly_sorted['Residual'], 
                'o-', label=pretty_model(model_name), color=colors[i], linewidth=2, markersize=6)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Residual (mm)', fontsize=12)
    ax2.set_title('Monthly Average Residuals by Model', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_NAMES, rotation=45)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/annual_cycle_comparison.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    print("Saved annual cycle comparison plot")


def plot_monthly_time_series(
    monthly_avgs: Dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Create individual time series plots for each month showing all models
    with standard deviation error bars.
    
    Parameters
    ----------
    monthly_avgs : Dict[str, pd.DataFrame]
        Monthly averages for each model with standard deviations
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model names and colors
    model_names = list(monthly_avgs.keys())
    n_models = len(model_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Create a 4x3 grid for 12 months
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    axes = axes.flatten()
    
    for month_idx in range(12):
        ax = axes[month_idx]
        month = month_idx + 1
        
        x_positions = np.arange(n_models)
        width = 0.35
        
        actual_vals = []
        actual_stds = []
        pred_vals = []
        pred_stds = []
        
        for model_name in model_names:
            monthly = monthly_avgs[model_name]
            row = monthly[monthly['Month'] == month].iloc[0]
            actual_vals.append(row['Actual_TWSA'])
            actual_stds.append(row.get('Std_Actual', 0))
            pred_vals.append(row['Predicted_TWSA'])
            pred_stds.append(row.get('Std_Predicted', 0))
        
        # Plot bars with error bars
        bars1 = ax.bar(x_positions - width/2, actual_vals, width, 
                       yerr=actual_stds, label='Actual', color='steelblue',
                       capsize=3, alpha=0.8)
        bars2 = ax.bar(x_positions + width/2, pred_vals, width,
                       yerr=pred_stds, label='Predicted', color='coral',
                       capsize=3, alpha=0.8)
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title(f"{MONTH_NAMES[month_idx]}", fontsize=11, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([pretty_model(m).replace(' + ', '\n+') for m in model_names], 
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('TWS (mm)', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        if month_idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    plt.suptitle('Monthly TWS Comparison by Model (with Std Dev)', 
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_time_series_all_months.png", 
               dpi=600, bbox_inches='tight')
    plt.close()
    
    print("Saved monthly time series comparison plot")


def plot_seasonal_time_series(
    seasonal_avgs: Dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Create time series plots for each season showing all models
    with standard deviation error bars.
    
    Parameters
    ----------
    seasonal_avgs : Dict[str, pd.DataFrame]
        Seasonal averages for each model with standard deviations
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model names and colors
    model_names = list(seasonal_avgs.keys())
    n_models = len(model_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    season_names = list(SEASONS.keys())
    
    # Create a 2x2 grid for 4 seasons
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for season_idx, season_name in enumerate(season_names):
        ax = axes[season_idx]
        
        x_positions = np.arange(n_models)
        width = 0.35
        
        actual_vals = []
        actual_stds = []
        pred_vals = []
        pred_stds = []
        
        for model_name in model_names:
            seasonal = seasonal_avgs[model_name]
            row = seasonal[seasonal['Season'] == season_name].iloc[0]
            actual_vals.append(row['Actual_TWSA'])
            actual_stds.append(row.get('Std_Actual', 0))
            pred_vals.append(row['Predicted_TWSA'])
            pred_stds.append(row.get('Std_Predicted', 0))
        
        # Plot bars with error bars
        bars1 = ax.bar(x_positions - width/2, actual_vals, width, 
                       yerr=actual_stds, label='Actual', color='steelblue',
                       capsize=4, alpha=0.8)
        bars2 = ax.bar(x_positions + width/2, pred_vals, width,
                       yerr=pred_stds, label='Predicted', color='coral',
                       capsize=4, alpha=0.8)
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title(f"{season_name}", fontsize=12, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([pretty_model(m).replace(' + ', '\n+') for m in model_names], 
                          rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('TWS Anomaly (mm)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        if season_idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    plt.suptitle('Seasonal TWS Comparison by Model (with Std Dev)', 
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seasonal_time_series_all_seasons.png", 
               dpi=600, bbox_inches='tight')
    plt.close()
    
    print("Saved seasonal time series comparison plot")


def plot_model_time_series_with_std(
    predictions: Dict[str, pd.DataFrame],
    monthly_avgs: Dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Create individual time series plots for each model showing monthly cycle
    with standard deviation shading.
    
    Parameters
    ----------
    predictions : Dict[str, pd.DataFrame]
        Raw predictions for each model
    monthly_avgs : Dict[str, pd.DataFrame]
        Monthly averages with standard deviations
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(monthly_avgs.keys())
    n_models = len(model_names)
    
    # Create subplot grid
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        monthly = monthly_avgs[model_name].sort_values('Month')
        months = monthly['Month'].values
        
        # Plot actual with std shading
        actual_mean = monthly['Actual_TWSA'].values
        actual_std = monthly.get('Std_Actual', pd.Series([0]*12)).values
        ax.fill_between(months, actual_mean - actual_std, actual_mean + actual_std,
                       alpha=0.3, color='steelblue', label='Actual ± Std')
        ax.plot(months, actual_mean, 'o-', color='steelblue', 
               linewidth=2, markersize=6, label='Actual')
        
        # Plot predicted with std shading
        pred_mean = monthly['Predicted_TWSA'].values
        pred_std = monthly.get('Std_Predicted', pd.Series([0]*12)).values
        ax.fill_between(months, pred_mean - pred_std, pred_mean + pred_std,
                       alpha=0.3, color='coral', label='Predicted ± Std')
        ax.plot(months, pred_mean, 's-', color='coral', 
               linewidth=2, markersize=6, label='Predicted')
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"{pretty_model(model_name)}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Month', fontsize=10)
        ax.set_ylabel('TWS Anomaly (mm)', fontsize=10)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES, rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Hide extra axes
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Monthly TWS Cycle by Model (with Std Dev Shading)', 
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_monthly_cycles_with_std.png", 
               dpi=600, bbox_inches='tight')
    plt.close()
    
    print("Saved model monthly cycles with std dev plot")


def plot_seasonal_cycle_with_std(
    seasonal_avgs: Dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Create seasonal cycle plot for each model with standard deviation.
    
    Parameters
    ----------
    seasonal_avgs : Dict[str, pd.DataFrame]
        Seasonal averages with standard deviations
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(seasonal_avgs.keys())
    n_models = len(model_names)
    season_names = list(SEASONS.keys())
    n_seasons = len(season_names)
    
    # Create subplot grid
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    season_short_names = ['Pre-Mon', 'Monsoon', 'Post-Mon', 'Winter']
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        seasonal = seasonal_avgs[model_name]
        
        x_positions = np.arange(n_seasons)
        
        # Get values in season order
        actual_vals = []
        actual_stds = []
        pred_vals = []
        pred_stds = []
        
        for season in season_names:
            row = seasonal[seasonal['Season'] == season].iloc[0]
            actual_vals.append(row['Actual_TWSA'])
            actual_stds.append(row.get('Std_Actual', 0))
            pred_vals.append(row['Predicted_TWSA'])
            pred_stds.append(row.get('Std_Predicted', 0))
        
        # Plot with error bars
        ax.errorbar(x_positions, actual_vals, yerr=actual_stds, 
                   fmt='o-', color='steelblue', linewidth=2, markersize=8,
                   capsize=5, capthick=2, label='Actual ± Std')
        ax.errorbar(x_positions, pred_vals, yerr=pred_stds,
                   fmt='s-', color='coral', linewidth=2, markersize=8,
                   capsize=5, capthick=2, label='Predicted ± Std')
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"{pretty_model(model_name)}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Season', fontsize=10)
        ax.set_ylabel('TWS Anomaly (mm)', fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(season_short_names, rotation=30, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Hide extra axes
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Seasonal TWS Cycle by Model (with Std Dev)', 
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_seasonal_cycles_with_std.png", 
               dpi=600, bbox_inches='tight')
    plt.close()
    
    print("Saved model seasonal cycles with std dev plot")


def save_summary_tables(
    monthly_avgs: Dict[str, pd.DataFrame],
    seasonal_avgs: Dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Save summary tables of monthly and seasonal averages.
    
    Parameters
    ----------
    monthly_avgs : Dict[str, pd.DataFrame]
        Monthly averages for each model
    seasonal_avgs : Dict[str, pd.DataFrame]
        Seasonal averages for each model
    output_dir : str
        Output directory for files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Monthly summary
    monthly_records = []
    for model_name, monthly in monthly_avgs.items():
        for _, row in monthly.iterrows():
            monthly_records.append({
                'Model': model_name,
                'Month': int(row['Month']),
                'Month_Name': MONTH_NAMES[int(row['Month']) - 1],
                'Actual_TWSA_mm': row['Actual_TWSA'],
                'Actual_Std_mm': row.get('Std_Actual', np.nan),
                'Predicted_TWSA_mm': row['Predicted_TWSA'],
                'Predicted_Std_mm': row.get('Std_Predicted', np.nan),
                'Residual_mm': row['Residual'],
                'Residual_Std_mm': row.get('Std_Residual', np.nan),
                'Sample_Count': row.get('Count', np.nan)
            })
    
    monthly_df = pd.DataFrame(monthly_records)
    monthly_df.to_csv(f"{output_dir}/monthly_averages_all_models.csv", index=False)
    
    # Seasonal summary
    seasonal_records = []
    for model_name, seasonal in seasonal_avgs.items():
        for _, row in seasonal.iterrows():
            seasonal_records.append({
                'Model': model_name,
                'Season': row['Season'],
                'Actual_TWSA_mm': row['Actual_TWSA'],
                'Predicted_TWSA_mm': row['Predicted_TWSA'],
                'Residual_mm': row['Residual'],
                'Sample_Count': row['Count']
            })
    
    seasonal_df = pd.DataFrame(seasonal_records)
    seasonal_df.to_csv(f"{output_dir}/seasonal_averages_all_models.csv", index=False)
    
    print(f"Saved summary tables to {output_dir}")


def main(predictions_dir=None, shapefile=None, output_dir=None):
    """Main function to generate monthly and seasonal maps."""
    # Configuration
    predictions_dir = "../Results/figures/temporal_holdout"
    from gridded_config import BASIN_SHAPEFILE
    shapefile_path = BASIN_SHAPEFILE
    output_dir = "../Results/figures/monthly_seasonal_maps"
    
    # Check paths
    if not os.path.exists(predictions_dir):
        print(f"Error: Predictions directory not found: {predictions_dir}")
        print("Please run temporal holdout analysis first.")
        return
    
    if not os.path.exists(shapefile_path):
        print(f"Error: Shapefile not found: {shapefile_path}")
        return
    
    print("="*60)
    print("GENERATING MONTHLY AND SEASONAL TWS MAPS")
    print("="*60)
    
    # Load data
    print("\n1. Loading predictions...")
    predictions = load_predictions(predictions_dir)
    
    if not predictions:
        print("Error: No prediction files found!")
        return
    
    print("\n2. Loading basin shapefile...")
    basin_gdf = load_basin_shapefile(shapefile_path)
    
    # Calculate averages
    print("\n3. Calculating monthly averages...")
    monthly_avgs = calculate_monthly_averages(predictions)
    
    print("\n4. Calculating seasonal averages...")
    seasonal_avgs = calculate_seasonal_averages(predictions)
    
    # Generate plots
    print("\n5. Generating monthly maps...")
    for data_type in ['Predicted_TWSA', 'Actual_TWSA', 'Residual']:
        plot_monthly_maps(monthly_avgs, basin_gdf, output_dir, data_type)
    
    print("\n6. Generating seasonal maps...")
    for data_type in ['Predicted_TWSA', 'Actual_TWSA', 'Residual']:
        plot_seasonal_maps(seasonal_avgs, basin_gdf, output_dir, data_type)
    
    print("\n7. Generating model comparison maps...")
    plot_model_comparison_maps(monthly_avgs, basin_gdf, output_dir)
    plot_seasonal_comparison_maps(seasonal_avgs, basin_gdf, output_dir)
    
    print("\n8. Generating annual cycle plot...")
    plot_annual_cycle(monthly_avgs, output_dir)
    
    print("\n9. Generating monthly time series plots...")
    plot_monthly_time_series(monthly_avgs, output_dir)
    
    print("\n10. Generating seasonal time series plots...")
    plot_seasonal_time_series(seasonal_avgs, output_dir)
    
    print("\n11. Generating model-wise monthly cycles with std...")
    plot_model_time_series_with_std(predictions, monthly_avgs, output_dir)
    
    print("\n12. Generating model-wise seasonal cycles with std...")
    plot_seasonal_cycle_with_std(seasonal_avgs, output_dir)
    
    print("\n13. Saving summary tables...")
    save_summary_tables(monthly_avgs, seasonal_avgs, output_dir)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"Output saved to: {output_dir}")
    print("="*60)
    
    # Print summary
    print("\nGenerated files:")
    print("  - monthly_maps_*.png: Monthly maps for each model")
    print("  - seasonal_maps_*.png: Seasonal maps for each model")
    print("  - model_comparison_*.png: All models compared for each month/season")
    print("  - annual_cycle_comparison.png: Line plot of annual TWS cycle")
    print("  - monthly_time_series_all_months.png: Bar plots by month with std dev")
    print("  - seasonal_time_series_all_seasons.png: Bar plots by season with std dev")
    print("  - model_monthly_cycles_with_std.png: Monthly cycles per model with shading")
    print("  - model_seasonal_cycles_with_std.png: Seasonal cycles per model with error bars")
    print("  - monthly_averages_all_models.csv: Summary table with std dev")
    print("  - seasonal_averages_all_models.csv: Summary table with std dev")


if __name__ == "__main__":
    import argparse
    from gridded_config import BASIN_SHAPEFILE
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument('--predictions-dir', default='../Results/figures/temporal_holdout')
    _ap.add_argument('--shapefile', default=BASIN_SHAPEFILE)
    _ap.add_argument('--output-dir', default='../Results/figures/monthly_seasonal_maps')
    _a = _ap.parse_args()
    main(_a.predictions_dir, _a.shapefile, _a.output_dir)
