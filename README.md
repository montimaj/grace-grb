# Explainable Artificial intelligence Based Temporal Downscaling and Forecasting of Terrestrial Water Storage Using Hydroclimatic Data in Ganges River Basin

## Citation
Kaushik, P. R., Majumdar, S., Lenczuk, A., Banerjee, S., Kumar, Y. S., & Thakur, P. K. (2026). Explainable Artificial intelligence Based Temporal Downscaling and Forecasting of Terrestrial Water Storage Using Hydroclimatic Data in Ganges River Basin. _In prep. for Journal of Hydrology: Regional Studies._

## Abstract
The climate of our planet is going through rapid and unforeseen changes, resulting in increased frequency, duration, and severity of droughts, which have enduring effects on flora and fauna, ecosystems, communities, and individuals. Therefore, monitoring climate patterns in various places has become increasingly significant. 
Additionally, water resources are changing at an unprecedented rate, with the dry areas getting drier and the wet wetter. These changes are strongly dependent on extreme climate events such as droughts and floods. 
Thus, hydrological dynamics assessment and its association with climate teleconnections during major hydrological events is highly important and required for the basin. Ongoing technological advances in remote sensing methods, have successfully used gravity data provided by the Gravity Recovery and Climate Experiment (GRACE) and its successor GRACE Follow-On (GRACE-FO) missions to assess the variability of terrestrial water storage (TWS). 
In our study, we apply GRACEderived TWS anomalies to characterize trends and variations of water storage across different seasonal and meteorological periods such as monsoon and spring in the Ganges River basin. The Ganges River basin covers 26% of India’s land area and plays a critical role in supporting the food and water security of this region. 
To predict and assess TWS changes for the upcoming years and to derive daily TWS changes, we rely on machine learning and deep learning methods such as extra gradient boosting and long short-term memory (LSTM) for monthly GRACE-derived TWS. We assess the reliability of obtained results by comparison to daily GRACE Institute of Geodesy at Graz University of Technology (ITSG) solutions. 
The impact of climate teleconnections such as El NiñoSouthern Oscillation (ENSO) and Niño 3.4 on TWS changes is also investigated to improve the understanding of hydrological dynamics during extreme climatic conditions in the Ganges basin. 
The results show significant GRACE-derived TWS variations in response to the monsoon and climatic periods. The hybrid deep learning model consisting of attention-based LSTM successfully imputed the GRACE missing months as well as predicted TWS variations up to 12 months. 
Moreover, resampled daily TWS data exhibited strong relationship to ITSG daily data in the Ganges basin with a high Pearson’s correlation coefficient of 0.9. The presented integration of artificial intelligence with remote sensing methodologies provides key implications for managing water resources in the river basin for the next few months. 
It also provides more relevant and timely information for operational decision-making, especially in areas with climate changing scenarios such as the Ganges basin.

<img src="Data/Readme_Figs/Graphical_Abstract.png" width="600"/>

## Project Structure

```
grace-grb/
├── README.md                           # Main readme
├── LICENSE                             # License file
├── Data/                               # Input data files
│   ├── All_Data.xlsx                   # Predictor variables (SMS, ET, rainfall, runoff, GWSA)
│   ├── TWS_JPL.xlsx                    # GRACE TWS data
│   ├── Ganga Basin Shapefile/          # Basin boundary shapefiles
│   ├── Outputs/                        # Processed data outputs
│   └── Readme_Figs/                    # Figures for documentation
|   └── archive/                        # Original standalone scripts that are not used
├── Results/                            # Analysis results and figures
│   └── figures/                        # Generated plots and maps
│       ├── temporal_holdout/           # Temporal analysis outputs
│       ├── random_holdout/             # Random analysis outputs
│       ├── spatial_holdout/            # Spatial analysis outputs
│       └── monthly_seasonal_maps/      # Monthly/seasonal TWS maps
├── main/                               # Main codebase
│   ├── README.md                       # Detailed methods documentation
│   ├── models.py                       # ML model wrapper classes (LSTM, BiLSTM, XGBoost, etc.)
│   ├── utils.py                        # Utility functions (metrics, plotting, SHAP analysis)
│   ├── holdout_random.py               # Random holdout analysis
│   ├── holdout_temporal.py             # Temporal holdout analysis
│   ├── holdout_spatial.py              # Spatial/grouped holdout analysis
│   ├── run_analysis.py                 # Main CLI entry point
│   ├── generate_monthly_maps.py        # Monthly/seasonal TWS map generation
│   ├── gee_download.py                 # Google Earth Engine data download
```

## Running the project

### 1. Download and install Anaconda/Miniconda
Either [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for installing the Python 3 packages. 
It is recommended to install the latest version of Anaconda or miniconda (Python >= 3.10). If Anaconda or miniconda is already installed, skip this step. 

**For Windows users:** Once installed, open the Anaconda terminal (called Ananconda Prompt), and run ```conda init powershell``` to add ```conda``` to Windows PowerShell path.

**For Linux/Mac users:** Make sure ```conda``` is added to path. Typically, conda is automatically added to path after installation. It may be necessary to restart the current shell session to add conda to path.

The conda package manager can be updated by running the following command: ```conda update conda```

Anaconda is a Python distribution and environment manager. Miniconda is a free minimal installer for conda. These will help in installing the correct packages and Python version to run the codes.

### 2. Clone or download the repository

Download the repository from the compressed file link at the top right of the repository webpage, or clone the repository using Git.
Unzip all zipped files.  Several of the input datasets in this repository are zipped for efficient storage and must be unzipped before they can be used to run this project.

#### Repository disk space requirements

| Component | Size | Description |
|-----------|------|-------------|
| Data/ | ~150 MB | Input data (GRACE TWS, GLDAS variables, shapefiles) |
| Results/ | ~150 MB | Generated outputs (figures, predictions, maps) |
| main/ | <1 MB | Source code |
| **Total** | **~500 MB** | Full repository with all outputs |

**Note:** The Results folder size will vary depending on how many analyses are run and which models are used.

### 3. Creating the conda environment and installing packages
Open Linux/Mac terminal or Windows PowerShell and run the following:
```
conda create -y -n grace-grb python=3.12
conda activate grace-grb
conda install -y -c conda-forge rioxarray gdal geopandas lightgbm py-xgboost earthengine-api rasterstats seaborn openpyxl pytorch dask-ml dask-jobqueue swifter shap
```

### 4. Google Earth Engine Authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) some of the predictor datasets from the GEE
data repository. After completing step 3, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). The Google Cloud CLI tools
may be required for this GEE authentication step. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk).

A GCloud project needs to be set up online (e.g., ```grace-grb```), with the GEE API service enabled (https://console.cloud.google.com/). Then set a default project using ```gcloud config set project grace-grb```. Additionally, you may need to run ```gcloud auth application-default set-quota-project grace-grb``` if prompted by the GCloud CLI. 
After that, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). 

### 5. Running the code

See [main/README.md](main/README.md) for detailed documentation on methods and API reference.

#### Quick Start

```bash
cd main

# Run all analyses (random, temporal, spatial holdout) with all models
python run_analysis.py --analysis all --compare

# Run specific analysis type
python run_analysis.py --analysis temporal

# Run with specific models only
python run_analysis.py -a random -m bilstm_attention xgboost lightgbm

# Generate monthly/seasonal TWS maps (after running temporal analysis)
python generate_monthly_maps.py
```

#### Available Models
- `bilstm_attention` - BiLSTM with Attention mechanism
- `bilstm` - Bidirectional LSTM
- `lstm` - Standard LSTM
- `xgboost` - XGBoost Regressor
- `lightgbm` - LightGBM Regressor
- `random_forest` - Random Forest Regressor

#### Output Files
Results are saved to `Results/figures/` including:
- Model predictions (CSV) with train/test splits
- Performance plots (actual vs predicted, residuals)
- SHAP analysis plots (feature importance, dependence)
- Model comparison summaries
- Monthly/seasonal TWS maps with statistics
