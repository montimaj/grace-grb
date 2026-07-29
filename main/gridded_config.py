"""
Configuration for the gridded (spatial + temporal) GRACE downscaling pipeline.

This module is the single source of truth for:
  * the study domain and the three native grids we work on,
  * every variable we pull from Earth Engine, with its unit conversion made
    explicit and machine-readable,
  * the provenance of the five predictors used by the basin-scale paper, so the
    gridded rebuild reproduces them exactly rather than approximately.

Grid policy
-----------
Each source dataset is downloaded on its OWN native grid and regridded locally.
We deliberately do not let Earth Engine resample, because the whole point of the
exercise is to control how coarse information is spread to fine cells.

  ERA5-Land  0.1  deg  transform [0.1,  0, -180.05, 0, -0.1, 90.05]
  GLDAS      0.25 deg  transform [0.25, 0, -180.0,  0, -0.25, 90.0 ]
  GRACE      0.5  deg  transform [0.5,  0, -180.0,  0, -0.5,  90.0 ]

Note the ERA5-Land origin is offset by half a cell (-180.05, not -180.0): its
cell *centres* fall on multiples of 0.1 deg. Snapping is done outward from the
basin bounding box so no basin cell is ever clipped.

Units
-----
Unit handling in the basin-scale pipeline had two defects that this module
fixes and records:

  1. GRACE `lwe_thickness` is in CENTIMETRES. `gee_download.py` multiplied by
     100 to reach millimetres; the correct factor is 10. Separately,
     `Data/TWS_JPL.xlsx` holds raw centimetres but is labelled millimetres
     throughout the manuscript.
  2. GLDAS V021 NOAH is 3-hourly. Summing the 8 sub-daily `Evap_tavg` images and
     multiplying by 86400 overestimates daily ET by 8x. We use V022 CLSM, which
     is already daily (one image per day), so `mean() * 86400` is exact.

Every VariableSpec below carries `scale`, `offset` and `out_units`, and
`build_cube.py` writes them into the netCDF attributes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Earth Engine
# --------------------------------------------------------------------------

# The basin-scale scripts hard-coded project='grace-grb', which no longer
# resolves ("Project 'projects/grace-grb' not found or deleted"). The working
# project is 'grace-grb-ml'.
#
# Resolution order: explicit env var -> the project recorded in the earthengine
# credentials file -> this literal fallback. The credentials file can carry a
# null project after a re-authentication, which is why the fallback has to be a
# real project rather than a placeholder.
EE_PROJECT_ENV = 'EE_PROJECT'
EE_PROJECT_FALLBACK = 'grace-grb-ml'
EE_HIGH_VOLUME = 'https://earthengine-highvolume.googleapis.com'


def resolve_ee_project() -> str:
    """Return the Earth Engine cloud project to initialise with."""
    env = os.environ.get(EE_PROJECT_ENV)
    if env:
        return env
    cred = os.path.expanduser('~/.config/earthengine/credentials')
    if os.path.exists(cred):
        try:
            import json
            with open(cred) as fh:
                project = json.load(fh).get('project')
            if project:
                return project
        except (ValueError, OSError):
            pass
    return EE_PROJECT_FALLBACK


def ee_initialize(project: Optional[str] = None) -> None:
    """Initialise Earth Engine against the high-volume endpoint."""
    import ee
    ee.Initialize(project=project or resolve_ee_project(), opt_url=EE_HIGH_VOLUME)


# --------------------------------------------------------------------------
# Study domain
# --------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(_HERE, '..', 'Data'))
BASIN_SHAPEFILE = os.path.join(DATA_DIR, 'Ganga Basin Shapefile', 'Ganga_basin.shp')

GRIDDED_DIR = os.path.join(DATA_DIR, 'Gridded')
RAW_TIF_DIR = os.path.join(GRIDDED_DIR, 'raw')
CUBE_DIR = os.path.join(GRIDDED_DIR, 'cubes')

# Product period. GLDAS V022 CLSM (the actual source of the paper's SMS and ET
# columns) begins 2003-01-01, which is why those columns are NaN for 2002 in
# All_Data.xlsx (8005 non-null of 8401 rows). GRACE mascons begin 2002-04 but
# the predictors do not, so the gridded product starts in 2003.
START_DATE = '2000-01-01'
END_DATE = '2025-12-31'

# GLDAS 2.1 NOAH begins 2000-01-01 and ERA5-Land in 1950, so the predictors span
# the full window. GRACE itself runs 2002-04 to 2024-09, so months outside that
# are RECONSTRUCTIONS (before/after the observation record) rather than gap-fills
# between anchors -- see the forward/backward holdout scenarios, which measure
# those two cases separately.
ANALYSIS_START = '2000-01-01'
ANALYSIS_END = END_DATE

# Widening the training domain beyond the Ganga is a one-line change here: set
# DOMAIN_BBOX to e.g. (65.0, 5.0, 100.0, 35.0) for South Asia. Everything
# downstream derives its grids from this box.
DOMAIN_BBOX: Optional[Tuple[float, float, float, float]] = None  # None -> basin bounds


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """A regular lon/lat grid, pinned to a source dataset's native alignment."""

    name: str
    res: float
    x_min: float          # western cell EDGE
    y_max: float          # northern cell EDGE
    n_x: int
    n_y: int
    crs: str = 'EPSG:4326'

    @property
    def x_max(self) -> float:
        return self.x_min + self.res * self.n_x

    @property
    def y_min(self) -> float:
        return self.y_max - self.res * self.n_y

    @property
    def crs_transform(self) -> List[float]:
        """Affine transform in Earth Engine order."""
        return [self.res, 0.0, self.x_min, 0.0, -self.res, self.y_max]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_y, self.n_x

    @property
    def n_cells(self) -> int:
        return self.n_x * self.n_y

    def lon_centers(self):
        import numpy as np
        return self.x_min + self.res * (np.arange(self.n_x) + 0.5)

    def lat_centers(self):
        import numpy as np
        return self.y_max - self.res * (np.arange(self.n_y) + 0.5)

    def ee_region(self):
        """Bounding rectangle as an ee.Geometry, matching the cell edges."""
        import ee
        return ee.Geometry.Rectangle(
            [self.x_min, self.y_min, self.x_max, self.y_max],
            proj=self.crs, geodesic=False,
        )

    def describe(self) -> str:
        return (f'{self.name:10s} {self.res:g} deg  '
                f'lon [{self.x_min:.2f}, {self.x_max:.2f}]  '
                f'lat [{self.y_min:.2f}, {self.y_max:.2f}]  '
                f'{self.n_x} x {self.n_y} = {self.n_cells} cells')


def snap_grid(
    name: str,
    bbox: Tuple[float, float, float, float],
    res: float,
    origin_x: float,
    origin_y: float,
    pad_cells: int = 0,
) -> Grid:
    """
    Build a Grid whose cell edges lie on the source dataset's native lattice and
    which fully contains `bbox` (snapping strictly outward), optionally padded.

    Parameters
    ----------
    bbox : (min_lon, min_lat, max_lon, max_lat)
    origin_x, origin_y : any cell edge of the native lattice (e.g. -180.05, 90.05)
    pad_cells : extra rings of cells added on every side. Used for the coarse
        grids so that every fine cell is covered by a coarse cell even where the
        fine grid's half-cell offset pushes it outside.
    """
    import math

    min_lon, min_lat, max_lon, max_lat = bbox
    # Work in units of cells from the lattice origin, snapping outward.
    i0 = math.floor((min_lon - origin_x) / res) - pad_cells
    i1 = math.ceil((max_lon - origin_x) / res) + pad_cells
    # y decreases as the row index grows, so the northern edge is the floor.
    j0 = math.floor((origin_y - max_lat) / res) - pad_cells
    j1 = math.ceil((origin_y - min_lat) / res) + pad_cells
    return Grid(
        name=name,
        res=res,
        x_min=origin_x + res * i0,
        y_max=origin_y - res * j0,
        n_x=int(i1 - i0),
        n_y=int(j1 - j0),
    )


def load_basin(dissolve: bool = False):
    """
    The Ganga basin boundary, always in EPSG:4326.

    Single reader for the shapefile. Five modules previously each called
    `gpd.read_file(...)` with their own CRS handling (one omitted `.to_crs`
    entirely), and two hardcoded the path rather than using BASIN_SHAPEFILE --
    the same one-thing-implemented-twice pattern that let the malformed-month
    bug survive its first fix.

    Parameters
    ----------
    dissolve : return a single unified geometry instead of the GeoDataFrame,
        for point-in-polygon tests.
    """
    import geopandas as gpd

    gdf = gpd.read_file(BASIN_SHAPEFILE).to_crs('EPSG:4326')
    if dissolve:
        return (gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all')
                else gdf.geometry.unary_union)
    return gdf


def basin_bbox() -> Tuple[float, float, float, float]:
    """Bounding box of the study domain in EPSG:4326."""
    if DOMAIN_BBOX is not None:
        return DOMAIN_BBOX
    return tuple(float(v) for v in load_basin().total_bounds)  # type: ignore[return-value]


# Native lattice origins, read off each collection's projection transform.
LATTICE = {
    'era5': (0.1, -180.05, 90.05),
    'gldas': (0.25, -180.0, 90.0),
    'grace': (0.5, -180.0, 90.0),
}


def build_grids(bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Grid]:
    """
    Construct the three native grids for the domain.

    The two coarse grids are padded by one cell so that they fully enclose the
    ERA5-Land grid: ERA5-Land's half-cell offset means its outermost edges
    (e.g. 21.45 S) can fall outside an unpadded 0.5 deg grid (21.50 S), which
    would leave fine cells with no parent mascon.
    """
    bbox = bbox or basin_bbox()
    grids: Dict[str, Grid] = {}
    for key, (res, ox, oy) in LATTICE.items():
        pad = 0 if key == 'era5' else 1
        grids[key] = snap_grid(key, bbox, res, ox, oy, pad_cells=pad)
    return grids


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VariableSpec:
    """One downloaded field, with its unit conversion stated explicitly."""

    name: str                   # name in the output cube
    band: str                   # band name in the source collection
    grid: str                   # 'era5' | 'gldas' | 'grace'
    scale: float = 1.0          # out = raw * scale + offset
    offset: float = 0.0
    out_units: str = ''
    long_name: str = ''
    # How to collapse the images that fall inside one output time step.
    # 'mean' is correct for every daily-native collection used here.
    reducer: str = 'mean'
    note: str = ''


ERA5_COLLECTION = 'ECMWF/ERA5_LAND/DAILY_AGGR'
# GLDAS 2.1 NOAH, NOT 2.2 CLSM.
#
# GLDAS-2.2 CLSM (collection id ...DA1D, "DA" = Data Assimilation) ASSIMILATES
# GRACE terrestrial water storage. Its water-storage states therefore contain the
# very signal this project predicts:
#
#   TWS_tavg, GWS_tavg          are GRACE, not independent of it
#   SoilMoist_*_tavg            carry the GRACE assimilation increment directly
#   Evap_tavg                   is a flux computed from those updated states
#
# Using them as predictors makes the model partly predict GRACE from GRACE, and
# using TWS_tavg as "independent validation" is circular. GLDAS-2.1 NOAH is
# model-only (no assimilation), so it is genuinely independent -- and it runs
# from 2000-01 rather than 2003-02, which is what allows the 2000-2025 product.
#
# NOAH has no groundwater store, so there is no TWS/GWS band to misuse.
GLDAS_COLLECTION = 'NASA/GLDAS/V021/NOAH/G025/T3H'
GRACE_COLLECTION = 'NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI'

COLLECTIONS = {
    'era5': ERA5_COLLECTION,
    'gldas': GLDAS_COLLECTION,
    'grace': GRACE_COLLECTION,
}

# ERA5-Land accumulations are metres of water equivalent per day -> x1000 = mm/day.
# Temperatures are kelvin -> -273.15 = degC.
# Volumetric soil water is m3/m3 (dimensionless), left as-is.
# ERA5-Land soil water is VOLUMETRIC (m3/m3), not a storage depth. Converting to
# millimetres requires the layer thickness:
#
#   layer 1   0-7 cm     ->   70 mm
#   layer 2   7-28 cm    ->  210 mm
#   layer 3  28-100 cm   ->  720 mm
#   layer 4 100-289 cm   -> 1890 mm
#
# Feeding the raw fraction to a model trained on GLDAS millimetres would be a
# silent 3-order-of-magnitude scale error, so the depths are applied here as the
# `scale` factor and the units recorded as mm.
ERA5_SOIL_LAYER_MM = {'vsw1': 70.0, 'vsw2': 210.0, 'vsw3': 720.0, 'vsw4': 1890.0}

ERA5_VARS: List[VariableSpec] = [
    VariableSpec('ppt', 'total_precipitation_sum', 'era5', 1000.0, 0.0, 'mm/day',
                 'Total precipitation',
                 note='Exact source of the paper\'s "rainfall" column.'),
    VariableSpec('runoff_surface', 'surface_runoff_sum', 'era5', 1000.0, 0.0, 'mm/day',
                 'Surface runoff',
                 note='Exact source of the paper\'s "runoff" column, and hence of "GWSA".'),
    VariableSpec('runoff_subsurface', 'sub_surface_runoff_sum', 'era5', 1000.0, 0.0, 'mm/day',
                 'Sub-surface runoff',
                 note='Needed for the water-balance daily disaggregation.'),
    VariableSpec('et_total', 'total_evaporation_sum', 'era5', -1000.0, 0.0, 'mm/day',
                 'Total evaporation (sign-flipped to positive = water leaving the surface)',
                 note='ERA5-Land reports evaporation as negative downward flux; '
                      'the -1000 factor makes it positive mm/day.'),
    VariableSpec('pet', 'potential_evaporation_sum', 'era5', -1000.0, 0.0, 'mm/day',
                 'Potential evaporation (sign-flipped)'),
    VariableSpec('swe', 'snow_depth_water_equivalent', 'era5', 1000.0, 0.0, 'mm',
                 'Snow depth water equivalent'),
    VariableSpec('snowmelt', 'snowmelt_sum', 'era5', 1000.0, 0.0, 'mm/day', 'Snowmelt'),
    VariableSpec('tmin', 'temperature_2m_min', 'era5', 1.0, -273.15, 'degC',
                 'Daily minimum 2 m air temperature'),
    VariableSpec('tmax', 'temperature_2m_max', 'era5', 1.0, -273.15, 'degC',
                 'Daily maximum 2 m air temperature'),
    VariableSpec('vsw1', 'volumetric_soil_water_layer_1', 'era5', 70.0, 0.0, 'mm',
                 'Soil water storage 0-7 cm (volumetric x layer thickness)'),
    VariableSpec('vsw2', 'volumetric_soil_water_layer_2', 'era5', 210.0, 0.0, 'mm',
                 'Soil water storage 7-28 cm (volumetric x layer thickness)'),
    VariableSpec('vsw3', 'volumetric_soil_water_layer_3', 'era5', 720.0, 0.0, 'mm',
                 'Soil water storage 28-100 cm (volumetric x layer thickness)'),
    VariableSpec('vsw4', 'volumetric_soil_water_layer_4', 'era5', 1890.0, 0.0, 'mm',
                 'Soil water storage 100-289 cm (volumetric x layer thickness)'),
]

# GLDAS V022 CLSM DA1D is DAILY (one image per day), so mean() over a one-day
# window returns that image and `Evap_tavg * 86400` is an exact kg/m2/s ->
# mm/day conversion. This is the 8x bug avoided; see module docstring.
# GLDAS 2.1 NOAH is 3-HOURLY (8 images per day), so the temporal reducer is not
# uniform -- getting this wrong is exactly how the historical 8x ET error arose:
#
#   *_inst  instantaneous states (kg/m2)          -> mean over the day
#   *_tavg  3-hourly MEAN of a rate (kg/m2/s)     -> mean over the day, then x86400
#   *_acc   3-hourly ACCUMULATION (kg/m2 per 3h)  -> SUM over the day
#
# Summing a *_tavg rate and then multiplying by 86400 overstates it eightfold;
# averaging a *_acc accumulation understates it eightfold.
GLDAS_VARS: List[VariableSpec] = [
    VariableSpec('sms', 'SoilMoi0_10cm_inst', 'gldas', 1.0, 0.0, 'mm',
                 'Surface soil moisture, 0-10 cm', reducer='mean',
                 note='Replaces GLDAS 2.2 CLSM SoilMoist_S_tavg (0-2 cm), which '
                      'carried the GRACE assimilation increment. Different layer '
                      'depth, so values are not directly comparable to the '
                      'earlier basin-scale analysis.'),
    VariableSpec('rzsm', 'RootMoist_inst', 'gldas', 1.0, 0.0, 'mm',
                 'Root-zone soil moisture', reducer='mean'),
    VariableSpec('sm_10_40', 'SoilMoi10_40cm_inst', 'gldas', 1.0, 0.0, 'mm',
                 'Soil moisture 10-40 cm', reducer='mean'),
    VariableSpec('sm_40_100', 'SoilMoi40_100cm_inst', 'gldas', 1.0, 0.0, 'mm',
                 'Soil moisture 40-100 cm', reducer='mean'),
    VariableSpec('sm_100_200', 'SoilMoi100_200cm_inst', 'gldas', 1.0, 0.0, 'mm',
                 'Soil moisture 100-200 cm', reducer='mean'),
    VariableSpec('et', 'Evap_tavg', 'gldas', 86400.0, 0.0, 'mm/day',
                 'Evapotranspiration', reducer='mean',
                 note='3-hourly mean rate; daily mean x 86400. Summing instead '
                      'would overstate by 8x.'),
    VariableSpec('pet', 'PotEvap_tavg', 'gldas', 86400.0, 0.0, 'W/m2->mm/day',
                 'Potential evaporation', reducer='mean'),
    VariableSpec('swe_gldas', 'SWE_inst', 'gldas', 1.0, 0.0, 'mm',
                 'Snow water equivalent', reducer='mean'),
    VariableSpec('canopy', 'CanopInt_inst', 'gldas', 1.0, 0.0, 'mm',
                 'Canopy interception', reducer='mean'),
    VariableSpec('qs_gldas', 'Qs_acc', 'gldas', 1.0, 0.0, 'mm/day',
                 'Surface runoff', reducer='sum',
                 note='3-hourly ACCUMULATION: sum the 8 sub-daily images, no '
                      'time-rate factor.'),
    VariableSpec('qsb_gldas', 'Qsb_acc', 'gldas', 1.0, 0.0, 'mm/day',
                 'Baseflow', reducer='sum', note='3-hourly accumulation; sum.'),
]

# GRACE JPL mascon: lwe_thickness is CENTIMETRES of equivalent water height.
# x10 -> mm. The basin-scale pipeline used x100.
GRACE_VARS: List[VariableSpec] = [
    VariableSpec('tws', 'lwe_thickness', 'grace', 10.0, 0.0, 'mm',
                 'GRACE/GRACE-FO liquid water equivalent thickness anomaly',
                 note='Source units are cm; x10 gives mm. gee_download.py used x100.'),
    VariableSpec('tws_uncertainty', 'uncertainty', 'grace', 10.0, 0.0, 'mm',
                 'GRACE mascon uncertainty'),
]

ALL_VARS: List[VariableSpec] = ERA5_VARS + GLDAS_VARS + GRACE_VARS
VARS_BY_NAME: Dict[str, VariableSpec] = {v.name: v for v in ALL_VARS}


def vars_for_grid(grid_key: str) -> List[VariableSpec]:
    return [v for v in ALL_VARS if v.grid == grid_key]


# --------------------------------------------------------------------------
# Predictor set used by the model
# --------------------------------------------------------------------------

# The five columns of the basin-scale paper, reproduced exactly on the grid.
# Provenance was established by testing for linear identity against the GEE
# downloads (max residual ~1e-14 in every case):
#
#   SMS      = GLDAS V022 CLSM SoilMoist_S_tavg              (surface layer)
#   ET       = GLDAS V022 CLSM Evap_tavg * 86400
#   rainfall = ERA5-Land total_precipitation_sum * 1000
#   runoff   = ERA5-Land surface_runoff_sum * 1000
#   GWSA     = runoff - mean(runoff)                          <-- not groundwater
#
# GWSA is de-meaned surface runoff. In the basin-mean case the offset was a
# single scalar, making it perfectly collinear with `runoff` (r = 1.00000) and
# rendering the SHAP split between the two arbitrary. On the grid the offset is
# per-pixel, so GWSA becomes a runoff anomaly relative to local climatology,
# which is non-degenerate and genuinely informative. It is retained under its
# original name for continuity with the manuscript; the name is a misnomer.
# Predictors, all from ERA5-Land -- ONE homogeneous source, 1950 to present,
# with no GRACE assimilation anywhere in its production chain.
#
# This replaces the GLDAS 2.2 CLSM fields the basin-scale analysis used. Those
# were disqualified on discovering that GLDAS-2.2 assimilates GRACE: its soil
# moisture carried the assimilation increment and its TWS/GWS bands simply ARE
# GRACE, so the model was partly predicting the target from the target. GLDAS 2.1
# NOAH would also have been open-loop, but ERA5-Land avoids a model splice
# entirely and is native to the 0.1 degree product grid, so no regridding of the
# predictors is required at all.
#
#   sms             = vsw1  (0-7 cm soil water storage, mm)
#   rzsm            = vsw1 + vsw2 + vsw3  (0-100 cm, mm)   [derived]
#   et              = total_evaporation_sum, sign-flipped to positive mm/day
#   ppt             = total_precipitation_sum
#   runoff_surface  = surface_runoff_sum
#   gwsa            = runoff_surface minus its per-pixel temporal mean [derived]
PREDICTORS: List[str] = ['sms', 'et', 'ppt', 'runoff_surface', 'gwsa']

# Derived on the grid rather than downloaded.
DERIVED = {
    'gwsa': dict(
        source='runoff_surface',
        method='subtract_pixelwise_temporal_mean',
        units='mm/day',
        long_name='Surface runoff anomaly relative to per-pixel long-term mean',
        note='Reproduces the paper\'s "GWSA" column definition. Despite the name '
             'this field contains no groundwater information.',
    ),
}

# Antecedent windows, in days. The basin-scale model used lags=7 only. Storage
# variables carry memory far longer than a week, and the temporal holdout's
# failure to extrapolate the secular decline is consistent with too short a
# memory, so the gridded model also gets multi-month antecedent means.
LAG_DAYS: Sequence[int] = (1, 2, 3, 5, 7, 14, 30, 60, 90, 180, 365)
ROLLING_WINDOWS: Sequence[int] = (7, 30, 90, 180, 365)

TARGET = 'tws'


# --------------------------------------------------------------------------
# Download layout
# --------------------------------------------------------------------------

# One GeoTIFF per (grid, variable, month) for ERA5-Land, per (grid, month) for
# the small grids. Sized to stay far under Earth Engine's ~48 MB download cap:
# a month of one ERA5-Land variable over the Ganga is 179 x 101 x 31 x 4 B
# ~= 2.2 MB.
DOWNLOAD_JOBS = 12
MAX_RETRIES = 8


def tif_path(grid_key: str, var_name: str, month: str) -> str:
    """Location of one downloaded monthly GeoTIFF stack (`month` is 'YYYY-MM')."""
    return os.path.join(RAW_TIF_DIR, grid_key, var_name, f'{var_name}_{month}.tif')


def cube_path(grid_key: str) -> str:
    return os.path.join(CUBE_DIR, f'{grid_key}_cube.nc')


def month_range(start: str = START_DATE, end: str = END_DATE) -> List[str]:
    """List of 'YYYY-MM' strings covering [start, end]."""
    import pandas as pd
    return [d.strftime('%Y-%m')
            for d in pd.date_range(start=start, end=end, freq='MS')]


def ensure_dirs() -> None:
    for d in (GRIDDED_DIR, RAW_TIF_DIR, CUBE_DIR, STATIC_DIR):
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    bbox = basin_bbox()
    print(f'Domain bbox: {tuple(round(v, 4) for v in bbox)}')
    print(f'Period     : {START_DATE} -> {END_DATE} ({len(month_range())} months)\n')
    for grid in build_grids(bbox).values():
        print('  ' + grid.describe())
    print(f'\nPredictors : {PREDICTORS}')
    print(f'Target     : {TARGET}')
    print(f'Downloads  : {len(ERA5_VARS)} ERA5 + {len(GLDAS_VARS)} GLDAS + '
          f'{len(GRACE_VARS)} GRACE fields')


# --------------------------------------------------------------------------
# Static / annual covariates
# --------------------------------------------------------------------------
#
# These supply WITHIN-MASCON spatial texture, which the dynamic predictors give
# only through their own climatology. They are chosen by mechanism, not by
# availability:
#
#   water table depth   controls specific yield and drainage, so it governs how
#                       much storage change a given flux produces
#   irrigated cropland  proxy for groundwater abstraction -- the driver of the
#                       secular trend that no reanalysis simulates and that the
#                       model currently cannot predict (LOMO trend R^2 ~ 0.11)
#   AWC, root depth     control the storage range available to the soil column
#
# None of them assimilates GRACE. GLOBGM is PCR-GLOBWB driven (validated against
# GRACE, not assimilating it), so it is independent in the sense that matters --
# but it is a MODEL OUTPUT, not an observation, and carries its own structural
# error. That belongs in the paper.
#
# WARNING, the reason `kind` exists: C3S land cover is CATEGORICAL. Requesting
# the class field at anything coarser than its native 309 m makes Earth Engine
# average the class CODES through its overview pyramids. Measured over this
# basin that returns values like 13, 15, 18, 23 -- none of which are LCCS
# classes, and all of which look like ordinary data. Class covariates must
# therefore be one-hot masked at native scale FIRST and only then aggregated,
# which `kind='annual_class'` enforces.

C3S_COLLECTION = 'projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS'
C3S_FIRST_YEAR, C3S_LAST_YEAR = 2000, 2022   # frozen after C3S_LAST_YEAR


@dataclass(frozen=True)
class StaticSpec:
    """A time-invariant (or annual) covariate on the 0.1 degree product grid."""

    name: str
    asset: str
    kind: str                    # 'image' | 'annual_class'
    band: Optional[str] = None
    class_value: Optional[int] = None
    scale: float = 1.0
    out_units: str = ''
    long_name: str = ''
    note: str = ''


STATIC_VARS: List[StaticSpec] = [
    StaticSpec('wtd', 'projects/sat-io/open-datasets/GLOBGM/STEADY-STATE/globgm-wtd-ss',
               'image', band='b1', out_units='m',
               long_name='Water table depth, GLOBGM steady state',
               note='Model output (PCR-GLOBWB), not an observation. Independent of '
                    'GRACE: GLOBGM is validated against GRACE, not assimilating it. '
                    'The TRANSIENT GLOBGM collection ends 2015-12 and is deliberately '
                    'NOT used - a predictor that vanishes partway through the record '
                    'is worse than a static one.'),
    StaticSpec('awc', 'projects/sat-io/open-datasets/FAO/HWSD_V2_SMU',
               'image', band='AWC', out_units='mm/m',
               long_name='Available water capacity, HWSD v2'),
    StaticSpec('root_depth', 'projects/sat-io/open-datasets/FAO/HWSD_V2_SMU',
               'image', band='ROOT_DEPTH', out_units='cm',
               long_name='Obstacle-to-roots depth, HWSD v2'),
    StaticSpec('elevation', 'NASA/NASADEM_HGT/001', 'image', band='elevation',
               out_units='m', long_name='NASADEM elevation',
               note='Averaged from 30 m to 0.1 deg.'),
    # NOTE: mean slope is deliberately NOT included.
    #
    # Three Earth Engine formulations were tried (native 30 m -> 0.1 deg, an
    # intermediate 1 km projection, and a 0.01 deg export differentiated locally)
    # and all three returned 400 at the pixel-fetch stage -- a server-side
    # compute timeout on ee.Terrain.slope over 30 m across a basin this size,
    # not a request-size limit.
    #
    # `elevation_std` carries the same information at this resolution. Both are
    # measures of relief, and over an 11 km cell they are near-redundant: mean
    # slope IS essentially sub-grid relief divided by cell width. Adding a second
    # collinear terrain covariate would cost a degree of freedom against only
    # ~19 independent mascons and buy nothing. If slope is wanted explicitly,
    # export the DEM in tiles and differentiate offline.
    StaticSpec('elevation_std', 'NASA/NASADEM_HGT/001', 'terrain_sd', band='elevation',
               out_units='m', long_name='Sub-grid elevation standard deviation (roughness)',
               note='Within-cell relief. At 11 km this separates the Himalayan '
                    'margin from the plain far better than mean elevation alone.'),
    StaticSpec('upa_log', 'MERIT/Hydro/v1_0_1', 'image_log', band='upa',
               out_units='log10(km2)', long_name='Log upstream drainage area',
               note='Used INSTEAD of a topographic wetness index. TWI = ln(a/tan b) '
                    'is a hillslope quantity defined at 30-100 m; averaged over an '
                    '11 km cell it degenerates -- its slope term duplicates '
                    'elevation_std and its contributing-area term collapses to '
                    '"is there a large river here". log(upa) states that directly '
                    'and interpretably. Log because upa spans ~9 orders of '
                    'magnitude, from hillslope cells to the whole Ganga.'),
    StaticSpec('hnd', 'MERIT/Hydro/v1_0_1', 'image', band='hnd',
               out_units='m', long_name='Height above nearest drainage',
               note='Floodplain vs upland separation. Unlike TWI this remains '
                    'meaningful when averaged, since it is a height difference '
                    'rather than a ratio of scale-dependent terms.'),
    StaticSpec('crop_irrigated', C3S_COLLECTION, 'annual_class', class_value=20,
               out_units='fraction', long_name='Irrigated cropland fraction (LCCS 20)',
               note='One-hot at native 309 m, then area-averaged to 0.1 deg. '
                    'Multiply by cell area for km^2.'),
    StaticSpec('crop_rainfed', C3S_COLLECTION, 'annual_class', class_value=10,
               out_units='fraction', long_name='Rainfed cropland fraction (LCCS 10)'),
]

STATIC_BY_NAME: Dict[str, StaticSpec] = {v.name: v for v in STATIC_VARS}

# Covariates offered to the model. Empty by default: each must EARN its place by
# improving held-out TREND skill in the leave-one-mascon-out test, because with
# ~19 independent mascons a covariate correlated with the depletion trend can
# manufacture apparent skill. See `downscale_covariate_gate.py`.
# Covariates the model actually uses.
#
# Resolved at call time by `resolve_active_covariates()`, which PREFERS the
# survivor list written by `downscale_covariate_gate.py`. Pipeline order is:
# gate measures held-out skill -> writes covariate_survivors.json -> the model
# picks it up automatically. No manual promotion step.
#
# Automated but not silent: the gate writes the full comparison table alongside
# the survivor list, every run logs which covariates were resolved and from
# where, and the choice is recorded in the product netCDF. Set
# COVARIATES_OVERRIDE or the GRACE_COVARIATES env var to pin a set and ignore
# the gate.
ACTIVE_COVARIATES: List[str] = []
COVARIATES_OVERRIDE: Optional[List[str]] = None
SURVIVOR_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'Results', 'downscaling',
    'covariate_survivors.json')


def resolve_active_covariates(verbose: bool = False) -> List[str]:
    """
    Which covariates the model should use, and where that decision came from.

    Order: explicit override -> GRACE_COVARIATES env var -> gate survivors ->
    the ACTIVE_COVARIATES literal (empty, i.e. baseline).
    """
    if COVARIATES_OVERRIDE is not None:
        chosen, src = list(COVARIATES_OVERRIDE), 'COVARIATES_OVERRIDE'
    elif os.environ.get('GRACE_COVARIATES'):
        chosen = [c for c in os.environ['GRACE_COVARIATES'].split(',') if c.strip()]
        src = 'GRACE_COVARIATES env var'
    elif os.path.exists(SURVIVOR_JSON):
        try:
            import json
            with open(SURVIVOR_JSON) as fh:
                chosen = list(json.load(fh).get('survivors', []))
            src = 'covariate_survivors.json (leave-one-mascon-out gate)'
        except (ValueError, OSError):
            chosen, src = list(ACTIVE_COVARIATES), 'ACTIVE_COVARIATES (survivor file unreadable)'
    else:
        chosen, src = list(ACTIVE_COVARIATES), 'ACTIVE_COVARIATES (gate has not run)'

    known = {v.name for v in STATIC_VARS}
    unknown = [c for c in chosen if c not in known]
    if unknown:
        raise ValueError(f'unknown covariates {unknown}; known: {sorted(known)}')
    if verbose:
        print(f'  covariates: {chosen or "(none)"}  [from {src}]')
    return chosen

STATIC_DIR = os.path.join(GRIDDED_DIR, 'static')


def static_tif_path(name: str, year: Optional[int] = None) -> str:
    suffix = f'_{year}' if year is not None else ''
    return os.path.join(STATIC_DIR, f'{name}{suffix}.tif')
