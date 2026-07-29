"""
Ingest the quality-controlled Indian groundwater-level dataset for use as
INDEPENDENT validation of the downscaled 0.1 degree TWSA product.

Source
------
Kuruva, S.K., Suryawanshi, M.R., Shakya, A. et al. "Quality controlled, reliable
groundwater level data with corresponding specific yield over India."
Sci Data 12, 1609 (2025). https://doi.org/10.1038/s41597-025-05899-5
Data: https://doi.org/10.6084/m9.figshare.29293877 (v3, CC BY 4.0)

These wells are never used as predictors. They are the only observation in this
project that GRACE did not produce, so they are held out entirely and used to
answer the one question GRACE-internal cross-validation cannot: does the
fine-scale spatial structure we synthesise correspond to anything real?

Three properties of the published files drive the choices here
--------------------------------------------------------------
1. `Station Code` is stored in 3-significant-figure scientific notation
   (`1.41E+14`), collapsing 2,759 wells onto 182 distinct codes. It is unusable
   as a key. Latitude/longitude is unique for every row in every file, so wells
   are identified by rounded coordinates instead.

2. The `Filtered_GWLs_2000_2024` files contain only wells whose records run to
   2024 -- 399 nationally, of which just 7 lie in the Ganga basin. The
   `ref_sy_2000_2022` files carry 2,759 wells (691 in-basin) AND the
   `Reference_Sy` column needed to convert water levels to storage. We take the
   larger set as primary and treat the long records as an out-of-period check.

3. Observations are QUARTERLY (Jan/May/Aug/Nov), not monthly. The wells can
   therefore validate spatial pattern and seasonal amplitude, but say nothing
   about daily structure; that remains the temporal closure test's job.

Baseline
--------
JPL RL06 mascons are anomalies relative to the mean of 2004.0-2010.0. Well
levels must be anomalised over the SAME window or every comparison carries a
constant offset that would be misread as bias. Wells without enough baseline
coverage are dropped rather than referenced to a different period.

Sign convention
---------------
The published levels are DEPTH TO WATER in metres below ground, so storage falls
as the number rises:

    GWS anomaly [mm] = -Sy * (depth - depth_baseline) * 1000
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import gridded_config as cfg

GW_DIR = os.path.join(cfg.DATA_DIR, 'Groundwater', 'gwl_india',
                      'Quality_controlled_groundwater_levels_over_India')

# Primary: dug wells tap the unconfined aquifer, where specific yield is the
# correct storage coefficient. Bore/tube wells may screen confined units whose
# storativity is orders of magnitude smaller, which would corrupt the
# conversion, so they are excluded by default.
DUG_WELL_FILE = os.path.join(
    GW_DIR, 'Output', 'CGWB_India_filtered_Dug_wells_GWLs_ref_sy_2000_2022.csv')
ALL_WELL_FILE = os.path.join(
    GW_DIR, 'Output', 'CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv')
LONG_RECORD_FILE = os.path.join(
    GW_DIR, 'Output', 'Filtered_GWLs_2000_2024', 'CGWB_filtered_wells_2000_2024.csv')
HYDROGEO_MAP = os.path.join(GW_DIR, 'Hydrogeological_map', 'Hydrogeological_map.tif')

# JPL RL06 mascon anomaly baseline.
BASELINE_START = '2004-01-01'
BASELINE_END = '2009-12-31'
MIN_BASELINE_OBS = 8       # of 24 possible quarterly readings in 2004-2009
MIN_TOTAL_OBS = 40         # of 92 possible readings in 2000-2022

_MONTHS = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
           'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
_TIME_COL = re.compile(r'^(' + '|'.join(_MONTHS) + r')-(\d{2})$')

META_COLS = [
    'Station Name', 'State', 'District', 'Tehsil', 'Block', 'Village',
    'Latitude', 'Longitude', 'Type of Well', 'Aquifer Type', 'Well Depth',
    'Reference_Sy',
]


@dataclass
class WellData:
    """Wells and their quarterly groundwater storage anomalies."""

    meta: pd.DataFrame     # one row per well: well_id, lat, lon, sy, ...
    obs: pd.DataFrame      # long: well_id, date, depth_m, gws_anomaly_mm

    def __len__(self) -> int:
        return len(self.meta)

    def describe(self) -> str:
        return (f'{len(self.meta)} wells, {len(self.obs)} observations, '
                f'{self.obs.date.min():%Y-%m} to {self.obs.date.max():%Y-%m}')


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _time_columns(columns) -> List[Tuple[str, pd.Timestamp]]:
    """Quarterly measurement columns and the dates they encode."""
    out = []
    for col in columns:
        m = _TIME_COL.match(str(col).strip())
        if m:
            month, yy = _MONTHS[m.group(1)], int(m.group(2))
            year = 2000 + yy          # files span 2000-2024
            out.append((col, pd.Timestamp(year=year, month=month, day=1)))
    return out


def _well_ids(df: pd.DataFrame) -> pd.Series:
    """
    Stable identifier built from coordinates.

    `Station Code` cannot be used: 3-significant-figure scientific notation in
    the source CSVs collapses 2,759 wells onto 182 codes.
    """
    return (df.Latitude.round(4).map('{:.4f}'.format) + '_'
            + df.Longitude.round(4).map('{:.4f}'.format))


def load_raw(path: str = DUG_WELL_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Download the dataset from '
            f'https://doi.org/10.6084/m9.figshare.29293877 into Data/Groundwater/.'
        )
    return pd.read_csv(path, low_memory=False)


def clip_to_basin(df: pd.DataFrame) -> pd.DataFrame:
    """Keep wells inside the Ganga basin polygon."""
    import geopandas as gpd

    poly = cfg.load_basin(dissolve=True)
    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs='EPSG:4326')
    return pd.DataFrame(pts[pts.within(poly)].drop(columns='geometry'))


# --------------------------------------------------------------------------
# Storage anomalies
# --------------------------------------------------------------------------

def build(
    path: str = DUG_WELL_FILE,
    basin_only: bool = True,
    baseline: Tuple[str, str] = (BASELINE_START, BASELINE_END),
    min_baseline_obs: int = MIN_BASELINE_OBS,
    min_total_obs: int = MIN_TOTAL_OBS,
    verbose: bool = True,
) -> WellData:
    """
    Load wells and convert depth-to-water into groundwater storage anomalies on
    the GRACE mascon baseline.

    Returns a `WellData` whose `obs` frame carries `gws_anomaly_mm`, directly
    comparable with the groundwater component derived from the downscaled TWSA
    product (TWSA minus modelled soil moisture, snow and canopy storage).
    """
    raw = load_raw(path)
    n_start = len(raw)
    if basin_only:
        raw = clip_to_basin(raw)
    n_basin = len(raw)

    time_cols = _time_columns(raw.columns)
    if not time_cols:
        raise ValueError(f'no quarterly measurement columns found in {path}')

    raw = raw.copy()
    raw['well_id'] = _well_ids(raw)
    if raw.well_id.duplicated().any():
        raise ValueError('duplicate well coordinates; identifier is not unique')

    keep_meta = ['well_id'] + [c for c in META_COLS if c in raw.columns]
    meta = raw[keep_meta].rename(columns={
        'Latitude': 'lat', 'Longitude': 'lon', 'Reference_Sy': 'sy',
        'Well Depth': 'well_depth_m', 'Type of Well': 'well_type',
        'Aquifer Type': 'aquifer_type', 'Station Name': 'name',
    }).set_index('well_id')

    obs = (raw[['well_id'] + [c for c, _ in time_cols]]
           .melt(id_vars='well_id', var_name='col', value_name='depth_m')
           .assign(date=lambda d: d.col.map(dict(time_cols)))
           .drop(columns='col')
           .dropna(subset=['depth_m']))
    obs = obs[np.isfinite(obs.depth_m)]

    # Baseline mean per well, on the GRACE anomaly window.
    b0, b1 = pd.Timestamp(baseline[0]), pd.Timestamp(baseline[1])
    in_base = obs[(obs.date >= b0) & (obs.date <= b1)]
    base = in_base.groupby('well_id').depth_m.agg(['mean', 'count'])
    base.columns = ['depth_baseline_m', 'n_baseline']

    counts = obs.groupby('well_id').size().rename('n_obs')
    meta = meta.join(base).join(counts)

    if 'sy' not in meta.columns:
        raise ValueError(
            f'{os.path.basename(path)} has no Reference_Sy column. Use the '
            f'ref_sy files, or join specific yield on by coordinates.'
        )

    good = (
        meta.n_baseline.fillna(0).ge(min_baseline_obs)
        & meta.n_obs.fillna(0).ge(min_total_obs)
        & meta.sy.notna() & meta.sy.gt(0)
    )
    dropped = (~good).sum()
    meta = meta[good]
    obs = obs[obs.well_id.isin(meta.index)]

    # depth rises  ->  water table falls  ->  storage decreases
    obs = obs.merge(
        meta[['depth_baseline_m', 'sy']], left_on='well_id', right_index=True)
    obs['gws_anomaly_mm'] = (
        -obs.sy * (obs.depth_m - obs.depth_baseline_m) * 1000.0)
    obs = (obs[['well_id', 'date', 'depth_m', 'gws_anomaly_mm']]
           .sort_values(['well_id', 'date'])
           .reset_index(drop=True))

    if verbose:
        print(f'  {os.path.basename(path)}')
        print(f'    {n_start} wells -> {n_basin} in basin -> {len(meta)} usable '
              f'({dropped} dropped for sparse baseline/record or missing Sy)')
        print(f'    {len(obs)} observations, {obs.date.min():%Y-%m} to '
              f'{obs.date.max():%Y-%m}, quarterly')
        print(f'    specific yield: median {meta.sy.median():.3f}, '
              f'range {meta.sy.min():.3f}-{meta.sy.max():.3f}')
        amp = obs.groupby('well_id').gws_anomaly_mm.agg(lambda s: s.max() - s.min())
        print(f'    GWS anomaly range per well: median {amp.median():.0f} mm, '
              f'p95 {amp.quantile(0.95):.0f} mm')

    return WellData(meta=meta.reset_index(), obs=obs)


def assign_grid_cells(wells: WellData, grid_key: str = 'era5') -> WellData:
    """
    Attach the row/col of the grid cell containing each well, plus its flat
    index, so observations can be matched to the downscaled product.
    """
    grid = cfg.build_grids()[grid_key]
    meta = wells.meta.copy()
    col = np.floor((meta.lon.to_numpy() - grid.x_min) / grid.res).astype(int)
    row = np.floor((grid.y_max - meta.lat.to_numpy()) / grid.res).astype(int)
    inside = (col >= 0) & (col < grid.n_x) & (row >= 0) & (row < grid.n_y)
    meta[f'{grid_key}_row'] = np.where(inside, row, -1)
    meta[f'{grid_key}_col'] = np.where(inside, col, -1)
    meta[f'{grid_key}_flat'] = np.where(inside, row * grid.n_x + col, -1)
    return WellData(meta=meta, obs=wells.obs)


def save(wells: WellData, out_dir: Optional[str] = None) -> Tuple[str, str]:
    out_dir = out_dir or os.path.join(cfg.GRIDDED_DIR, 'wells')
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, 'well_meta.csv')
    obs_path = os.path.join(out_dir, 'well_gws_anomaly.csv')
    wells.meta.to_csv(meta_path, index=False)
    wells.obs.to_csv(obs_path, index=False)
    return meta_path, obs_path


if __name__ == '__main__':
    print('Ganga basin groundwater wells (independent validation set)\n')

    print('Primary (dug wells, unconfined aquifer, Sy applicable):')
    dug = build(DUG_WELL_FILE)
    dug = assign_grid_cells(dug)

    print('\nAll well types, for a sensitivity check:')
    allw = build(ALL_WELL_FILE)

    grid = cfg.build_grids()['era5']
    occupied = dug.meta[dug.meta.era5_flat >= 0].era5_flat.nunique()
    print(f'\nCoverage: {occupied} distinct 0.1 deg cells hold at least one well, '
          f'of ~9,077 in-basin cells ({100 * occupied / 9077:.1f}%)')
    print(f'Well density: 1 per {1_004_724 / max(len(dug), 1):,.0f} km2')

    meta_path, obs_path = save(dug)
    print(f'\nwritten: {meta_path}\n         {obs_path}')
