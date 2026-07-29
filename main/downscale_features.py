"""
Feature construction for the spatial GRACE downscaling.

Design decision: what the model is actually asked to learn
----------------------------------------------------------
GRACE observes ~20 independent 3 degree mascons over the Ganga basin. Training
one sample per mascon per month would give ~5,000 samples with 20 spatial
degrees of freedom -- too thin to fit anything with spatial structure.

Instead we train on the 0.5 degree GRACE grid (364 in-basin cells), where the
PREDICTORS vary cell to cell but the TARGET is the parent mascon's value,
repeated across the cells it contains. The model therefore learns

    E[ TWSA | predictors ]

which is exactly the quantity we want to evaluate at 0.1 degrees. The repeated
target is not leakage PROVIDED cross-validation is blocked by mascon -- two
cells of one mascon share a target, so splitting them across folds would let the
model read the answer off its neighbour. `downscale_model.py` blocks on mascon
and additionally buffers out the neighbours, because GRACE's CRI filtering
correlates adjacent mascons.

Scale invariance, and its limit
-------------------------------
Applying a model fitted on 0.5 degree predictors to 0.1 degree predictors
assumes the predictor->TWSA relation is scale invariant. It is not exactly:
averaging shrinks variance, so 0.1 degree inputs range wider than anything seen
in training, and tree ensembles will clamp to their training envelope. We
mitigate by ALSO emitting features at several aggregation scales (0.5, 1.0 and
2.0 degrees) so the model sees both a cell's own value and its neighbourhood
context, and the extrapolation is over a genuinely observed range. The residual
mismatch is absorbed by the mass-conservation step, which pins every mascon's
aggregate to the observation.

Feature groups
--------------
  dynamic    the five paper predictors (sms, et, ppt, runoff_surface, gwsa),
             monthly means of the daily fields
  antecedent multi-month backward means -- storage carries memory far longer
             than the 7-day window the basin-scale model used, which is the
             most likely reason it could not extrapolate the secular decline
  context    the same fields averaged over 1 and 2 degree neighbourhoods
  climatology per-cell long-term mean and standard deviation of each predictor:
             static spatial covariates derived from the data itself, giving the
             model something to distinguish one cell from another with
  seasonal   sin/cos harmonics of day-of-year
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import gridded_config as cfg
import downscale_grid_ops as ops

# Raw latitude/longitude are NOT used as predictors.
#
# During training the target is constant within a mascon, so the best possible
# coordinate splits fall exactly ON mascon boundaries. A tree therefore learns
# the mascon partition itself and, evaluated at 0.1 degrees, prints those
# boundaries into the product as visible seams. Measured on a fitted XGBoost,
# lat ranked 6th of 62 features and lat+lon carried 4.8% of total gain.
#
# It is also memorisation of the training geometry rather than a transferable
# relationship: a coordinate split says "this place is dry", not "places like
# this are dry", so it cannot generalise to an unseen location. Spatial
# discrimination is left to the physical covariates -- per-pixel climatology and
# the neighbourhood context means -- which do transfer.
#
# `basin_frac` is dropped for a related reason: it spans 0.5-1.0 in training
# (0.5 degree cells) but 0-1 at prediction (0.1 degree cells), a distribution
# shift the model would read as signal.
USE_COORDINATE_FEATURES = False

# Backward windows in MONTHS for the monthly model.
ANTECEDENT_MONTHS: Sequence[int] = (1, 2, 3, 6, 12, 24)
# Neighbourhood context radii, in degrees.
CONTEXT_DEGREES: Sequence[float] = (1.0, 2.0)

# Antecedent Precipitation Index e-folding times, in MONTHS.
#
# The antecedent means above are boxcars: equal weight across the window, then a
# hard cutoff. Storage does not respond that way -- its memory of past rainfall
# decays roughly exponentially and never truly ends. API captures that with one
# parameter per timescale:
#
#     API_t = exp(-1/tau) * API_{t-1} + P_t
#
# The long tail matters specifically here. The model's central weakness is that
# it cannot predict the secular trend (leave-one-mascon-out R^2 ~ 0.11), because
# the trend is CUMULATIVE depletion, and no boxcar mean of 24 months or less can
# represent an accumulation over two decades. The 60- and 120-month terms are
# deliberately longer than any antecedent window for that reason.
API_TAU_MONTHS: Sequence[int] = (3, 12, 60, 120)

# Fields for which an API is built.
API_FIELDS: Sequence[str] = ('ppt', 'et')

# The WATER BALANCE gets an API too, and it is the physically motivated one.
#
# Storage change is driven by P - ET - Q, not by rainfall alone, so an
# exponentially damped accumulation of the water balance is the closest thing to
# a storage-memory term available from the predictors:
#
#     wb_api_tau  ~  a damped integral of (P - ET - Q)
#
# A plain cumulative sum would be better still in principle, but any bias in the
# flux integrates without bound over 26 years -- the finite tau damps that,
# which is precisely why an API is used rather than a cumsum.
#
# Only API columns are emitted for `wb`, not the full antecedent/context/
# climatology family. Against ~19 independent mascons every extra feature costs,
# and the accumulation is the part that carries information the existing
# predictors do not.
#
# Sub-surface runoff is omitted because it is not in the predictor set; the
# balance is therefore incomplete and its absolute level is not meaningful. Only
# its variation is, which is all the model uses.
WATER_BALANCE_API: bool = True


@dataclass
class FeatureSet:
    """Design matrix plus everything needed to interpret and re-map it."""

    X: pd.DataFrame
    y: np.ndarray                 # NaN where GRACE has a gap
    mascon: np.ndarray            # grouping variable for spatial CV
    cell: np.ndarray              # flat index into the grid
    time_index: np.ndarray        # index into the month axis
    dates: pd.DatetimeIndex
    grid_key: str

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.y)

    def describe(self) -> str:
        return (f'{len(self.X):,} samples x {self.X.shape[1]} features | '
                f'{self.observed.sum():,} with GRACE | '
                f'{len(np.unique(self.mascon))} mascons | '
                f'{len(self.dates)} months')


# --------------------------------------------------------------------------
# Reading cubes
# --------------------------------------------------------------------------

def cube_dates(grid_key: str) -> pd.DatetimeIndex:
    import netCDF4
    with netCDF4.Dataset(cfg.cube_path(grid_key)) as ds:
        return pd.to_datetime(ds['time'][:], unit='D', origin='1970-01-01')


def load_aux() -> Dict[str, np.ndarray]:
    import netCDF4
    path = os.path.join(cfg.CUBE_DIR, 'grids_aux.nc')
    with netCDF4.Dataset(path) as ds:
        return {k: np.asarray(ds[k][:]) for k in ds.variables}


def monthly_mean(grid_key: str, var: str, months: pd.DatetimeIndex) -> np.ndarray:
    """
    Monthly means of a daily cube variable, shape (n_months, n_y, n_x).

    Read month by month so a 3.85 GB cube never lands in memory at once.
    """
    import netCDF4

    dates = cube_dates(grid_key)
    with netCDF4.Dataset(cfg.cube_path(grid_key)) as ds:
        v = ds[var]
        n_y, n_x = v.shape[1], v.shape[2]
        out = np.full((len(months), n_y, n_x), np.nan, dtype='float32')
        period = dates.to_period('M')
        for i, m in enumerate(months):
            idx = np.flatnonzero(period == m.to_period('M'))
            if idx.size:
                block = np.asarray(v[idx[0]:idx[-1] + 1])
                with np.errstate(invalid='ignore'):
                    out[i] = np.nanmean(block, axis=0)
    return out


def load_grace_monthly(months: pd.DatetimeIndex) -> np.ndarray:
    """GRACE TWSA on its own 0.5 degree grid, (n_months, n_y, n_x), NaN in gaps."""
    import netCDF4

    dates = cube_dates('grace')
    with netCDF4.Dataset(cfg.cube_path('grace')) as ds:
        tws = np.asarray(ds['tws'][:])
    pos = {d.to_period('M'): i for i, d in enumerate(dates)}
    out = np.full((len(months),) + tws.shape[1:], np.nan, dtype='float32')
    for i, m in enumerate(months):
        j = pos.get(m.to_period('M'))
        if j is not None:
            out[i] = tws[j]
    return out


# --------------------------------------------------------------------------
# Predictor stack on a target grid
# --------------------------------------------------------------------------

# Which cube each of the five paper predictors comes from. `gwsa` is derived.
# Every predictor now comes from ERA5-Land, on the 0.1 degree product grid, so
# no predictor is regridded at all. Previously sms/et came from GLDAS 2.2 CLSM,
# which assimilates GRACE -- the model was partly predicting the target from the
# target, and the 0.25 -> 0.1 deg bilinear step introduced its own artefacts.
PREDICTOR_SOURCE = {
    'sms': ('era5', 'vsw1'),              # 0-7 cm soil water storage, mm
    'et': ('era5', 'et_total'),           # sign-flipped to positive mm/day
    'ppt': ('era5', 'ppt'),
    'runoff_surface': ('era5', 'runoff_surface'),
}


def predictor_stack(
    target_key: str,
    months: pd.DatetimeIndex,
    predictors: Sequence[str] = tuple(cfg.PREDICTORS),
) -> Dict[str, np.ndarray]:
    """
    Monthly predictor fields on `target_key`'s grid, each (n_months, n_y, n_x).

    Fields whose native grid is finer than the target are aggregated with the
    exact area-overlap operator; coarser ones are bilinearly interpolated.
    `gwsa` is derived on the target grid as surface runoff minus its per-cell
    long-term mean, reproducing the definition used by the basin-scale paper.
    """
    grids = cfg.build_grids()
    target = grids[target_key]

    stack: Dict[str, np.ndarray] = {}
    for name in predictors:
        if name == 'gwsa':
            continue
        src_key, var = PREDICTOR_SOURCE[name]
        field = monthly_mean(src_key, var, months)
        src = grids[src_key]

        if src_key == target_key:
            stack[name] = field
        elif src.res < target.res:
            w = ops.overlap_matrix(src, target)
            agg = ops.aggregate_area_weighted(field, w)
            stack[name] = agg.reshape(len(months), target.n_y, target.n_x)
        else:
            stack[name] = ops.regrid_bilinear(field, src, target)

    if 'gwsa' in predictors:
        runoff = stack['runoff_surface']
        with np.errstate(invalid='ignore'):
            stack['gwsa'] = runoff - np.nanmean(runoff, axis=0, keepdims=True)

    return stack


def _context_mean(field: np.ndarray, grid: cfg.Grid, radius_deg: float) -> np.ndarray:
    """Neighbourhood mean of a field, radius expressed in degrees."""
    sigma = max(radius_deg / grid.res / 2.0, 0.5)
    return ops.gaussian_smooth(field.astype('float64'), sigma).astype('float32')


def _api(field: np.ndarray, tau_months: float) -> np.ndarray:
    """
    Antecedent Precipitation Index with an exponential memory of `tau_months`.

        API_t = k * API_{t-1} + P_t,     k = exp(-1 / tau)

    Shifted by one step like the boxcar antecedents, so a sample never contains
    its own month.

    Initialised at the steady state implied by the mean of the first year, not by
    the first month. Two reasons. Starting from zero leaves a spin-up ramp that
    masquerades as a trend -- at tau = 120 months it takes a decade to equilibrate,
    which is exactly the backward-extrapolation period. And seeding from a single
    month makes the whole series hostage to whether the record happens to open on
    an anomalously wet or dry one; averaging the first 12 months removes the
    seasonal cycle from the seed.
    """
    n_t = field.shape[0]
    flat = field.reshape(n_t, -1)
    k = float(np.exp(-1.0 / float(tau_months)))

    # Seed window scales with the memory. A 12-month seed is far too short for a
    # 120-month filter: measured, a single 10x-wet opening month still displaced
    # the series by 134 units ten years later. Averaging over a window comparable
    # to tau makes the seed an estimate of the equilibrium the filter is heading
    # for, rather than an accident of where the record happens to start.
    warmup = min(max(12, int(round(tau_months))), n_t)
    with np.errstate(invalid='ignore'):
        seed = np.nan_to_num(np.nanmean(flat[:warmup], axis=0), nan=0.0)
    out = np.empty_like(flat)
    prev = seed / max(1.0 - k, 1e-9)
    for t in range(n_t):
        out[t] = prev
        prev = k * prev + np.nan_to_num(flat[t], nan=0.0)
    return out.reshape(field.shape)


def _antecedent(field: np.ndarray, window: int) -> np.ndarray:
    """
    Backward rolling mean over `window` months, ending at the previous month.

    Shifted by one so a sample never contains its own month twice, which would
    make the antecedent feature a near-duplicate of the contemporaneous one.
    """
    n_t = field.shape[0]
    flat = field.reshape(n_t, -1)
    out = np.full_like(flat, np.nan)
    frame = pd.DataFrame(flat)
    rolled = frame.shift(1).rolling(window, min_periods=max(1, window // 2)).mean()
    out[:] = rolled.to_numpy()
    return out.reshape(field.shape)



def covariate_stack(
    target_key: str,
    months: pd.DatetimeIndex,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Static / annual covariates on `target_key`'s grid, each (n_months, n_y, n_x).

    The cube is written on the 0.1 degree grid. For training on the 0.5 degree
    GRACE grid the fields are area-aggregated with the exact overlap operator --
    the same treatment the dynamic predictors get, so a covariate means the same
    thing at both scales.

    Annual covariates are broadcast month-to-year, holding the last available
    year for months beyond the source record (C3S ends 2022).
    """
    import netCDF4

    if not names:
        return {}
    path = os.path.join(cfg.CUBE_DIR, 'static_cube.nc')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found - run build_cube.py (needs gee_static_download.py first)')

    grids = cfg.build_grids()
    era5, target = grids['era5'], grids[target_key]
    weights = None if target_key == 'era5' else ops.overlap_matrix(era5, target)

    out: Dict[str, np.ndarray] = {}
    with netCDF4.Dataset(path) as ds:
        years = np.asarray(ds['year'][:]) if 'year' in ds.variables else None
        for name in names:
            if name not in ds.variables:
                raise KeyError(f'{name} not in static_cube.nc')
            var = ds[name]
            if var.ndim == 3:                       # annual
                stack = np.asarray(var[:])
                last = int(getattr(var, 'frozen_after', years[-1]))
                idx = [int(np.searchsorted(years, min(d.year, last))) for d in months]
                fine = stack[idx]
            else:                                   # static: repeat through time
                fine = np.repeat(np.asarray(var[:])[None], len(months), axis=0)

            if weights is None:
                out[name] = fine
            else:
                agg = ops.aggregate_area_weighted(fine.astype('float64'), weights)
                out[name] = agg.reshape(len(months), target.n_y, target.n_x)
    return out


def build_features(
    target_key: str = 'grace',
    start: str = cfg.ANALYSIS_START,
    end: str = cfg.ANALYSIS_END,
    predictors: Sequence[str] = tuple(cfg.PREDICTORS),
    min_basin_frac: float = 0.5,
    antecedent: Sequence[int] = ANTECEDENT_MONTHS,
    context: Sequence[float] = CONTEXT_DEGREES,
    covariates: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> FeatureSet:
    """
    Assemble the monthly design matrix on `target_key`'s grid.

    Only cells at least `min_basin_frac` inside the basin are emitted, so the
    model is not fitted on cells dominated by neighbouring basins.
    """
    months = pd.date_range(start, end, freq='MS')
    covariates = (cfg.resolve_active_covariates(verbose=verbose)
                  if covariates is None else list(covariates))
    grids = cfg.build_grids()
    grid = grids[target_key]
    aux = load_aux()

    basin_frac = aux[f'basin_frac_{target_key}']
    mascon_id = aux['mascon_id']
    if target_key != 'grace':
        raise NotImplementedError(
            'monthly training runs on the GRACE grid; use predict_stack() for 0.1 deg'
        )

    keep = (basin_frac >= min_basin_frac) & (mascon_id >= 0)
    cells = np.flatnonzero(keep.ravel())
    if verbose:
        print(f'  {len(cells)} of {grid.n_cells} cells retained '
              f'(basin fraction >= {min_basin_frac})')

    stack = predictor_stack(target_key, months, predictors)
    target = load_grace_monthly(months)

    columns: Dict[str, np.ndarray] = {}
    for name, field in stack.items():
        columns[name] = field.reshape(len(months), -1)[:, cells]
        for w in antecedent:
            columns[f'{name}_ante{w}m'] = _antecedent(field, w).reshape(
                len(months), -1)[:, cells]
        for r in context:
            columns[f'{name}_ctx{r:g}d'] = _context_mean(field, grid, r).reshape(
                len(months), -1)[:, cells]
        if name in API_FIELDS:
            for tau in API_TAU_MONTHS:
                columns[f'{name}_api{tau}m'] = _api(field, tau).reshape(
                    len(months), -1)[:, cells]
        with np.errstate(invalid='ignore'):
            clim_mean = np.nanmean(field, axis=0).ravel()[cells]
            clim_std = np.nanstd(field, axis=0).ravel()[cells]
        columns[f'{name}_clim_mean'] = np.tile(clim_mean, (len(months), 1))
        columns[f'{name}_clim_std'] = np.tile(clim_std, (len(months), 1))


    for name, field in covariate_stack('grace', months, covariates).items():
        columns[name] = field.reshape(len(months), -1)[:, cells]

    if WATER_BALANCE_API and {'ppt', 'et', 'runoff_surface'} <= set(stack):
        wb = stack['ppt'] - stack['et'] - stack['runoff_surface']
        for tau in API_TAU_MONTHS:
            columns[f'wb_api{tau}m'] = _api(wb, tau).reshape(
                len(months), -1)[:, cells]

    # Position and season.
    if USE_COORDINATE_FEATURES:
        lon = np.tile(grid.lon_centers(), (grid.n_y, 1)).ravel()[cells]
        lat = np.repeat(grid.lat_centers(), grid.n_x).ravel()[cells]
        columns['lon'] = np.tile(lon, (len(months), 1))
        columns['lat'] = np.tile(lat, (len(months), 1))
        columns['basin_frac'] = np.tile(basin_frac.ravel()[cells], (len(months), 1))

    doy = months.dayofyear.to_numpy()[:, None].astype('float64')
    for k in (1, 2):
        columns[f'sin{k}'] = np.tile(np.sin(2 * np.pi * k * doy / 365.25), (1, len(cells)))
        columns[f'cos{k}'] = np.tile(np.cos(2 * np.pi * k * doy / 365.25), (1, len(cells)))

    X = pd.DataFrame({k: v.ravel().astype('float32') for k, v in columns.items()})
    y = target.reshape(len(months), -1)[:, cells].ravel().astype('float64')
    mascon = np.tile(mascon_id.ravel()[cells], (len(months), 1)).ravel()
    cell = np.tile(cells, (len(months), 1)).ravel()
    tidx = np.repeat(np.arange(len(months)), len(cells))

    fs = FeatureSet(X=X, y=y, mascon=mascon, cell=cell, time_index=tidx,
                    dates=months, grid_key=target_key)
    if verbose:
        print('  ' + fs.describe())
    return fs


def predict_stack(
    months: pd.DatetimeIndex,
    feature_names: Sequence[str],
    reference_months: Optional[pd.DatetimeIndex] = None,
    predictors: Sequence[str] = tuple(cfg.PREDICTORS),
    antecedent: Sequence[int] = ANTECEDENT_MONTHS,
    context: Sequence[float] = CONTEXT_DEGREES,
    covariates: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    The same features evaluated on the 0.1 degree grid, for prediction.

    `reference_months` is the record the STATEFUL features are computed over;
    `months` selects which rows are returned. They differ when the caller chunks
    prediction to bound memory.

    This separation is essential. `gwsa` (runoff minus its temporal mean), the
    per-pixel climatologies, and the antecedent rolling means all depend on the
    window they are computed over. Building them from a 24-month chunk instead of
    the full 263-month record gives the model features it was never trained on:
    measured on the real cubes that shifted the 0.1 degree field by rms 48 mm
    (max 252 mm) against a field std of 120 mm, and pushed the 24-month
    antecedent feature from 2k to 115k NaNs per chunk, which XGBoost silently
    routes down its default branch.

    Returns (design matrix, in-basin cell indices). Column order is forced to
    match `feature_names` so the fitted model receives exactly what it expects.
    """
    covariates = (cfg.resolve_active_covariates()
                  if covariates is None else list(covariates))
    ref = reference_months if reference_months is not None else months
    if not pd.Index(months).isin(ref).all():
        raise ValueError('months must be a subset of reference_months')
    take = pd.Index(ref).get_indexer(pd.Index(months))

    grids = cfg.build_grids()
    grid = grids['era5']
    aux = load_aux()

    basin_frac = aux['basin_frac_era5']
    cells = np.flatnonzero((basin_frac > 0).ravel())

    stack = predictor_stack('era5', ref, predictors)

    columns: Dict[str, np.ndarray] = {}
    n_ref = len(ref)
    for name, field in stack.items():
        columns[name] = field.reshape(n_ref, -1)[take][:, cells]
        for w in antecedent:
            columns[f'{name}_ante{w}m'] = _antecedent(field, w).reshape(
                n_ref, -1)[take][:, cells]
        for r in context:
            columns[f'{name}_ctx{r:g}d'] = _context_mean(field, grid, r).reshape(
                n_ref, -1)[take][:, cells]
        if name in API_FIELDS:
            for tau in API_TAU_MONTHS:
                columns[f'{name}_api{tau}m'] = _api(field, tau).reshape(
                    n_ref, -1)[take][:, cells]
        with np.errstate(invalid='ignore'):
            columns[f'{name}_clim_mean'] = np.tile(
                np.nanmean(field, axis=0).ravel()[cells], (len(months), 1))
            columns[f'{name}_clim_std'] = np.tile(
                np.nanstd(field, axis=0).ravel()[cells], (len(months), 1))

    if WATER_BALANCE_API and {'ppt', 'et', 'runoff_surface'} <= set(stack):
        wb = stack['ppt'] - stack['et'] - stack['runoff_surface']
        for tau in API_TAU_MONTHS:
            columns[f'wb_api{tau}m'] = _api(wb, tau).reshape(
                n_ref, -1)[take][:, cells]

    for name, field in covariate_stack('era5', ref, covariates).items():
        columns[name] = field.reshape(n_ref, -1)[take][:, cells]

    if USE_COORDINATE_FEATURES:
        lon = np.tile(grid.lon_centers(), (grid.n_y, 1)).ravel()[cells]
        lat = np.repeat(grid.lat_centers(), grid.n_x).ravel()[cells]
        columns['lon'] = np.tile(lon, (len(months), 1))
        columns['lat'] = np.tile(lat, (len(months), 1))
        columns['basin_frac'] = np.tile(basin_frac.ravel()[cells], (len(months), 1))

    doy = months.dayofyear.to_numpy()[:, None].astype('float64')
    for k in (1, 2):
        columns[f'sin{k}'] = np.tile(np.sin(2 * np.pi * k * doy / 365.25), (1, len(cells)))
        columns[f'cos{k}'] = np.tile(np.cos(2 * np.pi * k * doy / 365.25), (1, len(cells)))

    X = pd.DataFrame({k: v.ravel().astype('float32') for k, v in columns.items()})
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise ValueError(f'prediction stack is missing training features: {missing}')
    return X[list(feature_names)], cells


if __name__ == '__main__':
    print('Building monthly training features on the GRACE grid\n')
    fs = build_features()
    print()
    print(f'feature columns ({fs.X.shape[1]}):')
    for i in range(0, len(fs.X.columns), 4):
        print('   ' + '  '.join(f'{c:24s}' for c in fs.X.columns[i:i + 4]))
    obs = fs.observed
    print(f'\ntarget: {obs.sum():,} observed of {len(fs.y):,} '
          f'({100 * obs.mean():.1f}%)')
    print(f'        range {np.nanmin(fs.y):.1f} to {np.nanmax(fs.y):.1f} mm, '
          f'std {np.nanstd(fs.y):.1f} mm')
    sizes = pd.Series(fs.mascon[obs]).value_counts()
    print(f'mascons: {len(sizes)}, samples per mascon '
          f'min {sizes.min():,} median {int(sizes.median()):,} max {sizes.max():,}')
