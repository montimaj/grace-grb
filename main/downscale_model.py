"""
Train the downscaling model, validate it spatially for real, and generate the
0.1 degree monthly TWSA product.

This module performs the SPATIAL DOWNSCALING: a predictor->TWSA relation is
learned across mascons and applied at 0.1 degrees. The mass-conservation step
that follows is a disaggregation -- it redistributes so every mascon mean is
preserved exactly. The DAILY step (downscale_daily.py) is disaggregation only.

Three things distinguish this from the basin-scale pipeline.

1. A GENUINE spatial holdout.
   The basin-scale `holdout_spatial.py` replicates one basin-mean series across
   fabricated locations with added noise; train and test share near-identical
   copies, so it leaks by construction and scores R^2 ~ 0.99. Here the folds are
   real, independent 3 degree mascons. Because JPL's CRI filtering correlates
   neighbouring mascons, each fold ALSO removes its spatial neighbours from the
   training set -- leave-one-mascon-out with a one-mascon buffer. With 19
   in-basin mascons a fold trains on roughly 12-13 of them.

2. Mass conservation against the observation.
   The model supplies fine-scale texture; GRACE supplies the mascon-scale
   truth. After predicting at 0.1 degrees we force every mascon's area-weighted
   aggregate back onto the observed value, using the exact area-overlap operator
   (36.7% of fine cells straddle a mascon boundary, so centre-in-cell assignment
   would leave the "conserved" product not actually reproducing GRACE).

   A naive correction is piecewise constant per mascon and prints 3 degree
   seams onto the output. Smoothing it breaks exact conservation, and one
   smoothed bump per mascon is still not enough -- with as many basis functions
   as constraints the solve is fully determined and reproduces whatever shape
   closes the system, seams included. We therefore lay Gaussian bumps on a
   lattice far finer than the mascons and take the MINIMUM-NORM solution of the
   resulting underdetermined system: the least-energy, hence smoothest,
   correction that still reproduces every mascon mean exactly. Measured on the
   time-mean field this cut the gradient excess along mascon edges from 12% to
   4% while leaving conservation exact to ~1e-11 mm.

   A consequence worth stating plainly: because the correction absorbs whatever
   the model cannot explain, the product reproduces GRACE at mascon scale BY
   CONSTRUCTION. Agreement with GRACE is therefore not evidence of skill. The
   evidence is the spatial holdout and the independent well comparison.

3. Baselines it has to beat.
   Bilinear interpolation of the GRACE field to 0.1 degrees, and the blocky
   piecewise-constant mascon field. A downscaling that cannot beat interpolation
   has added nothing but texture.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import gridded_config as cfg
import downscale_features as F
import downscale_grid_ops as ops

RESULTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'Results', 'downscaling'))

# Correction-basis width, in 0.1-degree cells. ~1.0 degree is a third of a
# mascon: wide enough to erase the 3-degree seams, narrow enough that a mascon's
# correction stays mostly inside it.
SMOOTH_SIGMA_CELLS = 10.0
# Width applied to the level+trend background, which starts life piecewise
# constant at 3 degrees and would otherwise step at every mascon edge.
BACKGROUND_SIGMA_CELLS = 12.0
# Minimum in-basin area for a mascon to be used as a conservation constraint.
# ~2000 km2 is about 2% of a 3-degree mascon; below that the constraint is
# imposed on so few cells that it distorts the surrounding field.
MIN_MASCON_AREA_KM2 = 2000.0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """
    MAE / RMSE / R^2 (squared Pearson) / NSE / PBIAS, plus the sample count.

    Delegates to `utils.calculate_metrics` so the gridded and basin-scale halves
    of the project cannot drift apart in how they score a model. Adds only `n`,
    and drops non-finite pairs first (GRACE has gap months, so the gridded
    arrays carry NaNs the basin-scale ones do not).
    """
    from utils import calculate_metrics

    ok = np.isfinite(y) & np.isfinite(p)
    y, p = np.asarray(y)[ok], np.asarray(p)[ok]
    if y.size < 2:
        return {k: float('nan') for k in ('MAE', 'RMSE', 'R2', 'NSE', 'PBIAS')} | {'n': int(y.size)}
    return {**calculate_metrics(y, p).to_dict(), 'n': int(y.size)}


# --------------------------------------------------------------------------
# Mascon adjacency
# --------------------------------------------------------------------------

def mascon_neighbours(mascon_id: np.ndarray) -> Dict[int, set]:
    """
    Adjacency between mascons, from 8-connectivity of their 0.5 degree cells.

    Used to buffer the spatial folds: GRACE's CRI filter mixes signal between
    adjacent mascons, so a neighbour left in the training set is a partial
    answer key for the held-out one.
    """
    nb: Dict[int, set] = {}
    n_y, n_x = mascon_id.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            a = mascon_id[max(0, dy):n_y + min(0, dy), max(0, dx):n_x + min(0, dx)]
            b = mascon_id[max(0, -dy):n_y + min(0, -dy), max(0, -dx):n_x + min(0, -dx)]
            ok = (a >= 0) & (b >= 0) & (a != b)
            for u, v in zip(a[ok].ravel(), b[ok].ravel()):
                nb.setdefault(int(u), set()).add(int(v))
                nb.setdefault(int(v), set()).add(int(u))
    return nb


# --------------------------------------------------------------------------
# Target decomposition
# --------------------------------------------------------------------------

def decompose_target(
    fs: F.FeatureSet, min_obs: int = 24
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[float, float]]]:
    """
    Split each mascon's observed TWSA into (level + linear trend) and anomaly.

    The leave-one-mascon-out diagnostic shows the predictor set transfers the
    SEASONAL cycle across space very well (R^2 = 0.85 at unseen mascons) and the
    trend hardly at all (R^2 = 0.11) -- yet the trend carries 53% of the
    variance in this basin. That is expected: the Ganga's secular decline is
    groundwater abstraction, and neither ERA5-Land nor GLDAS CLSM simulates
    pumping, so no predictor encodes it.

    Fitting the raw series therefore spends model capacity on a component that
    cannot be learned and contaminates the component that can. We instead fit
    only the anomaly, and take level and trend from GRACE itself.

    Returns (base, anomaly, coefficients) aligned with `fs` samples, where
    coefficients maps mascon -> (slope per month, intercept).
    """
    y, g, t = fs.y, fs.mascon, fs.time_index.astype('float64')
    base = np.full_like(y, np.nan, dtype='float64')
    coef: Dict[int, Tuple[float, float]] = {}

    for m in np.unique(g):
        seen = (g == m) & np.isfinite(y)
        if seen.sum() < min_obs:
            continue
        slope, intercept = np.polyfit(t[seen], y[seen], 1)
        coef[int(m)] = (float(slope), float(intercept))
        here = g == m
        base[here] = slope * t[here] + intercept

    return base, y - base, coef


def base_field_fine(
    coef: Dict[int, Tuple[float, float]],
    months: pd.DatetimeIndex,
    cells: np.ndarray,
) -> np.ndarray:
    """
    The level+trend background evaluated on the 0.1 degree grid.

    Built on the 0.5 degree GRACE grid and then bilinearly interpolated, so the
    background varies smoothly instead of stepping at mascon edges. The exact
    values are re-imposed afterwards by mass conservation; the smoothness is
    what stops 3 degree blocks printing through into the product.
    """
    grids = cfg.build_grids()
    grace, era5 = grids['grace'], grids['era5']
    aux = F.load_aux()
    mascon_id = aux['mascon_id'].ravel()

    t = np.arange(len(months), dtype='float64')
    coarse = np.full((len(months), grace.n_cells), np.nan)
    for m, (slope, intercept) in coef.items():
        sel = np.flatnonzero(mascon_id == m)
        if sel.size:
            coarse[:, sel] = (slope * t + intercept)[:, None]

    # Fill mascon-less cells by TRUE 2-D nearest neighbour.
    #
    # ffill/bfill along the flattened raster walks in row-major order, so a cell
    # at the end of one row inherits from the start of the next -- a donor that
    # can be most of the domain away in longitude. Use a distance transform so
    # the donor is the geometrically nearest valid cell.
    from scipy.ndimage import distance_transform_edt

    grid2 = coarse.reshape(len(months), grace.n_y, grace.n_x)
    valid = np.isfinite(grid2[0])
    if not valid.all():
        idx = distance_transform_edt(~valid, return_distances=False,
                                     return_indices=True)
        grid2 = grid2[:, idx[0], idx[1]]
    filled = grid2.reshape(len(months), -1)
    fine = ops.regrid_bilinear(
        filled.reshape(len(months), grace.n_y, grace.n_x), grace, era5)

    # Bilinear interpolation of a 3-degree piecewise-constant field still steps:
    # it just turns each mascon edge into a half-degree ramp. GRACE cannot say
    # WHERE inside a 3-degree cell a storage gradient sits, so a sharp seam at
    # the mascon boundary is an artefact of the parameterisation. Smoothing is
    # the honest representation of that ignorance. Mass conservation runs after
    # this and restores exact agreement regardless.
    fine = ops.gaussian_smooth(fine, BACKGROUND_SIGMA_CELLS)
    return fine.reshape(len(months), -1)[:, cells]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def make_model(name: str, seed: int = 42):
    if name == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.4,
            n_jobs=-1, random_state=seed)
    if name == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
            n_jobs=-1, random_state=seed, tree_method='hist')
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.6, n_jobs=-1,
            random_state=seed, verbose=-1)
    raise ValueError(f'unknown model {name}')


# --------------------------------------------------------------------------
# Spatial cross-validation
# --------------------------------------------------------------------------

@dataclass
class CVResult:
    per_fold: pd.DataFrame
    pooled: Dict[str, float]
    predictions: np.ndarray          # out-of-fold, aligned with observed samples
    y: np.ndarray
    mascon: np.ndarray


def leave_one_mascon_out(
    fs: F.FeatureSet,
    model_name: str = 'random_forest',
    buffer_neighbours: bool = True,
    target: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> CVResult:
    """
    Leave-one-mascon-out cross-validation with a one-mascon spatial buffer.

    This is the honest answer to "does the model transfer to a place it has
    never seen?", and it is the test the synthetic spatial holdout could not
    provide.

    `target` overrides fs.y, so the same routine evaluates either the raw TWSA
    or the decomposed anomaly that the product actually fits.
    """
    aux = F.load_aux()
    nb = mascon_neighbours(aux['mascon_id']) if buffer_neighbours else {}

    obs = fs.observed
    X = fs.X.to_numpy(dtype='float32')[obs]
    y = (fs.y if target is None else target)[obs]
    groups = fs.mascon[obs]

    folds = np.unique(groups)
    rows, oof = [], np.full(y.shape, np.nan)

    for m in folds:
        test = groups == m
        excluded = {int(m)} | (nb.get(int(m), set()) if buffer_neighbours else set())
        train = ~np.isin(groups, list(excluded))
        if train.sum() < 500 or test.sum() < 20:
            if verbose:
                print(f'    mascon {m:3d}: skipped '
                      f'(train {train.sum()}, test {test.sum()})')
            continue

        mdl = make_model(model_name)
        mdl.fit(X[train], y[train])
        pred = mdl.predict(X[test])
        oof[test] = pred

        row = metrics(y[test], pred)
        row.update(mascon=int(m), n_train=int(train.sum()),
                   n_excluded=len(excluded), n_train_mascons=int(
                       len(set(np.unique(groups).tolist()) - excluded)))
        rows.append(row)
        if verbose:
            print(f'    mascon {m:3d}: n={row["n"]:5d} trained on '
                  f'{row["n_train_mascons"]:2d} mascons  '
                  f'RMSE {row["RMSE"]:7.1f}  R2 {row["R2"]:5.2f}  '
                  f'NSE {row["NSE"]:6.2f}')

    per_fold = pd.DataFrame(rows)
    pooled = metrics(y, oof)

    # Separate LEVEL skill from PATTERN skill.
    #
    # A mascon's mean TWSA level is set largely by its cumulative groundwater
    # depletion history. ERA5-Land and GLDAS simulate no pumping, so nothing in
    # the predictor set carries that information and the model cannot be
    # expected to recover it for a location it has never seen -- which is what
    # the raw (absolute) metrics above measure.
    #
    # The product does not require it to. Mass conservation supplies every
    # mascon's level from the observation itself; the model only has to supply
    # the temporal and spatial PATTERN. The de-meaned metrics below measure that
    # component, by removing each mascon's own temporal mean from both series.
    # Both are reported: the absolute numbers are the honest headline, the
    # de-meaned ones say which part of the error the architecture actually
    # depends on.
    y_dm, p_dm = y.astype('float64').copy(), oof.astype('float64').copy()
    for m in np.unique(groups):
        sel = (groups == m) & np.isfinite(oof)
        if sel.sum() > 1:
            y_dm[sel] -= y[sel].mean()
            p_dm[sel] -= oof[sel].mean()
    pooled_dm = metrics(y_dm, p_dm)
    pooled = {**{f'{k}': v for k, v in pooled.items()},
              **{f'{k}_demeaned': v for k, v in pooled_dm.items()}}

    return CVResult(per_fold=per_fold, pooled=pooled, predictions=oof,
                    y=y, mascon=groups)


# --------------------------------------------------------------------------
# Prediction and mass conservation
# --------------------------------------------------------------------------

def conserve_mass(
    pred_fine: np.ndarray,
    obs_mascon: np.ndarray,
    weights,
    fine_shape: Tuple[int, int],
    cells: np.ndarray,
    sigma: float = SMOOTH_SIGMA_CELLS,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Force the fine field's mascon aggregates onto the observed GRACE values.

    Parameters
    ----------
    pred_fine  : (n_time, n_cells) prediction on in-basin fine cells
    obs_mascon : (n_time, n_mascons) observed GRACE, NaN where absent
    weights    : (n_mascons, n_fine_total) area-overlap operator
    cells      : indices of the in-basin fine cells within the full grid

    Returns the corrected field and a diagnostic of the residual conservation
    error after the final iteration.
    """
    n_t = pred_fine.shape[0]
    n_y, n_x = fine_shape
    w = weights[:, cells]
    row_sum = np.asarray(w.sum(axis=1)).ravel()

    # A mascon is only usable as a constraint if enough of it lies in the basin.
    #
    # `row_sum > 0` alone admits slivers: mascon 25 contributes a single 0.1 deg
    # cell of 0.41 km2, yet the constraint would force that one cell to equal the
    # mean of the whole 3-degree mascon. Measured, that one constraint moved
    # 4,326 of 9,538 cells by more than 10 mm (max 415 mm) -- and because the
    # residual is still driven to ~1e-12 mm either way, the conservation
    # diagnostic cannot see it. Cells in a rejected mascon still receive the
    # smooth correction bleeding in from neighbouring bumps.
    valid_mascon = row_sum >= MIN_MASCON_AREA_KM2
    n_dropped = int((row_sum > 0).sum() - valid_mascon.sum())

    def aggregate(field: np.ndarray) -> np.ndarray:
        return (w @ field.T).T / np.where(row_sum > 0, row_sum, np.nan)

    # Correct in an OVER-COMPLETE smooth basis, taking the minimum-norm
    # solution, so the result is smooth AND exactly conservative.
    #
    # Spreading a per-mascon residual through `alpha` directly gives a
    # piecewise-constant correction, which prints 3-degree seams onto the
    # product -- an artefact of the mascon parameterisation, not hydrology.
    # Using one smoothed bump PER MASCON is not enough either: with as many
    # basis functions as constraints the solve is fully determined, so it has no
    # freedom left and reproduces whatever shape closes the system, seams
    # included. (Measured that way the correction still carried 12% stronger
    # gradients along mascon edges than elsewhere.)
    #
    # Instead lay Gaussian bumps on a lattice much finer than the mascons, so
    # there are far MORE unknowns than constraints. The system
    #
    #     A x = r * row_sum,    A = W B     (n_mascons rows, n_bumps columns)
    #
    # is then underdetermined, and its minimum-norm solution
    # x = A^T (A A^T)^-1 rhs is the least-energy -- hence smoothest -- correction
    # that still reproduces every mascon mean exactly. Closure stays exact
    # because the constraint is imposed, not approximated. The row_sum factor is
    # required because the constraint is on the area-weighted MEAN, not the
    # total.
    step = max(int(round(sigma / 2.0)), 1)
    rows_c = np.arange(0, n_y, step)
    cols_c = np.arange(0, n_x, step)
    in_basin = np.zeros(n_y * n_x, dtype=bool)
    in_basin[cells] = True
    in_basin = in_basin.reshape(n_y, n_x)

    yy = np.arange(n_y)[:, None]
    xx = np.arange(n_x)[None, :]
    bumps = []
    for r0 in rows_c:
        for c0 in cols_c:
            # Skip centres far from the basin; their bump would be all zeros.
            lo_r, hi_r = max(0, r0 - 2 * step), min(n_y, r0 + 2 * step + 1)
            lo_c, hi_c = max(0, c0 - 2 * step), min(n_x, c0 + 2 * step + 1)
            if not in_basin[lo_r:hi_r, lo_c:hi_c].any():
                continue
            g = np.exp(-((yy - r0) ** 2 + (xx - c0) ** 2) / (2.0 * sigma ** 2))
            bumps.append(g.ravel()[cells])
    basis = np.column_stack(bumps) if bumps else np.zeros((len(cells), 1))

    A_full = w @ basis                          # (n_mascons, n_bumps)
    resid = obs_mascon - aggregate(pred_fine)
    corr = np.zeros((n_t, len(cells)))
    for t in range(n_t):
        ok = np.isfinite(resid[t]) & valid_mascon
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        A = A_full[idx]                         # (k, n_bumps)
        rhs = resid[t, idx] * row_sum[idx]
        gram = A @ A.T                          # (k, k), k <= ~35
        try:
            lam = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        corr[t] = basis @ (A.T @ lam)           # minimum-norm x = A^T lam
    out = pred_fine + corr

    agg = aggregate(out)
    err = np.abs(agg - obs_mascon)
    ok = np.isfinite(err)
    # Per-month record of whether GRACE constrained anything at all. 45 of 263
    # months have no observation, so the product there is unconstrained model
    # output; without this flag the netCDF advertises every month as conserved.
    observed_month = np.isfinite(obs_mascon).any(axis=1) & valid_mascon.any()
    diag = {
        'n_months_unconstrained': int((~observed_month).sum()),
        'grace_observed': observed_month,
        'n_mascons_dropped_small_footprint': n_dropped,
        'max_abs_error_mm': float(np.nanmax(err[ok])) if ok.any() else float('nan'),
        'mean_abs_error_mm': float(np.nanmean(err[ok])) if ok.any() else float('nan'),
        'n_mascon_months': int(ok.sum()),
    }
    return out, diag


def predict_fine(
    model,
    months: pd.DatetimeIndex,
    feature_names: Sequence[str],
    chunk_months: int = 24,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the fitted model on the 0.1 degree grid, in chunks so the design
    matrix (2.4 M rows x ~59 features) never lands in memory whole.

    `reference_months=months` is REQUIRED, not incidental: the chunk only selects
    which rows come back, while gwsa, the climatologies and the antecedent means
    are still computed over the whole record, exactly as in training. Without it
    the chunk size silently becomes a scientific parameter.
    """
    preds: List[np.ndarray] = []
    cells: Optional[np.ndarray] = None
    for i in range(0, len(months), chunk_months):
        block = months[i:i + chunk_months]
        X, cells = F.predict_stack(block, feature_names, reference_months=months)
        p = model.predict(X.to_numpy(dtype='float32'))
        preds.append(p.reshape(len(block), len(cells)))
        if verbose:
            print(f'    {block[0]:%Y-%m}..{block[-1]:%Y-%m}  '
                  f'{len(block) * len(cells):,} cells', flush=True)
    return np.concatenate(preds, axis=0), cells


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def interpolation_baselines(
    months: pd.DatetimeIndex, cells: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Bilinear and piecewise-constant interpolation of GRACE to 0.1 degrees.

    Any downscaling must beat these; they carry no information the mascon field
    does not already have.
    """
    grids = cfg.build_grids()
    grace, era5 = grids['grace'], grids['era5']
    tws = F.load_grace_monthly(months)
    aux = F.load_aux()
    parent = aux['parent_era5_to_grace']

    bilinear = ops.regrid_bilinear(np.nan_to_num(tws, nan=0.0), grace, era5)
    bilinear = np.where(
        ops.regrid_bilinear(np.isfinite(tws).astype('float64'), grace, era5) > 0.99,
        bilinear, np.nan)

    blocky = tws.reshape(len(months), -1)[:, parent.ravel()].reshape(
        len(months), era5.n_y, era5.n_x)

    return {
        'bilinear': bilinear.reshape(len(months), -1)[:, cells],
        'blocky': blocky.reshape(len(months), -1)[:, cells],
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_product(
    field: np.ndarray,
    months: pd.DatetimeIndex,
    cells: np.ndarray,
    path: str,
    diag: Dict[str, float],
    model_name: str,
) -> str:
    import netCDF4

    grids = cfg.build_grids()
    grid = grids['era5']
    os.makedirs(os.path.dirname(path), exist_ok=True)

    full = np.full((len(months), grid.n_y * grid.n_x), np.nan, dtype='float32')
    full[:, cells] = field
    full = full.reshape(len(months), grid.n_y, grid.n_x)

    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', len(months))
        ds.createDimension('lat', grid.n_y)
        ds.createDimension('lon', grid.n_x)

        tv = ds.createVariable('time', 'f8', ('time',))
        tv.units, tv.calendar = 'days since 1970-01-01', 'standard'
        tv[:] = (months - pd.Timestamp('1970-01-01')).days.to_numpy()
        for name, vals, units in [('lat', grid.lat_centers(), 'degrees_north'),
                                  ('lon', grid.lon_centers(), 'degrees_east')]:
            cv = ds.createVariable(name, 'f8', (name,))
            cv.units = units
            cv[:] = vals

        v = ds.createVariable('twsa', 'f4', ('time', 'lat', 'lon'),
                              zlib=True, complevel=4,
                              fill_value=np.float32(np.nan))
        v.units = 'mm'
        v.long_name = 'Spatially downscaled terrestrial water storage anomaly'
        v.baseline = '2004.0-2010.0 (JPL RL06 mascon convention)'
        v.model = model_name
        v.mass_conservation = (
            f'area-weighted mascon aggregates forced onto observed GRACE; '
            f'residual max {diag["max_abs_error_mm"]:.3g} mm, '
            f'mean {diag["mean_abs_error_mm"]:.3g} mm. NOT every month is '
            f'constrained - see the grace_observed flag.')
        v[:] = full

        obs_flag = np.asarray(diag.get('grace_observed', np.ones(len(months), bool)))
        gv = ds.createVariable('grace_observed', 'i1', ('time',))
        gv.long_name = 'GRACE observed this month and constrained the field'
        gv.description = ('0 means the month has NO GRACE observation: the field is '
                          'unconstrained model output, not a mass-conserved estimate. '
                          f'{int((~obs_flag).sum())} of {len(months)} months.')
        gv[:] = obs_flag.astype('i1')

        ds.title = 'Downscaled GRACE TWSA over the Ganga basin, 0.1 degrees, monthly'
        ds.resolution = '0.1 degree (ERA5-Land native grid)'
        ds.caveat = (
            'GRACE observes ~20 independent 3-degree mascons over this basin. '
            'Fine-scale structure is inferred from ERA5-Land/GLDAS predictors, '
            'not observed. Agreement with GRACE at mascon scale is imposed by '
            'construction and is not evidence of skill; see the leave-one-mascon-out '
            'holdout and the independent CGWB well comparison.')
        ds.created_by = 'main/downscale_model.py'
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='random_forest',
                    choices=['random_forest', 'xgboost', 'lightgbm'])
    ap.add_argument('--no-buffer', action='store_true',
                    help='disable the neighbour buffer (to quantify its effect)')
    ap.add_argument('--skip-product', action='store_true',
                    help='run cross-validation only')
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('1. Building monthly features on the GRACE grid')
    fs = F.build_features()

    print('\n2. Decomposing the target (level + trend from GRACE, anomaly to the model)')
    base, anom, coef = decompose_target(fs)
    obs = fs.observed
    print(f'  {len(coef)} mascons fitted | anomaly std {np.nanstd(anom[obs]):.1f} mm '
          f'of total {np.nanstd(fs.y[obs]):.1f} mm '
          f'({100 * np.nanvar(anom[obs]) / np.nanvar(fs.y[obs]):.0f}% of variance)')

    print(f'\n3. Leave-one-mascon-out spatial CV on the anomaly ({args.model}, '
          f'buffer={"off" if args.no_buffer else "on"})')
    cv = leave_one_mascon_out(fs, args.model, buffer_neighbours=not args.no_buffer,
                              target=anom)
    print('\n  pooled out-of-fold: ' +
          '  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}'
                    for k, v in cv.pooled.items() if not k.endswith('_demeaned')))
    cv.per_fold.to_csv(os.path.join(RESULTS_DIR, f'lomo_cv_{args.model}.csv'),
                       index=False)

    print('\n3b. Temporal holdout (forward block: train early, predict late)')
    # The spatial holdout answers "does this work somewhere it has not seen?".
    # It says nothing about "does this work at a TIME it has not seen?", which is
    # what the 2024-10 onward extrapolation depends on. The machinery lives in
    # downscale_uncertainty.gap_recovery_error, which hides contiguous blocks of
    # observed months and refits; mode='forward' hides the LAST ones, making it a
    # chronological holdout. It is reported here as a validation result and not
    # only consumed as an uncertainty term, because that is where a reader looks
    # for it.
    #
    # There is deliberately NO random holdout for this model. Monthly TWSA is
    # smooth and strongly autocorrelated, so a random month-level split leaves
    # each test month's neighbours in training. Measured on the basin-scale
    # series that leaked at R2 = 0.96 even after grouping by interpolation
    # segment, because the residual is intrinsic to the signal rather than to the
    # split. A random split here would manufacture an optimistic number with no
    # defensible reading.
    try:
        from downscale_uncertainty import gap_recovery_error
        fwd = gap_recovery_error(fs, anom, model_name=args.model, mode='forward',
                                 gap_length=11, verbose=False)
        if not fwd.empty:
            fwd.insert(0, 'regime', 'forward')
            fwd.to_csv(os.path.join(RESULTS_DIR,
                                    f'temporal_holdout_{args.model}.csv'), index=False)
            print('  RMSE by months beyond the training record: '
                  + ', '.join(f'+{int(r.depth)}mo {r.rmse:.0f}'
                              for _, r in fwd.iterrows()))
        else:
            print('  skipped: no contiguous observed block long enough')
    except Exception as err:  # noqa: BLE001 - diagnostic, never fatal
        print(f'  temporal holdout failed: {err}')

    if args.skip_product:
        return 0

    print('\n4. Fitting on all mascons and predicting the anomaly at 0.1 degrees')
    model = make_model(args.model)
    model.fit(fs.X.to_numpy(dtype='float32')[obs], anom[obs])

    months = fs.dates
    anom_fine, cells = predict_fine(model, months, list(fs.X.columns))
    base_fine = base_field_fine(coef, months, cells)
    raw = base_fine + anom_fine
    print(f'  background (level+trend) std {np.nanstd(base_fine):.1f} mm, '
          f'predicted anomaly std {np.nanstd(anom_fine):.1f} mm')

    print('\n5. Enforcing mass conservation against observed GRACE')
    aux = F.load_aux()
    grids = cfg.build_grids()
    w_mascon, mascon_ids = ops.mascon_weights(
        grids['era5'], grids['grace'], aux['mascon_id'],
        basin_frac_fine=aux['basin_frac_era5'])

    tws = F.load_grace_monthly(months)
    w_coarse, _ = ops.mascon_weights(grids['grace'], grids['grace'], aux['mascon_id'])
    obs_mascon = ops.aggregate_area_weighted(tws, w_coarse)

    conserved, diag = conserve_mass(
        raw, obs_mascon, w_mascon, (grids['era5'].n_y, grids['era5'].n_x), cells)
    print(f'  conservation residual: max {diag["max_abs_error_mm"]:.3g} mm, '
          f'mean {diag["mean_abs_error_mm"]:.3g} mm '
          f'over {diag["n_mascon_months"]:,} mascon-months')

    out = write_product(
        conserved, months, cells,
        os.path.join(RESULTS_DIR, f'twsa_0p1deg_monthly_{args.model}.nc'),
        diag, args.model)
    print(f'  written: {out}')

    print('\n6. Interpolation baselines (must be beaten)')
    base = interpolation_baselines(months, cells)
    for name, b in base.items():
        d = conserved - b
        ok = np.isfinite(d)
        print(f'  vs {name:9s}: mean |difference| {np.nanmean(np.abs(d[ok])):7.2f} mm, '
              f'spatial std added {np.nanmean(np.nanstd(conserved, axis=1) - np.nanstd(b, axis=1)):+7.2f} mm')

    with open(os.path.join(RESULTS_DIR, f'summary_{args.model}.json'), 'w') as fh:
        json.dump({'pooled_cv': cv.pooled, 'conservation': diag,
                   'model': args.model, 'n_features': fs.X.shape[1],
                   'n_mascons': int(len(np.unique(fs.mascon)))}, fh, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
