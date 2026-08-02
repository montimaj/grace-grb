"""
Per-pixel uncertainty for the 0.1 degree downscaled TWSA product.

This answers the reviewer comments that the basin-scale response letter deferred
to future work ("per-pixel skill maps are not computable from a single
basin-averaged series ... identified as the future gridded extension").

Three uncertainty terms, reported SEPARATELY as well as combined, because they
mean different things and only two of them are measurable against an
observation.

  sigma_grace     Mascon measurement uncertainty, from the GRACE product's own
                  `uncertainty` band. Common to every pixel inside a mascon.
                  This is a real observational error bar.

  sigma_transfer  Spatial-transfer error, from the leave-one-mascon-out
                  cross-validation: for each mascon and each calendar month,
                  the RMSE the model achieves at a location it never saw during
                  training. A measured quantity, resolved by mascon and season.

  sigma_within    Ensemble spread across model families, AFTER mass
                  conservation. Because conservation forces every member to
                  reproduce GRACE at mascon scale, whatever spread remains is
                  purely disagreement about how storage is distributed WITHIN a
                  mascon -- the fine-scale structure that GRACE cannot see.
                  Four members: two boosting (xgboost, lightgbm) and two bagging
                  (random_forest, xgboost_rf). See DEFAULT_MODELS.

The honest caveat, stated in the file's own metadata
----------------------------------------------------
`sigma_within` is a LOWER BOUND, not a calibrated error. GRACE observes ~20
numbers per month over this basin; there is no observation of within-mascon
structure to validate against, so nothing can calibrate it. Models that share
predictors and training data will agree more than they are jointly correct.
Treat it as "how much do defensible methods disagree", not "how wrong is this".

The independent CGWB well comparison (see `wells_ingest.py`) is the only test
that can say whether the fine-scale structure is real.

Usage
-----
    python downscale_uncertainty.py                    # full ensemble
    python downscale_uncertainty.py --models xgboost lightgbm
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import gridded_config as cfg
import downscale_features as F
import downscale_grid_ops as ops
import downscale_model as M

# Four members, spanning both tree families rather than three of one.
#
# `sigma_within` is the spread ACROSS these after mass conservation, so which
# members are present is not a wiring detail -- it sets a band shipped in the
# product. Two boosting implementations (xgboost, lightgbm) and two bagging ones
# (random_forest, xgboost_rf) is a more honest sample of "defensible methods"
# than 2:1, and a standard deviation over four values is less jumpy than over
# three.
#
# Be careful how much is claimed for it: xgboost_rf bags much as random_forest
# does, differing in split finding and L2 regularisation rather than in
# principle. Adding it widens the sample without making the members independent,
# and the LOWER BOUND caveat below is unaffected -- four correlated members still
# agree more than they are jointly correct.
#
# THE MLP IS DELIBERATELY NOT A MEMBER, though `--models` will accept it.
#
# It exists for the model COMPARISON (downscale_model.py, downscale_holdouts.py),
# where a non-tree function class is exactly the point. Putting it in the
# ensemble is a different decision, because `sigma_within` is a band shipped in
# the product: a member that is structurally unlike the others would widen the
# spread, and a reader could not tell whether that reflected genuine uncertainty
# about within-mascon structure or simply an architecture that suits this design
# less well.
#
# Revisit if the comparison shows the MLP performing comparably on held-out
# skill. Until then, adding it would inflate a number we already label a lower
# bound, in a direction that flatters the uncertainty estimate rather than the
# product.
DEFAULT_MODELS: Tuple[str, ...] = ('random_forest', 'xgboost', 'lightgbm',
                                   'xgboost_rf')


# --------------------------------------------------------------------------
# Transfer error, resolved by mascon and season
# --------------------------------------------------------------------------

def transfer_error(
    fs: F.FeatureSet,
    anom: np.ndarray,
    model_names: Sequence[str],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Leave-one-mascon-out RMSE per (mascon, calendar month), per model.

    Seasonal resolution matters: the LOMO diagnostic shows the predictors
    reproduce the monsoon cycle far better than the dry-season recession, so a
    single scalar error bar would be wrong in both seasons.
    """
    rows: List[Dict[str, float]] = []
    oof_by_model: Dict[str, np.ndarray] = {}

    obs = fs.observed
    month_of = fs.dates.month.to_numpy()[fs.time_index][obs]
    groups = fs.mascon[obs]

    for name in model_names:
        if verbose:
            print(f'  LOMO CV: {name}', flush=True)
        cv = M.leave_one_mascon_out(fs, name, target=anom, verbose=False)
        oof_by_model[name] = cv.predictions
        err = cv.predictions - cv.y

        # Remove each mascon-month's mean error before forming the RMSE.
        #
        # sigma_transfer is published as the uncertainty of the POST-conservation
        # product, but the LOMO error is measured before conservation. Mass
        # conservation removes the mascon-month mean exactly, so leaving it in
        # double-counts an error the product no longer carries.
        _t = fs.time_index[obs]
        _key = pd.DataFrame({'g': groups, 't': _t, 'e': err})
        err = (err - _key.groupby(['g', 't']).e.transform('mean').to_numpy())
        for m in np.unique(groups):
            for cm in range(1, 13):
                sel = (groups == m) & (month_of == cm) & np.isfinite(err)
                if sel.sum() >= 5:
                    rows.append({
                        'model': name, 'mascon': int(m), 'calendar_month': cm,
                        'rmse': float(np.sqrt(np.mean(err[sel] ** 2))),
                        'n': int(sel.sum()),
                    })
    return pd.DataFrame(rows), oof_by_model


def transfer_error_field(
    table: pd.DataFrame,
    months: pd.DatetimeIndex,
    cells: np.ndarray,
) -> np.ndarray:
    """
    Map the (mascon, calendar month) RMSE table onto the 0.1 degree grid.

    Averaged across ensemble members, then broadcast to every pixel of the
    parent mascon. Piecewise constant by construction -- this term genuinely has
    no sub-mascon resolution, and smoothing it would imply detail it lacks.
    """
    aux = F.load_aux()
    mascon_of_fine = aux['mascon_id'].ravel()[aux['parent_era5_to_grace'].ravel()][cells]

    pivot = (table.groupby(['mascon', 'calendar_month']).rmse.mean()
             .unstack(fill_value=np.nan))
    fallback = float(table.rmse.mean())

    out = np.full((len(months), len(cells)), fallback, dtype='float32')
    for i, d in enumerate(months):
        cm = d.month
        if cm not in pivot.columns:
            continue
        lookup = pivot[cm]
        vals = np.array([lookup.get(int(m), np.nan) for m in mascon_of_fine])
        out[i] = np.where(np.isfinite(vals), vals, fallback)
    return out


# --------------------------------------------------------------------------
# GRACE measurement uncertainty
# --------------------------------------------------------------------------

def grace_uncertainty_field(
    months: pd.DatetimeIndex, cells: np.ndarray
) -> np.ndarray:
    """
    The GRACE product's own `uncertainty` band, on the 0.1 degree grid.

    Piecewise constant within a mascon: it is a mascon-scale error bar, and
    every pixel inherits at least that much uncertainty.
    """
    import netCDF4

    aux = F.load_aux()
    parent = aux['parent_era5_to_grace'].ravel()[cells]

    dates = F.cube_dates('grace')
    with netCDF4.Dataset(cfg.cube_path('grace')) as ds:
        unc = np.asarray(ds['tws_uncertainty'][:])
    pos = {d.to_period('M'): i for i, d in enumerate(dates)}

    flat = unc.reshape(unc.shape[0], -1)
    out = np.full((len(months), len(cells)), np.nan, dtype='float32')
    for i, d in enumerate(months):
        j = pos.get(d.to_period('M'))
        if j is not None:
            out[i] = flat[j, parent]

    # Carry the median forward through GRACE gaps rather than leaving holes,
    # flagged in the output metadata.
    med = np.nanmedian(out)
    return np.where(np.isfinite(out), out, med).astype('float32')


# --------------------------------------------------------------------------
# Gap-fill skill
# --------------------------------------------------------------------------

def gap_recovery_error(
    fs: F.FeatureSet,
    anom: np.ndarray,
    model_name: str = 'xgboost',
    gap_length: int = 11,
    n_gaps: int = 6,
    seed: int = 20,
    mode: str = 'interior',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    How wrong is a gap-filled month, as a function of depth into the gap?

    The product is a SINGLE gap-filled field: ~44 of 263 months have no GRACE
    observation, so mass conservation cannot act and they are model output
    anchored only by the extrapolated level+trend. That is a legitimate and
    useful thing to publish, but "the model can reconstruct the gap" is an
    assertion until it is measured -- and leave-one-mascon-out cannot measure it,
    because it tests transfer across SPACE, not across a temporal blackout.

    So we manufacture blackouts. Contiguous blocks of `gap_length` OBSERVED
    months are hidden, the model is refitted without them, and the withheld
    observations are compared against the reconstruction at mascon scale. The
    real GRACE/GRACE-FO gap is ~11 months, so that is the default block.

    Error grows with depth into a gap: month 1 is nearly interpolation between
    two anchors, month 6 is genuine extrapolation. The returned RMSE-by-depth is
    what `sigma_gap_field` turns into a per-month uncertainty, so the product
    reports a wider error bar exactly where it is least constrained.

    Three regimes, because they are different claims and must not share an error
    bar. GRACE runs 2002-04 to 2024-09 while the product spans 2000-2025, so:

      mode='interior'  a month bracketed by observations on BOTH sides (the
                       GRACE/GRACE-FO gap). The trend is interpolated.
      mode='forward'   a month AFTER the last observation (2024-10 onward).
                       The trend is extrapolated with no closing anchor.
      mode='backward'  a month BEFORE the first observation (2000-01 to 2002-03).
                       Extrapolated backwards.

    The interior test hides blocks from the middle; the forward and backward
    tests hide the last / first `gap_length` observed months and refit, so the
    error they report is genuine out-of-record extrapolation error. Reusing the
    interior curve for the extrapolated periods would understate them badly --
    interior months are half-anchored, out-of-record months are not anchored at
    all.

    Returns a frame with columns [depth, rmse, rmse_raw, n].
    """
    if mode not in ('interior', 'forward', 'backward'):
        raise ValueError(f"mode must be interior/forward/backward, got {mode!r}")
    rng = np.random.default_rng(seed)
    obs = fs.observed
    t_all = fs.time_index
    observed_months = np.unique(t_all[obs])

    # Candidate gap starts: blocks fully inside the observed record.
    obs_set = set(observed_months.tolist())
    if mode == 'forward':
        # Hide the LAST gap_length observed months; depth counts forward from
        # the new end of the training record.
        starts = [int(observed_months[-gap_length])] if len(observed_months) > gap_length else []
    elif mode == 'backward':
        starts = [int(observed_months[0])] if len(observed_months) > gap_length else []
    else:
        starts = [m for m in observed_months
                  if all((m + k) in obs_set for k in range(gap_length))]
    if not starts:
        if verbose:
            print('  no contiguous observed block long enough for a gap test')
        return pd.DataFrame(columns=['depth', 'rmse', 'n'])
    # Non-overlapping blocks. Sampling starts independently lets two gaps share
    # months (a 6-month block at t=46 and another at t=49), which correlates the
    # error estimates and makes n_gaps overstate how much evidence there is.
    if mode in ('forward', 'backward'):
        chosen = [int(x) for x in starts]
        order = []
    else:
        order = rng.permutation(len(starts))
    if mode == 'interior':
        chosen, claimed = [], set()
    for k in order:
        st = int(starts[k])
        span = set(range(st, st + gap_length))
        if span & claimed:
            continue
        chosen.append(st)
        claimed |= span
        if len(chosen) >= n_gaps:
            break
    if verbose and len(chosen) < n_gaps:
        print(f'    only {len(chosen)} non-overlapping gaps available '
              f'(asked for {n_gaps})')

    rows: List[Dict[str, float]] = []
    X_all = fs.X.to_numpy(dtype='float32')
    for gi, start in enumerate(chosen):
        hidden = set(range(int(start), int(start) + gap_length))
        train = obs & ~np.isin(t_all, list(hidden))
        test = obs & np.isin(t_all, list(hidden))
        if train.sum() < 1000 or test.sum() < 10:
            continue
        # Refit the level+trend background on the surviving months only. The
        # `anom` passed in was decomposed over the whole record, so the hidden
        # months helped fit the line that is then subtracted from them -- which
        # is precisely the information a real blackout does not have. On the
        # forward split of downscale_holdouts.py the same leak was worth 13.9 mm
        # RMSE, and here it would propagate into sigma_gap, understating the
        # error bar on exactly the months that carry no observation.
        _, anom_g, _ = M.decompose_target(fs, fit_mask=train)
        ok = np.isfinite(anom_g)
        tr, te = train & ok, test & ok
        if tr.sum() < 1000 or te.sum() < 10:
            continue
        mdl = M.make_model(model_name)
        mdl.fit(X_all[tr], anom_g[tr])
        pred = mdl.predict(X_all[te])
        err = pred - anom_g[te]
        test = te
        # Depth = distance from the training edge. Backward gaps count from the
        # far end, so month 1 is always the one nearest surviving training data.
        if mode == 'backward':
            depth = int(start) + gap_length - t_all[test]
        else:
            depth = t_all[test] - int(start) + 1
        for d in range(1, gap_length + 1):
            sel = depth == d
            if sel.sum() >= 5:
                rows.append({'gap': gi, 'depth': int(d),
                             'sq': float(np.mean(err[sel] ** 2)),
                             'n': int(sel.sum())})
        if verbose:
            print(f'    synthetic gap {gi + 1}/{len(chosen)} at t={int(start)}: '
                  f'RMSE {np.sqrt(np.mean(err ** 2)):.1f} mm', flush=True)

    if not rows:
        return pd.DataFrame(columns=['depth', 'rmse', 'n'])
    df = pd.DataFrame(rows)
    out = (df.groupby('depth')
             .apply(lambda g: pd.Series({'rmse': float(np.sqrt(np.average(g.sq, weights=g.n))),
                                         'n': int(g.n.sum())}), include_groups=False)
             .reset_index())
    # Enforce a non-decreasing curve. A reconstruction further from its last
    # anchor cannot be BETTER constrained than a nearer one, so any dip is
    # sampling noise; letting it through would report a narrower error bar
    # deeper into a blackout. The running maximum is the conservative reading.
    out = out.sort_values('depth').reset_index(drop=True)
    out['rmse_raw'] = out['rmse']
    out['rmse'] = np.maximum.accumulate(out['rmse'].to_numpy())
    if verbose:
        print('  gap-recovery RMSE by depth (running max): '
              + ', '.join(f'm{int(r.depth)} {r.rmse:.0f}' for _, r in out.iterrows()))
        if not np.allclose(out.rmse, out.rmse_raw):
            print('    (raw curve was non-monotonic; dips treated as sampling noise)')
    return out


def sigma_gap_field(
    tables: Dict[str, pd.DataFrame],
    months: pd.DatetimeIndex,
    observed: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """
    Extra uncertainty carried by months GRACE did not observe.

    Zero where GRACE constrained the field. Elsewhere the value comes from the
    curve matching that month's REGIME, not from a single pooled number:

      before the first observation  -> the 'backward' curve
      after the last observation    -> the 'forward' curve
      between two observations      -> the 'interior' curve

    That distinction is the point. An interior gap month is bracketed by
    observations and its trend is interpolated; an out-of-record month has no
    closing anchor and its trend is extrapolated indefinitely. Giving both the
    same error bar would understate the second badly.

    Depth counts from the nearest observation, so the bar widens the further the
    reconstruction runs from its anchor. Beyond the deepest tested horizon the
    deepest measured value is held -- conservative only if the error saturates,
    which is recorded in the output metadata.
    """
    out = np.zeros((len(months), n_cells), dtype='float32')
    obs_idx = np.flatnonzero(observed)
    if obs_idx.size == 0:
        return out
    first_obs, last_obs = int(obs_idx[0]), int(obs_idx[-1])

    curves = {}
    for key, tbl in tables.items():
        if tbl is not None and not tbl.empty:
            curves[key] = (dict(zip(tbl.depth.astype(int), tbl.rmse.astype(float))),
                           int(tbl.depth.max()))
    if not curves:
        return out

    for i in range(len(months)):
        if observed[i]:
            continue
        if i < first_obs:
            regime, depth = 'backward', first_obs - i
        elif i > last_obs:
            regime, depth = 'forward', i - last_obs
        else:
            prev = obs_idx[obs_idx < i].max()
            regime, depth = 'interior', i - int(prev)
        by_depth, deepest = curves.get(regime) or curves.get('interior') or next(iter(curves.values()))
        out[i] = by_depth.get(min(depth, deepest), by_depth[deepest])
    return out


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------

def run_ensemble(
    model_names: Sequence[str] = DEFAULT_MODELS,
    seeds: Sequence[int] = (20,),
    verbose: bool = True,
) -> Dict[str, object]:
    """Fit every member, predict at 0.1 degrees, conserve mass, collect spread."""
    print('1. Features and target decomposition')
    fs = F.build_features(verbose=verbose)
    base, anom, coef = M.decompose_target(fs)
    obs = fs.observed
    months = fs.dates

    print(f'\n2. Spatial-transfer error by leave-one-mascon-out '
          f'({len(model_names)} members)')
    table, _ = transfer_error(fs, anom, model_names, verbose=verbose)
    print(f'  transfer RMSE: median {table.rmse.median():.1f} mm, '
          f'IQR {table.rmse.quantile(0.25):.1f}-{table.rmse.quantile(0.75):.1f} mm')

    print('\n3. Gap-fill skill (synthetic 11-month blackouts)')
    gap_tables = {}
    for mode in ('interior', 'forward', 'backward'):
        print(f'  {mode}:')
        gap_tables[mode] = gap_recovery_error(fs, anom, mode=mode, verbose=verbose)
    gap_tbl = pd.concat(
        [t.assign(regime=k) for k, t in gap_tables.items() if not t.empty],
        ignore_index=True) if any(not t.empty for t in gap_tables.values()) else pd.DataFrame()

    print('\n4. Fitting members and predicting at 0.1 degrees')
    aux = F.load_aux()
    grids = cfg.build_grids()
    w_mascon, _ = ops.mascon_weights(
        grids['era5'], grids['grace'], aux['mascon_id'],
        basin_frac_fine=aux['basin_frac_era5'])
    w_coarse, _ = ops.mascon_weights(grids['grace'], grids['grace'], aux['mascon_id'])
    obs_mascon = ops.aggregate_area_weighted(F.load_grace_monthly(months), w_coarse)

    X = fs.X.to_numpy(dtype='float32')[obs]
    # (n_families, n_seeds) grid of fields, so the two sources of spread can be
    # separated rather than mixed. See the sigma_within / sigma_seed note below.
    grid_fields: List[List[np.ndarray]] = []
    cells: Optional[np.ndarray] = None
    diag: Dict[str, float] = {}

    for name in model_names:
        per_seed: List[np.ndarray] = []
        for sd in seeds:
            label = f'  {name}' + (f' (seed {sd})' if len(seeds) > 1 else '')
            print(label, flush=True)
            mdl = M.make_model(name, seed=sd)
            mdl.fit(X, anom[obs])
            anom_fine, cells = M.predict_fine(mdl, months, list(fs.X.columns),
                                              verbose=False)
            raw = M.base_field_fine(coef, months, cells) + anom_fine
            conserved, diag = M.conserve_mass(
                raw, obs_mascon, w_mascon,
                (grids['era5'].n_y, grids['era5'].n_x), cells)
            per_seed.append(conserved)
            print(f'    conservation residual max {diag["max_abs_error_mm"]:.2e} mm')
        grid_fields.append(per_seed)

    # (n_families, n_seeds, n_time, n_cells)
    stack = np.stack([np.stack(s) for s in grid_fields])
    mean = stack.mean(axis=(0, 1))

    # TWO spreads, kept apart.
    #
    # `sigma_within` is disagreement BETWEEN MODEL FAMILIES. Averaging over seeds
    # first removes each family's own sampling noise from that comparison, so the
    # term means what its name says.
    #
    # `sigma_seed` is the spread WITHIN a family across random seeds -- bootstrap
    # draws in the forest, subsample/colsample draws in the boosters -- averaged
    # over families. Previously every member was fitted at one fixed seed, so
    # this variation was absent from the reported uncertainty entirely and
    # `sigma_within` was a lower bound in a way that was never stated.
    #
    # They are reported separately rather than summed into one number because
    # they answer different questions: "would another defensible method give this
    # answer?" and "would another run of THIS method?". Folding them together
    # would leave a reader unable to recover either.
    family_means = stack.mean(axis=1)                     # (n_families, t, cells)
    within = (family_means.std(axis=0, ddof=1) if len(model_names) > 1
              else np.zeros_like(mean))
    if len(seeds) > 1:
        # std over seeds within each family, then averaged across families
        sigma_seed = stack.std(axis=1, ddof=1).mean(axis=0)
    else:
        sigma_seed = None

    print('\n5. Assembling uncertainty terms')
    sigma_grace = grace_uncertainty_field(months, cells)
    sigma_transfer = transfer_error_field(table, months, cells)

    # Months GRACE did not observe are model output, not a mass-conserved
    # estimate. Without this term the product would report the SAME error bar
    # where it is most and least constrained.
    observed_month = np.asarray(diag.get('grace_observed',
                                         np.ones(len(months), bool)))
    sigma_gap = sigma_gap_field(gap_tables, months, observed_month, len(cells))

    # sigma_seed joins the quadrature sum only when it was actually measured.
    # With one seed it is not zero-but-known, it is UNMEASURED, and adding a
    # zero would understate sigma_total while looking like a result.
    terms = [sigma_grace, sigma_transfer, within, sigma_gap]
    if sigma_seed is not None:
        terms.append(sigma_seed)
    sigma_total = np.sqrt(sum(t ** 2 for t in terms))

    n_gap = int((~observed_month).sum())
    print(f'  {n_gap} of {len(months)} months are gap-filled (no GRACE observation)')
    for label, arr in [('sigma_grace', sigma_grace),
                       ('sigma_transfer', sigma_transfer),
                       ('sigma_within', within),
                       *(( ('sigma_seed', sigma_seed),) if sigma_seed is not None else ()),
                       ('sigma_gap', sigma_gap),
                       ('sigma_total', sigma_total)]:
        print(f'  {label:15s} median {np.nanmedian(arr):7.1f} mm   '
              f'p90 {np.nanpercentile(arr, 90):7.1f} mm')

    return dict(mean=mean, sigma_grace=sigma_grace, sigma_transfer=sigma_transfer,
                sigma_within=within, sigma_gap=sigma_gap, sigma_seed=sigma_seed,
                sigma_total=sigma_total,
                cells=cells, months=months, table=table, members=model_names,
                seeds=list(seeds),
                diag=diag, gap_table=gap_tbl, grace_observed=observed_month)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write(result: Dict[str, object], path: str) -> str:
    import netCDF4

    grids = cfg.build_grids()
    grid = grids['era5']
    months: pd.DatetimeIndex = result['months']          # type: ignore[assignment]
    cells: np.ndarray = result['cells']                  # type: ignore[assignment]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def expand(flat: np.ndarray) -> np.ndarray:
        full = np.full((len(months), grid.n_y * grid.n_x), np.nan, dtype='float32')
        full[:, cells] = flat
        return full.reshape(len(months), grid.n_y, grid.n_x)

    fields = {
        'twsa': (result['mean'], 'mm',
                 'Downscaled TWSA, ensemble mean, mass-conserved to GRACE'),
        'sigma_total': (result['sigma_total'], 'mm',
                        'Combined 1-sigma uncertainty (quadrature sum below)'),
        'sigma_grace': (result['sigma_grace'], 'mm',
                        'GRACE mascon measurement uncertainty (observational)'),
        'sigma_transfer': (result['sigma_transfer'], 'mm',
                           'Leave-one-mascon-out transfer RMSE by mascon and season'),
        'sigma_within': (result['sigma_within'], 'mm',
                         'Ensemble spread after mass conservation (within-mascon)'),
        'sigma_gap': (result['sigma_gap'], 'mm',
                      'Extra uncertainty where GRACE did not observe (gap-filled months)'),
    }
    # Only written when measured (more than one seed). A single-seed run omits
    # the variable entirely rather than shipping zeros, so its absence is
    # legible: the term was not estimated, as against estimated to be nil.
    if result.get('sigma_seed') is not None:
        fields['sigma_seed'] = (
            result['sigma_seed'], 'mm',
            'Within-family spread across random seeds, averaged over families')

    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', len(months))
        ds.createDimension('lat', grid.n_y)
        ds.createDimension('lon', grid.n_x)
        tv = ds.createVariable('time', 'f8', ('time',))
        tv.units, tv.calendar = 'days since 1970-01-01', 'standard'
        tv[:] = (months - pd.Timestamp('1970-01-01')).days.to_numpy()
        for nm, vals, un in [('lat', grid.lat_centers(), 'degrees_north'),
                             ('lon', grid.lon_centers(), 'degrees_east')]:
            cv = ds.createVariable(nm, 'f8', (nm,))
            cv.units = un
            cv[:] = vals

        for name, (arr, units, long_name) in fields.items():
            v = ds.createVariable(name, 'f4', ('time', 'lat', 'lon'),
                                  zlib=True, complevel=4,
                                  fill_value=np.float32(np.nan))
            v.units = units
            v.long_name = long_name
            v[:] = expand(np.asarray(arr))

        ds['sigma_within'].caveat = (
            'LOWER BOUND, not a calibrated error. GRACE resolves ~20 independent '
            'mascons over this basin, so there is no observation of within-mascon '
            'structure to calibrate against. Members sharing predictors and '
            'training data agree more than they are jointly correct. Read as '
            '"how much do defensible methods disagree", not "how wrong is this".')
        ds['sigma_grace'].gap_handling = (
            'Median substituted where the GRACE record has a gap.')
        obs_flag = np.asarray(result.get('grace_observed',
                                          np.ones(len(months), bool)))
        gv = ds.createVariable('grace_observed', 'i1', ('time',))
        gv.long_name = 'GRACE observed this month'
        gv.description = (
            'This is a SINGLE GAP-FILLED product. 0 means GRACE made no '
            'observation that month, so the field is a reconstruction: the '
            'extrapolated level+trend plus the modelled seasonal anomaly, with '
            'no mass constraint. Those months carry the extra sigma_gap term, '
            'measured by hiding contiguous blocks of observed months and '
            'reconstructing them. Mask on this variable for observation-only use.')
        gv[:] = obs_flag.astype('i1')
        ds.n_months_gap_filled = int((~obs_flag).sum())
        ds.ensemble_members = ', '.join(result['members'])   # type: ignore[arg-type]
        for _k, _v in M.provenance().items():
            setattr(ds, f'provenance_{_k}', str(_v))
        ds.ensemble_seeds = ', '.join(str(s) for s in result.get('seeds', []))
        if result.get('sigma_seed') is None:
            ds.sigma_seed_status = ('not estimated: a single seed was used, so '
                                    'within-family stochastic spread is absent from '
                                    'sigma_total and sigma_within is a lower bound '
                                    'in that respect too')
        ds.title = ('Downscaled GRACE TWSA with per-pixel uncertainty, '
                    'Ganga basin, 0.1 degree, monthly')
        ds.baseline = '2004.0-2010.0 (JPL RL06 mascon convention)'
        ds.created_by = 'main/downscale_uncertainty.py'
        ds.caveat = (
            'Fine-scale structure is inferred from ERA5-Land/GLDAS predictors, not '
            'observed. Agreement with GRACE at mascon scale is imposed by mass '
            'conservation and is not evidence of skill.')
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', nargs='+', type=int, default=[20],
                    help='Random seeds per family. More than one measures '
                         'sigma_seed (within-family stochastic spread); with a '
                         'single seed that term is NOT emitted, because it would '
                         'be unmeasured rather than zero. Cost is one full fit '
                         'per family per seed.')
    ap.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS),
                    choices=['random_forest', 'xgboost', 'lightgbm', 'xgboost_rf', 'mlp'])
    args = ap.parse_args()

    result = run_ensemble(args.models, seeds=args.seeds)
    out = write(result, os.path.join(
        M.RESULTS_DIR, 'twsa_0p1deg_monthly_with_uncertainty.nc'))
    result['table'].to_csv(                                  # type: ignore[union-attr]
        os.path.join(M.RESULTS_DIR, 'transfer_rmse_by_mascon_season.csv'), index=False)
    result['gap_table'].to_csv(                              # type: ignore[union-attr]
        os.path.join(M.RESULTS_DIR, 'gap_recovery_rmse_by_depth.csv'), index=False)
    print(f'\nwritten: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
