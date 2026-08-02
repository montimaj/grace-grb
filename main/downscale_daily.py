"""
Daily 0.1 degree TWSA by temporal DISAGGREGATION of the monthly product.

Disaggregation, not downscaling. Nothing is fitted at the daily scale: the
monthly mean is preserved exactly and the within-month shape is supplied by the
ERA5-Land water balance. There is no learned transfer function and no information
beyond what ERA5-Land already contains, so calling this "daily downscaling" would
claim skill the method does not have.

The monthly field is the anchor: it is mass-conserved to GRACE, so it carries
whatever the observation says. This module adds WITHIN-MONTH variation from
predictors that are genuinely daily, under a constraint that leaves the monthly
means untouched:

    twsa_daily(p, d) = twsa_monthly(p, m(d)) + delta(p, d)
    with  mean over the days of month m  of  delta(p, .)  ==  0

so re-aggregating the daily product to monthly returns the monthly product
exactly, and through it, GRACE.

Why NOT a daily machine-learning model
--------------------------------------
The obvious alternative is to train an ML model on daily predictors. It cannot
work honestly here. GRACE observes monthly, so no daily target exists; training
one requires interpolating the monthly series to daily, and the model then
learns to reproduce that interpolation. Its "daily skill" would measure how well
it fits an assumption. The basin-scale pipeline does exactly this and is explicit
that daily skill cannot be validated from it -- which is why the temporal closure
test exists.

So the daily signal is taken from data that IS daily.

Two independent routes, deliberately
------------------------------------
flux    cumulative water balance, from ERA5-Land P - ET - Qs - Qsb. Storage
        change integrates the flux imbalance, so a running sum within the month
        is the natural daily shape.

state   the modelled stores themselves, from ERA5-Land soil water layers 1-4
        plus snow water equivalent. No integration, no accumulated bias.

They share a driving dataset but not a derivation: one integrates fluxes, the
other reads states. Where they agree, the daily shape is robust to that choice.
Where they diverge, the disagreement is itself the honest uncertainty estimate,
and `daily_method_spread` records it per pixel and day.

Neither is validated against daily observations, because none exist. The wells
are quarterly. What CAN be checked is monthly closure, which is exact by
construction here, and the plausibility of the daily amplitude against the
storage change implied by the fluxes.

Usage
-----
    python downscale_daily.py                      # both methods + spread
    python downscale_daily.py --method flux
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import gridded_config as cfg
import downscale_features as F
import downscale_model as M
import figure_captions

# ERA5-Land fields used by each route.
FLUX_IN = 'ppt'
FLUX_OUT = ('et_total', 'runoff_surface', 'runoff_subsurface')
STATE_VARS = ('vsw1', 'vsw2', 'vsw3', 'vsw4', 'swe')


def _daily_block(var: str, days: pd.DatetimeIndex, cells: np.ndarray) -> np.ndarray:
    """One variable over `days`, flattened to (n_days, n_cells)."""
    import netCDF4

    dates = F.cube_dates('era5')
    pos = pd.Index(dates).get_indexer(days)
    if (pos < 0).any():
        missing = days[pos < 0]
        raise KeyError(f'{len(missing)} days absent from era5_cube, '
                       f'e.g. {missing[0]:%Y-%m-%d}')
    with netCDF4.Dataset(cfg.cube_path('era5')) as ds:
        block = np.asarray(ds[var][pos[0]:pos[-1] + 1])
    return block.reshape(len(days), -1)[:, cells]


def daily_shape(method: str, days: pd.DatetimeIndex,
                cells: np.ndarray) -> np.ndarray:
    """
    Within-month storage variation, de-meaned per pixel per month.

    Returns (n_days, n_cells). The de-meaning is what makes the disaggregation
    conservative: adding a month-mean-zero field cannot move the monthly mean.
    """
    if method == 'flux':
        # Storage integrates the flux imbalance, so the daily shape is the
        # running sum of P - ET - Q within the month. Restarting the sum each
        # month is deliberate: a continuous integral would accumulate ERA5-Land's
        # bias without bound over 26 years, and the monthly anchor already
        # supplies the level.
        net = _daily_block(FLUX_IN, days, cells)
        for var in FLUX_OUT:
            net = net - _daily_block(var, days, cells)
        raw = np.empty_like(net)
        for _, idx in pd.Series(range(len(days)), index=days).groupby(
                days.to_period('M')):
            sl = idx.to_numpy()
            raw[sl] = np.cumsum(net[sl], axis=0)
    elif method == 'state':
        # The stores themselves. No integration, so no accumulated bias, but it
        # only sees water ERA5-Land actually models -- no groundwater.
        raw = None
        for var in STATE_VARS:
            b = _daily_block(var, days, cells)
            raw = b if raw is None else raw + b
    else:
        raise ValueError(f"method must be 'flux' or 'state', got {method!r}")

    # De-mean within each calendar month, per pixel.
    out = np.empty_like(raw)
    for _, idx in pd.Series(range(len(days)), index=days).groupby(
            days.to_period('M')):
        sl = idx.to_numpy()
        with F.allnan_ok():
            out[sl] = raw[sl] - np.nanmean(raw[sl], axis=0, keepdims=True)
    return out


def disaggregate(monthly: np.ndarray, months: pd.DatetimeIndex,
                 days: pd.DatetimeIndex, cells: np.ndarray,
                 method: str) -> np.ndarray:
    """Monthly anchor + de-meaned daily shape, (n_days, n_cells)."""
    pos = pd.Index(months.to_period('M')).get_indexer(days.to_period('M'))
    if (pos < 0).any():
        raise KeyError('daily range extends beyond the monthly product')
    return monthly[pos] + daily_shape(method, days, cells)


def check_closure(daily: np.ndarray, days: pd.DatetimeIndex,
                  monthly: np.ndarray, months: pd.DatetimeIndex) -> Dict[str, float]:
    """Re-aggregate the daily field and compare against the monthly anchor."""
    agg = np.full_like(monthly, np.nan)
    pos = pd.Index(months.to_period('M')).get_indexer(days.to_period('M'))
    for i in range(len(months)):
        sel = pos == i
        if sel.any():
            with F.allnan_ok():
                agg[i] = np.nanmean(daily[sel], axis=0)
    err = np.abs(agg - monthly)
    ok = np.isfinite(err)
    return {'max_abs_mm': float(np.nanmax(err[ok])) if ok.any() else float('nan'),
            'mean_abs_mm': float(np.nanmean(err[ok])) if ok.any() else float('nan')}


def write(daily: Dict[str, np.ndarray], days: pd.DatetimeIndex,
          cells: np.ndarray, path: str, closure: Dict[str, Dict[str, float]],
          grace_observed_daily: Optional[np.ndarray] = None) -> str:
    import netCDF4

    grid = cfg.build_grids()['era5']
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def expand(flat):
        full = np.full((len(days), grid.n_y * grid.n_x), np.nan, dtype='float32')
        full[:, cells] = flat
        return full.reshape(len(days), grid.n_y, grid.n_x)

    primary = 'flux' if 'flux' in daily else next(iter(daily))
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', len(days))
        ds.createDimension('lat', grid.n_y)
        ds.createDimension('lon', grid.n_x)
        tv = ds.createVariable('time', 'f8', ('time',))
        tv.units, tv.calendar = 'days since 1970-01-01', 'standard'
        tv[:] = (days - pd.Timestamp('1970-01-01')).days.to_numpy()
        for nm, vals, un in [('lat', grid.lat_centers(), 'degrees_north'),
                             ('lon', grid.lon_centers(), 'degrees_east')]:
            cv = ds.createVariable(nm, 'f8', (nm,)); cv.units = un; cv[:] = vals

        chunks = (min(64, len(days)), grid.n_y, grid.n_x)
        for name, arr in daily.items():
            v = ds.createVariable(f'twsa_{name}', 'f4', ('time', 'lat', 'lon'),
                                  zlib=True, complevel=4, shuffle=True,
                                  chunksizes=chunks,
                                  fill_value=np.float32(np.nan))
            v.units = 'mm'
            v.long_name = f'Daily TWSA, {name} disaggregation'
            v.closure_max_mm = closure[name]['max_abs_mm']
            v.closure_mean_mm = closure[name]['mean_abs_mm']
            v[:] = expand(arr)

        if len(daily) > 1:
            names = list(daily)
            spread = np.abs(daily[names[0]] - daily[names[1]]) / 2.0
            v = ds.createVariable('daily_method_spread', 'f4',
                                  ('time', 'lat', 'lon'), zlib=True, complevel=4,
                                  chunksizes=chunks, fill_value=np.float32(np.nan))
            v.units = 'mm'
            v.long_name = ('Half the absolute difference between the flux and '
                           'state disaggregations')
            v.description = (
                'Uncertainty in the WITHIN-MONTH shape only; the monthly mean is '
                'identical in both. No daily observation exists to validate '
                'either route, so the disagreement between two independent '
                'derivations is the available estimate.')
            v[:] = expand(spread)

        if grace_observed_daily is not None:
            gv = ds.createVariable('grace_observed', 'i1', ('time',))
            gv.long_name = 'GRACE observed the month this day belongs to'
            gv[:] = grace_observed_daily.astype('i1')

        for _k, _v in M.provenance().items():
            setattr(ds, f'provenance_{_k}', str(_v))
        ds.title = ('Daily GRACE TWSA, Ganga basin, 0.1 degree: spatially downscaled, temporally disaggregated')
        ds.primary_variable = f'twsa_{primary}'
        ds.method = (
            'Monthly mass-conserved product plus a within-month shape that is '
            'de-meaned per pixel per month, so monthly means are unchanged and '
            'the daily field re-aggregates to GRACE exactly.')
        ds.caveat = (
            'Daily variation is INFERRED from ERA5-Land, not observed. GRACE is '
            'monthly and the CGWB wells are quarterly, so no daily observation '
            'exists against which to validate the sub-monthly shape. A daily ML '
            'model was deliberately not used: with no daily target it would be '
            'trained on an interpolation of the monthly product and would only '
            'learn to reproduce that assumption.')
    return path


def plot_temporal_closure(daily: Dict[str, np.ndarray], days: pd.DatetimeIndex,
                    monthly: np.ndarray, months: pd.DatetimeIndex,
                    cells: np.ndarray, out_dir: str) -> Optional[str]:
    """
    The temporal-closure figure: daily re-aggregated to monthly, against the anchor.

    A reviewer asked for exactly this — re-aggregate the predicted daily field
    back to monthly and compare against GRACE — and asked what result to expect.

    The expected result is EXACT AGREEMENT, and the figure is drawn to show that
    rather than to imply a test was passed. The within-month shape is de-meaned
    per pixel per month before it is added to the anchor, so the monthly mean is
    preserved identically; the monthly product is in turn mass-conserved to
    observed GRACE at mascon scale. A visible departure here would mean a coding
    error, not a hydrological finding.

    Drawing it anyway is worthwhile: it is the check a reader would otherwise
    have to take on trust, and the residual panel makes the magnitude of the
    floating-point disagreement legible (~1e-4 mm against a signal of ~100 mm).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import DPI, SCI_AQUA, SCI_BLUE, SCI_INK, SCI_MUTED, SCI_ORANGE

    method = 'flux' if 'flux' in daily else next(iter(daily))
    d = daily[method]
    pos = pd.Index(months.to_period('M')).get_indexer(days.to_period('M'))
    agg = np.full_like(monthly, np.nan)
    for i in range(len(months)):
        sel = pos == i
        if sel.any():
            with F.allnan_ok():
                agg[i] = np.nanmean(d[sel], axis=0)

    with F.allnan_ok():
        anchor_series = np.nanmean(monthly, axis=1)
        agg_series = np.nanmean(agg, axis=1)
    resid = agg_series - anchor_series

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True,
                                  gridspec_kw={'height_ratios': [2.4, 1]})
    ax.plot(months, anchor_series, '-', color=SCI_BLUE, lw=2.0,
            label='Monthly product (mass-conserved to GRACE)')
    ax.plot(months, agg_series, '--', color=SCI_ORANGE, lw=1.6,
            label=f'Daily field re-aggregated to monthly ({method})')
    ax.set_ylabel('Basin-mean TWSA (mm)', color=SCI_INK)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('Temporal closure: daily → monthly, against the monthly anchor',
                 fontweight='bold', color=SCI_INK, loc='left')

    ax2.plot(months, resid, '-', color=SCI_AQUA, lw=1.4)
    ax2.axhline(0, color=SCI_MUTED, lw=1)
    ax2.set_ylabel('Residual (mm)', color=SCI_INK)
    for a in (ax, ax2):
        a.grid(True, alpha=0.3)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    ax2.set_title(f'max |residual| = {np.nanmax(np.abs(resid)):.2e} mm — '
                  'floating-point error, not skill',
                  fontsize=9, color=SCI_MUTED, loc='left')

    fig.tight_layout()
    p = os.path.join(out_dir, 'Fig_temporal_closure.png')
    # Caption recorded, not drawn -- see figure_captions. This one matters more
    # than most: the figure looks like a validation and is not one.
    figure_captions.record(
        p,
        'Not to be confused with the basin-scale temporal closure test of '
        'temporal_closure_validation.py, which is retired: that one compared a '
        'model FREE to disagree with GRACE, and could fail. This one cannot. '
        'Agreement here is EXACT BY CONSTRUCTION: the within-month shape is '
        'de-meaned per pixel per month, so re-aggregating returns the anchor '
        'identically. This figure documents that the arithmetic holds; it is '
        'not evidence that the daily variation is correct, which no available '
        'observation can establish.')
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'   temporal-closure figure: {os.path.basename(p)}')
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--product', default=None, help='monthly product netCDF')
    ap.add_argument('--method', nargs='+', default=['flux', 'state'],
                    choices=['flux', 'state'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import netCDF4

    path = args.product or os.path.join(
        M.RESULTS_DIR, 'twsa_0p1deg_monthly_with_uncertainty.nc')
    if not os.path.exists(path):
        path = os.path.join(M.RESULTS_DIR, 'twsa_0p1deg_monthly_xgboost.nc')
    if not os.path.exists(path):
        raise FileNotFoundError('no monthly product; run downscale_model.py first')

    print(f'1. Monthly anchor: {os.path.basename(path)}')
    with netCDF4.Dataset(path) as ds:
        ds['twsa'].set_auto_mask(False)
        monthly_full = np.asarray(ds['twsa'][:])
        months = pd.to_datetime(ds['time'][:], unit='D', origin='1970-01-01')
        obs_flag = (np.asarray(ds['grace_observed'][:]).astype(bool)
                    if 'grace_observed' in ds.variables else None)

    aux = F.load_aux()
    cells = np.flatnonzero((aux['basin_frac_era5'] > 0).ravel())
    monthly = monthly_full.reshape(len(months), -1)[:, cells]

    era5_days = F.cube_dates('era5')
    # Half-open on the NEXT month's first day, not on this month's last day.
    #
    # `months[-1] + MonthEnd(1)` lands on 2025-12-31, and `<` then excluded it:
    # the record lost its final day, and -- worse -- December's monthly mean was
    # being enforced over 30 days instead of 31, so the closure constraint was
    # applied to a month the product did not fully contain.
    days = era5_days[(era5_days >= months[0]) &
                     (era5_days < months[-1] + pd.offsets.MonthBegin(1))]
    print(f'   {len(months)} months -> {len(days)} days, {len(cells):,} cells')

    daily: Dict[str, np.ndarray] = {}
    closure: Dict[str, Dict[str, float]] = {}
    for i, method in enumerate(args.method, start=2):
        print(f'{i}. {method} disaggregation')
        d = disaggregate(monthly, months, days, cells, method)
        c = check_closure(d, days, monthly, months)
        print(f'   monthly closure: max {c["max_abs_mm"]:.3g} mm, '
              f'mean {c["mean_abs_mm"]:.3g} mm  (must be ~0 by construction)')
        daily[method], closure[method] = d, c

    if len(daily) > 1:
        a, b = list(daily.values())[:2]
        sp = np.abs(a - b) / 2.0
        print(f'\n   method spread: median {np.nanmedian(sp):.1f} mm, '
              f'p90 {np.nanpercentile(sp, 90):.1f} mm  '
              f'(within-month shape only)')

    obs_daily = None
    if obs_flag is not None:
        pos = pd.Index(months.to_period('M')).get_indexer(days.to_period('M'))
        obs_daily = obs_flag[pos]

    plot_temporal_closure(daily, days, monthly, months, cells, M.RESULTS_DIR)

    out = args.out or os.path.join(M.RESULTS_DIR, 'twsa_0p1deg_daily.nc')
    write(daily, days, cells, out, closure, obs_daily)
    # M.RESULTS_DIR, not out_dir: this main() has no `out_dir` binding -- the
    # figure is written to M.RESULTS_DIR directly (see the plot call above), and
    # `out` is a FILE path, not a directory.
    caps = figure_captions.write_index(
        M.RESULTS_DIR, title='Figure captions — daily disaggregation')
    print(f'\nwritten: {out}  ({os.path.getsize(out) / 1e6:.0f} MB)')
    if caps:
        print(f'         {caps}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
