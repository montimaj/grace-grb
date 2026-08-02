"""
Independent validation of the 0.1 degree product against CGWB groundwater wells.

Everything else in this project is validated against GRACE, or against models
that supplied the predictors. This is the only test using an observation the
downscaling never saw, and it is the only one that can say whether the
fine-scale structure we synthesise corresponds to anything real.

The comparison
--------------
GRACE measures TOTAL water storage. Wells measure the groundwater component
only. To compare them the other stores have to come off:

    GWS = TWSA - (root-zone soil moisture + snow water equivalent [+ canopy])

with the subtracted terms expressed, like TWSA, as anomalies about the same
2004.0-2010.0 baseline. What remains is compared with

    GWS_well = -Sy * (depth - depth_baseline) * 1000       [mm]

Two definitions of those stores are available (see STORE_SETS): ERA5-Land layers
1-3 plus snow at native 0.1 degrees, which is the DEFAULT, or GLDAS 2.1 NOAH
root-zone plus snow plus canopy at 0.25 degrees. NOT GLDAS 2.2 CLSM, which
assimilates GRACE and was dropped from this project entirely.

Note the residual carries the errors of the subtracted model stores as well as
of the downscaling, so disagreement is an upper bound on downscaling error.

What is actually being tested
-----------------------------
Not "is the product right" -- mass conservation already pins mascon means to
GRACE, so any method would score similarly on basin-wide averages. The test
that matters is whether the product's WITHIN-MASCON pattern beats what you get
from interpolation alone. Both are therefore scored, on identical samples:

    downscaled  vs  bilinear interpolation of GRACE

If the downscaled field does not beat interpolation at the well locations, the
fine structure is decoration.

Caveats that belong in any caption
----------------------------------
* Wells are QUARTERLY (Jan/May/Aug/Nov), so this validates spatial pattern and
  seasonal amplitude, never daily structure.
* Specific yield is a single reference value per well; its uncertainty maps
  directly onto the storage conversion.
* A dug well samples a few hundred metres; a 0.1 degree cell is ~120 km^2.
  Point-to-pixel mismatch alone guarantees scatter.
* Well coverage is 621 of ~9,077 in-basin cells (6.8%), concentrated in the
  populated plains, so this says little about the Himalayan margin.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import gridded_config as cfg
import downscale_features as F
import downscale_grid_ops as ops
import downscale_model as M
import wells_ingest as W

BASELINE = (W.BASELINE_START, W.BASELINE_END)

# The non-groundwater stores subtracted from TWSA, and where they come from.
#
# ROOT-ZONE, not FULL PROFILE, in both definitions. A full-profile soil moisture
# already contains the modelled groundwater store, so subtracting it removes part
# of the very quantity the wells are supposed to validate and the residual is not
# groundwater at all. Both options below stop above the water table.
#
# 'gldas'  GLDAS 2.1 NOAH RootMoist_inst + SWE + canopy interception, at 0.25 deg
#          and bilinearly upsampled to 0.1 deg. INDEPENDENT of the predictors,
#          which are all ERA5-Land -- so the residual does not inherit ERA5's
#          soil-moisture bias on both sides of the subtraction. NOAH has no
#          groundwater store at all, so nothing here can double-count it.
#          Its weakness is resolution: the subtracted field carries no structure
#          finer than 0.25 deg, in a comparison made at 0.1 deg.
#
# 'era5'   ERA5-Land layers 1-3 (0-100 cm, thicknesses 70/210/720 mm) + snow
#          water equivalent, NATIVE 0.1 deg, no regridding. Matches the product
#          grid and the 0-100 cm root-zone convention `gee_download.py` already
#          uses for `rzsm_era5`. No canopy term exists in the download; canopy
#          interception is ~mm against hundreds of mm of soil moisture, so the
#          omission is small but real. Its weakness is the mirror of GLDAS's:
#          predictors and subtracted stores now share a model.
#
# Neither is obviously right, so both are selectable with `--stores`.
STORE_SETS: Dict[str, Dict[str, object]] = {
    'gldas': {'grid': 'gldas', 'vars': ('rzsm', 'swe_gldas', 'canopy')},
    'era5': {'grid': 'era5', 'vars': ('vsw1', 'vsw2', 'vsw3', 'swe')},
}
# ERA5 is the default because it is native to the product's own 0.1 degree grid.
# Subtracting a 0.25 degree field inside a comparison made at 0.1 degrees puts a
# resolution mismatch into the groundwater residual before the test runs. Its
# cost is the shared model noted above; `--stores gldas` is the alternative that
# avoids it.
#
# These are ALTERNATIVES, not an experiment. Every released number uses the
# default. No cross-store comparison is run or reported.
DEFAULT_STORES = 'era5'

# Kept for backward compatibility with callers that imported the old name.
STORE_VARS = STORE_SETS['gldas']['vars']


def storage_anomaly(months: pd.DatetimeIndex, cells: np.ndarray,
                    stores: str = DEFAULT_STORES) -> np.ndarray:
    """
    Non-groundwater storage on the 0.1 degree grid, as anomalies about the GRACE
    baseline. Returns (n_months, n_cells) in mm.

    Regridding happens only when the source grid is not already the target: the
    ERA5 set is native 0.1 deg and is used as-is.
    """
    if stores not in STORE_SETS:
        raise ValueError(f'stores must be one of {sorted(STORE_SETS)}, got {stores!r}')
    spec = STORE_SETS[stores]
    grids = cfg.build_grids()
    src, era5 = grids[spec['grid']], grids['era5']

    total = None
    for var in spec['vars']:
        field = F.monthly_mean(spec['grid'], var, months)
        fine = field if spec['grid'] == 'era5' else ops.regrid_bilinear(field, src, era5)
        total = fine if total is None else total + fine

    flat = total.reshape(len(months), -1)[:, cells]
    in_base = ((months >= pd.Timestamp(BASELINE[0]))
               & (months <= pd.Timestamp(BASELINE[1])))
    with F.allnan_ok():
        baseline = np.nanmean(flat[in_base], axis=0, keepdims=True)
    return flat - baseline


def gldas_storage_anomaly(months: pd.DatetimeIndex, cells: np.ndarray) -> np.ndarray:
    """Deprecated alias for `storage_anomaly(..., stores='gldas')`."""
    return storage_anomaly(months, cells, stores='gldas')


def match_wells(
    product_path: Optional[str] = None, verbose: bool = True,
    stores: str = DEFAULT_STORES,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Join well observations to the co-located product pixel, for both the
    downscaled field and the bilinear-interpolation baseline.
    """
    import netCDF4

    wells = W.assign_grid_cells(W.build(W.DUG_WELL_FILE, verbose=verbose))
    meta = wells.meta[wells.meta.era5_flat >= 0].set_index('well_id')

    path = product_path or os.path.join(
        M.RESULTS_DIR, 'twsa_0p1deg_monthly_with_uncertainty.nc')
    if not os.path.exists(path):
        path = os.path.join(M.RESULTS_DIR, 'twsa_0p1deg_monthly_xgboost.nc')
    if not os.path.exists(path):
        raise FileNotFoundError('no gridded product found; run downscale_model.py')

    with netCDF4.Dataset(path) as ds:
        for name in ('twsa', 'sigma_total'):
            if name in ds.variables:
                ds[name].set_auto_mask(False)
        twsa = np.asarray(ds['twsa'][:])
        sigma = (np.asarray(ds['sigma_total'][:])
                 if 'sigma_total' in ds.variables else None)
        months = pd.to_datetime(ds['time'][:], unit='D', origin='1970-01-01')

    grids = cfg.build_grids()
    era5 = grids['era5']
    aux = F.load_aux()
    cells = np.flatnonzero((aux['basin_frac_era5'] > 0).ravel())

    flat_twsa = twsa.reshape(len(months), -1)
    flat_sigma = sigma.reshape(len(months), -1) if sigma is not None else None

    base = M.interpolation_baselines(months, cells)
    bilinear_full = np.full((len(months), era5.n_cells), np.nan)
    bilinear_full[:, cells] = base['bilinear']

    store_field = np.full((len(months), era5.n_cells), np.nan)
    store_field[:, cells] = storage_anomaly(months, cells, stores=stores)

    # Mascon label per fine cell, so skill can be reported per mascon.
    mascon_fine = aux['mascon_id'].ravel()[aux['parent_era5_to_grace'].ravel()]

    month_pos = {m.to_period('M'): i for i, m in enumerate(months)}
    obs = wells.obs[wells.obs.well_id.isin(meta.index)].copy()
    obs['tidx'] = obs.date.dt.to_period('M').map(month_pos)
    obs = obs.dropna(subset=['tidx'])
    obs['tidx'] = obs.tidx.astype(int)
    obs['flat'] = obs.well_id.map(meta.era5_flat).astype(int)

    ti, fi = obs.tidx.to_numpy(), obs.flat.to_numpy()
    obs['gws_downscaled'] = flat_twsa[ti, fi] - store_field[ti, fi]
    obs['gws_bilinear'] = bilinear_full[ti, fi] - store_field[ti, fi]
    if flat_sigma is not None:
        obs['sigma_total'] = flat_sigma[ti, fi]
    obs['mascon'] = mascon_fine[fi]
    obs['lat'] = obs.well_id.map(meta.lat)
    obs['lon'] = obs.well_id.map(meta.lon)

    obs = obs.dropna(subset=['gws_downscaled', 'gws_bilinear', 'gws_anomaly_mm'])
    return obs, {'product': path, 'months': months,
                 'n_wells': obs.well_id.nunique(), 'stores': stores}


def score(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Skill against the wells, pooled and per-well, for both methods.

    The per-well correlation is the informative one: it asks whether the product
    reproduces each location's TEMPORAL behaviour, which is not fixed by mass
    conservation and is therefore a genuine test.
    """
    rows = []
    for label, col in [('Downscaled', 'gws_downscaled'),
                       ('Bilinear interpolation', 'gws_bilinear')]:
        pooled = M.metrics(obs.gws_anomaly_mm.to_numpy(), obs[col].to_numpy())
        per_well = []
        for _, g in obs.groupby('well_id'):
            if len(g) >= 12 and g[col].std() > 0 and g.gws_anomaly_mm.std() > 0:
                per_well.append(np.corrcoef(g.gws_anomaly_mm, g[col])[0, 1])
        per_well = np.array(per_well)
        rows.append({
            'method': label,
            'RMSE_mm': pooled['RMSE'],
            'MAE_mm': pooled['MAE'],
            'R2_pooled': pooled['R2'],
            'median_well_r': float(np.median(per_well)) if per_well.size else np.nan,
            'frac_wells_r_gt_0.5': (float((per_well > 0.5).mean())
                                    if per_well.size else np.nan),
            'n_obs': pooled['n'],
            'n_wells': int(obs.well_id.nunique()),
        })
    return pd.DataFrame(rows)


def uncertainty_calibration(obs: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Is the stated uncertainty honest?

    Compares |product - well| against the product's own sigma. A well-calibrated
    1-sigma should bracket ~68% of residuals. Expect UNDER-coverage here: the
    well residual also contains specific-yield error, point-to-pixel mismatch and
    GLDAS store error, none of which sigma claims to cover.
    """
    if 'sigma_total' not in obs.columns:
        return None
    resid = (obs.gws_downscaled - obs.gws_anomaly_mm).to_numpy()
    sigma = obs.sigma_total.to_numpy()
    ok = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < 100:
        return None
    z = resid[ok] / sigma[ok]
    return {
        'coverage_1sigma': float((np.abs(z) <= 1).mean()),
        'coverage_2sigma': float((np.abs(z) <= 2).mean()),
        'median_abs_z': float(np.median(np.abs(z))),
        'rmse_mm': float(np.sqrt(np.mean(resid[ok] ** 2))),
        'median_sigma_mm': float(np.median(sigma[ok])),
        'n': int(ok.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--product', default=None)
    ap.add_argument('--stores', default=DEFAULT_STORES, choices=sorted(STORE_SETS),
                    help='Which non-groundwater stores to subtract from TWSA.')
    args = ap.parse_args()

    print('Independent validation against CGWB dug wells\n')
    obs, info = match_wells(args.product, stores=args.stores)
    print(f'stores subtracted: {args.stores} '
          f'({", ".join(STORE_SETS[args.stores]["vars"])})')
    print(f'\nproduct: {os.path.basename(str(info["product"]))}')
    print(f'matched: {len(obs):,} well-months across {info["n_wells"]} wells\n')

    table = score(obs)
    print(table.to_string(index=False, float_format=lambda v: f'{v:9.3f}'))

    win = table.set_index('method')
    delta = (win.loc['Downscaled', 'median_well_r']
             - win.loc['Bilinear interpolation', 'median_well_r'])
    print(f'\nper-well temporal correlation, downscaled minus interpolation: '
          f'{delta:+.3f}')
    print('  (positive means the fine-scale structure carries real information;'
          '\n   zero or negative means it is decoration)')

    cal = uncertainty_calibration(obs)
    if cal:
        print('\nuncertainty calibration:')
        print(f'  1-sigma coverage {cal["coverage_1sigma"]:.1%} '
              f'(nominal 68%), 2-sigma {cal["coverage_2sigma"]:.1%} (nominal 95%)')
        print(f'  median sigma {cal["median_sigma_mm"]:.0f} mm vs '
              f'residual RMSE {cal["rmse_mm"]:.0f} mm over {cal["n"]:,} matches')
        print('  Under-coverage is expected: the residual also carries specific-yield\n'
              '  error, point-to-pixel mismatch and GLDAS store error.')

    out_dir = M.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'well_validation_summary.csv'), index=False)
    obs.to_csv(os.path.join(out_dir, 'well_validation_matches.csv'), index=False)
    print(f'\nwritten to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
