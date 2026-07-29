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

    GWS = TWSA - (root-zone soil moisture + snow water equivalent + canopy)

with the subtracted terms taken from GLDAS CLSM and, like TWSA, expressed as
anomalies about the same 2004.0-2010.0 baseline. What remains is compared with

    GWS_well = -Sy * (depth - depth_baseline) * 1000       [mm]

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
# GLDAS stores subtracted from TWSA to isolate groundwater.
#
# ROOT-ZONE, not PROFILE. GLDAS CLSM's `SoilMoist_P_tavg` is the full soil
# profile, which already contains the modelled groundwater store: subtracting it
# removes part of the very quantity the wells are supposed to validate, so the
# residual is not groundwater at all. `SoilMoist_RZ_tavg` stops above the water
# table, leaving TWSA - (root-zone SM + snow + canopy) ~ groundwater.
STORE_VARS = ('rzsm', 'swe_gldas', 'canopy')


def gldas_storage_anomaly(months: pd.DatetimeIndex, cells: np.ndarray) -> np.ndarray:
    """
    Soil moisture + snow + canopy storage on the 0.1 degree grid, as anomalies
    about the GRACE baseline. Returns (n_months, n_cells) in mm.
    """
    grids = cfg.build_grids()
    gldas, era5 = grids['gldas'], grids['era5']

    total = None
    for var in STORE_VARS:
        field = F.monthly_mean('gldas', var, months)
        fine = ops.regrid_bilinear(field, gldas, era5)
        total = fine if total is None else total + fine

    flat = total.reshape(len(months), -1)[:, cells]
    in_base = ((months >= pd.Timestamp(BASELINE[0]))
               & (months <= pd.Timestamp(BASELINE[1])))
    with np.errstate(invalid='ignore'):
        baseline = np.nanmean(flat[in_base], axis=0, keepdims=True)
    return flat - baseline


def match_wells(
    product_path: Optional[str] = None, verbose: bool = True
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

    stores = np.full((len(months), era5.n_cells), np.nan)
    stores[:, cells] = gldas_storage_anomaly(months, cells)

    # Mascon label per fine cell, so skill can be reported per mascon.
    mascon_fine = aux['mascon_id'].ravel()[aux['parent_era5_to_grace'].ravel()]

    month_pos = {m.to_period('M'): i for i, m in enumerate(months)}
    obs = wells.obs[wells.obs.well_id.isin(meta.index)].copy()
    obs['tidx'] = obs.date.dt.to_period('M').map(month_pos)
    obs = obs.dropna(subset=['tidx'])
    obs['tidx'] = obs.tidx.astype(int)
    obs['flat'] = obs.well_id.map(meta.era5_flat).astype(int)

    ti, fi = obs.tidx.to_numpy(), obs.flat.to_numpy()
    obs['gws_downscaled'] = flat_twsa[ti, fi] - stores[ti, fi]
    obs['gws_bilinear'] = bilinear_full[ti, fi] - stores[ti, fi]
    if flat_sigma is not None:
        obs['sigma_total'] = flat_sigma[ti, fi]
    obs['mascon'] = mascon_fine[fi]
    obs['lat'] = obs.well_id.map(meta.lat)
    obs['lon'] = obs.well_id.map(meta.lon)

    obs = obs.dropna(subset=['gws_downscaled', 'gws_bilinear', 'gws_anomaly_mm'])
    return obs, {'product': path, 'months': months, 'n_wells': obs.well_id.nunique()}


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
    args = ap.parse_args()

    print('Independent validation against CGWB dug wells\n')
    obs, info = match_wells(args.product)
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
