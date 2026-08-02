"""
Export the basin-mean GRACE TWSA series used as the target by the basin-scale
analysis, derived from the same gridded GRACE cube the 0.1 degree downscaling
uses.

Why this replaces Data/TWS_JPL.xlsx
-----------------------------------
`TWS_JPL.xlsx` has no recorded provenance and three defects we had to patch
around rather than explain:

  * 20 rows carry the month name 'January ' (trailing space) and one carries
    'july' (lowercase). An exact-match month map turns all 21 into NaT, so
    8.9% of the record -- almost every January from 2003 to 2024 -- was silently
    dropped from the published analysis.
  * The values are native CENTIMETRES while every figure and table labels them
    millimetres, so all reported MAE/RMSE were ten times too small.
  * Its basin mean correlates only r = 0.94 with a properly area-weighted mean
    of the JPL mascon product it is supposed to come from. We cannot account for
    the difference.

This module computes the series directly from
`NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI`, already downloaded on its native
0.5 degree grid by `gee_gridded_download.py` and assembled by `build_cube.py`.
The result is reproducible from code, has a verified unit conversion (cm -> mm,
factor 10), and is the SAME series the gridded pipeline uses -- so the
basin-scale and 0.1 degree analyses no longer rest on different GRACE data.

A note on "55 km"
-----------------
The 0.5 degree grid spacing is not the resolution. JPL RL06 mascons are 3 degree
equal-area cells resampled onto that grid: 364 grid cells over the Ganga basin
carry only 20 distinct values. Choosing this product over the spreadsheet buys
provenance and internal consistency, not spatial detail.

Weighting
---------
Cells are weighted by exact spherical area times the fraction of the cell inside
the basin polygon, so partially covered cells at the boundary contribute in
proportion to the area they actually contribute -- which a plain mean over
"cells whose centre is inside" would get wrong by ~10% of basin area.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import gridded_config as cfg
import downscale_features as F
import downscale_grid_ops as ops

OUT_CSV = os.path.join(cfg.DATA_DIR, 'TWS_GRACE_GEE.csv')


def basin_mean_series(start: str = cfg.START_DATE,
                      end: str = cfg.END_DATE) -> pd.DataFrame:
    """
    Area-weighted basin-mean GRACE TWSA, in millimetres.

    Returns a frame with Date, Year, Month, TWS (mm) and the number of
    contributing 0.5 degree cells. Months GRACE did not observe are omitted, so
    the series carries its gaps honestly rather than interpolating them.
    """
    months = pd.date_range(start, end, freq='MS')
    grids = cfg.build_grids()
    aux = F.load_aux()

    weight = aux['basin_frac_grace'] * ops.cell_area_km2(grids['grace'])
    tws = F.load_grace_monthly(months)              # mm, cm->mm applied in the cube

    ok = np.isfinite(tws) & (weight[None] > 0)
    num = np.nansum(np.where(ok, tws, 0.0) * weight[None], axis=(1, 2))
    den = np.where(ok, weight[None], 0.0).sum(axis=(1, 2))

    with np.errstate(invalid='ignore', divide='ignore'):
        series = np.where(den > 0, num / den, np.nan)

    out = pd.DataFrame({
        'Date': months,
        'TWS': series,
        'n_cells': ok.sum(axis=(1, 2)),
    }).dropna(subset=['TWS'])
    out['Year'] = out.Date.dt.year
    out['Month'] = out.Date.dt.strftime('%B')
    return out[['Date', 'Year', 'Month', 'TWS', 'n_cells']].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=OUT_CSV)
    ap.add_argument('--compare', action='store_true',
                    help='report agreement with the legacy TWS_JPL.xlsx')
    args = ap.parse_args()

    df = basin_mean_series()
    df.to_csv(args.out, index=False)
    print(f'{len(df)} observed months, {df.Date.min():%Y-%m} to {df.Date.max():%Y-%m}')
    print(f'  TWSA {df.TWS.min():.1f} to {df.TWS.max():.1f} mm, std {df.TWS.std():.1f} mm')
    print(f'  Januaries: {(df.Date.dt.month == 1).sum()}')
    print(f'  written: {args.out}')

    if args.compare:
        legacy = os.path.join(cfg.DATA_DIR, 'TWS_JPL.xlsx')
        if os.path.exists(legacy):
            from utils import TWS_CM_TO_MM
            t = pd.read_excel(legacy)
            names = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                     'August', 'September', 'October', 'November', 'December']
            mm = {m.lower(): i + 1 for i, m in enumerate(names)}
            t['Month_Num'] = t.Month.astype(str).str.strip().str.lower().map(mm)
            t['Date'] = pd.to_datetime(
                dict(year=t.Year, month=t.Month_Num, day=1))
            t['legacy_mm'] = t.TWS * TWS_CM_TO_MM
            m = df.merge(t[['Date', 'legacy_mm']], on='Date')
            r = np.corrcoef(m.TWS, m.legacy_mm)[0, 1]
            d = m.TWS - m.legacy_mm
            print(f'\n  vs legacy TWS_JPL.xlsx over {len(m)} shared months:')
            print(f'    r = {r:.4f}   mean difference {d.mean():+.1f} mm   '
                  f'std {d.std():.1f} mm   max |diff| {d.abs().max():.1f} mm')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
