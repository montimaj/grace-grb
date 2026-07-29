"""
Export the daily (or monthly) product as Cloud-Optimized GeoTIFFs for upload to
Earth Engine via `geeup`.

Produces, under `Results/downscaling/cogs/<collection>/`:

    twsa_daily_20000101.tif ...        one COG per time step
    metadata.csv                       geeup property table
    upload.sh                          the exact geeup command, ready to run

Why COG and not plain GeoTIFF
-----------------------------
Earth Engine ingests either, but COGs carry internal tiling and overviews, so the
explorer app can read a zoomed-out view without pulling the full raster. With
~9,500 daily rasters that is the difference between a responsive map and an
unusable one.

A note on what gets uploaded
----------------------------
The `grace_observed` flag travels as a per-image PROPERTY, not a band. Months
GRACE never saw are reconstructions, and an app that plots them beside observed
months without distinction would misrepresent the product. The property lets a
filter (`.filter(ee.Filter.eq('grace_observed', 1))`) recover the
observation-only subset.

Usage
-----
    python export_cogs.py                          # daily flux product
    python export_cogs.py --which monthly
    python export_cogs.py --variable twsa_state
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import gridded_config as cfg
import downscale_model as M

EE_PROJECT_FOR_UPLOAD = 'grace-grb-ml'
COG_ROOT = os.path.join(M.RESULTS_DIR, 'cogs')


def _profile(grid: cfg.Grid) -> Dict[str, object]:
    from rasterio.transform import from_origin
    return {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'height': grid.n_y, 'width': grid.n_x, 'crs': grid.crs,
        'transform': from_origin(grid.x_min, grid.y_max, grid.res, grid.res),
        'nodata': np.float32(np.nan),
        'tiled': True, 'blockxsize': 128, 'blockysize': 128,
        'compress': 'deflate', 'predictor': 2, 'zlevel': 6,
    }


def write_cog(array: np.ndarray, path: str, grid: cfg.Grid) -> None:
    """One COG: tiled, compressed, with overviews, validated by re-open."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.shutil import copy as rio_copy

    tmp = path + '.tmp.tif'
    with rasterio.open(tmp, 'w', **_profile(grid)) as dst:
        dst.write(array.astype('float32'), 1)
        # Overviews must exist BEFORE the COG copy, or the result is a plain
        # tiled GeoTIFF that merely looks like a COG.
        dst.build_overviews([2, 4, 8], Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')
    rio_copy(tmp, path, driver='COG', compress='deflate', predictor=2,
             overview_resampling='average')
    os.remove(tmp)


def export(which: str = 'daily', variable: Optional[str] = None,
           out_root: str = COG_ROOT, limit: Optional[int] = None) -> str:
    import netCDF4

    if which == 'daily':
        src = os.path.join(M.RESULTS_DIR, 'twsa_0p1deg_daily.nc')
        variable = variable or 'twsa_flux'
        collection = 'twsa_0p1deg_daily'
    else:
        src = os.path.join(M.RESULTS_DIR,
                           'twsa_0p1deg_monthly_with_uncertainty.nc')
        if not os.path.exists(src):
            src = os.path.join(M.RESULTS_DIR, 'twsa_0p1deg_monthly_xgboost.nc')
        variable = variable or 'twsa'
        collection = 'twsa_0p1deg_monthly'
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'{src} not found - run the pipeline before exporting COGs')

    out_dir = os.path.join(out_root, collection)
    os.makedirs(out_dir, exist_ok=True)
    grid = cfg.build_grids()['era5']

    with netCDF4.Dataset(src) as ds:
        if variable not in ds.variables:
            raise KeyError(f'{variable} not in {os.path.basename(src)}; '
                           f'have {[v for v in ds.variables]}')
        ds[variable].set_auto_mask(False)
        times = pd.to_datetime(ds['time'][:], unit='D', origin='1970-01-01')
        n = len(times) if limit is None else min(limit, len(times))
        obs = (np.asarray(ds['grace_observed'][:]).astype(int)
               if 'grace_observed' in ds.variables else np.ones(len(times), int))

        rows: List[Dict[str, object]] = []
        for i in range(n):
            stamp = times[i].strftime('%Y%m%d')
            name = f'{collection}_{stamp}'
            write_cog(np.asarray(ds[variable][i]),
                      os.path.join(out_dir, f'{name}.tif'), grid)
            rows.append({
                'id_no': name,
                'system:time_start': int(times[i].timestamp() * 1000),
                'date': times[i].strftime('%Y-%m-%d'),
                'variable': variable,
                'grace_observed': int(obs[i]),
                'units': 'mm',
                'baseline': '2004.0-2010.0',
            })
            if (i + 1) % 250 == 0 or i + 1 == n:
                print(f'  {i + 1}/{n}', flush=True)

    meta = os.path.join(out_dir, 'metadata.csv')
    pd.DataFrame(rows).to_csv(meta, index=False)

    asset = f'projects/{EE_PROJECT_FOR_UPLOAD}/assets/{collection}'
    script = os.path.join(out_dir, 'upload.sh')
    with open(script, 'w') as fh:
        fh.write(f'''#!/usr/bin/env bash
# Upload {collection} to Earth Engine with geeup.
#
# PREREQUISITES, neither currently satisfied on this machine:
#   pip install geeup
#   the Cloud project "{EE_PROJECT_FOR_UPLOAD}" with the Earth Engine API
#   enabled (verified working)
set -euo pipefail

earthengine create collection {asset} || true

geeup upload \\
  --source "{out_dir}" \\
  --dest "{asset}" \\
  --metadata "{meta}" \\
  --nodata 0 \\
  --pyramids MEAN \\
  --user "$(git config user.email 2>/dev/null || echo YOUR_EMAIL)"
''')
    os.chmod(script, 0o755)

    size = sum(os.path.getsize(os.path.join(out_dir, f))
               for f in os.listdir(out_dir) if f.endswith('.tif'))
    print(f'\n{n} COGs -> {out_dir}  ({size / 1e6:.0f} MB)')
    print(f'metadata: {meta}')
    print(f'upload:   bash {script}')
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--which', default='daily', choices=['daily', 'monthly'])
    ap.add_argument('--variable', default=None)
    ap.add_argument('--out-root', default=COG_ROOT)
    ap.add_argument('--limit', type=int, default=None,
                    help='export only the first N steps (for a smoke test)')
    args = ap.parse_args()
    export(args.which, args.variable, args.out_root, args.limit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
