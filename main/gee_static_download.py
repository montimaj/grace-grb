"""
Download the static / annual covariates onto the 0.1 degree product grid.

Two kinds, handled differently on purpose.

CONTINUOUS images (`kind='image'`) -- water table depth, AWC, root depth -- are
area-averaged to 0.1 degrees. Averaging is meaningful for these.

CATEGORICAL land cover (`kind='annual_class'`) is NOT. Requesting the C3S class
field at anything coarser than its native 309 m makes Earth Engine average the
class CODES through its overview pyramids. Measured over the Ganga basin at
1 km that returns values like 13, 15, 18, 23 -- none of which exist in the LCCS
legend, and every one of which looks like ordinary data. At native 309 m the same
region returns only {10, 11, 20, 130, 190, 200, 210}, all valid.

So class covariates are ONE-HOT MASKED AT NATIVE SCALE first:

    img.eq(20)                              # boolean, at 309 m
       .reduceResolution(mean, maxPixels)   # -> fraction of the coarse cell
       .reproject(era5 0.1 deg grid)

which yields the fraction of each 0.1 degree cell occupied by that class. Multiply
by cell area (`downscale_grid_ops.cell_area_km2`) for km^2.

C3S runs 2000-2022. Years after that reuse 2022 rather than splicing in MODIS: a
sensor/scheme change at 2022 would land inside the forward-extrapolation period,
where GRACE is already absent and the product is least constrained, and the
forward holdout could not then separate covariate discontinuity from
extrapolation error. The reuse is recorded as `frozen_after` in the cube.

Usage
-----
    python gee_static_download.py             # everything missing
    python gee_static_download.py --vars wtd  # one covariate
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import gridded_config as cfg
from gee_gridded_download import NODATA, _fetch


def _static_image(spec: cfg.StaticSpec, grid: cfg.Grid, year: Optional[int]):
    """Build the 0.1 degree image for one covariate."""
    import ee

    if spec.kind == 'image_log':
        # Upstream area spans ~9 orders of magnitude (a single hillslope cell to
        # the entire Ganga), so the LOG is taken at native resolution BEFORE
        # averaging. Averaging first would let the few river cells dominate the
        # mean completely and erase all structure elsewhere.
        # MAX then log, not log then mean.
        #
        # Averaging log(upa) returns the GEOMETRIC MEAN over ~1,369 native cells,
        # almost all of which are hillslope with a tiny contributing area. The
        # handful of river cells that carry the signal are drowned: measured that
        # way the largest 0.1 deg cell in the Ganga basin came out at 0.46 km2,
        # for a river draining ~10^6. The meaningful quantity is the LARGEST
        # drainage passing through the cell, so reduce with max first.
        img = ee.Image(spec.asset).select(spec.band)
        out = (img.toFloat()
                  .reduceResolution(reducer=ee.Reducer.max(), maxPixels=4096,
                                    bestEffort=True)
                  .reproject(crs=grid.crs, crsTransform=grid.crs_transform)
                  .max(1e-3).log10())
    elif spec.kind == 'image':
        img = ee.Image(spec.asset).select(spec.band)
        # Continuous: area-average from native resolution to the product grid.
        out = (img.toFloat()
                  .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=4096,
                                    bestEffort=True)
                  .reproject(crs=grid.crs, crsTransform=grid.crs_transform))
    elif spec.kind in ('terrain', 'terrain_sd'):
        dem = ee.Image(spec.asset).select(spec.band)
        # Derive at NATIVE 30 m, aggregate afterwards. Deriving slope from an
        # already-coarsened DEM measures the slope of the 11 km surface, which is
        # near zero everywhere and would erase the Himalayan front.
        if spec.kind == 'terrain':
            # Slope is NOT computed in Earth Engine. ee.Terrain.slope on a global
            # 30 m mosaic reduced 370x to 0.1 deg exceeds the request limit, and
            # forcing an intermediate metre-scale projection onto an EPSG:4326
            # image fails too (400 Bad Request, both attempted).
            #
            # Instead this exports elevation on a 10x FINER grid (0.01 deg, ~1 km)
            # and `assemble_static` differentiates it locally, then averages to
            # 0.1 deg. Same result, fully under our control, and it keeps the
            # important property: slope is measured on a ~1 km surface, not on the
            # 11 km product grid, where the Himalayan front would flatten to
            # near zero.
            fine_grid = cfg.Grid(name='dem10', res=grid.res / 10,
                                 x_min=grid.x_min, y_max=grid.y_max,
                                 n_x=grid.n_x * 10, n_y=grid.n_y * 10)
            return (dem.toFloat()
                       .reduceResolution(ee.Reducer.mean(), maxPixels=4096,
                                         bestEffort=True)
                       .reproject(crs=fine_grid.crs,
                                  crsTransform=fine_grid.crs_transform)
                       .multiply(spec.scale).unmask(NODATA)), fine_grid
            reducer = ee.Reducer.mean()
        else:
            fine = dem.toFloat()
            reducer = ee.Reducer.stdDev()
        out = (fine.reduceResolution(reducer=reducer, maxPixels=65535,
                                     bestEffort=True)
                   .reproject(crs=grid.crs, crsTransform=grid.crs_transform))
    elif spec.kind == 'annual_class':
        src = (ee.ImageCollection(spec.asset)
               .filter(ee.Filter.stringContains('system:index', f'P1Y-{year}-'))
               .first())
        # ONE-HOT AT NATIVE SCALE. This is the whole point: never let the class
        # code itself be resampled.
        mask = ee.Image(src).select(0).eq(spec.class_value).toFloat()
        out = (mask.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=4096,
                                     bestEffort=True)
                   .reproject(crs=grid.crs, crsTransform=grid.crs_transform))
    else:
        raise ValueError(f'unknown kind {spec.kind!r}')

    return out.multiply(spec.scale).unmask(NODATA)


def download_one(spec: cfg.StaticSpec, grid: cfg.Grid,
                 year: Optional[int] = None, overwrite: bool = False) -> Tuple[str, str]:
    import ee

    path = cfg.static_tif_path(spec.name, year)
    if os.path.exists(path) and os.path.getsize(path) > 0 and not overwrite:
        return 'skip', path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    last: Optional[Exception] = None
    for attempt in range(cfg.MAX_RETRIES):
        try:
            built = _static_image(spec, grid, year)
            img, out_grid = built if isinstance(built, tuple) else (built, grid)
            url = img.getDownloadURL({
                'format': 'GEO_TIFF', 'crs': out_grid.crs,
                'crs_transform': out_grid.crs_transform,
                'dimensions': [out_grid.n_x, out_grid.n_y],
            })
            payload = _fetch(url)
            tmp = path + '.part'
            with open(tmp, 'wb') as fh:
                fh.write(payload)
            os.replace(tmp, path)
            return 'ok', path
        except Exception as err:  # noqa: BLE001 - retried, then surfaced
            last = err
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f'{spec.name} {year or ""}: failed after '
                       f'{cfg.MAX_RETRIES} attempts ({last})')


def plan(var_names: Optional[List[str]] = None
         ) -> List[Tuple[cfg.StaticSpec, Optional[int]]]:
    jobs: List[Tuple[cfg.StaticSpec, Optional[int]]] = []
    for spec in cfg.STATIC_VARS:
        if var_names and spec.name not in var_names:
            continue
        if spec.kind == 'annual_class':
            for year in range(cfg.C3S_FIRST_YEAR, cfg.C3S_LAST_YEAR + 1):
                jobs.append((spec, year))
        else:
            jobs.append((spec, None))
    return [(s, y) for s, y in jobs
            if not (os.path.exists(cfg.static_tif_path(s.name, y))
                    and os.path.getsize(cfg.static_tif_path(s.name, y)) > 0)]


def run(var_names: Optional[List[str]] = None, workers: int = 8,
        dry_run: bool = False) -> int:
    cfg.ensure_dirs()
    grid = cfg.build_grids()['era5']
    print('  ' + grid.describe())

    jobs = plan(var_names)
    print(f'{len(jobs)} covariate rasters missing')
    if dry_run or not jobs:
        return 0

    cfg.ee_initialize()
    failures: List[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_one, s, grid, y): (s.name, y) for s, y in jobs}
        for fut in as_completed(futs):
            name, year = futs[fut]
            done += 1
            try:
                fut.result()
            except Exception as err:  # noqa: BLE001
                failures.append(f'{name} {year or ""}: {err}')
            if done % 10 == 0 or done == len(jobs):
                print(f'  {done}/{len(jobs)} ({len(failures)} failed)', flush=True)

    if failures:
        for line in failures[:10]:
            print('  FAILED ' + line)
        return 1
    print('  complete')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vars', nargs='+', default=None,
                    choices=[v.name for v in cfg.STATIC_VARS])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    return run(args.vars, args.workers, args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
