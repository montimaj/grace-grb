"""
Gridded Earth Engine download for the spatial GRACE downscaling pipeline.

Where `gee_download.py` reduces every field to one basin-mean number per time
step, this module keeps the fields gridded. Each dataset is downloaded on its
OWN native lattice (ERA5-Land 0.1 deg, GLDAS 0.25 deg, GRACE 0.5 deg) with an
explicit `crsTransform`, so Earth Engine never resamples on our behalf and the
regridding decisions stay in `build_cube.py` where they can be inspected.

Layout: one GeoTIFF per (grid, variable, month), with one band per day named
`d01 .. d31`. Over the Ganga basin a month of one ERA5-Land field is
179 x 101 x 31 x 4 B ~= 2.2 MB, far under Earth Engine's ~48 MB download cap,
and each file is independently resumable.

Masked pixels and absent days are written as the sentinel NODATA (-9999) rather
than silently becoming zero; `build_cube.py` converts the sentinel back to NaN.

Usage
-----
    python gee_gridded_download.py                 # everything, resumable
    python gee_gridded_download.py --grids grace   # one grid only
    python gee_gridded_download.py --vars ppt et   # named fields only
    python gee_gridded_download.py --dry-run       # report what is missing
"""

from __future__ import annotations

import argparse
import calendar
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests

import gridded_config as cfg

NODATA = -9999.0


# --------------------------------------------------------------------------
# Earth Engine image construction
# --------------------------------------------------------------------------

def _daily_band(ic, spec: cfg.VariableSpec, start: str, end: str, band_name: str):
    """
    One day of one variable as a single-band image, in output units.

    Days with no source image yield a fully masked band rather than an error, so
    a month with a gap (GRACE has several) still produces a well-formed stack.
    """
    import ee

    sub = ic.filterDate(start, end).select(spec.band)
    reducer = {'mean': sub.mean, 'sum': sub.sum, 'first': sub.first}[spec.reducer]
    img = ee.Image(
        ee.Algorithms.If(sub.size().gt(0), reducer(), ee.Image.constant(0).selfMask())
    )
    return (ee.Image(img)
            .multiply(spec.scale)
            .add(spec.offset)
            .rename(band_name)
            .toFloat())


def _month_stack(spec: cfg.VariableSpec, month: str):
    """
    A month of one variable as a multi-band image.

    Daily grids get one band per day (`d01 .. d31`). GRACE is monthly, so it
    gets a single band `m01` covering the whole month.
    """
    import ee

    ic = ee.ImageCollection(cfg.COLLECTIONS[spec.grid])
    year, mon = (int(x) for x in month.split('-'))
    n_days = calendar.monthrange(year, mon)[1]
    nxt = f'{year + 1}-01-01' if mon == 12 else f'{year}-{mon + 1:02d}-01'

    if spec.grid == 'grace':
        return _daily_band(ic, spec, f'{month}-01', nxt, 'm01').unmask(NODATA)

    bands = []
    for day in range(1, n_days + 1):
        d0 = f'{year}-{mon:02d}-{day:02d}'
        d1 = nxt if day == n_days else f'{year}-{mon:02d}-{day + 1:02d}'
        bands.append(_daily_band(ic, spec, d0, d1, f'd{day:02d}'))

    stack = bands[0]
    for band in bands[1:]:
        stack = stack.addBands(band)
    return stack.unmask(NODATA)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

_RETRYABLE = (
    requests.exceptions.RequestException,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    OSError,
)


def _fetch(url: str, timeout: int = 300) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.content
    # Earth Engine sometimes wraps a GEO_TIFF response in a zip archive.
    if payload[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith('.tif')]
            if not names:
                raise ValueError(f'zip response contained no GeoTIFF: {zf.namelist()}')
            payload = zf.read(names[0])
    return payload


def download_one(
    spec: cfg.VariableSpec,
    month: str,
    grid: cfg.Grid,
    overwrite: bool = False,
) -> Tuple[str, str]:
    """
    Download one (variable, month) GeoTIFF. Returns (status, path).

    Status is 'skip' if the file already exists, 'ok' on success. Failures raise
    after `cfg.MAX_RETRIES` attempts so the caller can report them rather than
    leaving a silently truncated archive.
    """
    import ee

    path = cfg.tif_path(spec.grid, spec.name, month)
    if os.path.exists(path) and os.path.getsize(path) > 0 and not overwrite:
        return 'skip', path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    last_err: Optional[Exception] = None
    for attempt in range(cfg.MAX_RETRIES):
        try:
            image = _month_stack(spec, month)
            url = image.getDownloadURL({
                'format': 'GEO_TIFF',
                'crs': grid.crs,
                'crs_transform': grid.crs_transform,
                'dimensions': [grid.n_x, grid.n_y],
            })
            payload = _fetch(url)
            tmp = path + '.part'
            with open(tmp, 'wb') as fh:
                fh.write(payload)
            os.replace(tmp, path)
            return 'ok', path
        except (ee.EEException, ValueError, *_RETRYABLE) as err:  # noqa: B014
            last_err = err
            time.sleep(min(2 ** attempt, 60))

    raise RuntimeError(f'{spec.name} {month}: failed after '
                       f'{cfg.MAX_RETRIES} attempts ({last_err})')


def plan(
    grids: Dict[str, cfg.Grid],
    grid_keys: List[str],
    var_names: Optional[List[str]],
    months: List[str],
) -> List[Tuple[cfg.VariableSpec, str]]:
    """Every (variable, month) pair that is still missing from disk."""
    jobs: List[Tuple[cfg.VariableSpec, str]] = []
    for key in grid_keys:
        for spec in cfg.vars_for_grid(key):
            if var_names and spec.name not in var_names:
                continue
            for month in months:
                path = cfg.tif_path(spec.grid, spec.name, month)
                if not (os.path.exists(path) and os.path.getsize(path) > 0):
                    jobs.append((spec, month))
    return jobs


def run(
    grid_keys: Optional[List[str]] = None,
    var_names: Optional[List[str]] = None,
    start: str = cfg.START_DATE,
    end: str = cfg.END_DATE,
    workers: int = cfg.DOWNLOAD_JOBS,
    dry_run: bool = False,
) -> int:
    cfg.ensure_dirs()
    grids = cfg.build_grids()
    grid_keys = grid_keys or list(cfg.LATTICE)
    months = cfg.month_range(start, end)

    for key in grid_keys:
        print('  ' + grids[key].describe())
    print(f'  period {start} -> {end}  ({len(months)} months)\n')

    jobs = plan(grids, grid_keys, var_names, months)
    total = sum(len(cfg.vars_for_grid(k)) if not var_names
                else len([v for v in cfg.vars_for_grid(k) if v.name in var_names])
                for k in grid_keys) * len(months)
    print(f'{len(jobs)} of {total} files missing')
    if dry_run or not jobs:
        return 0

    cfg.ee_initialize()

    done = 0
    failures: List[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, spec, month, grids[spec.grid]): (spec.name, month)
            for spec, month in jobs
        }
        for fut in as_completed(futures):
            name, month = futures[fut]
            done += 1
            try:
                fut.result()
            except Exception as err:  # noqa: BLE001 - reported, not swallowed
                failures.append(f'{name} {month}: {err}')
            if done % 50 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9)
                print(f'  {done}/{len(jobs)}  ({rate:.1f}/s, ETA {eta / 60:.1f} min, '
                      f'{len(failures)} failed)', flush=True)

    if failures:
        print(f'\n{len(failures)} downloads FAILED:', file=sys.stderr)
        for line in failures[:20]:
            print('  ' + line, file=sys.stderr)
        if len(failures) > 20:
            print(f'  ... and {len(failures) - 20} more', file=sys.stderr)
        print('Re-run to retry only the missing files.', file=sys.stderr)
        return 1

    print(f'\nComplete in {(time.time() - t0) / 60:.1f} min')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grids', nargs='+', choices=list(cfg.LATTICE), default=None,
                    help='restrict to these native grids')
    ap.add_argument('--vars', nargs='+', default=None,
                    help='restrict to these variable names')
    ap.add_argument('--start', default=cfg.START_DATE)
    ap.add_argument('--end', default=cfg.END_DATE)
    ap.add_argument('--workers', type=int, default=cfg.DOWNLOAD_JOBS)
    ap.add_argument('--dry-run', action='store_true',
                    help='report what is missing without downloading')
    args = ap.parse_args()

    return run(grid_keys=args.grids, var_names=args.vars, start=args.start,
               end=args.end, workers=args.workers, dry_run=args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
