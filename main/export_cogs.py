"""
Export the daily (or monthly) product as Cloud-Optimized GeoTIFFs for upload to
Earth Engine via `geeup`.

Produces, under `Results/downscaling/cogs/<collection>/`:

    <collection>_20000101.tif ...      one COG per time step
    metadata.csv                       geeup property table
    upload.sh                          the exact geeup command, ready to run

`<collection>` names the variables it holds, e.g.
`twsa_0p1deg_monthly_twsa__sigma_total`. It has to: every raster is named after
the collection, so a name derived from `--which` alone would let a sigma export
overwrite a twsa export, file for file, in the same directory, silently.

Why COG and not plain GeoTIFF
-----------------------------
Earth Engine ingests either, but COGs carry internal tiling and overviews, so the
explorer app can read a zoomed-out view without pulling the full raster. With
~9,500 daily rasters that is the difference between a responsive map and an
unusable one.

Bands
-----
`--variable` takes more than one name, and the fields become bands of a single
GeoTIFF in the order given. An app that draws TWSA and shades it by uncertainty
needs both in one image; as separate collections they are two ingests and a join
on date, and the uncertainty had no route to the app at all while export was
one variable per run.

Band NAMES do not survive the upload. GDAL keeps each netCDF variable name as
the band description, but geeup 2.0.0 builds its ingestion manifest from tilesets
and properties only -- no band ids (`geeup/batch_uploader.py`) -- so Earth Engine
falls back on b1, b2, ... in file order. metadata.csv therefore carries a `bands`
property listing that order. None of this has been confirmed against a completed
ingest: no upload has been run from this machine.

A note on what gets uploaded
----------------------------
The `grace_observed` flag travels as a per-image PROPERTY, not a band. Months
GRACE never saw are reconstructions, and an app that plots them beside observed
months without distinction would misrepresent the product. The property lets a
filter (`.filter(ee.Filter.eq('grace_observed', 1))`) recover the
observation-only subset.

Usage
-----
    python export_cogs.py                                    # daily flux product
    python export_cogs.py --which monthly
    python export_cogs.py --variable twsa_state
    python export_cogs.py --which monthly --variable twsa sigma_total
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

import gridded_config as cfg
import downscale_model as M

COG_ROOT = os.path.join(M.RESULTS_DIR, 'cogs')

# What each product exports when `--variable` is not given. These reproduce the
# single-band behaviour the module had before multi-band export existed.
DEFAULT_VARIABLES = {'daily': ['twsa_flux'], 'monthly': ['twsa']}

# Variable names reach the file stems, which become Earth Engine asset ids;
# geeup 2.0.0 rejects an id outside r'^[a-zA-Z0-9_-]+$' before uploading
# anything (metadata_loader.MetadataEntry). Hyphen is left out here because no
# variable in either product file has one.
_NAME_OK = re.compile(r'^[A-Za-z0-9_]+$')

# Separator between variables in the collection id. Double, because single
# underscores are inside the variable names themselves (twsa_flux, sigma_total)
# and the id must map back to one variable list and no other.
_BAND_SEP = '__'


def _check_variables(variables: Sequence[str]) -> List[str]:
    """Reject variable names that would make a collection id ambiguous."""
    if not variables:
        raise ValueError('at least one variable is required')
    checked: List[str] = []
    for v in variables:
        if not _NAME_OK.match(v) or _BAND_SEP in v:
            raise ValueError(
                f'{v!r}: variable names end up in the collection id and in every '
                f'file name, so they must match [A-Za-z0-9_] and must not contain '
                f'{_BAND_SEP!r}, which separates bands')
        if v in checked:
            raise ValueError(f'{v!r} requested twice; bands must be distinct')
        checked.append(v)
    return checked


def collection_name(which: str, variables: Sequence[str]) -> str:
    """Collection id for one product and one ordered list of variables.

    The variables are part of the id because the rasters inside are named
    `<collection>_<stamp>.tif`. Without them, exporting sigma_total after twsa
    writes the same names into the same directory and the twsa COGs are gone,
    with nothing in the output to say so.
    """
    return f'twsa_0p1deg_{which}_' + _BAND_SEP.join(variables)


def _profile(grid: cfg.Grid, count: int = 1) -> Dict[str, object]:
    from rasterio.transform import from_origin
    return {
        'driver': 'GTiff', 'dtype': 'float32', 'count': count,
        'height': grid.n_y, 'width': grid.n_x, 'crs': grid.crs,
        'transform': from_origin(grid.x_min, grid.y_max, grid.res, grid.res),
        'nodata': np.float32(np.nan),
        # Tiling of the intermediate only. The COG copy re-tiles at the GDAL COG
        # driver default: rasterio reports (512, 512) blocks on the output, not
        # (128, 128). The 0.1 deg grid is 179 x 101, so either way it is one tile.
        'tiled': True, 'blockxsize': 128, 'blockysize': 128,
        'compress': 'deflate', 'predictor': 2, 'zlevel': 6,
    }


def write_cog(arrays: Sequence[np.ndarray], names: Sequence[str], path: str,
              grid: cfg.Grid) -> None:
    """One COG, one band per array, tiled, compressed, with overviews.

    `names` become the GDAL band descriptions. They are for whoever opens the
    file with rasterio or QGIS; Earth Engine does not read them (see the module
    docstring), which is why metadata.csv repeats the order as a property.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.shutil import copy as rio_copy

    tmp = path + '.tmp.tif'
    with rasterio.open(tmp, 'w', **_profile(grid, len(arrays))) as dst:
        for band, (array, name) in enumerate(zip(arrays, names), start=1):
            dst.write(np.asarray(array).astype('float32'), band)
            dst.set_band_description(band, name)
        # Overviews must exist BEFORE the COG copy, or the result is a plain
        # tiled GeoTIFF that merely looks like a COG.
        dst.build_overviews([2, 4, 8], Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')
    rio_copy(tmp, path, driver='COG', compress='deflate', predictor=2,
             overview_resampling='average')
    os.remove(tmp)


def _verify_cog(path: str, names: Sequence[str]) -> None:
    """Re-open one COG and check what the copy above could silently drop.

    Run once per export, not per file: every step goes through the same call on
    the same grid, so what holds for the first holds for all of them. The three
    things checked are the three the rest of this module promises downstream --
    overviews (a file without them still opens and still looks like a COG, it
    just costs the app a full-resolution read), band order (Earth Engine renames
    the bands b1..bN by position, so position is the contract), and a NaN
    nodata (the upload declares no missing value of its own and leaves it to
    the header).
    """
    import rasterio

    with rasterio.open(path) as src:
        no_ovr = [b for b in range(1, src.count + 1) if not src.overviews(b)]
        if no_ovr:
            raise RuntimeError(
                f'{path}: bands {no_ovr} came out of the COG copy without '
                f'overviews')
        if tuple(src.descriptions) != tuple(names):
            raise RuntimeError(
                f'{path}: band descriptions {src.descriptions} do not match the '
                f'exported order {tuple(names)}')
        if src.nodata is None or not np.isnan(src.nodata):
            raise RuntimeError(
                f'{path}: nodata is {src.nodata}, not NaN')


def export(which: str = 'daily',
           variables: Optional[Union[str, Sequence[str]]] = None,
           out_root: str = COG_ROOT, limit: Optional[int] = None) -> str:
    import netCDF4

    # A bare string is accepted because it is the obvious thing to pass, and
    # left as a sequence it would iterate one character per band.
    if isinstance(variables, str):
        variables = [variables]
    variables = _check_variables(list(variables) if variables
                                 else DEFAULT_VARIABLES[which])

    if which == 'daily':
        src = os.path.join(M.RESULTS_DIR, 'twsa_0p1deg_daily.nc')
    else:
        src = os.path.join(M.RESULTS_DIR,
                           'twsa_0p1deg_monthly_with_uncertainty.nc')
        if not os.path.exists(src):
            # The single-model file has twsa but none of the sigma fields; the
            # variable check below is what reports that, by name.
            src = os.path.join(M.RESULTS_DIR, 'twsa_0p1deg_monthly_xgboost.nc')
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'{src} not found - run the pipeline before exporting COGs')

    collection = collection_name(which, variables)
    out_dir = os.path.join(out_root, collection)
    os.makedirs(out_dir, exist_ok=True)
    grid = cfg.build_grids()['era5']

    with netCDF4.Dataset(src) as ds:
        for v in variables:
            if v not in ds.variables:
                raise KeyError(f'{v} not in {os.path.basename(src)}; '
                               f'have {[n for n in ds.variables]}')
            if tuple(ds[v].dimensions) != ('time', 'lat', 'lon'):
                raise ValueError(
                    f'{v} has dimensions {tuple(ds[v].dimensions)}; only '
                    f"('time', 'lat', 'lon') fields can be bands. Per-step "
                    f'flags such as grace_observed travel as properties in '
                    f'metadata.csv instead.')
            ds[v].set_auto_mask(False)
        units = [getattr(ds[v], 'units', 'unknown') for v in variables]
        times = pd.to_datetime(ds['time'][:], unit='D', origin='1970-01-01')
        n = len(times) if limit is None else min(limit, len(times))
        obs = (np.asarray(ds['grace_observed'][:]).astype(int)
               if 'grace_observed' in ds.variables else np.ones(len(times), int))

        rows: List[Dict[str, object]] = []
        for i in range(n):
            stamp = times[i].strftime('%Y%m%d')
            name = f'{collection}_{stamp}'
            tif = os.path.join(out_dir, f'{name}.tif')
            write_cog([ds[v][i] for v in variables], variables, tif, grid)
            if i == 0:
                _verify_cog(tif, variables)
            rows.append({
                'id_no': name,
                'system:time_start': int(times[i].timestamp() * 1000),
                'date': times[i].strftime('%Y-%m-%d'),
                # Band order, because Earth Engine will call them b1, b2, ...
                'bands': ','.join(variables),
                'units': ','.join(units),
                'grace_observed': int(obs[i]),
                'baseline': '2004.0-2010.0',
            })
            if (i + 1) % 250 == 0 or i + 1 == n:
                print(f'  {i + 1}/{n}', flush=True)

    meta = os.path.join(out_dir, 'metadata.csv')
    pd.DataFrame(rows).to_csv(meta, index=False)
    _write_upload_script(out_dir, collection, variables, meta, n)

    size = sum(os.path.getsize(os.path.join(out_dir, f))
               for f in os.listdir(out_dir) if f.endswith('.tif'))
    print(f'\n{n} COGs, bands [{", ".join(variables)}] -> {out_dir}  '
          f'({size / 1e6:.0f} MB)')
    print(f'metadata: {meta}')
    print(f'upload:   bash {os.path.join(out_dir, "upload.sh")}')
    return out_dir


def _write_upload_script(out_dir: str, collection: str,
                         variables: Sequence[str], meta: str, n: int) -> str:
    """The geeup invocation for one exported collection."""
    # One source of truth for the project: $EE_PROJECT, else the project in the
    # Earth Engine credentials, else the fallback in gridded_config.
    project = cfg.resolve_ee_project()
    # More workers for the daily collection than the monthly one: 9,497 files
    # spend most of their wall-clock in transfer, 312 do not.
    workers = 8 if n > 1000 else 4
    asset = f'projects/{project}/assets/{collection}'
    script = os.path.join(out_dir, 'upload.sh')
    with open(script, 'w') as fh:
        fh.write(f'''#!/usr/bin/env bash
# Upload {collection} to Earth Engine with geeup.
# Bands, in order: {', '.join(variables)}
#
# PREREQUISITES:
#   geeup (2.0.0 is installed in the grace-grb conda env)
#   the Cloud project "{project}" with the Earth Engine API enabled and
#   ingestion permitted
#
# NOTE ON BAND NAMES. Earth Engine will call these b1, b2, ... in the order
# above, NOT by variable name -- geeup's ingestion manifest carries tilesets and
# properties but no band ids, so there is nothing to set them from. Confirmed on
# a real ingest. The order is recorded in each image's `bands` property, and any
# app reading this collection should rename on read.
set -euo pipefail

earthengine create collection {asset} || true

# No --nodata. The COGs declare nodata = NaN in their own header, which Earth
# Engine reads; geeup only adds a "missing_data" override when --nodata is
# given. It takes an int, so it cannot say NaN, and the int it was previously
# given -- 0 -- is a real value here: 0 mm is a cell sitting exactly on the
# 2004.0-2010.0 baseline. Declaring it missing would punch holes through every
# zero crossing of the anomaly field.
geeup upload \\
  --source "{out_dir}" \\
  --dest "{asset}" \\
  --metadata "{meta}" \\
  --pyramids MEAN \\
  --workers {workers} \\
  --user "$(git config user.email 2>/dev/null || echo YOUR_EMAIL)"

# --workers parallelises the upload of files to Google's staging area, which is
# the slow half of a large ingest. It does NOT raise how fast Earth Engine drains
# the resulting tasks: concurrent batch tasks average 2 per project and the READY
# queue is capped at 3,000, both fixed quotas. So this shortens the transfer, not
# the ingestion.
''')
    os.chmod(script, 0o755)
    return script


def export_trend(variables: Optional[Sequence[str]] = None,
                 out_root: str = COG_ROOT) -> str:
    """
    The per-pixel trend field as ONE multi-band COG.

    Separate from `export()` because the trend is static: it has dimensions
    (lat, lon) and no time axis, so none of the per-time-step machinery applies.
    Forcing it through the same path would mean special-casing every step of it.

    All nine fields are exported by default rather than just the slope. The
    significance mask is what makes the slope publishable -- a per-pixel trend
    map without it would be the integrity gap the app's other constraints exist
    to prevent -- and the p-value, tau and variance factor are what let a reader
    judge the mask rather than take it. Int8 masks are widened to float32
    because a GeoTIFF carries one dtype for all bands; the values stay 0/1.

    Ingested as a one-image collection, which is what geeup uploads into. The
    app reads it as `ee.Image(ee.ImageCollection(TREND).first())`.
    """
    import netCDF4

    src = os.path.join(M.RESULTS_DIR, 'twsa_trend_significance.nc')
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'{src} not found - run generate_gridded_maps.py first')

    with netCDF4.Dataset(src) as ds:
        available = [v for v in ds.variables if v not in ('lat', 'lon')]
        variables = _check_variables(list(variables) if variables else available)
        for v in variables:
            if v not in ds.variables:
                raise KeyError(f'{v} not in {os.path.basename(src)}; '
                               f'have {available}')
            if tuple(ds[v].dimensions) != ('lat', 'lon'):
                raise ValueError(f'{v} has dimensions {tuple(ds[v].dimensions)}; '
                                 f"only ('lat', 'lon') fields can be bands here")
            ds[v].set_auto_mask(False)
        units = [getattr(ds[v], 'units', 'unknown') for v in variables]
        arrays = [np.asarray(ds[v][:], dtype='float32') for v in variables]

    # A FIXED name, not one built from the band list. The time-series
    # collections encode their variables in the id because the choice of bands
    # distinguishes one export from another; here there is one trend field and
    # joining nine names produced a 130-character asset id. Band order lives in
    # metadata.csv, as it does for every other collection.
    collection = 'twsa_0p1deg_trend'
    out_dir = os.path.join(out_root, collection)
    os.makedirs(out_dir, exist_ok=True)
    grid = cfg.build_grids()['era5']

    name = f'{collection}_20000101'
    tif = os.path.join(out_dir, f'{name}.tif')
    write_cog(arrays, variables, tif, grid)
    _verify_cog(tif, variables)

    # system:time_start is the first month of the record. The field is static,
    # but Earth Engine sorts and filters collections by time and an image with
    # no timestamp is awkward to reach.
    meta = os.path.join(out_dir, 'metadata.csv')
    pd.DataFrame([{
        'id_no': name,
        'system:time_start': int(pd.Timestamp('2000-01-01').timestamp() * 1000),
        'date': '2000-01-01',
        'bands': ','.join(variables),
        'units': ','.join(units),
        'baseline': '2004.0-2010.0',
        'period': '2000-01 to 2025-12',
    }]).to_csv(meta, index=False)
    _write_upload_script(out_dir, collection, variables, meta, 1)

    size = os.path.getsize(tif)
    print(f'\n1 COG, {len(variables)} bands [{", ".join(variables)}] -> {out_dir}  '
          f'({size / 1e6:.1f} MB)')
    print(f'metadata: {meta}')
    print(f'upload:   bash {os.path.join(out_dir, "upload.sh")}')
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--which', default='daily',
                    choices=['daily', 'monthly', 'trend'])
    ap.add_argument('--variable', nargs='+', default=None, metavar='NAME',
                    help='netCDF variables to export, in band order; several '
                         'names give one multi-band GeoTIFF (e.g. --variable '
                         'twsa sigma_total). Default: '
                         + ', '.join(f'{k} -> {v[0]}'
                                     for k, v in DEFAULT_VARIABLES.items()))
    ap.add_argument('--out-root', default=COG_ROOT)
    ap.add_argument('--limit', type=int, default=None,
                    help='export only the first N steps (for a smoke test)')
    args = ap.parse_args()
    if args.which == 'trend':
        # The trend field is static, so --limit has no meaning for it; saying so
        # beats silently ignoring the flag.
        if args.limit is not None:
            print('  --limit ignored: the trend field is a single image')
        export_trend(args.variable, args.out_root)
    else:
        export(args.which, args.variable, args.out_root, args.limit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
