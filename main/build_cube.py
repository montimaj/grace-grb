"""
Assemble the downloaded GeoTIFF stacks into analysis-ready netCDF cubes.

Produces, under `Data/Gridded/cubes/`:

  era5_cube.nc    (time: daily, lat: 101, lon: 179)   13 ERA5-Land fields
  gldas_cube.nc   (time: daily, lat:  42, lon:  74)   10 GLDAS CLSM fields
  grace_cube.nc   (time: monthly, lat: 22, lon: 39)   GRACE mascon TWSA + uncertainty
  grids_aux.nc    static geometry shared by everything downstream

Band ordering
-------------
Earth Engine's GeoTIFF export does not preserve band descriptions, so day
identity rests on band ORDER, which `gee_gridded_download._month_stack` fixes as
d01..dN. Every file is checked against the calendar length of its month before
being written; a mismatch aborts rather than silently shifting a month's dates.

The mascon partition
--------------------
JPL mascons are 3 degree equal-area cells served on a 0.5 degree grid, so many
grid cells carry byte-identical values. We recover the true partition by
grouping 0.5 degree cells whose *entire time series* is identical. This yields
the exact number of independent GRACE observations in the domain -- 20 over the
Ganga basin -- which is what the spatial cross-validation must be blocked on.
Treating the 364 grid cells as independent would overstate the sample by ~18x.

Usage
-----
    python build_cube.py              # build everything that is missing
    python build_cube.py --force      # rebuild from scratch
    python build_cube.py --aux-only   # rebuild only grids_aux.nc
"""

from __future__ import annotations

import argparse
import calendar
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import gridded_config as cfg
# Areas and regridding live in downscale_grid_ops; build_cube reuses them
# rather than keeping a second implementation that could drift.
from downscale_grid_ops import cell_area_km2

warnings.filterwarnings('ignore', message='.*TIFFReadDirectory.*')

EARTH_RADIUS_KM = 6371.0088
FILL = np.float32(np.nan)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def basin_fraction(grid: cfg.Grid, oversample: int = 16) -> np.ndarray:
    """
    Fraction of each cell's area lying inside the basin polygon, in [0, 1].

    Computed by rasterising the polygon `oversample`x finer than the target grid
    and block-averaging. A binary centre-in-polygon test would drop ~10% of the
    basin's area at 0.5 degrees, which matters when we force the downscaled
    field to conserve mass against each mascon.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    gdf = cfg.load_basin()
    fine_res = grid.res / oversample
    transform = from_origin(grid.x_min, grid.y_max, fine_res, fine_res)
    fine = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(grid.n_y * oversample, grid.n_x * oversample),
        transform=transform, fill=0, dtype='uint8', all_touched=False,
    )
    return (fine.reshape(grid.n_y, oversample, grid.n_x, oversample)
                .mean(axis=(1, 3)).astype('float32'))


def parent_index(fine: cfg.Grid, coarse: cfg.Grid) -> np.ndarray:
    """
    For each fine cell, the flat index of the coarse cell containing its centre.

    Shape (fine.n_y, fine.n_x), values in [0, coarse.n_cells). Used to aggregate
    0.1 degree predictions up to mascon scale and to broadcast mascon-level
    corrections back down.
    """
    lon = fine.lon_centers()
    lat = fine.lat_centers()
    col = np.floor((lon - coarse.x_min) / coarse.res).astype(int)
    row = np.floor((coarse.y_max - lat) / coarse.res).astype(int)
    if col.min() < 0 or col.max() >= coarse.n_x or row.min() < 0 or row.max() >= coarse.n_y:
        raise ValueError(
            f'{fine.name} grid is not fully covered by {coarse.name}; '
            f'increase pad_cells in gridded_config.build_grids()'
        )
    return (row[:, None] * coarse.n_x + col[None, :]).astype('int32')


def mascon_labels(tws: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Recover the JPL mascon partition from a stack of GRACE grid fields.

    `tws` has shape (n_time, n_y, n_x). Cells belonging to one 3 degree mascon
    carry identical values at every time step, so grouping cells by their full
    time series recovers the partition exactly.

    Returns (labels, n_mascons); labels is (n_y, n_x) with -1 for all-NaN cells.
    """
    n_t, n_y, n_x = tws.shape
    flat = tws.reshape(n_t, -1).T                      # (n_cells, n_time)
    labels = np.full(flat.shape[0], -1, dtype='int32')
    valid = ~np.isnan(flat).all(axis=1)

    # Round to a precision far finer than the signal but coarser than float
    # noise, so bit-level differences do not split a mascon.
    keys = np.round(np.nan_to_num(flat[valid], nan=-9e9), 6)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    labels[valid] = inverse.astype('int32')
    return labels.reshape(n_y, n_x), int(inverse.max()) + 1 if valid.any() else 0


# --------------------------------------------------------------------------
# Reading the GeoTIFF stacks
# --------------------------------------------------------------------------

def read_month(spec: cfg.VariableSpec, month: str, grid: cfg.Grid) -> np.ndarray:
    """
    One month of one variable as (n_steps, n_y, n_x) float32 with NaN gaps.

    Asserts the band count matches the calendar, since day identity depends on
    band order (see module docstring).
    """
    import rasterio

    path = cfg.tif_path(spec.grid, spec.name, month)
    if not os.path.exists(path):
        raise FileNotFoundError(f'missing {path} - run gee_gridded_download.py')

    year, mon = (int(x) for x in month.split('-'))
    expected = 1 if spec.grid == 'grace' else calendar.monthrange(year, mon)[1]

    with rasterio.open(path) as src:
        if src.count != expected:
            raise ValueError(
                f'{path}: {src.count} bands but {expected} expected for {month}. '
                f'Delete the file and re-download.'
            )
        if (src.height, src.width) != grid.shape:
            raise ValueError(
                f'{path}: shape {(src.height, src.width)} != grid {grid.shape}'
            )
        arr = src.read().astype('float32')

    arr[arr <= -9998.0] = np.nan
    return arr


# --------------------------------------------------------------------------
# Cube construction
# --------------------------------------------------------------------------

def build_grid_cube(
    grid_key: str,
    grid: cfg.Grid,
    months: List[str],
    force: bool = False,
) -> str:
    """Assemble one grid's variables into a compressed netCDF cube."""
    import netCDF4

    out = cfg.cube_path(grid_key)
    if os.path.exists(out) and not force:
        print(f'  {os.path.basename(out)} exists (use --force to rebuild)')
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)

    specs = cfg.vars_for_grid(grid_key)
    monthly = grid_key == 'grace'
    if monthly:
        times = pd.to_datetime([f'{m}-01' for m in months])
        steps_per_month = [1] * len(months)
    else:
        times = pd.date_range(cfg.START_DATE, cfg.END_DATE, freq='D')
        steps_per_month = [calendar.monthrange(*(int(x) for x in m.split('-')))[1]
                           for m in months]
    offsets = np.concatenate([[0], np.cumsum(steps_per_month)])
    if offsets[-1] != len(times):
        raise ValueError(f'{grid_key}: {offsets[-1]} steps assembled vs {len(times)} expected')

    tmp = out + '.part'
    with netCDF4.Dataset(tmp, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', len(times))
        ds.createDimension('lat', grid.n_y)
        ds.createDimension('lon', grid.n_x)

        tv = ds.createVariable('time', 'f8', ('time',))
        tv.units = 'days since 1970-01-01'
        tv.calendar = 'standard'
        tv[:] = (times - pd.Timestamp('1970-01-01')).days.to_numpy()

        for name, vals, units, ln in [
            ('lat', grid.lat_centers(), 'degrees_north', 'latitude of cell centre'),
            ('lon', grid.lon_centers(), 'degrees_east', 'longitude of cell centre'),
        ]:
            cv = ds.createVariable(name, 'f8', (name,))
            cv.units, cv.long_name = units, ln
            cv[:] = vals

        ds.title = f'Gridded {grid_key} cube for GRACE spatial downscaling'
        ds.source_collection = cfg.COLLECTIONS[grid_key]
        ds.grid_transform = str(grid.crs_transform)
        ds.crs = grid.crs
        ds.created_by = 'main/build_cube.py'
        ds.history = ('Downloaded on the native lattice by gee_gridded_download.py; '
                      'no Earth Engine resampling applied.')

        for spec in specs:
            var = ds.createVariable(
                spec.name, 'f4', ('time', 'lat', 'lon'),
                zlib=True, complevel=4, shuffle=True,
                chunksizes=(min(64, len(times)), grid.n_y, grid.n_x),
                fill_value=np.float32(np.nan),
            )
            var.units = spec.out_units
            var.long_name = spec.long_name
            var.source_band = spec.band
            var.scale_applied = spec.scale
            var.offset_applied = spec.offset
            if spec.note:
                var.provenance_note = spec.note

            for i, month in enumerate(months):
                var[offsets[i]:offsets[i + 1], :, :] = read_month(spec, month, grid)
            print(f'    {spec.name:18s} {spec.out_units:8s} written', flush=True)

    os.replace(tmp, out)
    size_gb = os.path.getsize(out) / 1e9
    print(f'  {os.path.basename(out)}  {len(specs)} vars, '
          f'{len(times)} steps, {size_gb:.2f} GB')
    return out


def build_aux(grids: Dict[str, cfg.Grid], force: bool = False) -> str:
    """
    Static geometry: cell areas, basin fractions, mascon partition, and the
    fine->coarse parent maps that mass conservation depends on.
    """
    import netCDF4

    out = os.path.join(cfg.CUBE_DIR, 'grids_aux.nc')
    if os.path.exists(out) and not force:
        print(f'  {os.path.basename(out)} exists (use --force to rebuild)')
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)

    grace_cube = cfg.cube_path('grace')
    if not os.path.exists(grace_cube):
        raise FileNotFoundError('build the grace cube before grids_aux.nc')
    with netCDF4.Dataset(grace_cube) as ds:
        tws = np.asarray(ds['tws'][:])
    labels, n_mascons = mascon_labels(tws)

    tmp = out + '.part'
    with netCDF4.Dataset(tmp, 'w', format='NETCDF4') as ds:
        ds.title = 'Static grid geometry for GRACE spatial downscaling'
        ds.created_by = 'main/build_cube.py'
        ds.n_mascons = n_mascons

        for key, grid in grids.items():
            ds.createDimension(f'lat_{key}', grid.n_y)
            ds.createDimension(f'lon_{key}', grid.n_x)
            dims = (f'lat_{key}', f'lon_{key}')

            for name, vals, units, ln in [
                (f'lat_{key}', grid.lat_centers(), 'degrees_north', 'latitude'),
                (f'lon_{key}', grid.lon_centers(), 'degrees_east', 'longitude'),
            ]:
                cv = ds.createVariable(name, 'f8', (name,))
                cv.units, cv.long_name = units, ln
                cv[:] = vals

            area = ds.createVariable(f'area_{key}', 'f4', dims, zlib=True)
            area.units = 'km2'
            area.long_name = 'exact spherical cell area'
            area[:] = cell_area_km2(grid)

            frac = ds.createVariable(f'basin_frac_{key}', 'f4', dims, zlib=True)
            frac.units = '1'
            frac.long_name = 'fraction of cell area inside the Ganga basin'
            frac.method = 'polygon rasterised 16x finer than the grid, block-averaged'
            frac[:] = basin_fraction(grid)

        lab = ds.createVariable('mascon_id', 'i4', ('lat_grace', 'lon_grace'), zlib=True)
        lab.long_name = 'JPL 3-degree mascon partition recovered from identical time series'
        lab.description = ('0.5 degree GRACE cells sharing a byte-identical time series '
                           'belong to one 3 degree mascon; -1 marks all-NaN cells.')
        lab.n_mascons = n_mascons
        lab[:] = labels

        for fine_key in ('era5', 'gldas'):
            for coarse_key in ('grace',):
                pv = ds.createVariable(
                    f'parent_{fine_key}_to_{coarse_key}', 'i4',
                    (f'lat_{fine_key}', f'lon_{fine_key}'), zlib=True,
                )
                pv.long_name = (f'flat index into the {coarse_key} grid of the cell '
                                f'containing each {fine_key} cell centre')
                pv[:] = parent_index(grids[fine_key], grids[coarse_key])

    os.replace(tmp, out)

    in_basin = basin_fraction(grids['grace']) > 0
    n_basin_mascons = len(np.unique(labels[in_basin & (labels >= 0)]))
    print(f'  grids_aux.nc  {n_mascons} mascons in the domain, '
          f'{n_basin_mascons} intersecting the basin')
    return out


def build_static_cube(force: bool = False) -> str:
    """
    Assemble the static / annual covariates into `static_cube.nc` (0.1 deg grid).

    Static covariates get one 2-D field. Annual ones (C3S cropland) get a
    (year, lat, lon) stack for C3S_FIRST_YEAR..C3S_LAST_YEAR; later years reuse
    the last available one, recorded as `frozen_after` so the reuse is visible.
    Splicing MODIS in at 2022 instead would put a sensor/scheme discontinuity
    inside the forward-extrapolation period, where GRACE is already absent.
    """
    import netCDF4
    import rasterio

    out = os.path.join(cfg.CUBE_DIR, 'static_cube.nc')
    if os.path.exists(out) and not force:
        print(f'  {os.path.basename(out)} exists (use --force to rebuild)')
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)

    grid = cfg.build_grids()['era5']
    years = list(range(cfg.C3S_FIRST_YEAR, cfg.C3S_LAST_YEAR + 1))

    def read(path):
        with rasterio.open(path) as src:
            a = src.read(1).astype('float32')
        a[a <= -9998.0] = np.nan
        return a

    tmp = out + '.part'
    with netCDF4.Dataset(tmp, 'w', format='NETCDF4') as ds:
        ds.createDimension('lat', grid.n_y)
        ds.createDimension('lon', grid.n_x)
        ds.createDimension('year', len(years))
        for nm, vals, un in [('lat', grid.lat_centers(), 'degrees_north'),
                             ('lon', grid.lon_centers(), 'degrees_east')]:
            cv = ds.createVariable(nm, 'f8', (nm,)); cv.units = un; cv[:] = vals
        ds.createVariable('year', 'i4', ('year',))[:] = years

        for spec in cfg.STATIC_VARS:
            if spec.kind == 'annual_class':
                v = ds.createVariable(spec.name, 'f4', ('year', 'lat', 'lon'),
                                      zlib=True, complevel=4,
                                      fill_value=np.float32(np.nan))
                v[:] = np.stack([read(cfg.static_tif_path(spec.name, y))
                                 for y in years])
                v.frozen_after = cfg.C3S_LAST_YEAR
            else:
                v = ds.createVariable(spec.name, 'f4', ('lat', 'lon'),
                                      zlib=True, complevel=4,
                                      fill_value=np.float32(np.nan))
                v[:] = read(cfg.static_tif_path(spec.name))
            v.units = spec.out_units
            v.long_name = spec.long_name
            v.source_asset = spec.asset
            if spec.note:
                v.provenance_note = spec.note

        ds.title = 'Static and annual covariates, 0.1 degree Ganga grid'
        ds.created_by = 'main/build_cube.py:build_static_cube'
        ds.caveat = ('Categorical land cover was one-hot masked at its native '
                     '309 m and only then averaged. Requesting the class field '
                     'at 10 km makes Earth Engine average class CODES, returning '
                     'values absent from the LCCS legend.')
    os.replace(tmp, out)
    n_ann = sum(1 for v in cfg.STATIC_VARS if v.kind == 'annual_class')
    print(f'  static_cube.nc  {len(cfg.STATIC_VARS) - n_ann} static + '
          f'{n_ann} annual x {len(years)} yr')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true', help='rebuild existing cubes')
    ap.add_argument('--grids', nargs='+', choices=list(cfg.LATTICE), default=None)
    ap.add_argument('--aux-only', action='store_true')
    args = ap.parse_args()

    cfg.ensure_dirs()
    grids = cfg.build_grids()
    months = cfg.month_range()

    if not args.aux_only:
        # GRACE first: grids_aux.nc derives the mascon partition from it.
        order = [k for k in ('grace', 'gldas', 'era5') if k in (args.grids or cfg.LATTICE)]
        for key in order:
            print(f'\n{key}:')
            build_grid_cube(key, grids[key], months, force=args.force)

    print('\nstatic covariates:')
    build_static_cube(force=args.force)

    print('\naux:')
    build_aux(grids, force=args.force)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
