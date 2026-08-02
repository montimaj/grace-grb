"""
Grid operators for the spatial GRACE downscaling pipeline.

Everything here is geometry: moving fields between the 0.1 deg (ERA5-Land),
0.25 deg (GLDAS) and 0.5 deg (GRACE) lattices, and aggregating 0.1 deg cells
into the 3 deg mascons that GRACE actually observes.

Why an exact area-overlap operator
----------------------------------
The lattices are NOT nested. ERA5-Land cell edges fall on x.x5 (its centres are
on multiples of 0.1), while GRACE edges fall on multiples of 0.5. So one fine
cell in every five straddles a coarse boundary. Assigning each fine cell wholly
to the coarse cell containing its centre misallocates half of those cells' area
-- about 4% of the domain's area lands in the wrong mascon.

That matters because the downscaled product is forced to conserve mass against
each mascon. If the aggregation operator used for the correction disagrees with
the true geometry, the "conserved" product does not actually reproduce GRACE.
`overlap_matrix` therefore computes exact rectangular area overlaps, which is
cheap: axis-aligned lon/lat boxes make the 2-D overlap a product of two 1-D
overlaps, and each fine cell touches at most four coarse cells.

Areas are spherical (R^2 * dlon * [sin(lat_n) - sin(lat_s)]), not planar, so
the Himalayan rows are not over-weighted relative to the delta.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import sparse

import gridded_config as cfg

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------
# Areas
# --------------------------------------------------------------------------

def cell_area_km2(grid: cfg.Grid) -> np.ndarray:
    """Exact spherical cell area, shape (n_y, n_x), km^2."""
    lat_n = np.deg2rad(grid.y_max - grid.res * np.arange(grid.n_y))
    lat_s = np.deg2rad(grid.y_max - grid.res * (np.arange(grid.n_y) + 1))
    band = EARTH_RADIUS_KM ** 2 * np.deg2rad(grid.res) * (np.sin(lat_n) - np.sin(lat_s))
    return np.repeat(band[:, None], grid.n_x, axis=1)


def _edges(start: float, res: float, n: int, descending: bool = False) -> np.ndarray:
    """Cell edges along one axis, length n + 1."""
    step = -res if descending else res
    return start + step * np.arange(n + 1)


def _overlap_1d(a0: np.ndarray, a1: np.ndarray,
                b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """
    Pairwise 1-D interval overlap length, shape (len(a0), len(b0)).

    Intervals are given as lower/upper bounds already sorted ascending.
    """
    lo = np.maximum(a0[:, None], b0[None, :])
    hi = np.minimum(a1[:, None], b1[None, :])
    return np.clip(hi - lo, 0.0, None)


def overlap_matrix(fine: cfg.Grid, coarse: cfg.Grid) -> sparse.csr_matrix:
    """
    Exact area-overlap operator, shape (coarse.n_cells, fine.n_cells).

    Entry (c, f) is the spherical area in km^2 shared by coarse cell c and fine
    cell f. Row sums give each coarse cell's area restricted to the fine grid's
    footprint; column sums give each fine cell's total area.

    Aggregation of a fine field to coarse is then a matrix product followed by
    normalisation -- see `aggregate_area_weighted`.
    """
    # Longitude overlap in degrees (linear in lon, so no spherical correction).
    fx = _edges(fine.x_min, fine.res, fine.n_x)
    cx = _edges(coarse.x_min, coarse.res, coarse.n_x)
    ov_lon = _overlap_1d(fx[:-1], fx[1:], cx[:-1], cx[1:])          # (n_x_f, n_x_c)

    # Latitude overlap expressed as a difference of sines, which converts the
    # lon-degree overlap into true spherical area when multiplied through.
    fy = _edges(fine.y_max, fine.res, fine.n_y, descending=True)
    cy = _edges(coarse.y_max, coarse.res, coarse.n_y, descending=True)
    fs = np.sin(np.deg2rad(fy))
    cs = np.sin(np.deg2rad(cy))
    # Rows run north -> south, so the "upper" bound in sine space is index i.
    ov_lat = _overlap_1d(fs[1:], fs[:-1], cs[1:], cs[:-1])          # (n_y_f, n_y_c)

    scale = EARTH_RADIUS_KM ** 2 * np.deg2rad(1.0)

    rows, cols, vals = [], [], []
    for jf in range(fine.n_y):
        jc = np.flatnonzero(ov_lat[jf])
        if jc.size == 0:
            continue
        for if_ in range(fine.n_x):
            ic = np.flatnonzero(ov_lon[if_])
            if ic.size == 0:
                continue
            area = scale * np.outer(ov_lat[jf, jc], ov_lon[if_, ic])
            f_flat = jf * fine.n_x + if_
            c_flat = (jc[:, None] * coarse.n_x + ic[None, :]).ravel()
            rows.append(c_flat)
            cols.append(np.full(c_flat.size, f_flat))
            vals.append(area.ravel())

    mat = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(coarse.n_cells, fine.n_cells),
    )
    mat.eliminate_zeros()
    return mat


def aggregate_area_weighted(
    field: np.ndarray,
    weights: sparse.csr_matrix,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Area-weighted aggregation of a fine field onto the coarse grid.

    Parameters
    ----------
    field : (n_time, n_y_f, n_x_f) or (n_y_f, n_x_f)
        Fine-resolution field. NaNs are excluded from both numerator and
        denominator, so a partially masked coarse cell still returns the mean of
        the fine cells that do exist rather than NaN.
    weights : (n_coarse, n_fine) area-overlap matrix from `overlap_matrix`
    valid : optional (n_y_f, n_x_f) mask, e.g. basin fraction > 0. Cells outside
        the mask contribute nothing.

    Returns
    -------
    (n_time, n_coarse) or (n_coarse,) array of area-weighted means.
    """
    squeeze = field.ndim == 2
    flat = field.reshape(1, -1) if squeeze else field.reshape(field.shape[0], -1)

    ok = np.isfinite(flat)
    if valid is not None:
        ok &= valid.reshape(1, -1).astype(bool)

    filled = np.where(ok, flat, 0.0)
    num = weights.dot(filled.T).T                  # (n_time, n_coarse)
    den = weights.dot(ok.astype('float64').T).T
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(den > 0, num / den, np.nan)
    return out[0] if squeeze else out


def disaggregate(coarse_values: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """
    Broadcast a coarse field onto the fine grid by parent lookup (piecewise
    constant). `parent` is the (n_y_f, n_x_f) index map from build_cube.

    Used for the mass-conservation correction before smoothing; the blocky
    result is deliberate at this stage.
    """
    squeeze = coarse_values.ndim == 1
    vals = coarse_values[None, :] if squeeze else coarse_values
    out = vals[:, parent.ravel()].reshape((vals.shape[0],) + parent.shape)
    return out[0] if squeeze else out


# --------------------------------------------------------------------------
# Mascon aggregation
# --------------------------------------------------------------------------

def mascon_weights(
    fine: cfg.Grid,
    grace: cfg.Grid,
    mascon_id: np.ndarray,
    basin_frac_fine: Optional[np.ndarray] = None,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Area-overlap operator from the fine grid straight onto mascons.

    Composes `overlap_matrix(fine, grace)` with the 0.5 deg -> mascon grouping
    recovered in build_cube, so a fine cell straddling two 0.5 deg cells of the
    SAME mascon contributes its full area once, while one straddling a genuine
    mascon boundary is split correctly between the two.

    Parameters
    ----------
    mascon_id : (n_y_c, n_x_c) integer labels, -1 for cells with no mascon.
    basin_frac_fine : optional (n_y_f, n_x_f) in [0, 1]; when supplied, each
        fine cell contributes only its in-basin area.

    Returns
    -------
    (weights, mascon_ids) where weights has shape (n_mascons, fine.n_cells) in
    km^2 and mascon_ids gives the label for each row.
    """
    w_fc = overlap_matrix(fine, grace)                       # (n_coarse, n_fine)

    labels = mascon_id.ravel()
    ids = np.unique(labels[labels >= 0])
    lookup = {int(m): i for i, m in enumerate(ids)}

    keep = labels >= 0
    rows = np.array([lookup[int(m)] for m in labels[keep]], dtype='int64')
    cols = np.flatnonzero(keep)
    group = sparse.csr_matrix(
        (np.ones(rows.size), (rows, cols)),
        shape=(ids.size, labels.size),
    )
    weights = (group @ w_fc).tocsr()

    if basin_frac_fine is not None:
        weights = weights.multiply(
            sparse.csr_matrix(basin_frac_fine.ravel()[None, :])
        ).tocsr()
        weights.eliminate_zeros()

    return weights, ids


# --------------------------------------------------------------------------
# Regridding
# --------------------------------------------------------------------------

def regrid_bilinear(field: np.ndarray, src: cfg.Grid, dst: cfg.Grid) -> np.ndarray:
    """
    Bilinear interpolation of a coarse field onto a finer grid, on cell centres.

    Used for GLDAS 0.25 deg -> 0.1 deg, where the ratio is non-integer so block
    replication would alias. Edge cells are clamped rather than extrapolated.
    NaNs propagate to any target cell whose source stencil contains one.
    """
    squeeze = field.ndim == 2
    stack = field[None] if squeeze else field

    sx, sy = src.lon_centers(), src.lat_centers()
    dx, dy = dst.lon_centers(), dst.lat_centers()

    # Fractional source coordinates of each destination centre.
    fx = np.clip((dx - sx[0]) / src.res, 0, src.n_x - 1)
    fy = np.clip((sy[0] - dy) / src.res, 0, src.n_y - 1)

    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = np.minimum(x0 + 1, src.n_x - 1)
    y1 = np.minimum(y0 + 1, src.n_y - 1)
    wx = (fx - x0)[None, None, :]
    wy = (fy - y0)[None, :, None]

    out = (stack[:, y0[:, None], x0[None, :]] * (1 - wy) * (1 - wx)
           + stack[:, y0[:, None], x1[None, :]] * (1 - wy) * wx
           + stack[:, y1[:, None], x0[None, :]] * wy * (1 - wx)
           + stack[:, y1[:, None], x1[None, :]] * wy * wx)
    return out[0] if squeeze else out


def gaussian_smooth(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    """
    NaN-aware Gaussian smoothing, applied per time step.

    Used to take the edge off the mass-conservation correction, which is
    piecewise constant per mascon and would otherwise print 3 degree blocks onto
    the final product. Normalising by the smoothed validity mask keeps values
    unbiased next to the basin boundary.
    """
    from scipy.ndimage import gaussian_filter

    squeeze = field.ndim == 2
    stack = field[None] if squeeze else field

    out = np.empty_like(stack)
    for t in range(stack.shape[0]):
        layer = stack[t]
        ok = np.isfinite(layer).astype('float64')
        num = gaussian_filter(np.where(ok > 0, layer, 0.0), sigma_cells, mode='nearest')
        den = gaussian_filter(ok, sigma_cells, mode='nearest')
        with np.errstate(invalid='ignore', divide='ignore'):
            out[t] = np.where(den > 1e-6, num / den, np.nan)
    return out[0] if squeeze else out


if __name__ == '__main__':
    grids = cfg.build_grids()
    fine, coarse = grids['era5'], grids['grace']

    print('Grids:')
    for g in grids.values():
        print('  ' + g.describe())

    w = overlap_matrix(fine, coarse)
    area_f = cell_area_km2(fine)
    print(f'\noverlap operator: {w.shape}, {w.nnz} non-zeros '
          f'({w.nnz / fine.n_cells:.2f} coarse cells touched per fine cell)')

    # Conservation checks: column sums must reproduce each fine cell's area, and
    # the total must match the sum over the fine grid.
    col = np.asarray(w.sum(axis=0)).ravel()
    rel = np.abs(col - area_f.ravel()) / area_f.ravel()
    print(f'column-sum vs exact cell area: max relative error {rel.max():.3e}')
    print(f'total area: operator {w.sum():.1f} km2 vs fine grid '
          f'{area_f.sum():.1f} km2  (rel {abs(w.sum() - area_f.sum()) / area_f.sum():.3e})')

    n_split = int((np.diff(w.tocsc().indptr) > 1).sum())
    print(f'fine cells straddling a coarse boundary: {n_split} of {fine.n_cells} '
          f'({100 * n_split / fine.n_cells:.1f}%)')

    # A constant field must aggregate to exactly that constant.
    const = np.full(fine.shape, 7.5)
    agg = aggregate_area_weighted(const, w)
    print(f'constant-field round trip: max deviation from 7.5 = '
          f'{np.nanmax(np.abs(agg - 7.5)):.3e}')
