"""
Maps from the 0.1 degree downscaled TWSA product.

Replaces the basin-scale figures produced by `generate_monthly_maps.py`, which
shade the whole basin polygon with a single number because the underlying
analysis had only one basin-mean series per time step. These read the gridded
netCDF and draw actual fields.

Colour decisions
----------------
TWSA is an ANOMALY: its job is polarity about zero, so it gets a diverging map
with two hues and a neutral midpoint (`RdBu`, blue = wetter, red = drier), and
limits are forced symmetric so zero really sits at the neutral point. Red/blue
is the safest diverging pair under the common colour-vision deficiencies, and
matches the convention in the GRACE literature.

Uncertainty is a MAGNITUDE with no sign: it gets a single-hue sequential ramp
(`Purples`, light to dark), deliberately a different hue family so an
uncertainty panel is never mistaken for an anomaly panel.

No rainbow maps anywhere, and no hue at a diverging midpoint.

Panels within a figure share one colour scale, otherwise the eye compares
colours that do not mean the same thing across panels.

Significance is a SECOND, categorical layer and is drawn as one: dots on the
pixels whose trend does not survive the test, not a second colour ramp and not a
change of shade. The trend figure is the only one that carries it, because it is
the only figure making a claim that can fail a test.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import gridded_config as cfg
import figure_captions
from plot_style import DPI, SCI_INK, SCI_MUTED
from stats_utils import MK_AUTOCORR_METHODS, mann_kendall_field

FIG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'Results', 'figures', 'gridded_maps'))

DIVERGING = 'RdBu'        # two hues, neutral midpoint - for signed anomalies
SEQUENTIAL = 'Purples'    # single hue, light to dark - for magnitudes

SEASONS = {
    'Winter (DJF)': (12, 1, 2),
    'Pre-monsoon (MAM)': (3, 4, 5),
    'Monsoon (JJAS)': (6, 7, 8, 9),
    'Post-monsoon (ON)': (10, 11),
}
# Month names come from utils, the single definition shared across the project.
from utils import MONTH_NAMES


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_product(path: Optional[str] = None) -> Dict[str, object]:
    """Read the gridded product, preferring the version carrying uncertainty."""
    import netCDF4

    from downscale_model import RESULTS_DIR
    if path is None:
        with_unc = os.path.join(RESULTS_DIR, 'twsa_0p1deg_monthly_with_uncertainty.nc')
        plain = os.path.join(RESULTS_DIR, 'twsa_0p1deg_monthly_xgboost.nc')
        path = with_unc if os.path.exists(with_unc) else plain
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found - run downscale_model.py (and optionally '
            f'downscale_uncertainty.py) first.')

    out: Dict[str, object] = {'path': path}
    with netCDF4.Dataset(path) as ds:
        for name in ds.variables:
            v = ds[name]
            v.set_auto_mask(False)
            out[name] = np.asarray(v[:])
        out['months'] = pd.to_datetime(ds['time'][:], unit='D', origin='1970-01-01')
        out['has_uncertainty'] = 'sigma_total' in ds.variables
    return out


def basin_outline():
    return cfg.load_basin()


# --------------------------------------------------------------------------
# Panel drawing
# --------------------------------------------------------------------------

def _draw(ax, field: np.ndarray, lon, lat, cmap: str, norm, gdf) -> object:
    """One map panel: field, basin outline, recessive frame."""
    extent = [lon.min() - 0.05, lon.max() + 0.05, lat.min() - 0.05, lat.max() + 0.05]
    im = ax.imshow(field, extent=extent, origin='upper', cmap=cmap, norm=norm,
                   interpolation='nearest', aspect='equal')
    gdf.boundary.plot(ax=ax, color=SCI_INK, linewidth=0.6, zorder=3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(SCI_MUTED)
        side.set_linewidth(0.4)
    return im


def _stipple(ax, mask: np.ndarray, lon, lat, every: int = 2) -> None:
    """
    Dot the pixels where `mask` is True — used to mark NOT significant.

    Dots rather than hatching. A hatch drawn over a region this shape reads as a
    texture applied to the colour underneath, so at a glance the reader sees a
    different shade of trend rather than a separate layer; discrete dots cannot
    be confused with the colour scale. Every `every`-th cell in each direction
    is marked, which at 0.1 degree keeps the dots individually resolvable at
    600 DPI instead of merging into a grey wash.

    The hairline white edge is what makes the dot survive both ends of a
    diverging ramp: an ink dot on the pale midpoint is obvious but on the dark
    red end of RdBu it is nearly the same value as the background.

    Row 0 of the field is the highest latitude (`origin='upper'` in `_draw`,
    and `lat_centers()` descends), so mask row r sits at lat[r].
    """
    r, c = np.nonzero(np.asarray(mask, dtype=bool))
    if r.size == 0:
        return
    keep = (r % every == 0) & (c % every == 0)
    ax.scatter(np.asarray(lon)[c[keep]], np.asarray(lat)[r[keep]],
               s=1.4, c=SCI_INK, marker='.', edgecolors='white',
               linewidths=0.12, alpha=0.85, zorder=4)


def panel_figure(
    fields: List[np.ndarray],
    titles: List[str],
    lon, lat, gdf,
    cbar_label: str,
    diverging: bool,
    out_path: str,
    ncols: int = 4,
    suptitle: Optional[str] = None,
    footnote: Optional[str] = None,
    vmax: Optional[float] = None,
    stipple: Optional[List[Optional[np.ndarray]]] = None,
) -> str:
    """
    Multi-panel map figure sharing a single colour scale.

    A shared scale is the point: panels drawn on independent scales cannot be
    compared, which is the most common failure in seasonal map sets.

    `stipple`, if given, is one boolean mask per panel (or None for a panel that
    needs none); True cells are dotted. It carries a second, categorical layer
    on top of a continuous field — significance over trend, in the one figure
    that has it.
    """
    stack = np.stack([np.asarray(f, dtype='float64') for f in fields])
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        raise ValueError('nothing finite to plot')

    if diverging:
        lim = vmax if vmax is not None else float(np.nanpercentile(np.abs(finite), 99))
        lim = max(lim, 1e-6)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cmap = DIVERGING
    else:
        hi = vmax if vmax is not None else float(np.nanpercentile(finite, 99))
        norm = plt.Normalize(vmin=0.0, vmax=max(hi, 1e-6))
        cmap = SEQUENTIAL

    n = len(fields)
    nrows = int(np.ceil(n / ncols))
    # A one-panel figure gets a larger tile. At 3.0 in the map is smaller than
    # its own colourbar label, and a standalone figure has no neighbours to be
    # consistent with.
    tile_w = 3.0 if ncols > 1 else 5.0
    tile_h = 2.5 if ncols > 1 else 4.0
    fig_w = tile_w * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, tile_h * nrows),
                             squeeze=False)
    im = None
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis('off')
            continue
        im = _draw(ax, fields[i], lon, lat, cmap, norm, gdf)
        if stipple is not None and stipple[i] is not None:
            _stipple(ax, stipple[i], lon, lat)
        ax.set_title(titles[i], fontsize=9, color=SCI_INK, pad=4)

    # The footnote is NOT drawn. It is recorded against this figure's path and
    # written to CAPTIONS.md beside the image, because a caption belongs in the
    # text a journal typesets rather than in six-point grey under the axes. The
    # sentence itself still lives here, next to the code that knows what the
    # panels contain -- see figure_captions.
    figure_captions.record(out_path, footnote or '')
    bottom = 0.04
    # Headroom is set as a FRACTION but consumed by text measured in POINTS, so
    # a short figure gets less of it. On the 4-inch one-panel tile 0.92 left the
    # panel title running through the suptitle; 0.87 clears it. Multi-panel
    # figures are 7.5 inches tall and were never affected.
    top = (0.87 if ncols == 1 else 0.92) if suptitle else 0.96
    fig.subplots_adjust(left=0.02, right=0.88, top=top,
                        bottom=bottom, wspace=0.05, hspace=0.18)
    cax = fig.add_axes([0.90, bottom + 0.05, 0.018, 0.86 - bottom])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=9, color=SCI_INK)
    cb.ax.tick_params(labelsize=8, colors=SCI_INK, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(SCI_MUTED)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=SCI_INK, y=0.98)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def climatology_maps(prod: Dict[str, object], out_dir: str) -> str:
    """Long-term mean TWSA for each calendar month."""
    twsa, months = prod['twsa'], prod['months']
    fields, titles = [], []
    for m in range(1, 13):
        sel = months.month == m
        with np.errstate(invalid='ignore'):
            fields.append(np.nanmean(twsa[sel], axis=0))
        titles.append(MONTH_NAMES[m - 1])
    return panel_figure(
        fields, titles, prod['lon'], prod['lat'], prod['gdf'],
        'TWSA (mm)', diverging=True,
        out_path=os.path.join(out_dir, 'Fig_monthly_climatology_0p1deg.png'),
        ncols=4, suptitle='Monthly mean terrestrial water storage anomaly, 0.1°',
        footnote=f'{months[0]:%Y-%m} to {months[-1]:%Y-%m}. '
                 f'Anomalies relative to the 2004.0–2010.0 GRACE baseline.')


def seasonal_maps(prod: Dict[str, object], out_dir: str) -> str:
    twsa, months = prod['twsa'], prod['months']
    fields, titles = [], []
    for label, mm in SEASONS.items():
        sel = np.isin(months.month, mm)
        with np.errstate(invalid='ignore'):
            fields.append(np.nanmean(twsa[sel], axis=0))
        titles.append(label)
    return panel_figure(
        fields, titles, prod['lon'], prod['lat'], prod['gdf'],
        'TWSA (mm)', diverging=True,
        out_path=os.path.join(out_dir, 'Fig_seasonal_mean_0p1deg.png'),
        ncols=4, suptitle='Seasonal mean terrestrial water storage anomaly, 0.1°',
        footnote=f'{months[0]:%Y-%m} to {months[-1]:%Y-%m}.')


def rms_map(prod: Dict[str, object], out_dir: str) -> str:
    """
    RMS of the TWSA time series at each pixel — the figure R1 asked for by name.

    A magnitude with no sign, so it gets the sequential ramp rather than the
    diverging one used for anomalies.

    Note what this is and is not. It is a property of the PRODUCT: how much
    storage varies at each location over the record. It is **not** an error map —
    scoring error per pixel would need a per-pixel observation, and GRACE
    supplies ~20 independent values over this basin. The distinction is stated in
    the caption because the two are easily confused at a glance.
    """
    twsa = prod['twsa']
    with np.errstate(invalid='ignore'):
        field = np.sqrt(np.nanmean(np.square(twsa), axis=0))
    return panel_figure(
        [field], ['RMS of TWSA time series'], prod['lon'], prod['lat'], prod['gdf'],
        'RMS TWSA (mm)', diverging=False,
        out_path=os.path.join(out_dir, 'Fig_rms_0p1deg.png'), ncols=1,
        suptitle='Amplitude of storage variability, 0.1 deg',
        footnote='RMS of the monthly anomaly at each pixel over '
                 f'{prod["months"][0]:%Y-%m}-{prod["months"][-1]:%Y-%m}. This is the '
                 'amplitude of the reconstructed signal, NOT a per-pixel error: '
                 'scoring error per pixel would require a per-pixel observation, '
                 'which GRACE does not provide.')


def mascon_skill_map(prod: Dict[str, object], out_dir: str,
                     model: str = 'xgboost') -> Optional[str]:
    """
    Leave-one-mascon-out skill painted onto the mascons it was measured on.

    This is the honest answer to the request for a spatial map of performance.
    Skill is defined at the scale at which the target is independent — the
    mascon — so the map has 19 patches rather than 9,538. Shading every pixel
    would imply a resolution of evidence the data cannot support.
    """
    from downscale_model import RESULTS_DIR
    import downscale_features as F

    path = os.path.join(RESULTS_DIR, f'lomo_cv_{model}.csv')
    if not os.path.exists(path):
        print(f'  skipped mascon skill map: {path} not found')
        return None
    per_fold = pd.read_csv(path)
    if 'mascon' not in per_fold.columns or 'R2' not in per_fold.columns:
        print('  skipped mascon skill map: unexpected columns')
        return None

    aux = F.load_aux()
    mid = np.asarray(aux['mascon_id'], dtype='float64')
    lookup = dict(zip(per_fold.mascon.astype(int), per_fold.R2.astype(float)))

    # Painted on the FINE grid, then clipped to the basin.
    #
    # Drawn natively on the 0.5 degree GRACE grid, each mascon is a block of
    # coarse cells and the coloured area spills well outside the catchment --
    # every cell a mascon touches is filled, so the map had a ragged rectangular
    # edge while every other figure in the set stops at the basin. The VALUES
    # here are still per mascon: `parent_era5_to_grace` gives each 0.1 degree
    # cell its enclosing GRACE cell, so a mascon's R2 is broadcast unchanged over
    # the fine cells it owns. Nothing is interpolated and no gradient is implied
    # inside a patch -- the resolution of the EVIDENCE is unchanged, only the
    # resolution at which its outline is cut.
    parent = np.asarray(aux['parent_era5_to_grace'])
    flat_mid = mid.ravel()
    fine_mascon = np.where(
        (parent >= 0) & (parent < flat_mid.size),
        flat_mid[np.clip(parent, 0, flat_mid.size - 1)], np.nan)

    skill = np.full(fine_mascon.shape, np.nan)
    for m, r2 in lookup.items():
        skill[fine_mascon == m] = r2
    # Clip to the basin exactly as the product fields are clipped.
    skill[~(np.asarray(aux['basin_frac_era5']) > 0)] = np.nan

    grids = cfg.build_grids()
    g = grids['era5']
    lon, lat = g.lon_centers(), g.lat_centers()
    return panel_figure(
        [skill], [f'Held-out R² ({len(lookup)} mascons)'], lon, lat, prod['gdf'],
        'Leave-one-mascon-out R²', diverging=False,
        out_path=os.path.join(out_dir, f'Fig_mascon_skill_{model}.png'), ncols=1,
        suptitle='Spatial transfer skill, by mascon',
        footnote='Each patch is one 3 deg GRACE mascon, coloured by the R² the model '
                 'achieved there when that mascon (and its neighbours) were withheld '
                 'from training. Skill is shown at mascon scale because that is the '
                 'scale at which the target is independent; a per-pixel version would '
                 'imply evidence that ~20 independent observations cannot support.')


def write_trend_fields(res, prod: Dict[str, object], path: str) -> str:
    """
    Persist the trend test as netCDF, on the same grid as the product.

    The slope used to be computed inside the figure and thrown away, so nothing
    downstream could check a number quoted from the map or difference the trend
    against anything else. Written next to the product, in the same conventions
    as `downscale_model.write_product`.
    """
    import netCDF4
    import downscale_model as M

    lat, lon = np.asarray(prod['lat']), np.asarray(prod['lon'])
    months = prod['months']
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fields = {
        'sen_slope': (res.slope, 'mm yr-1',
                      "Theil-Sen slope of the deseasonalised monthly anomaly"),
        'ols_slope': (res.ols_slope, 'mm yr-1',
                      'Least-squares slope of the raw monthly anomaly, for comparison'),
        'p_value': (res.p_value, '1',
                    'Two-sided Mann-Kendall p-value, variance corrected for '
                    'serial dependence'),
        'z_score': (res.z, '1', 'Continuity-corrected normal score of Mann-Kendall S'),
        'kendall_tau': (res.tau, '1', "Kendall's tau-a of the deseasonalised series"),
        'variance_factor': (res.variance_factor, '1',
                            'Var(S) inflation applied for serial dependence (n/n*)'),
    }
    masks = {
        'significant': (res.significant,
                        f'p < {res.alpha}, NO multiple-comparison correction'),
        'significant_fdr': (res.significant_fdr,
                            f'Benjamini-Hochberg FDR at {res.fdr_alpha} over the '
                            f'{res.n_tested} tested pixels'),
        'tested': (res.valid,
                   'Pixel had a finite value in every month and was tested'),
    }

    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.createDimension('lat', lat.size)
        ds.createDimension('lon', lon.size)
        for nm, vals, un in [('lat', lat, 'degrees_north'),
                             ('lon', lon, 'degrees_east')]:
            cv = ds.createVariable(nm, 'f8', (nm,))
            cv.units = un
            cv[:] = vals

        for name, (arr, units, long_name) in fields.items():
            v = ds.createVariable(name, 'f4', ('lat', 'lon'), zlib=True, complevel=4,
                                  fill_value=np.float32(np.nan))
            v.units, v.long_name = units, long_name
            v[:] = np.asarray(arr, dtype='float32')
        for name, (arr, long_name) in masks.items():
            v = ds.createVariable(name, 'i1', ('lat', 'lon'), zlib=True, complevel=4)
            v.long_name = long_name
            v[:] = np.asarray(arr, dtype='i1')

        ds['sen_slope'].caveat = (
            'The trend is inherited from GRACE by mass conservation. Its '
            'large-scale pattern is observed; the within-mascon detail comes '
            'from the smooth background interpolation, not from data.')
        ds['p_value'].caveat = (
            'Not a calibrated error rate. On simulated AR(1) series at this '
            'record length the corrected test still rejected a true null 18.8% '
            'of the time at a nominal 5% (stats_utils.trend_selftest). Read the '
            'field as an ordering of pixels by strength of evidence.')
        ds['significant_fdr'].caveat = (
            f'FDR treats the {res.n_tested} pixels as {res.n_tested} tests. '
            'GRACE resolves ~20 independent mascons over this basin, so the '
            'true number of independent trends tested is far smaller and this '
            'mask is still optimistic.')
        ds['variance_factor'].note = (
            f'Method was {res.method}. It is 1 everywhere unless the method is '
            'hamed_rao: tfpw filters the series instead of rescaling Var(S), '
            'and none applies no correction at all.')

        for k, v in res.summary().items():
            setattr(ds, f'test_{k}', str(v))
        ds.title = ('Per-pixel TWSA trend and Mann-Kendall significance, '
                    'Ganga basin, 0.1 degree')
        ds.period = f'{months[0]:%Y-%m} to {months[-1]:%Y-%m}'
        ds.source_product = os.path.basename(str(prod['path']))
        ds.created_by = 'main/generate_gridded_maps.py'
        for k, v in M.provenance().items():
            setattr(ds, f'provenance_{k}', str(v))
    return path


def trend_map(prod: Dict[str, object], out_dir: str,
              method: str = 'hamed_rao', alpha: float = 0.05) -> str:
    """
    Per-pixel Sen slope in mm/yr, with the non-significant pixels stippled.

    Two changes from the version that only drew a least-squares slope. The slope
    painted is Theil-Sen, because that is the estimator that belongs with the
    Mann-Kendall test drawn on top of it; the OLS slope is still computed and
    written to the netCDF so the two can be differenced. And the pixels whose
    trend does not survive the test are dotted, so a reader can see where the
    colour is carrying a claim and where it is not.

    Significance is Mann-Kendall after removing the calendar-month median and
    inflating Var(S) for serial dependence, then Benjamini-Hochberg across the
    field. `stats_utils.mann_kendall_field` documents what that does and does
    not fix; the short version, which the caption repeats, is that the test is
    still liberal under strong persistence and that the pixels are nothing like
    independent tests of anything.

    Worth stating in the caption for the same reason as before: the trend is
    inherited from GRACE by mass conservation, so its LARGE-SCALE pattern is
    observed while its within-mascon detail comes from the smooth background
    interpolation, not from data. The stippling says whether a pixel's own
    series trends; it does not say the trend was measured at that pixel.
    """
    from downscale_model import RESULTS_DIR

    twsa, months = prod['twsa'], prod['months']
    res = mann_kendall_field(twsa, months, method=method, alpha=alpha)
    nc = write_trend_fields(
        res, prod, os.path.join(RESULTS_DIR, 'twsa_trend_significance.nc'))
    pct = 100.0 * res.n_significant_fdr / max(res.n_tested, 1)
    print(f'  trend test: {res.n_tested} pixels, {res.n_significant} at '
          f'p<{alpha} uncorrected, {res.n_significant_fdr} after FDR ({pct:.0f}%), '
          f'{res.seconds:.1f} s -> {os.path.basename(nc)}')

    return panel_figure(
        [res.slope], ['Sen slope'], prod['lon'], prod['lat'], prod['gdf'],
        'TWSA trend (mm yr⁻¹)', diverging=True,
        out_path=os.path.join(out_dir, 'Fig_trend_0p1deg.png'), ncols=1,
        suptitle=f'TWSA trend, {months[0]:%Y}–{months[-1]:%Y}',
        stipple=[res.valid & ~res.significant_fdr],
        footnote='Theil-Sen slope per pixel. Dots mark pixels whose Mann-Kendall '
                 'trend does NOT survive Benjamini-Hochberg FDR at '
                 f'{res.fdr_alpha:g} ({res.n_tested - res.n_significant_fdr} of '
                 f'{res.n_tested}); the series is deseasonalised and Var(S) is '
                 'inflated for serial dependence (Hamed-Rao), without which a '
                 'persistent monthly series returns significance almost '
                 'everywhere. The test is still liberal under strong persistence, '
                 'and the pixels are not independent: the trend is inherited from '
                 'GRACE by mass conservation, so its large-scale pattern is '
                 'observed while its within-mascon detail is interpolated.')


def uncertainty_maps(prod: Dict[str, object], out_dir: str) -> Optional[str]:
    """The three uncertainty terms and their combination, on one shared scale."""
    if not prod.get('has_uncertainty'):
        return None
    names = [('sigma_total', 'Total'),
             ('sigma_grace', 'GRACE measurement'),
             ('sigma_transfer', 'Spatial transfer'),
             ('sigma_within', 'Within-mascon spread')]
    fields, titles = [], []
    for key, label in names:
        with np.errstate(invalid='ignore'):
            fields.append(np.nanmean(prod[key], axis=0))
        titles.append(label)
    return panel_figure(
        fields, titles, prod['lon'], prod['lat'], prod['gdf'],
        'Uncertainty, 1σ (mm)', diverging=False,
        out_path=os.path.join(out_dir, 'Fig_uncertainty_components_0p1deg.png'),
        ncols=4, suptitle='Per-pixel uncertainty, time-mean',
        footnote='Within-mascon spread is a LOWER BOUND: GRACE resolves ~20 mascons here, '
                 'so no observation constrains sub-mascon structure.')


def method_comparison(prod: Dict[str, object], out_dir: str,
                      when: str = '2015-06') -> str:
    """
    One month, three ways: downscaled, bilinear interpolation, raw mascon field.

    The comparison a reader needs in order to judge whether downscaling added
    anything beyond smoothing.
    """
    import downscale_model as M
    import downscale_features as DF

    months = prod['months']
    idx = int(np.flatnonzero(months.to_period('M') == pd.Period(when, 'M'))[0])

    aux = DF.load_aux()
    cells = np.flatnonzero((aux['basin_frac_era5'] > 0).ravel())
    base = M.interpolation_baselines(months, cells)

    shape = prod['twsa'].shape[1:]
    def expand(flat_row):
        full = np.full(shape[0] * shape[1], np.nan)
        full[cells] = flat_row
        return full.reshape(shape)

    fields = [prod['twsa'][idx], expand(base['bilinear'][idx]),
              expand(base['blocky'][idx])]
    titles = ['Downscaled (this work)', 'Bilinear interpolation',
              'GRACE mascons (native)']
    return panel_figure(
        fields, titles, prod['lon'], prod['lat'], prod['gdf'],
        'TWSA (mm)', diverging=True,
        out_path=os.path.join(out_dir, f'Fig_method_comparison_{when}.png'),
        ncols=3, suptitle=f'Downscaling versus interpolation, {when}',
        footnote='All three reproduce GRACE at mascon scale; they differ only in the '
                 'sub-mascon structure they imply.')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--product', default=None)
    ap.add_argument('--out-dir', default=FIG_DIR)
    ap.add_argument('--model', default='xgboost',
                    help='Which lomo_cv_<model>.csv to draw the mascon skill map from.')
    ap.add_argument('--month', default='2015-06',
                    help='month used for the method comparison figure')
    ap.add_argument('--trend-method', default='hamed_rao',
                    choices=list(MK_AUTOCORR_METHODS),
                    help='Serial-dependence correction for the trend test. '
                         '"none" is diagnostic only - it rejects a true null '
                         'roughly half the time on a persistent monthly series.')
    ap.add_argument('--trend-alpha', type=float, default=0.05,
                    help='Level for the uncorrected trend mask. The figure '
                         'stipples on the FDR mask, not this one.')
    args = ap.parse_args()

    prod = load_product(args.product)
    prod['gdf'] = basin_outline()
    print(f'product: {os.path.basename(str(prod["path"]))}  '
          f'{prod["twsa"].shape[0]} months, uncertainty='
          f'{prod.get("has_uncertainty")}')

    made = [climatology_maps(prod, args.out_dir),
            seasonal_maps(prod, args.out_dir),
            trend_map(prod, args.out_dir, args.trend_method, args.trend_alpha),
            rms_map(prod, args.out_dir),
            method_comparison(prod, args.out_dir, args.month)]
    skill = mascon_skill_map(prod, args.out_dir, args.model)
    if skill:
        made.append(skill)
    unc = uncertainty_maps(prod, args.out_dir)
    if unc:
        made.append(unc)

    caps = figure_captions.write_index(
        args.out_dir, title='Figure captions — gridded maps',
        preamble='Each caption states what the panel does *not* establish as '
                 'well as what it shows; carry them into the manuscript text '
                 'rather than re-deriving them.')

    print(f'\n{len(made)} figures written to {args.out_dir}:')
    for p in made:
        print(f'  {os.path.basename(p)}')
    print(f'  captions -> {os.path.basename(caps)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
