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
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import gridded_config as cfg
from plot_style import DPI, SCI_INK, SCI_MUTED

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
) -> str:
    """
    Multi-panel map figure sharing a single colour scale.

    A shared scale is the point: panels drawn on independent scales cannot be
    compared, which is the most common failure in seasonal map sets.
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows),
                             squeeze=False)
    im = None
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis('off')
            continue
        im = _draw(ax, fields[i], lon, lat, cmap, norm, gdf)
        ax.set_title(titles[i], fontsize=9, color=SCI_INK, pad=4)

    fig.subplots_adjust(left=0.02, right=0.88, top=0.92 if suptitle else 0.96,
                        bottom=0.10 if footnote else 0.04, wspace=0.05, hspace=0.18)
    cax = fig.add_axes([0.90, 0.15, 0.018, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=9, color=SCI_INK)
    cb.ax.tick_params(labelsize=8, colors=SCI_INK, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(SCI_MUTED)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=SCI_INK, y=0.98)
    if footnote:
        fig.text(0.02, 0.015, footnote, fontsize=7, color=SCI_MUTED, ha='left')

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


def trend_map(prod: Dict[str, object], out_dir: str) -> str:
    """
    Per-pixel linear trend in mm/yr.

    Worth stating in the caption: the trend is inherited from GRACE by mass
    conservation, so its LARGE-SCALE pattern is observed while its within-mascon
    detail comes from the smooth background interpolation, not from data.
    """
    twsa, months = prod['twsa'], prod['months']
    t = (months - months[0]).days.to_numpy() / 365.25
    flat = twsa.reshape(len(months), -1)
    ok = np.isfinite(flat).all(axis=0)
    slope = np.full(flat.shape[1], np.nan)
    if ok.any():
        design = np.vstack([t, np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(design, flat[:, ok], rcond=None)
        slope[ok] = coef[0]
    field = slope.reshape(twsa.shape[1:])
    return panel_figure(
        [field], ['Linear trend'], prod['lon'], prod['lat'], prod['gdf'],
        'TWSA trend (mm yr⁻¹)', diverging=True,
        out_path=os.path.join(out_dir, 'Fig_trend_0p1deg.png'), ncols=1,
        suptitle=f'TWSA trend, {months[0]:%Y}–{months[-1]:%Y}',
        footnote='Trend is inherited from GRACE by mass conservation; its large-scale '
                 'pattern is observed, its within-mascon detail is interpolated.')


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
    ap.add_argument('--month', default='2015-06',
                    help='month used for the method comparison figure')
    args = ap.parse_args()

    prod = load_product(args.product)
    prod['gdf'] = basin_outline()
    print(f'product: {os.path.basename(str(prod["path"]))}  '
          f'{prod["twsa"].shape[0]} months, uncertainty='
          f'{prod.get("has_uncertainty")}')

    made = [climatology_maps(prod, args.out_dir),
            seasonal_maps(prod, args.out_dir),
            trend_map(prod, args.out_dir),
            method_comparison(prod, args.out_dir, args.month)]
    unc = uncertainty_maps(prod, args.out_dir)
    if unc:
        made.append(unc)

    print(f'\n{len(made)} figures written to {args.out_dir}:')
    for p in made:
        print(f'  {os.path.basename(p)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
