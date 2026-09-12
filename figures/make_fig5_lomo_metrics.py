"""
Supplementary figure -- leave-one-mascon-out skill, metric by metric.

WHY THIS FIGURE EXISTS
----------------------
The Results section promises "the complete fold-wise spatial distribution of
RMSE, R-squared, NSE and bias", and for two revisions it pointed at a plate that
shows something else: `mascon_metric_maps.png` is the WELL comparison redrawn at
mascon scale, which covers the 10 of 19 mascons that hold enough CGWB wells to
score. That is a statement about where the monitoring network is, not about
where the model transfers, and it cannot answer the question the sentence asks.

This is the plate the sentence describes: the four metrics of the 19
leave-one-mascon-out folds, each fold scored by a model fitted without that
mascon and without its neighbours. Both plates are kept, because they answer
different questions and the paper makes a claim about each.

WHY FOUR SEPARATE COLOUR BARS
-----------------------------
`generate_gridded_maps.panel_figure` puts its panels on one shared scale, which
is right for the uncertainty terms because they share a unit. Here they do not:
two panels are millimetres and two are dimensionless, and bias is signed while
RMSE is not. A shared bar would make three of the four unreadable, so each panel
carries its own, and the diverging ramp is reserved for the one signed metric.

WHERE THE NUMBERS COME FROM
---------------------------
RMSE, R-squared and NSE are read from `lomo_cv_<model>.csv` exactly as written
by the cross-validation. Bias is not in that file -- its PBIAS column is empty,
because a percentage bias on an anomaly with a near-zero mean is meaningless --
so mean bias error is recomputed per mascon from the out-of-fold predictions in
`lomo_oof_<model>.csv`. The script checks that recomputing RMSE the same way
reproduces the stored value before it trusts the bias.

USAGE
-----
    python make_figS5_lomo_metrics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'main'))
import plot_style as ps                                        # noqa: E402

OUT = os.path.join(HERE, 'output')
STEM = 'FigS5_lomo_metrics'
RESULTS = os.path.join(REPO, 'Results', 'downscaling')
BASIN = os.path.join(REPO, 'Data', 'Ganga Basin Shapefile', 'Ganga_basin.shp')

INK, MUTED, GRID = ps.SCI_INK, ps.SCI_MUTED, ps.SCI_GRID

# (column, panel title, colour map, signed?, clip)
#
# NSE is clipped at -1. Two folds sit near -5 and -9, and on an unclipped scale
# they flatten the other seventeen into one colour -- the panel then says only
# "two mascons are bad", which the table already says. Clipped, it shows the
# gradient among the folds that are usable, and the arrow on the bar marks the
# ones that run past it. An NSE below zero already means "worse than predicting
# the mean", so nothing interpretable is lost below the cut.
PANELS = [
    ('RMSE', 'RMSE (mm)', 'viridis_r', False, None),
    ('R2', r'$\mathrm{R}^{2}$', 'viridis', False, None),
    ('NSE', 'NSE', 'viridis', False, (-1.0, 1.0)),
    ('MBE', 'Mean bias error (mm)', 'RdBu_r', True, None),
]


def load(model: str = 'xgboost'):
    """Per-mascon metrics, with bias recomputed from the out-of-fold file."""
    cv = pd.read_csv(os.path.join(RESULTS, f'lomo_cv_{model}.csv'))
    oof = pd.read_csv(os.path.join(RESULTS, f'lomo_oof_{model}.csv'))

    resid = oof.predicted - oof.observed
    per = oof.assign(resid=resid, sq=resid ** 2).groupby('mascon')
    derived = pd.DataFrame({'MBE': per.resid.mean(),
                            'RMSE_check': np.sqrt(per.sq.mean())}).reset_index()

    m = cv.merge(derived, on='mascon', validate='one_to_one')
    # Trust the recomputed bias only if the same rows reproduce the stored RMSE.
    gap = float(np.abs(m.RMSE - m.RMSE_check).max())
    if gap > 0.5:
        raise SystemExit(f'out-of-fold rows do not reproduce the stored RMSE '
                         f'(max difference {gap:.3f} mm); bias not trustworthy')
    print(f'  bias check: recomputed RMSE matches the stored value to '
          f'{gap:.4f} mm across {len(m)} mascons')
    return m


def mascon_fields(metrics: pd.DataFrame):
    """Broadcast each mascon's value over the 0.1 degree cells it owns."""
    import downscale_features as F
    import gridded_config as cfg

    aux = F.load_aux()
    mid = np.asarray(aux['mascon_id'], dtype='float64').ravel()
    parent = np.asarray(aux['parent_era5_to_grace'])
    fine = np.where((parent >= 0) & (parent < mid.size),
                    mid[np.clip(parent, 0, mid.size - 1)], np.nan)
    in_basin = np.asarray(aux['basin_frac_era5']) > 0

    fields = {}
    for col, _, _, _, _ in PANELS:
        f = np.full(fine.shape, np.nan)
        for m, v in zip(metrics.mascon.astype(int), metrics[col].astype(float)):
            f[fine == m] = v
        f[~in_basin] = np.nan
        fields[col] = f

    g = cfg.build_grids()['era5']
    return fields, g.lon_centers(), g.lat_centers()


def basin_outline():
    try:
        import geopandas as gpd
        return gpd.read_file(BASIN).to_crs(4326)
    except Exception as exc:                                   # pragma: no cover
        print(f'  (basin outline unavailable: {exc})')
        return None


def build(metrics: pd.DataFrame):
    fields, lon, lat = mascon_fields(metrics)
    extent = [lon.min() - 0.05, lon.max() + 0.05,
              lat.min() - 0.05, lat.max() + 0.05]
    gdf = basin_outline()

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    for ax, (col, title, cmap, signed, clip) in zip(axes.ravel(), PANELS):
        shape = (len(lat), len(lon))
        f = fields[col].reshape(shape)
        shown = f[::-1] if lat[0] < lat[-1] else f
        extend = 'neither'
        if signed:
            lim = float(np.nanmax(np.abs(f)))
            im = ax.imshow(shown, extent=extent, origin='upper', cmap=cmap,
                           norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                           interpolation='nearest')
        elif clip is not None:
            lo, hi = clip
            if np.nanmin(f) < lo:
                extend = 'min'
            im = ax.imshow(shown, extent=extent, origin='upper', cmap=cmap,
                           vmin=lo, vmax=hi, interpolation='nearest')
        else:
            im = ax.imshow(shown, extent=extent, origin='upper', cmap=cmap,
                           interpolation='nearest')
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color=INK, linewidth=0.6, zorder=3)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(title, fontsize=8.4, fontweight='bold', color=INK,
                     loc='left', pad=4)
        cb = fig.colorbar(im, ax=ax, fraction=0.031, pad=0.015, extend=extend)
        cb.ax.tick_params(labelsize=7.0, length=2, color=MUTED, labelcolor=INK)
        cb.outline.set_visible(False)

    fig.suptitle('Leave-one-mascon-out skill by mascon, all 19 folds',
                 fontsize=9.4, fontweight='bold', color=INK, y=0.985)
    fig.text(0.5, 0.012,
             'Each patch is one GRACE mascon, coloured by the score a model '
             'achieved there when that mascon and its neighbours were withheld '
             'from training.',
             fontsize=7.0, color=MUTED, ha='center')
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def main() -> int:
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42,
                         'ps.fonttype': 42})
    metrics = load()
    fig = build(metrics)
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, STEM)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=ps.DPI, bbox_inches='tight',
                    facecolor='white')
        print(f'written: {stem}.{ext}')
    cols = ['mascon', 'RMSE', 'R2', 'NSE', 'MBE']
    print()
    print(metrics[cols].sort_values('R2').to_string(
        index=False, float_format=lambda v: f'{v:.3f}'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
