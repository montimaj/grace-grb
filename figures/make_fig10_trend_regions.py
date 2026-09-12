"""
Figure 10 -- where the depletion is, and how the regions are defined.

WHY THIS FIGURE EXISTS
----------------------
The paper says depletion is concentrated in the north-west and quotes a mean
trend for that area. Until `trend_regions.py` was written there was no committed
definition of "that area", so the number could not be recomputed from the
archive. The regions are now boxes stated in code, and this figure draws them on
the trend field so a reader can see exactly what was averaged rather than take
the boundary on trust.

Panel (a) is the masked trend field with the boxes on top. Panel (b) is the
distribution of per-pixel trends inside each box, which carries what a single
mean cannot: that every significant pixel in the north-west is negative, that
excluding the Himalayan edge barely moves the result, and that the central and
lower plain is a different regime rather than a weaker version of the same one.

Both panels and the accompanying table read the same two files -- the archived
trend field and `trend_regions.summarise()` -- so figure, table and manuscript
cannot disagree.

USAGE
-----
    python make_fig10_trend_regions.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'main'))
import plot_style as ps                                        # noqa: E402
import trend_regions as tr                                     # noqa: E402
from generate_gridded_maps import _stipple                     # noqa: E402

OUT = os.path.join(HERE, 'output')
STEM = 'Fig10_trend_regions'
BASIN = os.path.join(REPO, 'Data', 'Ganga Basin Shapefile', 'Ganga_basin.shp')

INK, MUTED, GRID = ps.SCI_INK, ps.SCI_MUTED, ps.SCI_GRID
DIVERGING = 'RdBu'
# The four quadrants are a PARTITION ordered by depletion, so panel (b) fills
# each box from the same diverging ramp as the map rather than from four
# categorical hues. That ties the two panels together and sidesteps the
# colour-vision problem a four-hue categorical set runs into here: the best
# four-colour subset of the project palette separates two greens by only
# dE 15.6 for normal vision. Every row is named in both panels regardless, so
# no identity rests on colour.
SUBSET_C = '#4a3aa7'                      # the one nested box, drawn dashed
LABEL = {'basin': 'Basin (all four quadrants)',
         'northwest_plain': 'North-western plain',
         'northwest_plain_lowland': 'North-western plain, below 30°N',
         'northern_plain_himalayan_front': 'Northern plain and Himalayan front',
         'southern_plateau': 'Southern plateau',
         'eastern_plain_delta': 'Eastern plain and delta'}



def basin_outline():
    try:
        import geopandas as gpd
        return gpd.read_file(BASIN).to_crs(4326)
    except Exception as exc:                                   # pragma: no cover
        print(f'  (basin outline unavailable: {exc})')
        return None


def build():
    lat, lon, slope, sig, tested = tr.load_trend()
    df = tr.summarise().set_index('region')

    # Every TESTED pixel is painted and the ones that fail FDR are dotted, rather
    # than blanked. Blanking them says "no data" where the truth is "a trend was
    # estimated here but the evidence for it is weak", and it also erases the
    # basin outline wherever a run of weak pixels happens to reach the edge.
    masked = np.where(tested, slope, np.nan)
    weak = tested & ~sig
    lim = float(np.nanpercentile(np.abs(slope[sig]), 99))
    extent = [lon.min() - 0.05, lon.max() + 0.05,
              lat.min() - 0.05, lat.max() + 0.05]
    shown = masked[::-1] if lat[0] < lat[-1] else masked

    # Explicit geometry rather than tight_layout. Panel (b) carries long
    # two-line y labels that reach far into the left margin; with a tight
    # bounding box those labels set the crop, and a map centred on the FIGURE
    # then lands right of centre in the saved PNG. Positioning both panels by
    # hand and saving the full canvas keeps (a) centred on what the reader sees.
    FIGW, FIGH = 7.4, 6.9
    fig = plt.figure(figsize=(FIGW, FIGH))
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    MAP_H, MAP_BOTTOM, CB_W, CB_PAD = 0.44, 0.525, 0.015, 0.010
    map_w = MAP_H * aspect * (FIGH / FIGW)
    total = map_w + CB_PAD + CB_W
    left = (1.0 - total) / 2.0
    axm = fig.add_axes([left, MAP_BOTTOM, map_w, MAP_H])
    cax = fig.add_axes([left + map_w + CB_PAD, MAP_BOTTOM, CB_W, MAP_H])
    axb = fig.add_axes([0.405, 0.065, 0.55, 0.33])

    # ---- (a) the field, masked to significance, with the boxes on top -------
    im = axm.imshow(shown, extent=extent, origin='upper', cmap=DIVERGING,
                    norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                    interpolation='nearest')
    _stipple(axm, weak, lon, lat)
    gdf = basin_outline()
    if gdf is not None:
        gdf.boundary.plot(ax=axm, color=INK, linewidth=0.7, zorder=3)

    # The partition, drawn as its two dividing lines rather than four boxes:
    # four rectangles sharing edges read as clutter over a field this busy.
    axm.axhline(tr.SPLIT_LAT, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=4)
    axm.axvline(tr.SPLIT_LON, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=4)
    r = df.loc['northwest_plain_lowland']
    axm.add_patch(Rectangle((extent[0], r.lat_min),
                            tr.SPLIT_LON - extent[0], r.lat_max - r.lat_min,
                            fill=False, edgecolor=SUBSET_C, linewidth=1.3,
                            linestyle=(0, (3, 2)), zorder=5))

    def qlab(x, y, text, ha, va):
        axm.text(x, y, text, fontsize=7.4, fontweight='bold', color=INK,
                 ha=ha, va=va, zorder=6,
                 bbox=dict(fc='white', ec='none', alpha=0.72, pad=1.4))

    qlab(extent[0] + 0.25, extent[3] - 0.2, 'North-western plain', 'left', 'top')
    # The two eastern-quadrant labels share one centre so they read as a pair:
    # one in the empty band north of the basin, one in the band south of it.
    EAST_LABEL_X = 84.6
    qlab(EAST_LABEL_X, extent[3] - 0.15, 'Northern plain and Himalayan front',
         'center', 'top')
    qlab(extent[0] + 0.25, extent[2] + 0.2, 'Southern plateau', 'left', 'bottom')
    # Placed in the empty band south of the basin rather than on the delta,
    # which the right-aligned position used to overlap.
    qlab(EAST_LABEL_X, 21.85, 'Eastern plain and delta', 'center', 'bottom')
    axm.text(extent[0] + 0.25, 29.85, 'Below 30°N', fontsize=6.8,
             color=SUBSET_C, ha='left', va='top', zorder=6,
             bbox=dict(fc='white', ec='none', alpha=0.72, pad=1.0))

    axm.set_xlim(extent[0], extent[1]); axm.set_ylim(extent[2], extent[3])
    axm.set_aspect('equal')
    # Degrees east and north, not bare numbers: the axes carry the only statement
    # of units on the map, and the region names are quoted in degrees throughout.
    axm.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f'{v:g}\u00b0E'))
    axm.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f'{v:g}\u00b0N'))
    axm.tick_params(labelsize=7.4, color=MUTED, length=2.5, labelcolor=INK)
    for s in ('top', 'right'):
        axm.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        axm.spines[s].set_color(GRID)

    # Panel letters only. What each panel shows belongs in the caption, but the
    # caption still needs something to point at.
    # Both letters on one left margin. The map is centred and panel (b) is
    # pushed right to clear its labels, so anchoring each letter to its own axes
    # would leave them at different indents.
    LETTER_X = 0.048
    fig.text(LETTER_X, MAP_BOTTOM + MAP_H + 0.012, '(a)', fontsize=9.5,
             fontweight='bold', color=INK, va='bottom', ha='left')
    fig.text(LETTER_X, 0.065 + 0.33 + 0.012, '(b)', fontsize=9.5,
             fontweight='bold', color=INK, va='bottom', ha='left')

    cb = fig.colorbar(im, cax=cax, extend='both')
    cb.set_label('TWSA trend (mm yr$^{-1}$)', fontsize=7.8, color=INK)
    cb.ax.tick_params(labelsize=7.2, length=2, color=MUTED, labelcolor=INK)
    cb.outline.set_visible(False)

    # ---- (b) the distribution inside each box ------------------------------
    order = ['northwest_plain', 'northwest_plain_lowland',
             'northern_plain_himalayan_front', 'southern_plateau',
             'eastern_plain_delta', 'basin']
    data, ylabels = [], []
    for name in order:
        # Masks come from trend_regions, never re-derived here: the figure and
        # the table must be the same arithmetic, not two copies of it.
        box = tr.region_mask(name, lat, lon)
        v = slope[box & sig]
        data.append(v)
        # Both shares, not just the negative one: "76% negative" alone leaves the
        # reader to assume the remainder is positive, which is only true where no
        # pixel sits exactly at zero.
        ylabels.append(f'{LABEL[name]}\nn = {len(v):,}   mean ' + f'{v.mean():.1f}'.replace('-', '−') + '   '
                       f'{100 * (v < 0).mean():.0f}% negative, '
                       f'{100 * (v > 0).mean():.0f}% positive')

    bp = axb.boxplot(data, vert=False, widths=0.55, showfliers=False,
                     patch_artist=True, medianprops=dict(color=INK, lw=1.3),
                     whiskerprops=dict(color=MUTED, lw=0.9),
                     capprops=dict(color=MUTED, lw=0.9),
                     boxprops=dict(edgecolor=MUTED, lw=0.9))
    ramp = plt.get_cmap(DIVERGING)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    fills = [ramp(norm(v.mean())) for v in data]
    for patch, c, name in zip(bp['boxes'], fills, order):
        if name == 'basin':
            patch.set_facecolor('none')          # the reference, not a region
        else:
            patch.set_facecolor(c)
            patch.set_edgecolor(SUBSET_C if 'lowland' in name else MUTED)
            patch.set_linestyle((0, (3, 2)) if 'lowland' in name else '-')
    for i, v in enumerate(data, start=1):
        axb.plot(v.mean(), i, marker='D', ms=4.5, color=INK, zorder=5,
                 markeredgecolor='white', markeredgewidth=0.7)

    axb.axvline(0, color=MUTED, lw=0.8, zorder=1)
    # Every region reaches into positive trend, so the axis must be ticked there
    # too. The default locator stopped at zero because the largest whisker falls
    # short of the next multiple of 20, which hid the fact that the distributions
    # straddle zero at all.
    lo, hi = axb.get_xlim()
    hi = max(hi, 20.0)
    axb.set_xlim(lo, hi)
    axb.set_xticks(np.arange(np.ceil(lo / 20.0) * 20.0, hi + 1e-9, 20.0))
    axb.set_yticklabels(ylabels, fontsize=7.4, color=INK)
    axb.tick_params(labelsize=7.6, color=MUTED, length=2.5, labelcolor=INK)
    axb.set_xlabel('TWSA trend at FDR-significant pixels (mm yr$^{-1}$)',
                   fontsize=8.0, color=INK)
    axb.grid(axis='x', color=GRID, lw=0.6)
    axb.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        axb.spines[s].set_visible(False)
    axb.spines['bottom'].set_color(GRID)
    axb.invert_yaxis()
    return fig, df


def main() -> int:
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42,
                         'ps.fonttype': 42})
    fig, df = build()
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, STEM)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=ps.DPI, facecolor='white')
        print(f'written: {stem}.{ext}')
    cols = ['n_significant', 'mean_significant_mm_yr',
            'median_significant_mm_yr', 'pct_negative_significant']
    print()
    print(df[cols].to_string(float_format=lambda v: f'{v:.2f}'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
