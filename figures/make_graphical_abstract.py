"""
Graphical abstract — built from the product, not drawn around it.

WHAT THIS REPLACES, AND WHY
---------------------------
The previous abstract was clip art: a satellite, a brain, a cartoon India and an
invented bar chart. For a data paper that is the wrong artefact. A graphical
abstract is the first and often only figure a reader looks at, and for this work
the single most useful thing it can do is show the actual transformation the
paper performs, in real data, at both scales, on one colour scale.

THE ARGUMENT IT HAS TO MAKE, IN ORDER
-------------------------------------
1. GRACE constrains this basin as 19 blocks. That is the honest starting point,
   and the left panel shows it as blocks rather than as a smooth field.
2. The product resolves it to 9,538 cells, and the mascon means are reproduced
   exactly rather than approximately.
3. Most of what the right panel shows is therefore INFERRED. The abstract says
   so in the figure, not in a caption a reader may never reach.
4. What that inference is worth is a number, and the numbers shown are held-out
   ones — leave-one-mascon-out and an independent well network — never the
   mascon-scale agreement that conservation guarantees.

The same month and the same symmetric colour scale are used on both maps, and
all four panels are drawn on identical map limits, or the comparison would be
decorative rather than a comparison. Red is negative throughout, matching
`generate_gridded_maps.DIVERGING` and the app.

FORMAT IS A HARD CONSTRAINT, NOT A PREFERENCE
---------------------------------------------
Elsevier renders a graphical abstract into a 500 x 200 pixel window on
ScienceDirect and requires at least 1328 x 531 px at 300 dpi in that same 2.5:1
ratio. Everything here follows from "must still be legible 200 pixels high".
The canvas is 11 x 4.4 in at 300 dpi = 3300 x 1320 px: comfortably over the
minimum, but deliberately SMALLER than the 12.5 x 5.0 in it used to be, because
at a fixed point size a smaller canvas is a larger font once the whole thing is
scaled into a 500 px window. Arial throughout -- Elsevier permits Times, Arial,
Courier or Symbol, and matplotlib's DejaVu Sans default is none of them.

NOTHING MAY OVERLAP, AND THAT IS CHECKED
----------------------------------------
Every element sits in an explicit rectangle declared in the LAYOUT block below,
and `check_layout` re-measures the rendered bounding boxes of every text, axes
and colourbar (ticks and labels included) and fails if any two intersect or if
anything runs off the canvas. Hand-tuned figure coordinates drift the moment a
label changes length; the check is what keeps this honest.

Usage
-----
    python make_graphical_abstract.py
    python make_graphical_abstract.py --month 2016-06
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.patches import FancyArrowPatch                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, os.path.join(ROOT, 'main'))

INK, MUTED, GRID = '#1a1a1a', '#6b6b6b', '#d8dee3'
# RdBu: red is water lost, blue is water gained. Identical to
# generate_gridded_maps.DIVERGING and to the Earth Engine app, so the same basin
# never appears in opposite colours across the three.
DIVERGING = 'RdBu'
# Elsevier permits Times, Arial, Courier or Symbol. matplotlib's default DejaVu
# Sans is not on that list, so it is set explicitly rather than inherited.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# font.family alone is not enough. Anything inside $...$ is drawn by mathtext,
# which has its OWN font set and defaults to DejaVu Sans -- one subscript was
# enough to embed DejaVuSans-Oblique in the PDF next to Arial, which is exactly
# the thing Elsevier's font rule forbids. Point mathtext at Arial too.
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'
mpl.rcParams['mathtext.default'] = 'regular'

# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
# 11.0 x 4.4 in at 300 dpi = 3300 x 1320 px. Exactly the 2.5:1 Elsevier asks
# for, 2.5x the 1328 x 531 minimum in area.
FIGW, FIGH, DPI = 11.0, 4.4, 300

# One set of map limits for all four panels. The GRACE grid is slightly larger
# than the ERA5-Land grid, so without this the basin is drawn at two different
# sizes in the two panels the reader is being asked to compare.
LON0, LON1, LAT0, LAT1 = 73.2, 91.4, 21.3, 31.7
# An equal-aspect map in a box of the wrong shape gets letterboxed, and the
# left/right titles then float away from the map edges they label. Deriving the
# box width from the data aspect makes the box and the map the same rectangle.
MAP_ASPECT = (LON1 - LON0) / (LAT1 - LAT0)
HGT = 0.355                                   # map height, figure fraction
W = HGT * MAP_ASPECT * (FIGH / FIGW)          # map width, figure fraction

TEXT_L, TEXT_R = 0.014, 0.300                 # the left column, in full
FACT_X = 0.104                                # where each fact's caption starts

yBot, yTop = 0.115, 0.545                     # bottoms of the two map rows
xA = 0.318                                    # left map column
CW = 0.009                                    # colourbar width
BAR_PAD = 0.013                               # map edge -> colourbar
BAR_SLOT = 0.061                              # colourbar + its ticks + its label
ARROW_SLOT = 0.036
xBar1 = xA + W + BAR_PAD                      # right of the FIRST map
xB = xA + W + BAR_SLOT + ARROW_SLOT           # right map column
xBar2 = xB + W + BAR_PAD                      # right of the second column


def load(month: str):
    import netCDF4
    import downscale_features as F
    import gridded_config as cfg

    res = os.path.join(ROOT, 'Results', 'downscaling')
    ds = netCDF4.Dataset(os.path.join(res, 'twsa_0p1deg_monthly_with_uncertainty.nc'))
    times = pd.to_datetime(ds['time'][:], unit='D', origin='1970-01-01')
    idx = int(np.where(times.strftime('%Y-%m') == month)[0][0])
    obs = bool(np.asarray(ds['grace_observed'][:])[idx])
    if not obs:
        raise SystemExit(f'{month} is a reconstructed month; pick a GRACE-observed '
                         f'one so the left panel shows a real observation')

    for v in ('twsa', 'sigma_total'):
        ds[v].set_auto_mask(False)
    fine = np.asarray(ds['twsa'][idx])
    sigma = np.asarray(ds['sigma_total'][idx])

    aux = F.load_aux()
    coarse = F.load_grace_monthly(times)[idx]
    # The mascon field is constant within a mascon; blanking cells outside the
    # basin keeps the left panel about this basin rather than its bounding box.
    coarse = np.where(np.asarray(aux['basin_frac_grace']) > 0, coarse, np.nan)

    grids = cfg.build_grids()
    summary = json.load(open(os.path.join(res, 'summary_xgboost.json')))
    holdouts = pd.read_csv(os.path.join(res, 'holdouts_month_xgboost.csv'))
    wells = pd.read_csv(os.path.join(res, 'well_validation_by_scale.csv'))
    per_well = pd.read_csv(os.path.join(res, 'well_metrics_per_well.csv'))

    return dict(month=month, fine=fine, sigma=sigma, coarse=coarse,
                basin=cfg.load_basin(), grids=grids, summary=summary,
                holdouts=holdouts.set_index('scheme'), wells=wells,
                per_well=per_well,
                # Counted, not asserted: a hand-typed "9,538 cells" goes stale
                # the first time the grid or the basin mask changes.
                n_cells=int(np.isfinite(fine).sum()),
                n_mascons=int(summary['n_mascons']),
                n_wells=int(len(per_well)),
                n_recon=int((~np.asarray(
                    ds['grace_observed'][:]).astype(bool)).sum()),
                n_months=len(times))


def _frame(ax, title, subtitle):
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    # One line, not a title plus a subtitle underneath. The subtitle sat in the
    # gap between a map and the colourbar below it, which is exactly where a
    # horizontal bar wants to be -- they collided at every size tried.
    ax.set_title(title, fontsize=10.5, fontweight='bold', color=INK,
                 loc='left', pad=4)
    ax.set_title(subtitle, fontsize=8.8, color=MUTED, loc='right', pad=4)


def _map(ax, field, grid, basin, title, subtitle, vmin, vmax,
         cmap=DIVERGING):
    lon, lat = grid.lon_centers(), grid.lat_centers()
    ext = [lon[0], lon[-1], min(lat[0], lat[-1]), max(lat[0], lat[-1])]
    shown = field[::-1] if lat[0] < lat[-1] else field
    im = ax.imshow(shown, extent=ext, origin='upper', cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    basin.boundary.plot(ax=ax, color=INK, linewidth=0.7, zorder=3)
    _frame(ax, title, subtitle)
    return im


def build(d, out_stem):
    s, h, w = d['summary'], d['holdouts'], d['wells']
    basin_row = w[w.level == 'basin'].iloc[0]
    cv = s['pooled_cv']
    pct = 100 * d['n_recon'] / d['n_months']
    pw = d['per_well']

    vmax = float(np.nanpercentile(np.abs(d['fine']), 98))
    vmax = round(vmax / 50) * 50 or 200
    smax = float(np.nanpercentile(d['sigma'], 98))
    # Symmetric about zero: for a bias, zero is "unbiased" and must sit on the
    # neutral colour rather than two thirds along a stretched ramp.
    mmax = float(np.nanpercentile(np.abs(pw.MBE), 90))

    fig = plt.figure(figsize=(FIGW, FIGH))

    # ---- left: the claim and the numbers -----------------------------------
    # Two lines, not one. On a 2.5:1 canvas a single-line headline at a size
    # worth reading is wider than the column and runs under the first map.
    fig.text(TEXT_L, 0.960, 'GRACE water storage,\ndownscaled to 0.1°',
             fontsize=20, fontweight='bold', color=INK, va='top',
             linespacing=1.15)
    # "Ganges", not "Ganga", in the one figure ScienceDirect renders directly
    # under the paper title. Everywhere the plate is not sitting next to that
    # title the repository uses the official "Ganga"; see the note in README.md.
    fig.text(TEXT_L, 0.775, 'Ganges basin  ·  2000–2025  ·  monthly and daily',
             fontsize=10.2, color=MUTED, va='top')
    fig.text(TEXT_L, 0.700,
             f'{d["n_mascons"]} GRACE mascons → {d["n_cells"]:,} cells, every\n'
             'mascon mean reproduced exactly, with\n'
             'per-pixel uncertainty published.',
             fontsize=10.2, color=INK, va='top', linespacing=1.45)
    # The maps are one month, and the figure used to say nowhere which one.
    fig.text(TEXT_L, 0.540,
             f'Maps: {pd.Timestamp(d["month"]).strftime("%B %Y")}, '
             f'a GRACE-observed month',
             fontsize=9.4, color=MUTED, va='top')

    facts = [('%.0f mm' % cv['RMSE'], 'held out in SPACE', 'leave-one-mascon-out'),
             ('%.0f mm' % h.loc['forward', 'RMSE_mean'], 'held out in TIME',
              'out-of-record'),
             ('r %.2f' % basin_row.downscaled_r, 'vs WELLS',
              'independent; bilinear %.2f' % basin_row.bilinear_r),
             ('%.0f%%' % pct, 'NOT observed', 'months reconstructed')]
    for i, (val, head, sub) in enumerate(facts):
        y = 0.455 - i * 0.095
        fig.text(TEXT_L, y, val, fontsize=15.5, fontweight='bold', color=INK,
                 va='center')
        fig.text(FACT_X, y + 0.021, head.upper(), fontsize=9.0, color=MUTED,
                 fontweight='bold', va='center')
        fig.text(FACT_X, y - 0.022, sub, fontsize=9.0, color=MUTED,
                 va='center')

    # Full width, and below every map: the only strip of the canvas where a
    # sentence this long does not have to be broken into six lines.
    fig.text(TEXT_L, 0.022,
             'Mascon-scale agreement is imposed by mass conservation and is '
             'therefore not evidence of skill; every number above is held out.\n'
             'Fine structure is inferred from ERA5-Land covariates, not '
             'observed. Red is water lost, blue is water gained.',
             fontsize=8.6, color=MUTED, va='bottom', linespacing=1.45)

    # ---- right: 2 x 2 maps, VERTICAL colourbars ----------------------------
    # Vertical rather than horizontal. A horizontal bar has to live in the gap
    # under a map, which is the same gap the next row's title wants, and at this
    # aspect ratio there is not room for both. Turned on its side, each bar sits
    # beside its own panel and competes with nothing.
    axA = fig.add_axes([xA, yTop, W, HGT])
    axB = fig.add_axes([xB, yTop, W, HGT])
    axC = fig.add_axes([xA, yBot, W, HGT])
    axD = fig.add_axes([xB, yBot, W, HGT])

    imA = _map(axA, d['coarse'], d['grids']['grace'], d['basin'],
               'What GRACE sees', f'{d["n_mascons"]} mascons, 3°', -vmax, vmax)
    imB = _map(axB, d['fine'], d['grids']['era5'], d['basin'],
               'What we produce', f'{d["n_cells"]:,} cells, 0.1°', -vmax, vmax)
    ims = _map(axC, d['sigma'], d['grids']['era5'], d['basin'],
               'How uncertain', 'per-pixel σ', 0, smax, cmap='Purples')

    # The least flattering panel, kept on purpose: bias against wells the model
    # never used, at the scale where point error is largest.
    d['basin'].boundary.plot(ax=axD, color=INK, linewidth=0.7, zorder=1)
    sc = axD.scatter(pw.lon, pw.lat, c=pw.MBE, cmap='RdBu_r', vmin=-mmax,
                     vmax=mmax, s=2.2, edgecolor='none', zorder=3)
    axD.set_aspect(axC.get_aspect())
    _frame(axD, 'Checked against wells', f'{d["n_wells"]:,} wells')

    def vbar(mappable, x, y, label, extend):
        cax = fig.add_axes([x, y, CW, HGT])
        cax.set_label('<colorbar>')          # so check_layout can tell them apart
        cb = fig.colorbar(mappable, cax=cax, orientation='vertical',
                          extend=extend)
        # Short labels only. A rotated label longer than the bar it names
        # overhangs the panel above and below it; the sign convention lives in
        # the footnote instead, where it has a whole line to itself.
        cb.set_label(label, fontsize=8.8, labelpad=2)
        cb.ax.tick_params(labelsize=8.0, length=2, pad=1.5)
        return cb

    # Every panel gets its own bar, immediately to its right, and the layout is
    # a plain 2 x 2 of map-then-bar. The two TWSA bars are deliberately
    # identical rather than shared: drawing the same limits twice is what makes
    # "these two maps are on one scale" visible, and a single shared bar left
    # the top right of the canvas empty while the bottom right was full.
    vbar(imA, xBar1, yTop, 'TWSA (mm)', 'both')
    vbar(imB, xBar2, yTop, 'TWSA (mm)', 'both')
    vbar(ims, xBar1, yBot, 'σ$_{total}$ (mm)', 'max')
    vbar(sc, xBar2, yBot, 'well bias (mm)', 'both')

    # Drawn, not typed: '→' at 24 pt depends on the glyph being in Arial, and
    # an arrow is one of the few things here that must never fall back to a
    # substitute font.
    ax0, ay = xA + W + BAR_SLOT, yTop + HGT * 0.5
    fig.add_artist(FancyArrowPatch(
        (ax0 + 0.004, ay), (ax0 + ARROW_SLOT - 0.004, ay),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=13,
        linewidth=1.6, color=INK, shrinkA=0, shrinkB=0))

    for ext_ in ('png', 'pdf'):
        path = f'{out_stem}.{ext_}'
        fig.savefig(path, dpi=DPI, facecolor='white')
        print(f'written: {path}  ({FIGW * DPI:.0f} x {FIGH * DPI:.0f} px)')
    return fig


def write_preview(out_stem):
    """Write the 500 x 200 px thumbnail ScienceDirect will actually show.

    The point of the whole layout is that it survives this reduction, and the
    only way to know whether a font size is large enough is to look at it at the
    size the reader gets, not at the size it was authored.

    This is a proof, NOT the deliverable. It is 500 x 200 actual pixels and
    carries no dpi tag, because at thumbnail size only the pixel count means
    anything. Submit Graphical_Abstract.png (3300 x 1320 at 300 dpi) or the PDF.
    """
    try:
        from PIL import Image
    except ImportError:
        print('preview skipped: Pillow not installed')
        return
    with Image.open(f'{out_stem}.png') as img:
        img.convert('RGB').resize((500, 200), Image.LANCZOS).save(
            f'{out_stem}_preview_500x200.png')
    print(f'written: {out_stem}_preview_500x200.png  (500 x 200 px — legibility '
          f'proof only, do NOT submit this one)')


def check_layout(fig, tol=1.0):
    """Re-measure what was actually drawn and fail on any collision.

    Every rectangle in this figure is hand-placed, so the usual guard --
    constrained_layout -- is not available. This is the substitute: it walks the
    rendered artists, takes the tight bounding box of each (which for a
    colourbar includes its ticks and its label, the parts that actually collide)
    and reports any pair that intersects by more than `tol` pixels, plus
    anything that leaves the canvas.
    """
    # Measure at the dpi the file is written at, so `tol` means the same thing
    # here as it does in the delivered PNG.
    fig.set_dpi(DPI)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []

    for i, t in enumerate(fig.texts):
        label = (t.get_text().splitlines() or [''])[0][:34]
        boxes.append((f'text[{i}] {label!r}', t.get_window_extent(r)))
    for ax in fig.axes:
        name = 'cbar' if ax.get_label() == '<colorbar>' else 'axes'
        # The axes box and its two titles are measured apart: a title may
        # legally sit outside the box, but it must not sit on a neighbour.
        boxes.append((f'{name} {ax.get_position().x0:.3f}',
                      ax.get_tightbbox(r) if name == 'cbar'
                      else ax.get_window_extent()))
        if name == 'axes':
            for loc in ('left', 'right'):
                t = {'left': ax._left_title, 'right': ax._right_title}[loc]
                if t.get_text():
                    boxes.append((f'title[{loc}] {t.get_text()[:22]!r}',
                                  t.get_window_extent(r)))

    bad = []
    for (n1, b1), (n2, b2) in itertools.combinations(boxes, 2):
        ov = mpl.transforms.Bbox.intersection(b1, b2)
        if ov is not None and ov.width > tol and ov.height > tol:
            bad.append(f'  OVERLAP  {n1}  x  {n2}   '
                       f'({ov.width:.0f} x {ov.height:.0f} px)')

    fw, fh = fig.canvas.get_width_height()
    for n, b in boxes:
        if b.x0 < -tol or b.y0 < -tol or b.x1 > fw + tol or b.y1 > fh + tol:
            bad.append(f'  OFF-CANVAS  {n}  '
                       f'[{b.x0:.0f},{b.y0:.0f},{b.x1:.0f},{b.y1:.0f}]')

    px = f'{fw} x {fh} px'
    if bad:
        print(f'LAYOUT FAILED ({px}), {len(bad)} problem(s):')
        print('\n'.join(bad))
        return False
    print(f'layout clean: {len(boxes)} elements, no overlaps, {px}')
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--month', default='2016-06',
                    help='YYYY-MM; must be a GRACE-observed month')
    ap.add_argument('--no-check', action='store_true',
                    help='skip the overlap check (it is on by default)')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, 'Graphical_Abstract')
    fig = build(load(args.month), stem)
    write_preview(stem)
    ok = args.no_check or check_layout(fig)
    plt.close(fig)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
