"""
Figure 2 — method workflow, for the revised manuscript.

WHY THIS IS A REWRITE AND NOT A RESTYLE
---------------------------------------
The submitted Figure 2 describes a method the paper no longer uses. Every one of
its boxes is now wrong: GLDAS predictors (dropped — GLDAS 2.2 CLSM assimilates
GRACE, so the model was partly predicting the target from the target), "GWSA" as
a predictor (renamed `runoff_anom`; it contains no groundwater), daily-to-monthly
aggregation, a spatial holdout over synthetic replicated locations, and the
recurrent networks, which are not carried forward. It showed no spatial downscaling, no mass conservation,
no uncertainty decomposition and no daily disaggregation.

DESIGN
------
Category legend at the top, lane labels in grey down the left, one pastel fill
per category, and NEAR-BLACK TEXT EVERYWHERE. An earlier version coloured the
box text and the lane labels as well as the fills, which left five competing
hues and made the fills decorative rather than informative. Here the fill states
the category and nothing else has to.

Connectors are ORTHOGONAL: several boxes drop onto a horizontal bus, and one
arrow leaves the bus for the next stage. Diagonal arrows between box centres
crossed each other and crossed the boxes between them, and at this density they
read as noise rather than as flow.

WHAT THE DRAWING HAS TO CARRY
-----------------------------
* only the ANOMALY is fitted — level and trend come from GRACE;
* mass conservation is IMPOSED, not learned, so agreement with GRACE at mascon
  scale is not evidence of skill;
* the daily field is DISAGGREGATED — nothing is fitted at the daily scale;
* the model comparison is a real comparison — five candidates under one tuning
  protocol, ranked with a margin, rather than a preferred model asserted. An
  earlier draft dropped this to save space, which removed the one part of the
  method the reviewers asked to see substantiated.
"""

from __future__ import annotations

import os

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch,        # noqa: E402
                                Patch, Rectangle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')

MM = 1 / 25.4
# Sized for legibility at 100%, not for the smallest footprint that fits. The
# type is set first -- 7 pt detail, 8 pt headings -- and the plate is made large
# enough to hold it. A 183 mm plate with 5.9 pt detail was compact and hard to
# read on screen, which is where a reviewer will meet it.
FIG_W = 240 * MM
ASPECT = 0.70

INK = '#1a1a1a'
GREY = '#7a7a7a'
LINE = '#5c5c5c'

# TWO accents, not five.
#
# The earlier scheme gave every stage its own hue, and four of the six rows then
# had a single category across the whole row -- so the colour repeated what the
# row label on the left already said. Five hues and a five-entry legend were
# being spent to carry one binary distinction that shows up in exactly two rows:
# which boxes are IMPOSED rather than learned. That distinction is the paper's
# central claim -- level and trend are taken from GRACE and never fitted, mascon
# means are reproduced by construction -- and it was the one thing the palette
# buried, because 'Constraint' was just one of five equal-weight categories.
#
# So: neutral fills carry the flow, one accent marks what is imposed, and one
# marks what is published. Everything else is found by its row label, which is
# how a reader was locating it anyway.
NEUTRAL = ('#eef1f4', '#8d99a6')

CATS = {
    'Inputs':     NEUTRAL,
    'Model':      NEUTRAL,
    'Evaluation': NEUTRAL,
    # Imposed by construction, never fitted. Warm, because it is the exception
    # to everything else on the page.
    'Constraint': ('#f2c9be', '#b0524d'),
    # The deliverables. BLUE, not green -- and that is measured, not taste.
    #
    # The first version of this pair was a red and a green pastel, which is the
    # worst available choice: simulated, the two fills separated by dE 0.9 under
    # deuteranopia and 1.1 under protanopia, i.e. not at all, and each sat about
    # 3 from the neutral grey. The accents would have existed only for readers
    # with full colour vision. Red against blue moves the contrast onto the
    # blue-yellow axis, which both common deficiencies leave intact.
    #
    # Measured on these values (OKLab dE x100, Vienot dichromat simulation):
    #
    #                    constraint-product  constraint-grey  product-grey
    #     normal                    9.1            10.3            9.9
    #     protanopia                7.1            11.3            8.8
    #     deuteranopia              7.9             9.2           10.2
    #
    # Every pair clears the dE >= 6 floor and the vs-neutral separations clear 8,
    # so an accented box reads as accented under either deficiency. The
    # accent-to-accent figure sits below the dE >= 15 a chart would need if
    # colour were the sole carrier of identity -- here it is not: every box is
    # directly labelled, sits in a named row, and the borders separate by dE 14
    # to 23 under both simulations.
    'Products':   ('#bed6f2', '#2a5f9e'),
}

L, RIGHT = 0.163, 0.972
FULL = RIGHT - L


def box(ax, x, y, w, h, title, detail='', *, cat, fs_t=8.0, fs_d=7.0):
    fill, edge = CATS[cat]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.009',
        linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=3))
    if detail:
        ax.text(x + w / 2, y + h - 0.016, title, ha='center', va='top',
                fontsize=fs_t, color=INK, zorder=4, fontweight='bold')
        ax.text(x + w / 2, y + h - 0.048, detail, ha='center', va='top',
                fontsize=fs_d, color=INK, zorder=4, linespacing=1.6)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                fontsize=fs_t, color=INK, zorder=4, fontweight='bold')


def bus(ax, xs, y_top, y_bus, x_out, y_out):
    """
    Orthogonal manifold: drop from each source, run along a bus, one arrow out.

    Replaces one diagonal per source-target pair. With four inputs feeding two
    stages the diagonals crossed each other and passed through boxes.
    """
    for x in xs:
        ax.plot([x, x], [y_top, y_bus], color=LINE, lw=0.8, zorder=2)
    ax.plot([min(xs + [x_out]), max(xs + [x_out])], [y_bus, y_bus],
            color=LINE, lw=0.8, zorder=2)
    ax.add_patch(FancyArrowPatch((x_out, y_bus), (x_out, y_out),
                                 arrowstyle='-|>', mutation_scale=8, lw=0.8,
                                 color=LINE, shrinkA=0, shrinkB=1, zorder=2))


def drop(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle='-|>',
                                 mutation_scale=8, lw=0.8, color=LINE,
                                 shrinkA=1, shrinkB=1, zorder=2))


def across(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>',
                                 mutation_scale=8, lw=0.8, color=LINE,
                                 shrinkA=1, shrinkB=1, zorder=2))


def lane(ax, y, text):
    ax.text(0.004, y, text, ha='left', va='center', fontsize=7.4, color=GREY,
            fontweight='bold', linespacing=1.6)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * ASPECT))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # No title on the plate: the journal sets it in the caption, and a heading
    # here would be printed twice.
    # Two keys, and each says what the colour MEANS rather than naming the stage.
    # "Constraint" and "Products" were stage names, which the row labels already
    # give; a legend that repeats the row labels is not doing anything. These
    # name the property a reader cannot get anywhere else on the plate.
    #
    # Laid out by matplotlib rather than by hand. The hand-placed version
    # advanced x by a per-character width estimate, which under-measured the
    # longer label and printed the second swatch on top of the end of the first.
    # Estimating text width in axes fractions is not something to do by eye.
    keys = [('Constraint', 'constraint'),
            ('Products', 'published product')]
    ax.legend(handles=[Patch(facecolor=CATS[c][0], edgecolor=CATS[c][1],
                             linewidth=1.0, label=lab) for c, lab in keys],
              loc='lower center', bbox_to_anchor=(0.5, 0.965), ncol=2,
              frameon=False, fontsize=8.0, handlelength=1.6, handleheight=1.1,
              columnspacing=2.4, handletextpad=0.7)

    # ---------------- inputs ----------------
    y, h = 0.835, 0.105
    w = (FULL - 2 * 0.018) / 3
    xs = [L + i * (w + 0.018) for i in range(3)]
    lane(ax, y + h / 2, 'INPUTS')
    box(ax, xs[0], y, w, h, 'GRACE / GRACE-FO',
        'JPL RL06 mascons, 3°\n19 in-basin, monthly', cat='Inputs')
    box(ax, xs[1], y, w, h, 'ERA5-Land',
        '0.1°, daily → monthly\nP · ET · runoff · soil water', cat='Inputs')
    box(ax, xs[2], y, w, h, 'Covariates',
        'terrain · soil · water table\nirrigated cropland', cat='Inputs')

    # ---------------- target and features ----------------
    y2, h2 = 0.684, 0.100
    wa = FULL * 0.455
    lane(ax, y2 + h2 / 2, 'TARGET\nAND FEATURES')
    box(ax, L, y2, wa, h2, 'Target decomposition',
        'TWSA = level + trend·t + anomaly\n'
        'level and trend taken from GRACE: never fitted', cat='Constraint')
    # Families, not counts or window lengths. A workflow diagram that names the
    # antecedent windows and the exact column count has to be redrawn whenever
    # the design matrix changes -- and it changed twice during the revision. The
    # numbers live in METHODS and in the released ablation table, which are the
    # places a reader can check them against something.
    box(ax, L + wa + 0.020, y2, FULL - wa - 0.020, h2, 'Feature construction',
        'storage memory · neighbourhood context\n'
        'climatology · seasonality',
        cat='Inputs')

    drop(ax, xs[0] + w / 2, y, y2 + h2)
    bus(ax, [xs[1] + w / 2, xs[2] + w / 2], y, y - 0.030,
        L + wa + 0.020 + (FULL - wa - 0.020) / 2, y2 + h2)

    # ---------------- model comparison ----------------
    y3, h3 = 0.473, 0.160
    lane(ax, y3 + h3 / 2, 'MODEL\nCOMPARISON')
    box(ax, L, y3, FULL, h3, 'Five candidates under one protocol',
        'RandomForest · XGBoost · LightGBM · XGBRF · MLP\n'
        'Optuna on mascon-grouped CV with fold-level pruning; the MLP is scored at a\n'
        'fixed configuration from a released 21-point sweep, not searched each run\n'
        'ranked on the same folds; the margin is judged against each model’s spread',
        cat='Model', fs_d=6.9)
    bus(ax, [L + wa / 2, L + wa + 0.020 + (FULL - wa - 0.020) / 2], y2,
        y2 - 0.026, L + FULL / 2, y3 + h3)

    # ---------------- downscale and constrain ----------------
    y4, h4 = 0.317, 0.105
    wc = (FULL - 0.020) / 2
    lane(ax, y4 + h4 / 2, 'DOWNSCALE\nAND CONSTRAIN')
    box(ax, L, y4, wc, h4, 'Spatial downscaling',
        'the selected model is fitted across mascons\nand applied at 0.1°',
        cat='Model')
    box(ax, L + wc + 0.020, y4, wc, h4, 'Mass conservation',
        'minimum-norm correction; every mascon mean\n'
        'reproduced exactly: imposed, not learned', cat='Constraint')
    drop(ax, L + wc / 2, y3, y4 + h4)
    across(ax, L + wc, L + wc + 0.020, y4 + h4 / 2)

    # ---------------- products ----------------
    y5, h5 = 0.161, 0.105
    lane(ax, y5 + h5 / 2, 'PRODUCTS')
    box(ax, L, y5, wc, h5, '0.1° MONTHLY TWSA',
        # Mathtext, so the subscripts are real subscripts. \mathrm keeps the
        # subscript upright: a bare \sigma_{grace} sets 'grace' in maths
        # italic, which reads as a product of five variables.
        '2000–2025 · five per-pixel terms\n'
        r'$\sigma_{\mathrm{GRACE}}$, $\sigma_{\mathrm{transfer}}$, '
        r'$\sigma_{\mathrm{within}}$, $\sigma_{\mathrm{gap}}$ $\rightarrow$ '
        r'$\sigma_{\mathrm{total}}$',
        cat='Products')
    box(ax, L + wc + 0.020, y5, wc, h5, '0.1° DAILY TWSA',
        'water-balance shape; monthly mean preserved\n'
        'exactly: disaggregated, not downscaled', cat='Products')
    drop(ax, L + wc / 2, y4, y5 + h5)
    across(ax, L + wc, L + wc + 0.020, y5 + h5 / 2)

    # ---------------- evidence ----------------
    y6, h6 = 0.010, 0.100
    we = (FULL - 3 * 0.013) / 4
    xe = [L + i * (we + 0.013) for i in range(4)]
    lane(ax, y6 + h6 / 2, 'EVALUATION')
    box(ax, xe[0], y6, we, h6, 'Leave-one-mascon-out',
        '19 mascons, each with\nits neighbour buffer', cat='Evaluation',
        fs_t=7.2)
    box(ax, xe[1], y6, we, h6, 'Month holdouts',
        'random (optimistic)\nblocked · forward', cat='Evaluation', fs_t=7.6)
    box(ax, xe[2], y6, we, h6, 'CGWB wells',
        '656 wells, quarterly; per well,\nper mascon, basin mean', cat='Evaluation', fs_t=7.6)
    box(ax, xe[3], y6, we, h6, 'TreeSHAP',
        'by feature family;\nthe anomaly only', cat='Evaluation', fs_t=7.6)
    bus(ax, [L + wc / 2, L + wc + 0.020 + wc / 2], y5, y5 - 0.024,
        L + FULL / 2, y6 + h6)

    for e in ('pdf', 'png'):
        path = os.path.join(OUT, f'Fig2_workflow.{e}')
        fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f'written: {path}')
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
