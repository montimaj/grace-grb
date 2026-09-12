"""
Supplementary figure: what each temporal holdout actually withholds.

WHY THIS FIGURE EXISTS
----------------------
A reviewer asked for the exact months used by the random, blocked and forward
validation experiments, and in particular whether the blocked experiment covers
the GRACE/GRACE-FO mission gap. The answer is a sentence long but it is really a
statement about a timeline, and a timeline is easier to check than a list of
sixty dates in prose: the blocked experiment CANNOT cover the mission gap,
because a month has to carry an observation before it can be withheld and then
scored, and the gap contains none.

So this figure draws the observation record and the three designs against the
same axis. The reader can see that the blocked blocks sit on either side of the
2017-2018 blackout rather than over it, that the random draws sample the whole
record, and that the forward split is a clean chronological cut.

HOW THE MONTHS ARE OBTAINED
---------------------------
Not retyped from the paper. The selection logic here is the same seeded
arithmetic as `downscale_holdouts.random_months`, `blocked_months` and
`forward_months`, run against the observation flag stored in the released
product, so the figure cannot drift away from the experiment it describes. If
the seed or the record changes, this figure changes with it.

USAGE
-----
    python make_fig3_holdouts.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'main'))
import plot_style as ps                                        # noqa: E402

OUT = os.path.join(HERE, 'output')
STEM = 'Fig3_holdouts'
PRODUCT = os.path.join(REPO, 'Results', 'downscaling',
                       'twsa_0p1deg_monthly_xgboost.nc')

# The pipeline defaults, restated here so the figure is self-describing.
SEED, TEST_FRAC, REPEATS, BLOCK, N_BLOCKS = 20, 0.2, 5, 11, 5

INK, MUTED, GRID = ps.SCI_INK, ps.SCI_MUTED, ps.SCI_GRID
C_RANDOM, C_BLOCKED, C_FORWARD = ps.SCI_BLUE, ps.SCI_ORANGE, ps.SCI_AQUA
C_OBS = '#6f6f6f'          # observed months in the record strip
C_TRAIN = '#c9c9c9'        # training months of the forward split
C_OUTSIDE = '#f2efe6'      # framework months outside the GRACE record



def load_observed():
    """Return (months DatetimeIndex, observed boolean array) from the product."""
    import netCDF4 as nc
    with nc.Dataset(PRODUCT) as ds:
        obs = np.asarray(ds['grace_observed'][:]).astype(bool)
        t = np.asarray(ds['time'][:], float)
        units = ds['time'].units
    base = pd.Timestamp(units.split('since')[-1].strip().split()[0])
    months = pd.DatetimeIndex([base + pd.Timedelta(days=float(v)) for v in t])
    return months, obs


def holdouts(obs):
    """Reproduce the three designs exactly as downscale_holdouts.py draws them."""
    idx = np.flatnonzero(obs)                       # observed month indices
    n_test = max(int(round(len(idx) * TEST_FRAC)), 1)

    rng = np.random.default_rng(SEED)
    random_draws = [np.sort(rng.choice(idx, size=n_test, replace=False))
                    for _ in range(REPEATS)]

    mset = set(idx.tolist())
    starts = [m for m in idx if all((m + k) in mset for k in range(BLOCK))]
    rng = np.random.default_rng(SEED)
    chosen, claimed = [], set()
    for k in rng.permutation(len(starts)):
        st = int(starts[k])
        span = set(range(st, st + BLOCK))
        if span & claimed:
            continue
        chosen.append(st)
        claimed |= span
        if len(chosen) >= N_BLOCKS:
            break
    blocks = [np.arange(st, st + BLOCK) for st in chosen]

    forward = idx[-n_test:]
    train = idx[:-n_test]
    return random_draws, blocks, forward, train, idx, starts


def gap_span(obs):
    """Longest run of consecutive unobserved months inside the record."""
    idx = np.flatnonzero(obs)
    miss = np.flatnonzero(~obs)
    inner = miss[(miss > idx[0]) & (miss < idx[-1])]
    best, run = (0, 0), [inner[0]]
    for a, b in zip(inner, inner[1:]):
        if b == a + 1:
            run.append(b)
        else:
            if len(run) > best[1] - best[0] + 1:
                best = (run[0], run[-1])
            run = [b]
    if len(run) > best[1] - best[0] + 1:
        best = (run[0], run[-1])
    return best


def bar(ax, y, cols, color, h=0.62):
    """Draw one month-wide rectangle per index in `cols`."""
    for c in np.atleast_1d(cols):
        ax.add_patch(Rectangle((c - 0.5, y - h / 2), 1.0, h,
                               facecolor=color, edgecolor='none'))


def build(months, obs):
    random_draws, blocks, forward, train, idx, starts = holdouts(obs)
    g0, g1 = gap_span(obs)

    rows, labels = [], []

    def row(label):
        rows.append(len(rows))
        labels.append(label)
        return len(rows) - 1

    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    y_rec = row('GRACE/GRACE-FO record')
    bar(ax, y_rec, idx, C_OBS, h=0.62)
    # The record is not one clean block. Counting the outages here, rather than
    # asserting them in a caption, keeps the number tied to the flag itself --
    # and it is the reason only a minority of months can start an 11-month run.
    runs, cur = [], []
    for i in range(idx[0], idx[-1] + 1):
        if not obs[i]:
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    n_out = sum(len(r) for r in runs)
    # Left edge on the first observed month, so the sentence starts where the
    # strip it describes starts rather than out in the pre-record margin.
    ax.text(idx[0], y_rec + 0.52,
            f'{len(idx)} of {idx[-1] - idx[0] + 1} months observed; '
            f'{len(runs)} separate outages totalling {n_out} months, '
            f'the longest being the transition gap',
            fontsize=6.8, color=MUTED, va='top', ha='left')

    row('')                                         # spacer
    y_rand = [row(f'draw {i + 1}' if i else 'Random  (5 draws x 45 months)')
              for i in range(REPEATS)]
    for y, d in zip(y_rand, random_draws):
        bar(ax, y, d, C_RANDOM, h=0.52)

    row('')
    y_blk = row('Blocked  (5 x 11 months)')
    for b in blocks:
        bar(ax, y_blk, b, C_BLOCKED, h=0.62)

    row('')
    y_fwd = row('Forward  (1 split)')
    bar(ax, y_fwd, train, C_TRAIN, h=0.62)
    bar(ax, y_fwd, forward, C_FORWARD, h=0.62)

    # The product framework runs Jan 2000 to Dec 2025, but GRACE only observes
    # Apr 2002 to Sep 2024. Shading the two ends says why the axis extends past
    # the data: those months are reconstruction, not measurement, and together
    # with the in-record outages they make up the 85 unobserved months.
    n_before, n_after = int(idx[0]), int(len(obs) - 1 - idx[-1])
    for x0, x1 in ((-2, idx[0] - 0.5), (idx[-1] + 0.5, len(obs) + 1)):
        ax.axvspan(x0, x1, color=C_OUTSIDE, zorder=0, lw=0)

    # The mission gap, marked once and carried down the whole figure.
    ax.axvspan(g0 - 0.5, g1 + 0.5, color=INK, alpha=0.065, zorder=0, lw=0)
    ax.annotate(f'GRACE/GRACE-FO gap, {months[g0]:%b %Y} – {months[g1]:%b %Y}\n'
                f'{g1 - g0 + 1} months with no observation to withhold',
                xy=((g0 + g1) / 2, -0.55), ha='center', va='bottom',
                fontsize=7.2, color=INK, linespacing=1.35,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8,
                                shrinkA=0, shrinkB=2),
                xytext=((g0 + g1) / 2, -2.35))

    ax.set_ylim(len(rows) - 0.4, -3.1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8.2, color=INK)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if lab.startswith('draw'):
            t.set_color(MUTED); t.set_fontsize(7.4)
        elif lab:
            t.set_fontweight('bold')
    # Spacer rows carry no label, so they must not carry a tick either.
    for t, lab in zip(ax.yaxis.get_major_ticks(), labels):
        if not lab:
            t.tick1line.set_visible(False)

    # Five-year labels so both ends of the product framework are named: at the
    # two-year spacing used before, the axis stopped at 2024 and adding 2025
    # collided with it. Unlabelled yearly minor ticks keep the finer structure
    # readable, which is what the two-year labels were really for.
    yrs = pd.date_range(months[0], months[-1], freq='2YS').tolist()
    if yrs[-1] != months[-1].replace(month=1, day=1):
        yrs.append(months[-1].replace(month=1, day=1))
    ax.set_xticks([months.get_loc(y) for y in yrs])
    ax.set_xticklabels([f'{y:%Y}' for y in yrs], fontsize=8.2, color=INK,
                       rotation=45, ha='right', rotation_mode='anchor')
    # 2000 and 2025 bound the released product, not the observations; the rest
    # of the axis is ordinary time.
    for t, y in zip(ax.get_xticklabels(), yrs):
        if y.year in (months[0].year, months[-1].year):
            t.set_fontweight('bold')
    ax.set_xticks([months.get_loc(y)
                   for y in pd.date_range(months[0], months[-1], freq='YS')],
                  minor=True)
    ax.set_xlim(-2, len(months) + 1)
    ax.tick_params(length=2.5, color=MUTED)
    ax.tick_params(axis='x', which='minor', length=2.0, color=MUTED)
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)

    handles = [Rectangle((0, 0), 1, 1, fc=c, ec='none') for c in
               (C_OBS, C_RANDOM, C_BLOCKED, C_FORWARD, C_TRAIN, C_OUTSIDE)]
    ax.legend(handles,
              ['observed month', 'withheld: random', 'withheld: blocked',
               'withheld: forward', 'forward training months',
               f'outside the GRACE record ({n_before} months before, '
               f'{n_after} after)'],
              loc='upper center', bbox_to_anchor=(0.5, -0.115), ncol=3,
              frameon=False, fontsize=7.6, handlelength=1.4,
              handleheight=0.8, columnspacing=1.4, labelcolor=INK)

    fig.tight_layout()
    return fig, (random_draws, blocks, forward, train, idx, starts, g0, g1)


def main() -> int:
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42,
                         'ps.fonttype': 42})
    months, obs = load_observed()
    fig, info = build(months, obs)
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, STEM)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=ps.DPI, bbox_inches='tight',
                    facecolor='white')
        print(f'written: {stem}.{ext}')

    random_draws, blocks, forward, train, idx, starts, g0, g1 = info
    print(f'\nobserved months            : {len(idx)}  '
          f'({months[idx[0]]:%Y-%m} to {months[idx[-1]]:%Y-%m})')
    print(f'mission gap                : {months[g0]:%Y-%m} to {months[g1]:%Y-%m}')
    print(f'random, months per draw    : {len(random_draws[0])} x {REPEATS} draws')
    print(f'eligible blocked starts    : {len(starts)} of {len(idx)} observed months')
    for b in sorted(blocks, key=lambda x: x[0]):
        print(f'  blocked block            : {months[b[0]]:%Y-%m} to {months[b[-1]]:%Y-%m}')
    print(f'forward withheld           : {months[forward[0]]:%Y-%m} to '
          f'{months[forward[-1]]:%Y-%m}  ({len(forward)} months)')
    print(f'forward training           : {months[train[0]]:%Y-%m} to '
          f'{months[train[-1]]:%Y-%m}  ({len(train)} months)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
