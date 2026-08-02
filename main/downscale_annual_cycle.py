"""
Annual cycle of the downscaled product, scored where scoring is possible.

WHY THIS READS OUT-OF-FOLD PREDICTIONS AND NOT THE PRODUCT
----------------------------------------------------------
The obvious version of this figure -- basin-mean product against basin-mean
GRACE, month by month -- cannot say anything. Mass conservation pins the
area-weighted mascon mean to the observation, so the basin mean of the published
field IS GRACE's. Measured over the 227 observed months it agrees at

    RMSE 0.27 mm,  r = 1.00000

which would draw two curves exactly on top of each other and a residual panel
flat at a quarter of a millimetre. A reader would take that for extraordinary
skill. It is arithmetic.

So the comparison here uses LEAVE-ONE-MASCON-OUT predictions: for each mascon,
the value predicted by a model fitted without that mascon or any of its
neighbours. Those predictions are free to disagree with GRACE, so the agreement
between them means something. This is the same series `downscale_model.py`
scores pooled; what this module adds is WHEN in the year the error falls, which
a pooled number cannot express.

WHAT THE FOUR PANELS SAY
------------------------
    (a) annual cycle       climatological month means, predicted vs observed.
                           Does the model reproduce the shape of the year?
    (b) monthly residual   predicted minus observed, by calendar month. A flat
                           line at zero is the ideal; structure here is seasonal
                           bias, which panel (a) hides by plotting two curves
                           that are close on a wide axis.
    (c) seasonal means     the same, aggregated to the four Indian seasons, with
                           +/- 1 sd across years so the spread the mean was
                           taken over is visible.
    (d) monthly cycle      month means with +/- 1 sd bands, the interannual
                           spread that panel (a) averages away.

Seasons are imported from `generate_gridded_maps` rather than redefined, so this
figure cannot drift from the seasonal maps beside it.

ONE MODEL, NOT SIX
------------------
The superseded basin-scale figure drew six models per panel. That was a model
COMPARISON; the comparison now happens once, in `tune_gridded.py`, which ranks
every candidate on the same grouped CV and writes the winner to
`selected_model.txt`. Redrawing five losers here would invite a reader to
re-adjudicate a choice already made on stronger evidence than an annual cycle.

Usage
-----
    python downscale_annual_cycle.py
    python downscale_annual_cycle.py --model xgboost
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

import downscale_model as M
import figure_captions
from generate_gridded_maps import SEASONS
from plot_style import DPI, SCI_BLUE, SCI_GRID, SCI_INK, SCI_MUTED, SCI_ORANGE
from utils import MONTH_NAMES


def basin_series(oof: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the cell-month table to one basin value per month.

    TWSA is rebuilt as level+trend plus anomaly before averaging, because the
    anomaly alone is centred by construction and its annual cycle would sit
    about zero rather than about the basin's actual storage state.

    Averaging is unweighted across cells. The samples are GRACE 0.5 degree cells
    whose areas differ by under 4 % across this basin's latitude span, which is
    far below the spread being plotted; area weighting would change the curves
    by less than the line width and require carrying a weight the CV does not.
    """
    d = oof.copy()
    d['date'] = pd.to_datetime(d['date'])
    base = d['base'] if 'base' in d.columns else 0.0
    d['obs_twsa'] = base + d['observed']
    d['pred_twsa'] = base + d['predicted']
    g = d.groupby('date').agg(observed=('obs_twsa', 'mean'),
                              predicted=('pred_twsa', 'mean'),
                              n_cells=('cell', 'size')).reset_index()
    g['month'] = g.date.dt.month
    g['residual'] = g.predicted - g.observed
    return g.sort_values('date').reset_index(drop=True)


def _season_of(month: int) -> Optional[str]:
    for label, months in SEASONS.items():
        if month in months:
            return label
    return None


def _style(ax):
    ax.grid(True, color=SCI_GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SCI_MUTED)
    ax.tick_params(colors=SCI_MUTED, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(SCI_INK)


def plot_annual_cycle(series: pd.DataFrame, model_name: str,
                      out_dir: str) -> str:
    """The four panels. One y-axis each; never two."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), layout='constrained')
    (ax_a, ax_b), (ax_c, ax_d) = axes

    by_month = series.groupby('month').agg(
        obs=('observed', 'mean'), obs_sd=('observed', 'std'),
        pred=('predicted', 'mean'), pred_sd=('predicted', 'std'),
        res=('residual', 'mean'), res_sd=('residual', 'std'),
        n=('observed', 'size')).reindex(range(1, 13))
    x = np.arange(1, 13)
    # Abbreviated: utils.MONTH_NAMES holds full names, and twelve of those on a
    # half-width axis overlap into an unreadable band.
    names = [MONTH_NAMES[i - 1][:3] for i in x]

    # (a) annual cycle. Observed is dashed and drawn on top: it is the reference,
    # and a solid prediction under a dashed reference reads correctly even where
    # the two overlap, which is most of the year.
    ax_a.plot(x, by_month.pred, color=SCI_ORANGE, lw=2, marker='s', ms=5,
              label=f'Predicted ({model_name}, out-of-fold)')
    ax_a.plot(x, by_month.obs, color=SCI_INK, lw=2, ls='--', marker='o', ms=5,
              label='Observed (GRACE)')
    ax_a.set_ylabel('TWS anomaly (mm)', color=SCI_INK, fontsize=9)
    ax_a.legend(frameon=False, fontsize=8, loc='upper left')

    # (b) the residual, on its own axis. In (a) the two curves span ~250 mm, so a
    # 20 mm seasonal bias is invisible; here it is the whole plot.
    ax_b.axhline(0, color=SCI_MUTED, lw=1)
    ax_b.plot(x, by_month.res, color=SCI_BLUE, lw=2, marker='o', ms=5)
    ax_b.fill_between(x, by_month.res - by_month.res_sd,
                      by_month.res + by_month.res_sd,
                      color=SCI_BLUE, alpha=0.16, lw=0)
    ax_b.set_ylabel('Predicted − observed (mm)', color=SCI_INK, fontsize=9)

    # (c) seasons, with the spread across years the mean was taken over.
    series = series.assign(season=series.month.map(_season_of))
    order = [s for s in SEASONS if (series.season == s).any()]
    by_season = series.groupby('season').agg(
        obs=('observed', 'mean'), obs_sd=('observed', 'std'),
        pred=('predicted', 'mean'), pred_sd=('predicted', 'std')).reindex(order)
    xs = np.arange(len(order))
    ax_c.errorbar(xs - 0.06, by_season.obs, yerr=by_season.obs_sd, fmt='o--',
                  color=SCI_INK, lw=1.8, ms=5, capsize=4, label='Observed ± 1 sd')
    ax_c.errorbar(xs + 0.06, by_season.pred, yerr=by_season.pred_sd, fmt='s-',
                  color=SCI_ORANGE, lw=1.8, ms=5, capsize=4,
                  label='Predicted ± 1 sd')
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([s.replace(' (', '\n(') for s in order], fontsize=8)
    ax_c.set_ylabel('TWS anomaly (mm)', color=SCI_INK, fontsize=9)
    ax_c.legend(frameon=False, fontsize=8, loc='upper left')

    # (d) the interannual spread panel (a) averages away. Bands, not bars, so the
    # overlap between them is readable.
    for mean, sd, colour, label, ls in (
            (by_month.obs, by_month.obs_sd, SCI_INK, 'Observed ± 1 sd', '--'),
            (by_month.pred, by_month.pred_sd, SCI_ORANGE, 'Predicted ± 1 sd', '-')):
        ax_d.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15, lw=0)
        ax_d.plot(x, mean, color=colour, lw=2, ls=ls, label=label)
    ax_d.set_ylabel('TWS anomaly (mm)', color=SCI_INK, fontsize=9)
    ax_d.legend(frameon=False, fontsize=8, loc='upper left')

    for ax in (ax_a, ax_b, ax_d):
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_xlabel('Month', color=SCI_INK, fontsize=9)
    ax_c.set_xlabel('Season', color=SCI_INK, fontsize=9)

    titles = ['(a) Annual cycle, out-of-fold prediction vs GRACE',
              '(b) Mean residual by calendar month',
              '(c) Seasonal means',
              '(d) Monthly cycle with interannual spread']
    for ax, ttl in zip((ax_a, ax_b, ax_c, ax_d), titles):
        _style(ax)
        ax.set_title(ttl, fontweight='bold', color=SCI_INK, loc='left',
                     fontsize=10, pad=8)

    fig.suptitle(f'Basin annual cycle, leave-one-mascon-out ({model_name})',
                 fontweight='bold', color=SCI_INK, fontsize=13, x=0.006,
                 ha='left')

    p = os.path.join(out_dir, f'Fig_annual_cycle_{model_name}.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    figure_captions.record(
        p,
        'Basin-mean total water storage anomaly by calendar month, comparing '
        'GRACE against leave-one-mascon-out predictions — each mascon predicted '
        'by a model fitted without it or any of its neighbours. The published '
        'product is NOT used here: mass conservation pins its basin mean to '
        'GRACE (r = 1.00000, RMSE 0.27 mm over the 227 observed months), so '
        'plotting it would draw one curve twice. (a) climatological month means. '
        '(b) the same difference on its own axis, where a seasonal bias of a few '
        'tens of mm is legible rather than lost in the ~250 mm range of (a). '
        '(c) the four Indian seasons, ± 1 sd across years. (d) month means with '
        '± 1 sd interannual bands. Only the selected model is drawn; the '
        'model comparison is made in tune_gridded.py on grouped cross-'
        'validation, not on an annual cycle.')
    return p


def summarise(series: pd.DataFrame) -> Dict[str, float]:
    """Numbers for the text, since the figure no longer carries a caption."""
    # Bracket access throughout. `groupby(...).observed` is a GroupBy ATTRIBUTE
    # (the categorical flag), so attribute access silently returns a bool here
    # instead of the column of the same name.
    g = series.groupby('month')
    by_month = g['residual'].mean()
    obs_cycle, pred_cycle = g['observed'].mean(), g['predicted'].mean()
    worst = by_month.abs().idxmax()
    return {
        'n_months': int(len(series)),
        'RMSE': float(np.sqrt(np.mean(series['residual'] ** 2))),
        'MBE': float(series['residual'].mean()),
        'amplitude_obs': float(obs_cycle.max() - obs_cycle.min()),
        'amplitude_pred': float(pred_cycle.max() - pred_cycle.min()),
        'worst_month': int(worst),
        'worst_month_bias': float(by_month.loc[worst]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=None,
                    help='Defaults to the model named in selected_model.txt.')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or M.RESULTS_DIR
    model = args.model
    if model is None:
        sel = os.path.join(os.path.dirname(M.RESULTS_DIR), 'tuning',
                           'selected_model.txt')
        model = (open(sel).read().strip() if os.path.exists(sel) else 'xgboost')

    oof_path = os.path.join(out_dir, f'lomo_oof_{model}.csv')
    if not os.path.exists(oof_path):
        print(f'  {os.path.basename(oof_path)} not found — run '
              f'downscale_model.py --model {model} first.')
        return 1

    print(f'Basin annual cycle from out-of-fold predictions ({model})\n')
    oof = pd.read_csv(oof_path)
    series = basin_series(oof)
    print(f'  {len(oof):,} cell-months -> {len(series)} basin months, '
          f'{series.date.min():%Y-%m} to {series.date.max():%Y-%m}')

    s = summarise(series)
    print(f'\n  basin-mean out-of-fold error: RMSE {s["RMSE"]:.1f} mm, '
          f'MBE {s["MBE"]:+.1f} mm')
    print(f'  seasonal amplitude: observed {s["amplitude_obs"]:.0f} mm, '
          f'predicted {s["amplitude_pred"]:.0f} mm '
          f'({100 * s["amplitude_pred"] / s["amplitude_obs"]:.0f}% of observed)')
    print(f'  largest monthly bias: {MONTH_NAMES[s["worst_month"] - 1]} '
          f'{s["worst_month_bias"]:+.1f} mm')

    p = plot_annual_cycle(series, model, out_dir)
    series.to_csv(os.path.join(out_dir, f'annual_cycle_{model}.csv'), index=False)
    caps = figure_captions.write_index(
        out_dir, title='Figure captions — well comparison and annual cycle')
    print(f'\nwritten:\n  {p}\n  annual_cycle_{model}.csv')
    if caps:
        print(f'  {caps}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
