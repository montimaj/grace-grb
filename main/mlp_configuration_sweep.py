"""
The MLP's configuration, chosen by measurement and released in full.

Why this exists
---------------
The gridded comparison includes one neural model so that it is not four
variations on a single idea. That model loses to the tree ensembles, which is
the expected outcome for tabular data (Grinsztajn et al., 2022). The obvious
objection to any such result is that the network was simply not given a fair
configuration -- and because the MLP is deliberately excluded from the per-run
Optuna search (`tune_gridded.FIXED_CONFIG_MODELS`), that objection has to be
answered with evidence rather than with a search budget.

This module is that evidence. It sweeps the MLP's configuration along both axes
that matter, under the SAME grouped spatial cross-validation the models are
ranked by, and releases every point rather than only the winner. The result is
more disclosive than a fixed number of opaque TPE trials would be: a reader can
see the whole response surface and judge whether the adopted setting sits in a
sensible place.

What the sweeps establish
-------------------------
1. WIDTH is not the constraint and DEPTH is actively harmful. Skill is flat
   across an 8x range of hidden units but degrades sharply with a second hidden
   layer. So the small adopted network is the MLP's BEST case, not a handicap --
   the opposite of the usual "the baseline was under-sized" criticism.

   Width and depth are reported separately for a reason: a single "capacity"
   statistic averages a flat axis with a steep one and describes neither. An
   earlier version of this analysis pooled them and read the result as a
   monotonic capacity effect, which the wider sweep does not support.

2. LEARNING RATE is the axis that matters, and its optimum is INTERIOR to the
   swept range -- there is a turning point, not a grid edge. A sweep whose best
   point sits at a boundary has not found an optimum, only a stopping place, so
   the range is extended until the curve turns.

3. ALPHA is flat across four orders of magnitude, so the adopted value is not
   load-bearing.

The tree reference is run on identical folds so the comparison is not across
protocols.

Cost and cadence
----------------
~50 fits, roughly an hour. This is a ONE-OFF that settles a configuration, not
part of the per-run pipeline -- which is the entire point of fixing the MLP
rather than re-searching it on every run.

Usage
-----
    python mlp_configuration_sweep.py               # full sweep, 3 folds
    python mlp_configuration_sweep.py --folds 5     # matches the tuner exactly
    python mlp_configuration_sweep.py --plot-only   # redraw from the CSV
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import downscale_features as F
import downscale_model as M
import gridded_config as cfg
from plot_style import (BAR_3, DPI, SCI_BLUE, SCI_GRID, SCI_INK, SCI_MUTED,
                        pretty_model)

OUT_DIR = os.path.join(os.path.dirname(M.RESULTS_DIR), 'tuning')
CSV_PATH = os.path.join(OUT_DIR, 'mlp_configuration_sweep.csv')
FIG_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'Results', 'figures', 'Fig_mlp_configuration_sweep.png'))

# Hidden-layer stacks, shallow-to-deep. Widths span a 21x range in weight count
# so "capacity does not help" is a claim about a range, not two points.
ARCHITECTURES: Sequence[Tuple[int, ...]] = [
    (32,), (64,), (128,), (256,), (128, 64), (256, 128)]

# Learning rates bracket the optimum on BOTH sides -- see the module docstring
# on why a boundary optimum would not count.
LEARNING_RATES = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
ALPHAS = [1e-5, 1e-3, 1e-1]

# The architecture the regularisation axis is swept at, and the alpha the
# learning-rate axis is swept at. Both are the winners of their own sweep.
REF_ARCH: Tuple[int, ...] = (64,)
REF_ALPHA = 1e-3

TREE_REFERENCE = ('xgboost', 'random_forest')


def _cv(params: Optional[Dict], model: str, X, y, splits,
        seed: int) -> Tuple[float, float, List[int]]:
    """Pooled RMSE over the folds, wall time, and epochs used (MLP only)."""
    errs: List[np.ndarray] = []
    iters: List[int] = []
    t0 = time.time()
    for tr, te in splits:
        mdl = M.make_model(model, seed=seed, params=params)
        mdl.fit(X[tr], y[tr])
        errs.append((mdl.predict(X[te]) - y[te]) ** 2)
        if hasattr(mdl, 'named_steps'):
            iters.append(int(mdl.named_steps['mlp'].n_iter_))
    return float(np.sqrt(np.mean(np.concatenate(errs)))), time.time() - t0, iters


def sweep(folds: int = 3, seed: int = 20) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold

    fs = F.build_features(verbose=True)
    _, anom, _ = M.decompose_target(fs)
    obs = fs.observed
    X = fs.X.to_numpy(dtype='float32')[obs]
    y = anom[obs]
    groups = fs.mascon[obs]
    splits = list(GroupKFold(n_splits=folds).split(X, y, groups))
    print(f'\n{X.shape[0]:,} samples x {X.shape[1]} features, '
          f'{len(np.unique(groups))} mascons, GroupKFold({folds})\n')

    rows: List[Dict] = []

    # Stamped on every row. A sweep is only interpretable against the design
    # matrix it was run on, and this file's numbers were once quoted in METHODS
    # and the response letter after the predictor set changed underneath them.
    # Carrying the provenance in the CSV makes that mismatch visible, not silent.
    n_features = X.shape[1]
    predictors = ','.join(cfg.PREDICTORS)

    def record(axis, label, params, model='mlp', **extra):
        rmse, secs, iters = _cv(params, model, X, y, splits, seed)
        rows.append(dict(axis=axis, model=model, label=label, rmse=rmse,
                         seconds=round(secs, 1),
                         epochs_min=min(iters) if iters else None,
                         epochs_max=max(iters) if iters else None,
                         n_features=n_features, predictors=predictors, **extra))
        ep = f'  epochs {min(iters)}-{max(iters)}' if iters else ''
        print(f'  {label:26s} RMSE {rmse:8.3f}   {secs:5.0f}s{ep}', flush=True)
        return rmse

    print('1. Capacity, at the default learning rate and alpha')
    for arch in ARCHITECTURES:
        name = '-'.join(str(a) for a in arch)
        sizes = [X.shape[1]] + list(arch) + [1]
        record('capacity', name,
               dict(M.DEFAULT_PARAMS['mlp'], hidden_layer_sizes=arch),
               hidden_layer_sizes=name,
               n_weights=int(sum(sizes[i] * sizes[i + 1]
                                 for i in range(len(sizes) - 1))))

    arch_txt = '-'.join(str(a) for a in REF_ARCH)
    print(f'\n2. Learning rate and alpha, at ({arch_txt})')
    for alpha in ALPHAS:
        for lr in LEARNING_RATES:
            # A slower rate needs more epochs; without headroom the fit would
            # truncate and the point would measure the cap, not the rate.
            record('regularisation', f'alpha={alpha:g}, lr={lr:g}',
                   dict(M.DEFAULT_PARAMS['mlp'], hidden_layer_sizes=REF_ARCH,
                        alpha=alpha, learning_rate_init=lr, max_iter=3000),
                   hidden_layer_sizes=arch_txt, alpha=alpha, learning_rate=lr)

    # The reference uses each tree's ADOPTED configuration -- `params=None` sends
    # make_model through load_gridded_params, which reads the tuned JSON and
    # falls back to the hand-set defaults when it is absent.
    #
    # This is why the sweep belongs AFTER tuning in the pipeline. Comparing the
    # MLP's best of 21 configurations against trees left at hand-set defaults
    # would flatter the network on a technicality, and the whole point of this
    # module is that the MLP's loss should be hard to dismiss. Run before tuning,
    # it silently reverts to the defaults -- which is a valid comparison, just a
    # different and weaker one, so the CSV records which happened.
    tuned_available = os.path.exists(M.GRIDDED_PARAM_PATH)
    print(f'\n3. Tree reference, identical folds '
          f'({"adopted/tuned" if tuned_available else "hand-set defaults - no tuning file"})')
    for name in TREE_REFERENCE:
        record('reference', pretty_model(name), None, model=name,
               reference_config='adopted' if tuned_available else 'defaults')

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f'\nwritten: {CSV_PATH}')
    return df


def _summarise(df: pd.DataFrame) -> None:
    mlp = df[df.model == 'mlp']
    cap = mlp[mlp.axis == 'capacity']
    reg = mlp[mlp.axis == 'regularisation']
    best = mlp.loc[mlp.rmse.idxmin()]
    ref = df[df.axis == 'reference'].sort_values('rmse')

    print('\n' + '=' * 68)
    if 'n_features' in df.columns and df.n_features.notna().any():
        print(f'design matrix: {int(df.n_features.dropna().iloc[0])} features '
              f'[{df.predictors.dropna().iloc[0]}]')
    print(f'best MLP of {len(mlp)} configurations: {best.label}  '
          f'RMSE {best.rmse:.3f}')
    if len(cap) > 2:
        # WIDTH and DEPTH are reported separately because they do NOT behave the
        # same way, and a single "capacity" statistic averages the two into
        # something untrue of either. A rank correlation over all architectures
        # returns "no clear trend" here purely because a flat width axis and a
        # steep depth axis cancel.
        cap = cap.assign(n_layers=cap.hidden_layer_sizes.astype(str)
                         .str.count('-').add(1))
        flat = cap[cap.n_layers == 1].sort_values('n_weights')
        deep = cap[cap.n_layers > 1]
        if len(flat) > 1:
            spread = 100 * (flat.rmse.max() - flat.rmse.min()) / flat.rmse.min()
            print(f'  width: {flat.rmse.min():.3f}-{flat.rmse.max():.3f} over '
                  f'{flat.n_weights.max() / flat.n_weights.min():.0f}x '
                  f'({spread:.2f}%) -- '
                  f'{"FLAT, width is not the constraint" if spread < 2 else "width matters"}')
        if len(deep) and len(flat):
            pen = 100 * (deep.rmse.min() - flat.rmse.min()) / flat.rmse.min()
            print(f'  depth: best 2-layer {deep.rmse.min():.3f} vs best '
                  f'1-layer {flat.rmse.min():.3f} ({pen:+.2f}%) -- '
                  f'{"a second layer HURTS" if pen > 2 else "depth is neutral"}')
    if len(reg):
        by_lr = reg.groupby('learning_rate').rmse.min()
        best_lr = by_lr.idxmin()
        interior = best_lr not in (min(by_lr.index), max(by_lr.index))
        print(f'  best learning rate {best_lr:g} -- '
              f'{"INTERIOR to the swept range" if interior else "AT A GRID EDGE (extend it)"}')
        by_a = reg.groupby('alpha').rmse.min()
        print(f'  alpha spread {100 * (by_a.max() - by_a.min()) / by_a.min():.2f}% '
              f'over {max(ALPHAS) / min(ALPHAS):g}x')
    for _, r in ref.iterrows():
        print(f'  vs {r.label}: {100 * (best.rmse - r.rmse) / r.rmse:+.2f}% '
              f'({r.rmse:.3f})')
    print('=' * 68)


def plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cap = df[(df.axis == 'capacity')].sort_values('n_weights')
    reg = df[(df.axis == 'regularisation')]
    ref = df[df.axis == 'reference'].sort_values('rmse')
    if cap.empty or reg.empty:
        print('nothing to plot')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    # A BAND across the tree references, not a line at whichever won.
    #
    # Labelling one "best tree" asserts an ordering the data does not support:
    # on the 78-feature build the two references sat 0.028 mm apart (0.036%),
    # far inside their own fold-to-fold spread. The band states that the trees
    # cluster HERE, which is the only claim this figure needs -- what matters is
    # that the MLP sits outside it, not which tree edged the other.
    tree_lo = float(ref.rmse.min()) if len(ref) else None
    tree_hi = float(ref.rmse.max()) if len(ref) else None
    tree_label = (f'Tree models (n={len(ref)})' if len(ref) > 1
                  else (ref.iloc[0].label if len(ref) else ''))

    # -- (a) capacity ----------------------------------------------------
    # Two series, because width and depth behave differently: joining them into
    # one line against weight count would imply a single capacity trend that
    # the numbers do not show.
    cap = cap.assign(n_layers=cap.hidden_layer_sizes.astype(str)
                     .str.count('-').add(1))
    for lay, colour, mark, lab in ((1, SCI_BLUE, 'o', 'One hidden layer'),
                                   (2, BAR_3[1], 's', 'Two hidden layers')):
        grp = cap[cap.n_layers == lay].sort_values('n_weights')
        if grp.empty:
            continue
        ax1.plot(grp.n_weights, grp.rmse, '-', marker=mark, color=colour,
                 lw=2, ms=7, zorder=3, label=lab)
    for i, (_, r) in enumerate(cap.sort_values('n_weights').iterrows()):
        # Alternating offsets: adjacent points sit close on a log axis and
        # a constant offset makes neighbouring labels overlap.
        ax1.annotate(r.hidden_layer_sizes, (r.n_weights, r.rmse),
                     textcoords='offset points',
                     xytext=(0, 10) if i % 2 == 0 else (0, -15),
                     ha='center', fontsize=7, color=SCI_MUTED)
    if tree_lo is not None:
        if tree_hi - tree_lo < 1e-9:
            ax1.axhline(tree_lo, color=SCI_INK, lw=1.6, ls='--', zorder=2,
                        label=tree_label)
        else:
            ax1.axhspan(tree_lo, tree_hi, color=SCI_INK, alpha=0.13, lw=0,
                        zorder=1, label=tree_label)
            for _v in (tree_lo, tree_hi):
                ax1.axhline(_v, color=SCI_INK, lw=1.0, ls='--', zorder=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('Weights in the network', color=SCI_INK)
    ax1.set_ylabel('Grouped-CV RMSE (mm)', color=SCI_INK)
    ax1.set_title('a. Width is flat; depth hurts', fontsize=10,
                  color=SCI_INK, loc='left')

    # -- (b) learning rate, one line per alpha ---------------------------
    for colour, (alpha, grp) in zip(BAR_3, reg.groupby('alpha')):
        grp = grp.sort_values('learning_rate')
        ax2.plot(grp.learning_rate, grp.rmse, '-o', color=colour, lw=2, ms=6,
                 zorder=3, label=f'alpha = {alpha:g}')
    if tree_lo is not None:
        if tree_hi - tree_lo < 1e-9:
            ax2.axhline(tree_lo, color=SCI_INK, lw=1.6, ls='--', zorder=2,
                        label=tree_label)
        else:
            ax2.axhspan(tree_lo, tree_hi, color=SCI_INK, alpha=0.13, lw=0,
                        zorder=1, label=tree_label)
            for _v in (tree_lo, tree_hi):
                ax2.axhline(_v, color=SCI_INK, lw=1.0, ls='--', zorder=2)
    ax2.set_xscale('log')
    ax2.set_xlabel('Initial learning rate', color=SCI_INK)
    ax2.set_ylabel('Grouped-CV RMSE (mm)', color=SCI_INK)
    ax2.set_title('b. The optimum is interior, not at an edge', fontsize=10,
                  color=SCI_INK, loc='left')

    for ax in (ax1, ax2):
        ax.grid(True, color=SCI_GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(SCI_MUTED)
        ax.tick_params(colors=SCI_MUTED, labelsize=8)
        ax.legend(frameon=False, fontsize=7.5, loc='upper left')

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'written: {FIG_PATH}')


def check_stale(path: str = CSV_PATH) -> int:
    """
    Does the released sweep still describe the CURRENT design matrix?

    Cheap on purpose: it compares the predictor list recorded in the CSV against
    `cfg.PREDICTORS` and never builds a feature matrix, so the pipeline can call
    it on every run for free.

    This exists because the failure it catches already happened. The sweep was
    run on an 80-feature matrix, its numbers were quoted in METHODS and the
    response letter, and then `gwsa` was dropped from the predictors -- leaving
    published figures describing a design matrix the project no longer used, with
    nothing anywhere to flag it. A sweep that is deliberately NOT re-run on every
    pipeline pass needs something that notices when it has gone stale.
    """
    if not os.path.exists(path):
        print(f'  MLP sweep absent ({os.path.basename(path)}). The MLP ships a '
              f'FIXED configuration,\n  so its evidence should exist: run '
              f'`python mlp_configuration_sweep.py`.')
        return 1
    df = pd.read_csv(path)
    if 'predictors' not in df.columns or df.predictors.isna().all():
        print('  MLP sweep predates provenance stamping, so it cannot be matched '
              'to a design\n  matrix. Re-run `python mlp_configuration_sweep.py`.')
        return 1
    was = str(df.predictors.dropna().iloc[0]).split(',')
    now = list(cfg.PREDICTORS)
    if was != now:
        print(f'  STALE MLP sweep. It was measured on predictors {was},\n'
              f'  the current set is {now}. Every number it reports -- and every '
              f'figure quoting\n  them -- describes a design matrix this project '
              f'no longer uses.\n  Re-run: python mlp_configuration_sweep.py')
        return 1
    n = int(df.n_features.dropna().iloc[0]) if 'n_features' in df.columns else -1
    print(f'  MLP sweep matches the current predictor set ({n} features).')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds', type=int, default=3,
                    help='GroupKFold splits. 5 matches the tuner; 3 is cheaper '
                         'and the RANKING across configurations is what matters.')
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--plot-only', action='store_true',
                    help='Redraw the figure from an existing CSV.')
    ap.add_argument('--check-stale', action='store_true',
                    help='Exit non-zero if the released sweep was measured on a '
                         'different predictor set. Cheap; builds nothing.')
    args = ap.parse_args()

    if args.check_stale:
        return check_stale()

    if args.plot_only:
        if not os.path.exists(CSV_PATH):
            print(f'{CSV_PATH} does not exist - run the sweep first')
            return 1
        df = pd.read_csv(CSV_PATH)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = sweep(args.folds, args.seed)

    _summarise(df)
    plot(df)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
