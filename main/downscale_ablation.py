"""
Feature ablation for the gridded model: does the design matrix earn its size?

The design carries a large MEMORY block -- antecedent rolling means (predictors x
windows) plus exponential APIs. Measured on the real feature matrix, that block
is far smaller than it looks:

* PCA on the memory columns: 10 components explain 97.2% of variance, 14 reach
  99%. So ~42 columns encoded ~10-14 independent signals.
* Consecutive antecedent windows correlate 0.59-0.98. The 12- and 24-month
  windows sit at 0.96-0.98 -- the 24-month one is nearly free information.

Against ~19 independent mascons, that is a lot of parameters for the evidence
available. This module measures whether trimming costs anything.

The one rung that is not noise
------------------------------
`drop_runoff_anom` is the only configuration in this ladder measured to move the
score by more than the repeat-to-repeat spread: +0.66 mm against a spread of
0.35 mm, i.e. 1.9x. `thin_ante`, `lean` and `minimal` all land INSIDE the spread
and are indistinguishable from doing nothing -- `minimal` even scores marginally
*below* `full` while using 22 fewer features, which is what noise looks like, not
a finding.

`runoff_anom` is exactly `runoff_surface - runoff_surface_clim_mean` (residual
1.7e-6, float32 storage noise) with both parents already present, so what it buys
is a CENTRING effect rather than new information. Its own two climatology columns
are degenerate by construction and are no longer built at all -- see
`downscale_features.NO_CLIMATOLOGY`; the duplicate among them mattered beyond
waste, because two identical columns SPLIT SHAP credit and made the shared
feature read as about half as important as it is.

What is scored
--------------
The same objective `tune_gridded.py` optimises: RMSE of the ANOMALY under
GroupKFold grouped by mascon. Using one objective for tuning and ablation means
the two are directly comparable; using held-out skill rather than fit means a
configuration cannot win by having more parameters.

Every configuration is scored with the SAME hand-set xgboost hyperparameters, and
deliberately not with tuned ones: tuning is performed on the full matrix, so
tuned values would favour the full configuration over every trim. That makes this
module's answer independent of whether tuning has run, which is also why it can
sit before or after the tuning step without changing.

Each configuration is a full rebuild of the design matrix, not a column subset of
the full one. That matters: dropping `gwsa` from `PREDICTORS` also removes its
antecedent, context and climatology derivatives, which is the actual saving.

Reading the result
------------------
Compare against the SPREAD, not between point estimates. Every repeat uses the
same fold partition across configurations, so the comparison is PAIRED and a
paired test is the right one.

Result on the 69-feature build (3 repeats, xgboost defaults), i.e. after `gwsa`
was removed from the default predictors:

    config      n    RMSE            dRMSE vs full   dRMSE %
    full       69    78.02 +/- 0.35       --           --
    with_gwsa  80    77.36 +/- 0.33     -0.66        -0.85
    thin_ante  57    78.23 +/- 0.38     +0.21        +0.27
    lean       54    78.37 +/- 0.20     +0.35        +0.44
    minimal    47    78.00 +/- 0.62     -0.02        -0.03

Read the PAIRED columns, not this one. `minimal` sits 0.02 mm below `full` and a
sorted table makes that look like a win; it is a SIXTEENTH of `full`'s own
repeat-to-repeat spread, and the curve is non-monotonic (69 low, 57 high, 54
higher, 47 low again), which is what noise looks like when feature count is not
actually driving the result. Two things temper the table further:

* LINEAR REDUNDANCY IS NOT REDUNDANCY FOR A TREE. XGBoost splits on one feature
  at a time and cannot form `runoff - clim_mean` itself. But the mechanism is a
  CENTRING effect rather than new information: a per-pixel anomaly lets one
  global split threshold mean the same thing in every pixel, which is worth
  something to a model fitted on 19 mascons and applied to 9,538 cells.
  Collinearity diagnostics -- PCA, VIF, correlation matrices -- answer a question
  about linear models and do not transfer to this one.
* THE LADDER CANNOT RANK ITS OWN RUNGS. `minimal` (47 features) and `full`
  (69) are identical within the spread despite a 22-feature difference. The
  table can say whether a trim costs anything; it cannot order the trims against
  each other.

A first run with a single fold partition produced a non-monotonic curve and no
way to tell that it was noise; that is what `--repeats` exists to prevent -- and
why `RMSE_per_repeat`, `paired_d_vs_full`, `repeats_beating_full` and `paired_p`
are now written to the CSV rather than collapsed into a mean. Earlier versions
of this module asserted that a paired test was the right analysis and then did
not run one.

Usage
-----
    python downscale_ablation.py                     # the standard ladder
    python downscale_ablation.py --folds 4
    python downscale_ablation.py --configs full drop_runoff_anom lean
"""

from __future__ import annotations

import argparse
import contextlib
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import downscale_features as F
import downscale_model as M
import gridded_config as cfg
from tune_gridded import grouped_cv_rmse

ALL_PREDICTORS = tuple(cfg.PREDICTORS)
# `runoff_anom` IS a default predictor again (see gridded_config.PREDICTORS), so
# the rung measures what dropping it costs. Filtering rather than hardcoding a
# list keeps this correct if the default set changes.
NO_RUNOFF_ANOM = tuple(p for p in cfg.PREDICTORS if p != 'runoff_anom')

# label -> (predictors, antecedent windows, API taus)
#
# A ladder rather than a grid: each rung removes one KIND of redundancy, so the
# table reads as "what did this specific trim cost?" rather than as a search.
CONFIGS: Dict[str, Dict[str, object]] = {
    'full': dict(predictors=ALL_PREDICTORS, antecedent=(1, 2, 3, 6, 12, 24),
                 api=(3, 12, 60, 120),
                 note='current default'),
    'drop_runoff_anom': dict(predictors=NO_RUNOFF_ANOM,
                             antecedent=(1, 2, 3, 6, 12, 24),
                             api=(3, 12, 60, 120),
                             note='drops the per-pixel runoff anomaly; the only '
                                  'trim measured to clear the noise band'),
    'thin_ante': dict(predictors=ALL_PREDICTORS, antecedent=(1, 3, 12),
                      api=(3, 12, 60, 120),
                      note='drop windows correlated >0.94 with a neighbour'),
    'lean': dict(predictors=ALL_PREDICTORS, antecedent=(1, 3, 12), api=(3, 12, 60),
                 note='also drop tau=120mo, longer than the record can constrain'),
    'minimal': dict(predictors=ALL_PREDICTORS, antecedent=(1, 3), api=(3, 12),
                    note='deliberately too small, to bracket the cost'),
}


@contextlib.contextmanager
def _api_taus(taus: Sequence[int]):
    """
    Temporarily override the module-level API taus.

    `build_features` parameterises `predictors`, `antecedent` and `context` but
    reads `API_TAU_MONTHS` from the module. Rather than widen the signature for
    an experiment, this swaps the constant for the duration of one build.
    Confined to a context manager so a failure cannot leave the module mutated
    for whatever runs next in the same process.
    """
    old = F.API_TAU_MONTHS
    F.API_TAU_MONTHS = tuple(taus)
    try:
        yield
    finally:
        F.API_TAU_MONTHS = old


def evaluate(label: str, spec: Dict[str, object], folds: int, seed: int,
             repeats: int = 3, verbose: bool = True) -> Optional[Dict[str, object]]:
    """
    Build one configuration's design matrix and score it, REPEATEDLY.

    Repeats are not decoration. The first run of this ablation used a single
    fold assignment and a single model seed, and returned a NON-MONOTONIC curve:
    the deliberately-undersized `minimal` (47 features) beat both `lean` (54) and
    `thin_ante` (57). No account on which the dropped features carry information
    predicts that, so the between-configuration differences were not separable
    from run-to-run variation -- and with one number each there was no way to
    say so.

    Each repeat permutes which mascons fall in which fold AND reseeds the model,
    so the spread covers both sources. Compare configurations against that
    spread, not against each other's point estimates.
    """
    with _api_taus(spec['api']):
        fs = F.build_features(predictors=tuple(spec['predictors']),
                              antecedent=tuple(spec['antecedent']),
                              verbose=False)
    _, anom, _ = M.decompose_target(fs)
    obs = fs.observed
    X = fs.X.to_numpy(dtype='float32')[obs]
    y = anom[obs]
    groups = fs.mascon[obs]

    uniq = np.unique(groups)
    scores = []
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        # Relabel mascons so GroupKFold, which does not shuffle, sees a
        # different partition each repeat. The GROUPING is untouched -- a mascon
        # still never spans folds -- only which mascons share a fold.
        perm = {g: int(p) for g, p in zip(uniq, rng.permutation(len(uniq)))}
        g_perm = np.vectorize(perm.get)(groups)
        # Explicit hand-set defaults, NOT `params=None`.
        #
        # `None` would send make_model through load_gridded_params, which reads
        # the tuning JSON whenever one exists -- so this module would score with
        # defaults on a fresh tree and with tuned values on a re-run, silently,
        # while its own header printed "xgboost defaults" either way.
        #
        # Defaults are also the right choice on the merits, and this is the one
        # place it differs from mlp_configuration_sweep.py. That module compares
        # MODELS, so each belongs at its own best. This module compares FEATURE
        # SETS, so the model must be held constant and neutral. Hyperparameters
        # tuned on the FULL matrix are not neutral: applying them to a trimmed
        # configuration scores that trim with settings chosen for a larger
        # matrix, which disadvantages every trim and biases the table toward
        # "full is best" -- the very conclusion being tested.
        scores.append(grouped_cv_rmse(dict(M.DEFAULT_PARAMS['xgboost']),
                                      X, y, g_perm, 'xgboost',
                                      folds, seed + r))
    scores = np.asarray(scores)
    if verbose:
        print(f'  {label:12s} {fs.X.shape[1]:3d} features   '
              f'RMSE {scores.mean():8.4f} +/- {scores.std(ddof=1):.4f}  '
              f'(n={repeats}: ' + ', '.join(f'{s:.2f}' for s in scores) + ')',
              flush=True)
    return {'config': label, 'n_features': int(fs.X.shape[1]),
            'RMSE': float(scores.mean()), 'RMSE_sd': float(scores.std(ddof=1)),
            'RMSE_min': float(scores.min()), 'RMSE_max': float(scores.max()),
            # The per-repeat scores, kept rather than summarised away. Repeat r
            # uses the same permutation seed for EVERY config, so these are
            # PAIRED across configurations -- which is the whole basis for
            # comparing them, and it is lost the moment only a mean survives.
            'RMSE_per_repeat': ','.join(f'{s:.6f}' for s in scores),
            'n_repeats': repeats,
            'n_predictors': len(spec['predictors']),
            'antecedent': ','.join(str(a) for a in spec['antecedent']),
            'api_tau': ','.join(str(a) for a in spec['api']),
            'note': spec['note']}


def _print_verdict(df: pd.DataFrame) -> None:
    """
    State what the table does and does not support, in words.

    Written because the failure mode here is not a wrong number, it is a right
    number read too confidently: a configuration ahead by a twentieth of the
    repeat-to-repeat spread looks like a winner in a sorted table.
    """
    if 'paired_d_vs_full' not in df.columns:
        return
    trims = df[df.config != 'full'].dropna(subset=['paired_d_vs_full'])
    if trims.empty:
        return
    n = int(df.n_repeats.iloc[0])
    noise = float(df.RMSE_sd.median())
    print(f'\nVerdict  (repeat-to-repeat sd is ~{noise:.3f} mm; '
          f'{n} repeats)')
    for _, r in trims.sort_values('paired_d_vs_full').iterrows():
        d, w = r.paired_d_vs_full, r.repeats_beating_full
        if abs(d) < noise:
            call = f'INDISTINGUISHABLE from full (|d| < the {noise:.3f} mm sd)'
        elif d < 0:
            call = f'better than full, won {w:.0f}/{n} repeats'
        else:
            call = f'worse than full, won {w:.0f}/{n} repeats'
        print(f'  {r.config:12s} d = {d:+.4f} mm   {call}')
    if n < 5:
        print(f'  NOTE: {n} repeats cannot support a significance claim. '
              f'`repeats_beating_full`\n        is the honest column; the '
              f'p-value is reported because the method\n        promises a '
              f'paired test, not because it has power at this n.')


def _paired_vs_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired comparison of every configuration against `full`.

    The docstring at the top of this module has always said the comparison is
    paired and that a paired test is the right one. It was not actually computed,
    which left the table to be read off point estimates -- and point estimates
    here are misleading: the configurations differ by far less than they vary
    between repeats, so whichever happens to be lowest looks like a winner.

    Reports, per configuration: the mean paired difference, its spread, how many
    repeats it beat `full` in, and a paired t-test. The `wins` column is the one
    to trust at small `--repeats`; the p-value is included because it was
    promised, and is flagged rather than relied upon.
    """
    base_row = df.loc[df.config == 'full'].iloc[0]
    if 'RMSE_per_repeat' not in df.columns:
        return df
    base = np.array([float(v) for v in str(base_row.RMSE_per_repeat).split(',')])

    means, sds, wins, pvals = [], [], [], []
    for _, r in df.iterrows():
        vals = np.array([float(v) for v in str(r.RMSE_per_repeat).split(',')])
        if r.config == 'full' or len(vals) != len(base):
            means.append(np.nan); sds.append(np.nan)
            wins.append(np.nan); pvals.append(np.nan)
            continue
        d = vals - base                      # positive = the trim is WORSE
        means.append(float(d.mean()))
        sds.append(float(d.std(ddof=1)) if len(d) > 1 else np.nan)
        wins.append(int((d < 0).sum()))
        try:
            from scipy import stats
            pvals.append(float(stats.ttest_rel(vals, base).pvalue)
                         if len(d) > 1 and d.std() > 0 else np.nan)
        except Exception:                    # noqa: BLE001 - scipy is optional here
            pvals.append(np.nan)

    df['paired_d_vs_full'] = means
    df['paired_d_sd'] = sds
    df['repeats_beating_full'] = wins
    df['paired_p'] = pvals
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=3,
                    help='Repeats per configuration, each with a different fold\n'
                         'partition and model seed. Below 2 there is no error bar\n'
                         'and configurations cannot be told apart.')
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or M.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print('Feature ablation, mascon-grouped CV on the anomaly '
          f'(GroupKFold({args.folds}), xgboost defaults, '
          f'{args.repeats} repeats)\n')
    rows: List[Dict[str, object]] = []
    for label in args.configs:
        try:
            r = evaluate(label, CONFIGS[label], args.folds, args.seed,
                         repeats=args.repeats)
        except Exception as err:  # noqa: BLE001 - one bad config must not lose the rest
            print(f'  {label:12s} FAILED: {err}', flush=True)
            continue
        if r:
            rows.append(r)

    if not rows:
        print('no configuration completed')
        return 1

    df = pd.DataFrame(rows)
    if 'full' in df.config.values:
        base = float(df.loc[df.config == 'full', 'RMSE'].iloc[0])
        df['dRMSE_vs_full'] = df.RMSE - base
        df['dRMSE_pct'] = 100.0 * df.dRMSE_vs_full / base
        df['features_saved'] = int(df.loc[df.config == 'full', 'n_features'].iloc[0]) - df.n_features
        df = _paired_vs_full(df)

    path = os.path.join(out_dir, 'feature_ablation_xgboost.csv')
    df.to_csv(path, index=False)

    print('\nSummary  (dRMSE negative = the trim also reduced error)')
    cols = ['config', 'n_features', 'RMSE', 'RMSE_sd']
    if 'dRMSE_vs_full' in df:
        cols += ['dRMSE_vs_full', 'dRMSE_pct', 'features_saved']
    if 'paired_d_vs_full' in df:
        cols += ['paired_d_vs_full', 'repeats_beating_full', 'paired_p']
    print(df[cols].round(4).to_string(index=False))
    _print_verdict(df)
    for _, r in df.iterrows():
        print(f'  {r.config:12s} {r.note}')
    print(f'\nwritten: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
