"""
Optuna tuning for the model that BUILDS THE 0.1 degree PRODUCT.

Not to be confused with `tune_hyperparameters.py`, which tunes the six
basin-scale models on basin-mean series. Those models do not produce the
deliverable; for a long time the pipeline spent its entire tuning budget on them
while `downscale_model.make_model` used hand-set values that had never been
tuned at all. This module closes that gap.

What it optimises, and why that objective
-----------------------------------------
The objective is RMSE of the ANOMALY under GROUPED SPATIAL cross-validation,
grouping by mascon. Three deliberate choices:

* THE ANOMALY, not the raw field. The product takes level and trend from GRACE
  per mascon (see METHODS Sec. 4); the model only ever predicts the anomaly. Tuning
  against the raw field would optimise for reproducing a component the model is
  not responsible for, and would be dominated by it -- the anomaly is ~26% of
  total variance.

* GROUPED BY MASCON, so a fold's training data never contains the mascon it is
  scored on. Ungrouped k-fold would place pixels from the same mascon on both
  sides of the split; since the target is constant within a mascon-month, that
  is the target itself leaking into training.

* SPATIAL, not temporal. Transfer across mascons is what the model is asked to
  do at prediction time -- it is fitted on 0.5 degree cells and applied at 0.1
  degrees. `downscale_holdouts.py` measures temporal transfer separately.

Cost, and the one compromise
----------------------------
Full leave-one-mascon-out is 19 refits per trial, which at any useful trial
count is not affordable. This uses GroupKFold with `--folds` groups of mascons
(default 5), so a trial costs 5 refits rather than 19. The folds are therefore
coarser than the final LOMO evaluation, and the absolute RMSE here will not
match `lomo_cv_<model>.csv` exactly. That is fine -- tuning needs a consistent
RANKING of configurations, not a calibrated score. The tuned model is then
evaluated honestly by the real LOMO in `downscale_model.py`.

Note also that LOMO in `downscale_model.py` applies a one-mascon NEIGHBOUR
BUFFER, which this does not. The buffer makes the final evaluation stricter than
the tuning objective, so tuned scores here are optimistic relative to the
reported ones. Stated rather than corrected, because a buffered grouped CV would
cost most of the saving that makes tuning affordable.

Output
------
`../Results/tuning/gridded_best_params.json`, keyed by model name, in the shape
`downscale_model.load_gridded_params` reads. Recording the default score
alongside the tuned one is the point: if tuning does not beat the hand-set
values, that is the result and the defaults should stay.

Which model builds the product
------------------------------
`--all` tunes every candidate and then ranks them on this same grouped CV,
writing the winner to `selected_model.txt` beside the parameter JSON;
`run_full_pipeline.sh` reads that file rather than hardcoding a model. Ranking
uses each model's ADOPTED configuration, so a model is never credited with a
search that made it worse.

Tuning all of them is not only about picking a winner. Four of these models form
the uncertainty ensemble, whose `sigma_within` term is meant to measure
disagreement between model FAMILIES -- tuning one member and leaving the rest on
hand-set defaults would fold a tuned-vs-untuned artefact into that term.

The ranking shares the tuning's folds, so the winner's score is optimistic and
the ranking, not the number, is the usable output. Absolute skill comes from the
LOMO run, which selection never touches.

Usage
-----
    python tune_gridded.py --all                     # tune all five, then select
    python tune_gridded.py --models xgboost lightgbm --trials 60 --folds 4
    python tune_gridded.py --summarize               # rebuild the table only
    python tune_gridded.py --select                  # re-rank from the existing JSON
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

import downscale_features as F
import downscale_model as M

DEFAULT_OUT = M.GRIDDED_PARAM_PATH

# Models scored and ranked like the rest, but NOT searched on every run.
#
# The MLP is here because its configuration was settled by measurement once,
# and re-deriving it every run buys nothing. Swept over 21 configurations on
# this dataset under the same grouped CV (3 folds, 83,309 x 80):
#
#     one hidden layer      two hidden layers
#     (32,)  79.192         (128, 64)  84.591   [the old default]
#     (64,)  79.038  <-     (256, 128) 87.383
#     (128,) 78.917
#     (256,) 79.132
#
# WIDTH is flat (0.35% across an 8x range) and DEPTH is not (+7.2% for a second
# layer). So the small network is the MLP's BEST case, not a handicap -- which
# is the question a reviewer asks when a neural comparator loses. (64,) is
# adopted over the nominally-best (128,) because 0.15% is inside the flat band
# and (64,) is where the learning-rate axis was swept, so the adopted point lies
# on both swept lines.
#
# Learning rate is the axis that matters, and its optimum is INTERIOR to the
# swept range rather than at an edge (3e-4 -> 79.038, against 1e-4 -> 79.589 and
# 1e-3 -> 80.515), so the sweep is not simply stopping too early. alpha is flat
# to 0.01% over four orders of magnitude.
#
# XGBoost scores 77.234 on identical folds, so the gap from the adopted MLP is
# 2.34% -- several times any tuning gain measured on this problem (~0.8%).
# Searching the MLP every run costs ~16x a single scored fit and cannot close
# that.
#
# It stays in the ranking with a real, released number. `--search-fixed`
# overrides; `mlp_configuration_sweep.py` regenerates the full table.
FIXED_CONFIG_MODELS = frozenset({'mlp'})

# MLP architectures, keyed by a string so Optuna can persist the choice.
# Widths are listed shallow-to-deep; the key is the layer sizes joined by '-'.
_MLP_WIDTHS = {'64': (64,), '128': (128,),
               '128-64': (128, 64), '256-128': (256, 128)}


def _decode_params(model: str, params: Dict[str, object]) -> Dict[str, object]:
    """
    Turn Optuna's raw suggestions into constructor kwargs.

    Only the MLP needs this. `study.best_params` returns what was *suggested* --
    the architecture key, not the tuple -- so writing it straight to the JSON
    would store a value `MLPRegressor` cannot accept. Called on both the tuned
    params and the adopted set before either is serialised.
    """
    out = dict(params)
    key = out.get('hidden_layer_sizes')
    if model == 'mlp' and isinstance(key, str):
        out['hidden_layer_sizes'] = list(_MLP_WIDTHS[key])
    return out


def _spaces(trial, model: str) -> Dict[str, object]:
    """
    Search space per model.

    Ranges bracket the hand-set defaults rather than starting from library
    defaults, because the hand-set values are a considered starting point and a
    space that excludes them cannot report "tuning did not help".
    """
    if model == 'xgboost':
        # Ranges narrowed from the first full run, which cost ~38 min per model.
        # The chosen point was n_estimators=500 against a 1200 ceiling, so the
        # top half of that range was never competitive and only bought cost --
        # trial time scales linearly in n_estimators. max_depth is NOT narrowed:
        # the optimum sat AT the ceiling of 12, so the range is extended instead.
        # Shrinking a dimension whose optimum is on the boundary would be
        # cutting toward the answer rather than away from waste.
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 700, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 14),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        }
    if model == 'lightgbm':
        # The chosen point was n_estimators=200 -- the FLOOR of the old range --
        # so the ceiling of 1600 was pure waste and the floor was the binding
        # constraint. Lowered to 100 and capped at 600. num_leaves came out at
        # 249 against a 255 ceiling, so that one is widened, not narrowed.
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 400, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
    if model == 'mlp':
        # Width, depth (one or two hidden layers), regularisation and learning
        # rate. Depth is capped at two: with ~19 independent spatial units, a
        # deeper search would fit the fold assignment rather than the physics.
        #
        # The architecture is suggested as a STRING key rather than a tuple.
        # Optuna warns that tuple choices are not reliably persistable in a
        # study storage, and `study.best_params` returns whatever was suggested
        # -- so a tuple here would put a non-round-trippable value into the
        # released JSON. `_decode_params` maps the key back before storage.
        return {
            'hidden_layer_sizes': _MLP_WIDTHS[trial.suggest_categorical(
                'hidden_layer_sizes', list(_MLP_WIDTHS))],
            'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float(
                'learning_rate_init', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
            # max_iter is a SAFETY CAP, not the stopping rule: early stopping
            # decides. At the old cap of 300 every fit hit it and emitted a
            # ConvergenceWarning; measured on the real design matrix
            # (83,309 x 80), lifting the cap lets early stopping fire at 384
            # epochs for ~24% more time, and the held-out fold RMSE barely
            # moves (68.23 unconverged -> 68.78 converged). So the old cap was
            # not materially handicapping the network -- but a converged fit
            # lets us say that from measurement instead of assertion, and stops
            # every MLP result carrying a convergence warning.
            'max_iter': 1000, 'early_stopping': True, 'n_iter_no_change': 12,
        }
    if model == 'xgboost_rf':
        # No learning_rate: XGBRFRegressor fixes it at 1.0 and tuning it would
        # turn the forest back into shrunken boosting. Column sampling is per
        # NODE, which is the forest convention and a different knob from the
        # boosting model's colsample_bytree. Depth ranges higher than for
        # boosting because bagged trees are grown deep and averaged rather than
        # kept shallow and summed.
        return {
            # THIS is the model that made the first run unaffordable: it alone
            # ran over an hour, against ~38 min for each of the others. Cost is
            # roughly n_estimators x 2^max_depth, and a depth-20 forest of 800
            # bagged trees on 83,309 x 80 is the most expensive point any space
            # in this file can reach. Both ceilings are cut; depth 14 still
            # comfortably exceeds the depth-12 optimum the boosting search
            # found, and bagged trees do not need the extra depth to average.
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 6, 14),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        }
    return {
        # Tuning made this model WORSE than its hand-set defaults on the first
        # run (+0.24%), so the defaults are kept anyway; a smaller range costs
        # nothing here and the trial time is linear in n_estimators.
        'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.2, 1.0),
    }


def grouped_cv_rmse(params: Optional[Dict], X: np.ndarray, y: np.ndarray,
                    groups: np.ndarray, model: str, folds: int,
                    seed: int, trial=None, return_folds: bool = False):
    """
    Mean RMSE across GroupKFold folds, grouping by mascon.

    When `trial` is supplied the running mean is reported after each fold and
    the trial may be PRUNED partway through. A hopeless configuration then costs
    two folds instead of `folds`, which is where most of the tuning saving comes
    from -- the first full run spent over an hour on `xgboost_rf` alone, and
    every trial in it paid for all five folds no matter how bad it was.

    The baseline call passes `trial=None` and always runs to completion: the
    default score is the thing every tuned score is compared against, so it must
    never be a partial estimate.
    """
    from sklearn.model_selection import GroupKFold

    scores = []
    for step, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        mdl = M.make_model(model, seed=seed, params=params)
        mdl.fit(X[tr], y[tr])
        scores.append(M.metrics(y[va], mdl.predict(X[va]))['RMSE'])
        if trial is not None:
            import optuna
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if return_folds:
        return float(np.mean(scores)), scores
    return float(np.mean(scores))


def run(model: str = 'xgboost', trials: int = 15, folds: int = 5,
        seed: int = 20, out_path: str = DEFAULT_OUT,
        search_fixed: bool = False) -> Dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print('1. Features and target decomposition')
    fs = F.build_features(verbose=True)
    _, anom, _ = M.decompose_target(fs)
    obs = fs.observed
    X = fs.X.to_numpy(dtype='float32')[obs]
    y = anom[obs]
    groups = fs.mascon[obs]
    print(f'  tuning on {X.shape[0]:,} observed samples x {X.shape[1]} features, '
          f'{len(np.unique(groups))} mascons, GroupKFold({folds})\n')

    print('2. Baseline: the hand-set defaults, same folds')
    # Keep the per-fold scores, not just their mean. Their spread is how noisy a
    # single model's own CV estimate is, and that is the yardstick the
    # between-model spread has to beat before a ranking means anything. Without
    # it, "model A beats model B by 0.7 mm" has no scale attached.
    base, base_folds = grouped_cv_rmse(None, X, y, groups, model, folds, seed,
                                       return_folds=True)
    base_sd = float(np.std(base_folds))
    print(f'  default grouped-CV RMSE: {base:.4f}  '
          f'(fold-to-fold sd {base_sd:.4f})\n')

    if model in FIXED_CONFIG_MODELS and not search_fixed:
        # Scored, ranked and released like the others -- just not searched.
        # See FIXED_CONFIG_MODELS for the measurements behind this.
        print(f'3. No per-run search: {model} is a fixed-configuration '
              f'comparator.\n   Its configuration was chosen by a released '
              f'sweep, not by a per-run search;\n   re-searching it every run '
              f'would cost ~16x this and has no product to improve.\n'
              f'   Override with --search-fixed.\n')
        return _store(model, best=dict(M.DEFAULT_PARAMS[model]), best_rmse=base,
                      base=base, trials=0, folds=folds, out_path=out_path,
                      adopted='fixed', fold_sd=base_sd)

    print(f'3. Optuna, {trials} trials')

    def objective(trial):
        return grouped_cv_rmse(_spaces(trial, model), X, y, groups,
                               model, folds, seed, trial=trial)

    # Fold-level pruning. `n_startup_trials=5` lets a third of a 15-trial budget
    # run in full before any pruning decision is made, so the median it prunes
    # against is not itself an artefact of one lucky early trial;
    # `n_warmup_steps=2` means no trial dies before three folds have been seen.
    #
    # Both guards matter here specifically because the folds are GROUPS OF
    # MASCONS and their difficulty varies a lot -- a configuration that looks
    # poor on fold 0 may simply have drawn the hard mascons. Pruning on a single
    # fold would discard good configurations for that reason alone.
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    if n_pruned:
        print(f'  {n_pruned} of {trials} trials pruned early')

    # Decode here rather than at serialisation so the line printed below and
    # the value stored in the JSON are the same object.
    best = _decode_params(model, study.best_params)
    best_rmse = study.best_value
    # tuned MINUS default, so NEGATIVE means less error. Same convention as
    # tune_hyperparameters.py -- the plotted quantity is an error, and "-2.1%"
    # should read as "2.1% less error" without a mental inversion.
    delta_rmse = best_rmse - base
    delta_pct = 100.0 * delta_rmse / base if base else 0.0
    print(f'  default : {base:.4f}')
    print(f'  tuned   : {best_rmse:.4f}  (delta {delta_rmse:+.4f}, {delta_pct:+.2f}%; '
          f'negative = less error)')
    print(f'  best    : {best}')

    # Keeping the defaults when tuning does not beat them is the honest action,
    # and the JSON records which happened so the choice is auditable.
    use_tuned = best_rmse < base
    if not use_tuned:
        print('\n  tuning did NOT beat the hand-set defaults; keeping defaults.')

    return _store(model, best, best_rmse, base, trials, folds, out_path,
                  adopted='tuned' if use_tuned else 'defaults',
                  fold_sd=base_sd)


def _store(model: str, best: Dict, best_rmse: float, base: float, trials: int,
           folds: int, out_path: str, adopted: str,
           fold_sd: Optional[float] = None) -> Dict:
    """
    Merge one model's result into the shared JSON.

    Merging rather than overwriting is what lets the models be tuned one at a
    time, in separate invocations, and still end up in one comparable table.
    """
    delta_rmse = best_rmse - base
    delta_pct = 100.0 * delta_rmse / base if base else 0.0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as fh:
                payload = json.load(fh)
        except (ValueError, OSError):
            payload = {}
    payload[model] = {
        'best_params': best if adopted == 'tuned' else dict(M.DEFAULT_PARAMS[model]),
        'tuned_params': best,
        'adopted': adopted,
        'tuned_cv_rmse': best_rmse,
        'default_cv_rmse': base,
        'delta_rmse': delta_rmse,
        'delta_rmse_pct': delta_pct,
        'n_trials': trials,
        'fold_sd': fold_sd,
        'cv': f'GroupKFold(n_splits={folds}) by mascon, anomaly target, no neighbour buffer',
    }
    with open(out_path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f'\nwritten: {out_path}')
    return payload[model]


def summarize(path: str = DEFAULT_OUT) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f'{path} does not exist - run the tuning first')
        return None
    with open(path) as fh:
        raw = json.load(fh)
    rows = [{'model': k, 'adopted': v.get('adopted'),
             'default_CV_RMSE': round(v['default_cv_rmse'], 4),
             'tuned_CV_RMSE': round(v['tuned_cv_rmse'], 4),
             'dRMSE': round(v['tuned_cv_rmse'] - v['default_cv_rmse'], 4),
             'dRMSE_%': round(v.get('delta_rmse_pct',
                                    100.0 * (v['tuned_cv_rmse'] - v['default_cv_rmse'])
                                    / v['default_cv_rmse']), 2),
             'trials': v.get('n_trials')} for k, v in raw.items()]
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(path), 'gridded_tuning_summary.csv')
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f'\nwritten: {out}')
    return df


ALL_MODELS = ('random_forest', 'xgboost', 'lightgbm', 'xgboost_rf', 'mlp')


def select_best(path: str = DEFAULT_OUT,
                out_file: Optional[str] = None) -> Optional[str]:
    """
    Rank every tuned model on the same grouped CV and name a winner.

    Selection uses the ADOPTED configuration per model — tuned where tuning beat
    the hand-set defaults, defaults where it did not — so a model is never
    credited with a search that made it worse.

    One caveat, stated because it affects how the winning number should be read:
    the ranking uses the same mascon-grouped CV the tuning optimised, so the
    winner's score is optimistic by an unknown amount. Selecting on a fully
    independent split would need a third layer of cross-validation over ~19
    spatial units, which the data cannot support. The RANKING is the usable
    output; the winner's absolute skill comes from the leave-one-mascon-out run
    in `downscale_model.py`, which is not used for selection.
    """
    # Default the winner file NEXT TO the tuning JSON, not to a fixed path.
    # Deriving it from `path` keeps the pair together when --out is redirected;
    # a constant would write the winner of one run beside the parameters of
    # another, and the pipeline reads the two as a matched set.
    if out_file is None:
        out_file = os.path.join(os.path.dirname(os.path.abspath(path)),
                                'selected_model.txt')
    if not os.path.exists(path):
        print(f'{path} does not exist - run the tuning first')
        return None
    with open(path) as fh:
        raw = json.load(fh)
    scored = {k: min(v['tuned_cv_rmse'], v['default_cv_rmse'])
              for k, v in raw.items()
              if isinstance(v, dict) and 'tuned_cv_rmse' in v}
    if not scored:
        print('no scored models in the tuning file')
        return None
    missing = [m for m in ALL_MODELS if m not in scored]
    if missing:
        # A model that crashed mid-tuning leaves no entry, and selection would
        # then quietly range over a subset. Say so rather than let the log read
        # as though every candidate competed.
        print(f'  WARNING: selecting over {len(scored)} of {len(ALL_MODELS)} '
              f'candidates; absent: {", ".join(missing)}\n')

    # Name breaks exact ties so the winner never depends on JSON key order.
    order = sorted(scored, key=lambda m: (scored[m], m))
    best = order[0]
    spread = scored[order[-1]] - scored[best]
    print('Model selection, mascon-grouped CV RMSE of the adopted configuration:')
    for i, m in enumerate(order):
        gap = scored[m] - scored[best]
        mark = '  <- selected' if i == 0 else f'  (+{gap:.3f}, {100*gap/scored[best]:+.2f}%)'
        print(f'  {i+1}. {m:15s} {scored[m]:8.4f} [{raw[m].get("adopted","?")}]{mark}')
    print(f'\n  spread across models: {spread:.3f} mm '
          f'({100*spread/scored[best]:.2f}% of the best)')

    # Is that spread bigger than the noise on any single model's own score?
    #
    # The yardstick is each model's fold-to-fold standard deviation, measured on
    # the same folds. If the models differ by less than a typical model differs
    # from itself across folds, the ranking is not resolvable and saying so is
    # the honest report -- this is what R3-38's "uncertainty ranges" asks for.
    # A fixed percentage threshold cannot do this: it is a number chosen in
    # advance, and it does not know how noisy THIS dataset's folds are.
    sds = [v['fold_sd'] for v in raw.values()
           if isinstance(v, dict) and v.get('fold_sd') is not None]
    if sds:
        noise = float(np.median(sds))
        print(f'  typical fold-to-fold sd within one model: {noise:.3f} mm')
        if spread < noise:
            print('  NOTE: the models differ by LESS than a single model varies '
                  'across folds.\n        The ranking is not resolvable on these '
                  'data -- treat it as a way to\n        pick one model, not as '
                  'evidence that the winner is better. Prefer the\n        simpler '
                  'or better-understood model if other considerations apply.')
        else:
            print(f'  the spread is {spread/noise:.1f}x that noise, so the '
                  f'ordering is at least resolvable\n        (which is not the '
                  f'same as a large or important difference).')
    elif spread / scored[best] < 0.02:
        # Fallback for a tuning file written before fold_sd was recorded.
        print('  NOTE: under 2% separates best from worst, and no per-fold spread '
              'was recorded\n        to compare it against. Treat the ranking as '
              'weak evidence.')

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as fh:
        fh.write(best + '\n')
    print(f'\nwritten: {out_file}')
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=['xgboost'],
                    choices=list(ALL_MODELS),
                    help='Models to tune. Pass all five to make the selection '
                         'step meaningful.')
    ap.add_argument('--all', action='store_true',
                    help=f'Shorthand for --models {" ".join(ALL_MODELS)}')
    ap.add_argument('--trials', type=int, default=15,
                    help='Optuna trials PER MODEL. Each costs `folds` refits on '
                         'the full gridded feature set, so total cost is '
                         'models x trials x folds.')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--summarize', action='store_true',
                    help='Only rebuild the summary table from an existing JSON.')
    ap.add_argument('--select', action='store_true',
                    help='Only rank the tuned models and write the winner.')
    ap.add_argument('--search-fixed', action='store_true',
                    help=f'Also run a per-run search for the fixed-configuration '
                         f'models ({", ".join(sorted(FIXED_CONFIG_MODELS))}), whose '
                         f'settings were instead chosen by a released sweep.')
    args = ap.parse_args()

    if args.summarize:
        summarize(args.out)
        return 0
    if args.select:
        return 0 if select_best(args.out) else 1

    models = list(ALL_MODELS) if args.all else args.models
    for i, m in enumerate(models, start=1):
        print(f'\n{"="*68}\n[{i}/{len(models)}] {m}\n{"="*68}')
        run(m, args.trials, args.folds, args.seed, args.out,
            search_fixed=args.search_fixed)

    if len(models) > 1:
        print(f'\n{"="*68}\nSELECTION\n{"="*68}')
        select_best(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
