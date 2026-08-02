"""
Month-level holdouts for the 0.1 degree SPATIAL model: random, blocked, forward.

Leave-one-mascon-out asks "does this work somewhere it has not seen?". This
module asks the other question -- "does it work at a TIME it has not seen?" --
and answers it three ways, because the three are different claims and a single
number would hide the one that matters.

The leakage ladder
------------------
All three hold out whole MONTHS (never individual pixels: every pixel of a month
shares that month's GRACE constraint, so a pixel-level split would leave the
target itself in training). They differ only in WHICH months:

    random    months drawn at random          OPTIMISTIC, see below
    blocked   contiguous blocks, interior     honest interpolation
    forward   the last months, chronological  honest extrapolation

Read the gap between them, not any one alone. That gap IS the result: it
measures how much of the apparent skill comes from temporal autocorrelation
rather than from the predictors.

Why `random` is reported despite being optimistic
-------------------------------------------------
Monthly TWSA is smooth and strongly autocorrelated, so a random month split
leaves each test month's immediate neighbours in training. The model can
interpolate between them without using the predictors at all. On the basin-scale
series the same design leaked at R2 = 0.96 even after grouping by interpolation
segment, because the residual is intrinsic to the signal rather than to the
split.

This module reports it anyway, clearly labelled, for two reasons. It is the
holdout the first-round reviewers saw, so removing it silently invites the
question of what it would have shown. And quantifying the optimism is more
convincing than asserting it: `random` minus `forward` is the cost of pretending
a smooth series can be split at random.

Do NOT quote the random number as the model's skill. `forward` is the number
that describes the product's extrapolated months (2024-10 onward); `blocked` is
the number that describes its interior gap-filled months.

Relationship to the rest of the project
---------------------------------------
`downscale_uncertainty.gap_recovery_error` already measures blocked and forward
errors, but as a function of DEPTH into the gap, because that is what feeds
`sigma_gap`. This module reports pooled skill per scheme, which is what a
holdout table in a paper needs. The two agree by construction where they overlap
-- same refit, different aggregation.

Usage
-----
    python downscale_holdouts.py --model xgboost
    python downscale_holdouts.py --model xgboost --repeats 3 --test-frac 0.2
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import downscale_features as F
import downscale_model as M

SCHEMES = ('random', 'blocked', 'forward')


def _fit_score(fs: F.FeatureSet, X: np.ndarray, anom_all: np.ndarray,
               train: np.ndarray, test: np.ndarray,
               model_name: str) -> Optional[Dict[str, float]]:
    """
    One refit and score, done twice: leak-free, then the way it used to be done.

    The target here is an anomaly about a per-mascon level+trend line, and that
    line has to be fitted on something. Fitting it on every month -- which is
    what a single decompose_target call before the splits does -- lets each
    held-out month help build the background that is then subtracted from it
    before scoring. On the forward split that background moved by 27.5 mm on
    average and up to 76 mm across the test window, against a leave-one-mascon-
    out RMSE near 77 mm, so it flattered the result by a material margin. The
    forward split is the one that claims to be honest extrapolation, and it was
    the worst affected: it was extrapolating a trend fitted on the future.

    So the line is refitted on the training months alone and extrapolated over
    the held-out ones. Both numbers are returned -- the leak-free metrics plain,
    the old ones suffixed `_leaky` -- because the size of the gap is itself
    worth reporting rather than quietly correcting.
    """
    if train.sum() < 1000 or test.sum() < 50:
        return None

    # Leak-free: the level+trend background sees only the training months.
    _, anom_tr, _ = M.decompose_target(fs, fit_mask=train)
    ok = np.isfinite(anom_tr)
    tr_ok, te_ok = train & ok, test & ok
    if tr_ok.sum() < 1000 or te_ok.sum() < 50:
        return None
    mdl = M.make_model(model_name)
    mdl.fit(X[tr_ok], anom_tr[tr_ok])
    out = M.metrics(anom_tr[te_ok], mdl.predict(X[te_ok]))

    # The same split scored against the all-months background, for the record.
    mdl_l = M.make_model(model_name)
    mdl_l.fit(X[train], anom_all[train])
    for k, v in M.metrics(anom_all[test], mdl_l.predict(X[test])).items():
        out[f'{k}_leaky'] = v
    return out


def random_months(fs: F.FeatureSet, anom: np.ndarray, model_name: str,
                  test_frac: float, repeats: int, seed: int,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Hold out a random subset of observed months, refit, score. Repeated.

    Repeated because a single draw of ~45 months is a small sample and the
    spread across draws is itself informative -- if it is wide, the single
    number a paper would quote is not stable.
    """
    obs, t = fs.observed, fs.time_index
    months = np.unique(t[obs])
    n_test = max(int(round(len(months) * test_frac)), 1)
    X = fs.X.to_numpy(dtype='float32')
    rng = np.random.default_rng(seed)

    rows = []
    for r in range(repeats):
        held = rng.choice(months, size=n_test, replace=False)
        test = obs & np.isin(t, held)
        train = obs & ~np.isin(t, held)
        m = _fit_score(fs, X, anom, train, test, model_name)
        if m is None:
            continue
        rows.append({'scheme': 'random', 'repeat': r, 'n_test_months': len(held), **m})
        if verbose:
            print(f'    random  repeat {r}: R2 {m["R2"]:.3f}  RMSE {m["RMSE"]:.1f}', flush=True)
    return pd.DataFrame(rows)


def blocked_months(fs: F.FeatureSet, anom: np.ndarray, model_name: str,
                   block: int, n_blocks: int, seed: int,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Hold out contiguous INTERIOR blocks of observed months, one block per refit.

    Contiguous so a held-out month's neighbours are held out with it, which is
    what removes the interpolation shortcut. Interior means bracketed by
    surviving observations on both sides, so this measures gap-FILLING, not
    extrapolation -- the regime of the GRACE/GRACE-FO blackout.

    Blocks are chosen non-overlapping: sampling starts independently would let
    two blocks share months, correlating the estimates and overstating how much
    evidence `n_blocks` represents.
    """
    obs, t = fs.observed, fs.time_index
    months = np.unique(t[obs])
    mset = set(months.tolist())
    starts = [m for m in months if all((m + k) in mset for k in range(block))]
    if not starts:
        if verbose:
            print('    blocked: no contiguous observed block long enough')
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    claimed: set = set()
    for k in rng.permutation(len(starts)):
        st = int(starts[k])
        span = set(range(st, st + block))
        if span & claimed:
            continue
        chosen.append(st)
        claimed |= span
        if len(chosen) >= n_blocks:
            break

    X = fs.X.to_numpy(dtype='float32')
    rows = []
    for bi, st in enumerate(chosen):
        hidden = list(range(st, st + block))
        test = obs & np.isin(t, hidden)
        train = obs & ~np.isin(t, hidden)
        m = _fit_score(fs, X, anom, train, test, model_name)
        if m is None:
            continue
        rows.append({'scheme': 'blocked', 'repeat': bi, 'n_test_months': block, **m})
        if verbose:
            print(f'    blocked repeat {bi}: R2 {m["R2"]:.3f}  RMSE {m["RMSE"]:.1f}', flush=True)
    return pd.DataFrame(rows)


def forward_months(fs: F.FeatureSet, anom: np.ndarray, model_name: str,
                   test_frac: float, verbose: bool = True) -> pd.DataFrame:
    """
    Chronological split: train on the early record, predict the late months.

    Only one split is possible -- there is exactly one "last N months" -- so no
    repeats. This is the regime the product's post-2024-09 months live in, where
    the level and trend are extrapolated with no closing anchor.
    """
    obs, t = fs.observed, fs.time_index
    months = np.unique(t[obs])
    n_test = max(int(round(len(months) * test_frac)), 1)
    held = months[-n_test:]
    test = obs & np.isin(t, held)
    train = obs & ~np.isin(t, held)
    m = _fit_score(fs, fs.X.to_numpy(dtype='float32'), anom, train, test,
                   model_name)
    if m is None:
        return pd.DataFrame()
    if verbose:
        print(f'    forward         : R2 {m["R2"]:.3f}  RMSE {m["RMSE"]:.1f}', flush=True)
    return pd.DataFrame([{'scheme': 'forward', 'repeat': 0,
                          'n_test_months': len(held), **m}])


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and spread per scheme, ordered least to most honest.

    `*_leaky` columns carry the same splits scored against a level+trend
    background fitted on all months including the held-out ones. They are
    reported, not quoted: the gap between the two says how much of the old
    number was the test months vouching for themselves.
    """
    if per_run.empty:
        return per_run
    g = per_run.groupby('scheme')
    out = g.agg(n_refits=('R2', 'size'),
                R2_mean=('R2', 'mean'), R2_sd=('R2', 'std'),
                RMSE_mean=('RMSE', 'mean'), RMSE_sd=('RMSE', 'std'),
                MAE_mean=('MAE', 'mean'), NSE_mean=('NSE', 'mean'),
                R2_leaky_mean=('R2_leaky', 'mean'),
                RMSE_leaky_mean=('RMSE_leaky', 'mean')).reset_index()
    out['RMSE_leak_penalty'] = out.RMSE_mean - out.RMSE_leaky_mean
    order = {s: i for i, s in enumerate(SCHEMES)}
    out['_o'] = out.scheme.map(order)
    out = out.sort_values('_o').drop(columns='_o').reset_index(drop=True)
    out.insert(1, 'reading', out.scheme.map({
        'random': 'OPTIMISTIC - autocorrelation leak, do not quote as skill',
        'blocked': 'honest: interior gap-filling',
        'forward': 'honest: out-of-record extrapolation',
    }))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='xgboost',
                    choices=['xgboost', 'lightgbm', 'random_forest', 'xgboost_rf', 'mlp'])
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--repeats', type=int, default=5,
                    help='Random-split repeats (also the cap on blocked blocks).')
    ap.add_argument('--block', type=int, default=11,
                    help='Blocked-holdout length in months; 11 matches the '
                         'GRACE/GRACE-FO gap.')
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or M.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print('Month-level holdouts for the 0.1 degree spatial model\n')
    print('1. Features and target decomposition')
    fs = F.build_features(verbose=True)
    # All-months decomposition, kept only to score the `_leaky` comparison
    # columns. Every split refits its own background on training months alone
    # -- see _fit_score.
    _, anom, _ = M.decompose_target(fs)

    print(f'\n2. Holdouts ({args.model}, test fraction {args.test_frac})')
    parts = [
        random_months(fs, anom, args.model, args.test_frac, args.repeats, args.seed),
        blocked_months(fs, anom, args.model, args.block, args.repeats, args.seed),
        forward_months(fs, anom, args.model, args.test_frac),
    ]
    per_run = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if per_run.empty:
        print('  no holdout produced a usable split')
        return 1

    p1 = os.path.join(out_dir, f'holdouts_month_{args.model}_runs.csv')
    per_run.to_csv(p1, index=False)

    summary = summarise(per_run)
    p2 = os.path.join(out_dir, f'holdouts_month_{args.model}.csv')
    summary.to_csv(p2, index=False)

    print('\n3. Summary')
    print(summary[['scheme', 'n_refits', 'R2_mean', 'R2_sd',
                   'RMSE_mean', 'NSE_mean']].round(3).to_string(index=False))

    s = summary.set_index('scheme')
    if {'random', 'forward'} <= set(s.index):
        d = s.loc['random', 'R2_mean'] - s.loc['forward', 'R2_mean']
        print(f'\n  optimism of a random split: R2 {d:+.3f} '
              f'({s.loc["random", "R2_mean"]:.3f} random vs '
              f'{s.loc["forward", "R2_mean"]:.3f} forward)')
        print('  That gap is the result. Quote `forward` for the extrapolated '
              'months and `blocked` for the interior gaps.')

    print(f'\nwritten: {p2}\n         {p1}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
