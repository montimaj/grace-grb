"""
Decide which static covariates earn a place in the model.

Why a gate rather than "add them all and read SHAP"
---------------------------------------------------
SHAP measures a feature's contribution to the model's PREDICTIONS. It does not
measure whether that contribution generalises. A covariate that lets the model
fit the secular trend across only ~19 independent mascons will show up as highly
important precisely because the model leans on it -- SHAP rewards the overfit
rather than detecting it.

This project already produced the counter-example. Raw `lat` ranked 6th of 62
features by XGBoost gain, because the target is constant within a mascon so the
optimal latitude splits fall exactly on mascon boundaries. It was pure
memorisation of the training geometry: removing it moved held-out skill by 0.006
(R2 0.642 -> 0.636) and simultaneously removed visible seams from the product.
An attribution measure called it important; only a held-out test showed it was
worthless.

So each covariate is judged by whether it improves skill at a mascon the model
never saw, using the leave-one-mascon-out cross-validation in
`downscale_model.leave_one_mascon_out` with its one-mascon neighbour buffer.

What counts as an improvement
-----------------------------
Reported per covariate:

  dR2_anomaly   change in pooled out-of-fold R2 on the detrended anomaly, the
                component the model actually fits
  dR2_trend     change in R2 of the per-mascon TREND at held-out mascons

The second is the interesting one for irrigation extent. The predictors cannot
currently reproduce the secular decline (trend R2 ~ 0.11) because no reanalysis
simulates abstraction; irrigated area is a proxy for abstraction, so if it helps
anywhere it should help there. A covariate that lifts `dR2_anomaly` while leaving
`dR2_trend` flat is adding texture, not mechanism -- worth keeping but worth
describing accurately.

Usage
-----
    python downscale_covariate_gate.py                    # test each, one at a time
    python downscale_covariate_gate.py --cumulative       # greedy forward selection
    python downscale_covariate_gate.py --model xgboost
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import gridded_config as cfg
import downscale_features as F
import downscale_model as M


def _trend_r2(fs: F.FeatureSet, oof: np.ndarray, y: np.ndarray,
              groups: np.ndarray, t: np.ndarray) -> float:
    """R2 of the per-mascon linear trend, observed vs out-of-fold predicted."""
    obs_slope, pred_slope = [], []
    for m in np.unique(groups):
        sel = (groups == m) & np.isfinite(oof) & np.isfinite(y)
        if sel.sum() < 24:
            continue
        obs_slope.append(np.polyfit(t[sel], y[sel], 1)[0])
        pred_slope.append(np.polyfit(t[sel], oof[sel], 1)[0])
    if len(obs_slope) < 4:
        return float('nan')
    a, b = np.array(obs_slope), np.array(pred_slope)
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def evaluate(covariates: Sequence[str], model_name: str,
             verbose: bool = False) -> Dict[str, float]:
    """Leave-one-mascon-out skill for one covariate set."""
    fs = F.build_features(covariates=list(covariates), verbose=verbose)
    base, anom, _ = M.decompose_target(fs)
    cv = M.leave_one_mascon_out(fs, model_name, target=anom, verbose=False)

    obs = fs.observed
    t = fs.time_index[obs].astype('float64')
    return {
        'n_features': int(fs.X.shape[1]),
        'R2_anomaly': float(cv.pooled.get('R2', np.nan)),
        'NSE_anomaly': float(cv.pooled.get('NSE', np.nan)),
        'RMSE_anomaly': float(cv.pooled.get('RMSE', np.nan)),
        'R2_trend': _trend_r2(fs, cv.predictions, anom[obs], cv.mascon, t),
    }


def run(model_name: str = 'xgboost', cumulative: bool = False,
        candidates: Optional[List[str]] = None) -> pd.DataFrame:
    candidates = candidates or [v.name for v in cfg.STATIC_VARS]

    print(f'baseline (no covariates), {model_name}')
    base = evaluate([], model_name)
    print(f'  {base["n_features"]} features | anomaly R2 {base["R2_anomaly"]:.3f} '
          f'| trend R2 {base["R2_trend"]:.3f}\n')

    rows = [{'covariate': '(baseline)', **base, 'dR2_anomaly': 0.0, 'dR2_trend': 0.0}]
    kept: List[str] = []
    # The configuration each trial is scored against. In cumulative mode this is
    # the last ACCEPTED state, which is not the same as the previous row.
    #
    # It was `rows[-1]`, i.e. the previous trial whether or not it was kept. A
    # rejected covariate is not part of the model, so differencing against it
    # measures nothing: with zero survivors every trial after the first was
    # scored against the covariate before it. Measured, `awc` came out at -0.007
    # (against `wtd`) where the truth against baseline is -0.0029 -- the
    # single-covariate value. The bug bites precisely when a covariate is
    # rejected, so it never showed up while survivors were being found.
    accepted = rows[0]

    for name in candidates:
        trial = kept + [name] if cumulative else [name]
        res = evaluate(trial, model_name)
        ref = accepted if cumulative else rows[0]
        d_anom = res['R2_anomaly'] - ref['R2_anomaly']
        d_trend = res['R2_trend'] - ref['R2_trend']
        keep = (d_anom > 0.005) or (d_trend > 0.02)
        if cumulative and keep:
            kept.append(name)
            accepted = res
        rows.append({'covariate': name, **res,
                     'dR2_anomaly': d_anom, 'dR2_trend': d_trend,
                     'kept': bool(keep)})
        print(f'  {name:16s} anomaly R2 {res["R2_anomaly"]:.3f} ({d_anom:+.3f})   '
              f'trend R2 {res["R2_trend"]:.3f} ({d_trend:+.3f})   '
              f'{"KEEP" if keep else "drop"}', flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(M.RESULTS_DIR, exist_ok=True)
    path = os.path.join(M.RESULTS_DIR, f'covariate_gate_{model_name}.csv')
    out.to_csv(path, index=False)

    survivors = kept if cumulative else [r['covariate'] for r in rows[1:] if r.get('kept')]
    print(f'\nsurvivors: {survivors or "(none improved held-out skill)"}')
    print(f'written: {path}')
    print('\nSet gridded_config.ACTIVE_COVARIATES to the survivors, then rebuild.')
    with open(os.path.join(M.RESULTS_DIR, 'covariate_survivors.json'), 'w') as fh:
        json.dump({'model': model_name, 'cumulative': cumulative,
                   'survivors': survivors}, fh, indent=2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='xgboost',
                    choices=['random_forest', 'xgboost', 'lightgbm', 'xgboost_rf', 'mlp'])
    ap.add_argument('--cumulative', action='store_true',
                    help='greedy forward selection instead of one-at-a-time')
    ap.add_argument('--covariates', nargs='+', default=None)
    args = ap.parse_args()
    run(args.model, args.cumulative, args.covariates)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
