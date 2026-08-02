"""
Hyperparameter tuning with Optuna for the TWS downscaling models.

Design choices (consistent with the leakage-aware evaluation used elsewhere):

- Tuning uses **walk-forward (TimeSeriesSplit) cross-validation on the TRAINING
  data only**, so the search respects temporal ordering and never touches the
  held-out test period. This avoids the optimistic bias that a shuffled search
  would introduce on an autocorrelated series.
- Objective = mean RMSE across the walk-forward folds (minimised).
- TPE sampler + MedianPruner (per-fold reporting) stop weak trials early.
- Best parameters per model are written to JSON and can be loaded by the holdout
  runners via `load_tuned_params`.

Usage
-----
    # tune the fast tree models
    python tune_hyperparameters.py --models randomforest xgboost lightgbm --trials 60

    # tune neural nets (slower; fewer trials, capped epochs, pruning)
    python tune_hyperparameters.py --models lstm bilstm bilstm_attention --trials 20 --nn-epochs-max 30

Then run any holdout with the tuned parameters:
    python run_analysis.py --analysis temporal --tuned-params ../Results/tuning/best_params.json
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

import optuna
from sklearn.model_selection import TimeSeriesSplit

from models import get_model, set_seed
from utils import (
    load_and_preprocess_data, create_lagged_features,
    prepare_features_target, calculate_metrics,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

TREE_MODELS = {"xgboost", "xgb", "lightgbm", "lgbm", "randomforest", "rf", "random_forest"}
NN_MODELS = {"lstm", "bilstm", "bilstm_attention", "bilstm+attention"}

DEFAULT_PARAM_PATH = "../Results/tuning/best_params.json"


# =============================================================================
# Search spaces
# =============================================================================

def suggest_params(trial: "optuna.Trial", model_name: str, nn_epochs_max: int = 40) -> dict:
    """Sample a hyperparameter set for `model_name` from the trial."""
    m = model_name.lower().replace(" ", "_").replace("-", "_")

    if m in ("xgboost", "xgb"):
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
    if m in ("lightgbm", "lgbm"):
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
    if m in ("randomforest", "rf", "random_forest"):
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
        )
    if m in ("lstm", "bilstm", "bilstm_attention"):
        return dict(
            hidden_size=trial.suggest_categorical("hidden_size", [16, 32, 64, 96, 128]),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            epochs=trial.suggest_int("epochs", 20, nn_epochs_max, step=5),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        )
    raise ValueError(f"No search space for model: {model_name}")


# =============================================================================
# Objective + study
# =============================================================================

def _objective(trial, model_name, X, y, n_splits, seed, nn_epochs_max):
    params = suggest_params(trial, model_name, nn_epochs_max)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (tr, va) in enumerate(tscv.split(X)):
        try:
            model = get_model(model_name, seed=seed, **params)
            model.fit(X[tr], y[tr], verbose=False)
            pred = model.predict(X[va])
            rmse = calculate_metrics(y[va], pred).rmse
        except optuna.TrialPruned:
            raise
        except Exception as e:
            # A bad parameter combination should not crash the study.
            print(f"  trial {trial.number} fold {fold} failed: {e}")
            return float("inf")
        if not np.isfinite(rmse):
            return float("inf")
        scores.append(rmse)
        trial.report(float(np.mean(scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


def tune_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 60,
    n_splits: int = 4,
    seed: int = 20,
    timeout: float | None = None,
    nn_epochs_max: int = 40,
) -> "optuna.Study":
    """Run an Optuna study for one model; returns the completed study."""
    set_seed(seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        study_name=f"tune_{model_name}",
    )
    study.optimize(
        lambda t: _objective(t, model_name, X, y, n_splits, seed, nn_epochs_max),
        n_trials=n_trials, timeout=timeout, show_progress_bar=False,
    )
    return study


def default_cv_rmse(model_name, X, y, n_splits, seed):
    """Walk-forward CV RMSE with the library default parameters (baseline)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, va in tscv.split(X):
        model = get_model(model_name, seed=seed)
        model.fit(X[tr], y[tr], verbose=False)
        scores.append(calculate_metrics(y[va], model.predict(X[va])).rmse)
    return float(np.mean(scores))


# =============================================================================
# Persistence
# =============================================================================

def _delta_pct(entry: dict) -> float:
    """
    Percentage change in CV RMSE, tuned minus default. NEGATIVE means less error.

    Reads `delta_rmse_pct` when present and otherwise derives it from the two
    RMSEs, so a JSON written before the sign convention changed still summarises
    correctly rather than plotting the old `improvement_pct` upside down.
    """
    if "delta_rmse_pct" in entry:
        return float(entry["delta_rmse_pct"])
    base, tuned = entry.get("default_cv_rmse"), entry.get("best_cv_rmse")
    if base:
        return 100.0 * (tuned - base) / base
    return float("nan")


def save_tuned_params(results: dict, path: str = DEFAULT_PARAM_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved tuned parameters to {path}")


def load_tuned_params(path: str = DEFAULT_PARAM_PATH) -> dict:
    """
    Load {display_model_name: params_dict} for use by the holdout runners.

    The JSON stores results keyed by the CLI model name; this maps them to the
    display names the runners use (e.g. 'randomforest' -> 'RandomForest',
    'bilstm_attention' -> 'BiLSTM+Attention') and also keeps the CLI keys.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    name_map = {
        "randomforest": "RandomForest", "xgboost": "XGBoost", "lightgbm": "LightGBM",
        "lstm": "LSTM", "bilstm": "BiLSTM", "bilstm_attention": "BiLSTM+Attention",
    }
    out = {}
    for k, v in raw.items():
        params = v.get("best_params", v) if isinstance(v, dict) else v
        out[k] = params
        if k in name_map:
            out[name_map[k]] = params
    return out


# =============================================================================
# Data + driver
# =============================================================================

def load_training_features(predictor_file, tws_file, predictors=None, lags=7, test_size=0.2):
    """Load features and return the TRAINING portion (chronological)."""
    if predictors is None:
        predictors = ["SMS", "ET", "rainfall", "runoff", "GWSA"]
    merged = load_and_preprocess_data(predictor_file, tws_file, predictors)
    lagged = create_lagged_features(merged.copy(), predictors, lags=lags)
    X, y, feat = prepare_features_target(lagged, predictors, "TWS", lags)
    n_test = int(len(X) * test_size)
    return X[:-n_test], y[:-n_test], feat


def run_tuning(
    models,
    predictor_file="../Data/All_Data.csv",
    tws_file="../Data/TWS_GRACE_GEE.csv",
    n_trials=60,
    nn_trials=20,
    n_splits=4,
    seed=20,
    nn_epochs_max=40,
    out_path=DEFAULT_PARAM_PATH,
):
    X, y, _ = load_training_features(predictor_file, tws_file)
    print(f"Tuning on TRAINING data only: X={X.shape} (walk-forward {n_splits}-fold CV)\n")

    # Merge with any previously tuned models so separate runs accumulate.
    results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as _f:
                results = json.load(_f)
            print(f"Merging into existing results with {len(results)} model(s): {list(results)}\n")
        except Exception:
            results = {}
    for model_name in models:
        is_nn = model_name.lower().replace("-", "_") in NN_MODELS
        trials = nn_trials if is_nn else n_trials
        print(f"{'='*60}\nTuning {model_name}  ({trials} trials, {'NN' if is_nn else 'tree'})\n{'='*60}")

        base_rmse = default_cv_rmse(model_name, X, y, n_splits, seed)
        study = tune_model(model_name, X, y, n_trials=trials, n_splits=n_splits,
                           seed=seed, nn_epochs_max=nn_epochs_max)

        best = study.best_params
        best_rmse = study.best_value
        n_done = len([t for t in study.trials if t.state.name == "COMPLETE"])
        n_pruned = len([t for t in study.trials if t.state.name == "PRUNED"])
        # SIGN CONVENTION: tuned MINUS default, so a NEGATIVE number means the
        # tuning reduced RMSE. Reported this way because the quantity plotted is
        # an error: "-14.8%" reads directly as "14.8% less error", whereas the
        # old "improvement" convention inverted the sign of an error change and
        # made a worse model (RandomForest) show as "-0.1%".
        delta_rmse = best_rmse - base_rmse
        delta_pct = 100.0 * delta_rmse / base_rmse if base_rmse else 0.0

        results[model_name] = {
            "best_params": best,
            "best_cv_rmse": best_rmse,
            "default_cv_rmse": base_rmse,
            "delta_rmse": delta_rmse,
            "delta_rmse_pct": delta_pct,
            "n_trials": trials,
            "n_complete": n_done,
            "n_pruned": n_pruned,
            "cv": f"TimeSeriesSplit(n_splits={n_splits})",
        }
        print(f"  default CV RMSE : {base_rmse:.4f}")
        print(f"  tuned   CV RMSE : {best_rmse:.4f}  "
              f"(delta {delta_rmse:+.4f}, {delta_pct:+.1f}%; negative = less error; "
              f"{n_done} complete, {n_pruned} pruned)")
        print(f"  best params    : {best}\n")
        save_tuned_params(results, out_path)  # checkpoint after each model

    print(f"\n{'='*60}\nTUNING SUMMARY  (delta = tuned - default; negative = less error)\n{'='*60}")
    for m, r in results.items():
        print(f"  {m:18s} RMSE {r['default_cv_rmse']:.3f} -> {r['best_cv_rmse']:.3f} "
              f"({_delta_pct(r):+.1f}%)")
    return results


def summarize_tuning(json_path: str = DEFAULT_PARAM_PATH, output_dir: str | None = None):
    """
    Build a transparent summary of the tuning experiment (all models) from the
    saved JSON: a tidy CSV, a markdown table, and a publication-quality figure
    of default-vs-tuned walk-forward CV RMSE.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from plot_style import DPI, BAR_2, pretty_model

    if output_dir is None:
        output_dir = os.path.dirname(json_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path) as f:
        raw = json.load(f)

    order = ["randomforest", "xgboost", "lightgbm", "lstm", "bilstm", "bilstm_attention"]
    rows = []
    for k in [m for m in order if m in raw] + [m for m in raw if m not in order]:
        v = raw[k]
        base, tuned = v["default_cv_rmse"], v["best_cv_rmse"]
        rows.append({
            "Model": pretty_model(k),
            "Default CV RMSE (mm)": round(base, 3),
            "Tuned CV RMSE (mm)": round(tuned, 3),
            "dRMSE (mm)": round(tuned - base, 3),
            "dRMSE (%)": round(_delta_pct(v), 1),
            "Trials (complete/pruned)": f"{v['n_complete']}/{v['n_pruned']}",
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "tuning_summary.csv")
    df.to_csv(csv_path, index=False)

    # Markdown table (for the reviewer response / supplement).
    md = ["| " + " | ".join(df.columns) + " |",
          "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(x) for x in r.values) + " |")
    md_path = os.path.join(output_dir, "tuning_summary.md")
    open(md_path, "w").write("\n".join(md) + "\n")

    # Two panels. The left keeps the absolute RMSEs, without which a percentage
    # is unreadable; the right is the quantity actually being claimed --
    # tuned MINUS default, so bars BELOW zero are the improvements.
    x = np.arange(len(df)); w = 0.38
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(2.1 * len(df) + 4, 5), gridspec_kw={'width_ratios': [1.25, 1]})

    ax.bar(x - w/2, df["Default CV RMSE (mm)"], w, label="Default", color=BAR_2[0], alpha=0.95)
    ax.bar(x + w/2, df["Tuned CV RMSE (mm)"], w, label="Tuned (Optuna)", color=BAR_2[1], alpha=0.95)
    ax.set_xticks(x); ax.set_xticklabels(df["Model"], rotation=30, ha="right")
    ax.set_ylabel("Walk-forward CV RMSE (mm)")
    ax.set_title("Default vs Optuna-tuned", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    d = df["dRMSE (mm)"].to_numpy()
    # Colour by direction rather than by model: the sign is the message.
    ax2.bar(x, d, 0.6, color=np.where(d <= 0, BAR_2[1], BAR_2[0]), alpha=0.95)
    ax2.axhline(0, color="#0b0b0b", lw=1)
    for i, (dv, dp) in enumerate(zip(d, df["dRMSE (%)"])):
        ax2.text(i, dv, f"{dp:+.1f}%", ha="center",
                 va="top" if dv <= 0 else "bottom", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(df["Model"], rotation=30, ha="right")
    ax2.set_ylabel("$\\Delta$RMSE (mm), tuned $-$ default")
    ax2.set_title("Change in error: below zero = tuning helped", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    pad = max(np.abs(d).max(), 1e-9) * 0.25
    ax2.set_ylim(min(d.min(), 0) - pad, max(d.max(), 0) + pad)

    plt.suptitle("Hyperparameter tuning, walk-forward CV on the training split only",
                 fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "tuning_summary.png")
    plt.savefig(fig_path, dpi=DPI); plt.close()

    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}\n       {md_path}\n       {fig_path}")
    return df


def main():
    p = argparse.ArgumentParser(description="Optuna hyperparameter tuning for TWS models.")
    p.add_argument("--models", nargs="+",
                   default=["randomforest", "xgboost", "lightgbm"],
                   help="Models to tune.")
    p.add_argument("--trials", type=int, default=60, help="Trials per tree model.")
    p.add_argument("--nn-trials", type=int, default=20, help="Trials per neural-net model.")
    p.add_argument("--splits", type=int, default=4, help="Walk-forward CV folds.")
    p.add_argument("--nn-epochs-max", type=int, default=40, help="Max epochs sampled for NNs.")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--predictor-file", default="../Data/All_Data.csv")
    p.add_argument("--tws-file", default="../Data/TWS_GRACE_GEE.csv")
    p.add_argument("--out", default=DEFAULT_PARAM_PATH)
    p.add_argument("--summarize", action="store_true",
                   help="Only build the summary table/figure from an existing JSON (no tuning).")
    args = p.parse_args()

    if args.summarize:
        summarize_tuning(args.out)
        return

    run_tuning(
        models=args.models, predictor_file=args.predictor_file, tws_file=args.tws_file,
        n_trials=args.trials, nn_trials=args.nn_trials, n_splits=args.splits,
        seed=args.seed, nn_epochs_max=args.nn_epochs_max, out_path=args.out,
    )


if __name__ == "__main__":
    main()
