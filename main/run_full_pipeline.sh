#!/usr/bin/env bash
# Complete regeneration of every result in the project, from raw inputs.
#
#   ./run_full_pipeline.sh                 # the product (inputs must already exist)
#   ./run_full_pipeline.sh --with-download # also re-download from Earth Engine (~30 min, ~5 GB)
#   ./run_full_pipeline.sh --check         # preflight only: report what is missing, run nothing
#   ./run_full_pipeline.sh --skip-tuning   # reuse existing gridded hyperparameters
#   ./run_full_pipeline.sh --skip-ablation # skip the feature-ablation diagnostic
#   ./run_full_pipeline.sh --with-mlp-sweep # also re-derive the MLP's fixed config (~1 h)
#   ./run_full_pipeline.sh --trials=N      # Optuna trials for the gridded model (default 15)
#   ./run_full_pipeline.sh --with-legacy   # also run the superseded basin-scale analysis
#   ./run_full_pipeline.sh --from=STEP     # resume: skip every step before STEP
#
# RESUMING. --from=STEP skips every step up to the first whose command mentions
# STEP, then runs normally from there:
#
#   ./run_full_pipeline.sh --from=downscale_uncertainty.py
#
# It takes on trust that the skipped steps' outputs are present and current --
# preflight still checks the raw inputs, but nothing verifies that a skipped
# `require` ever succeeded. Use it to continue an interrupted run, not to skip
# a step that failed. It APPENDS to the log rather than truncating it, so the
# record of the earlier steps survives the resume.
#
# WHAT THIS PRODUCES. The deliverables are the 0.1 deg MONTHLY product and the
# DAILY disaggregation of it. Everything else here either feeds those two or
# validates them. The basin-scale analysis is legacy and off by default.
#
# Two classes of step, deliberately handled differently:
#
#   require <cmd>   ABORTS the whole script on failure. Used for anything later
#                   steps depend on. A failed target or cube must never be
#                   followed by ten steps quietly consuming a stale file.
#   run <cmd>       logs and CONTINUES. Used only for leaf steps whose failure
#                   cannot corrupt anything downstream.
#
# The old version used `run` for everything, so a failed step 0 would leave every
# subsequent step reading a stale or absent GRACE target and still report success.
set -uo pipefail
cd "$(dirname "$0")"
LOG="run_full_pipeline.log"

# Prints the leading comment block of this file, so the usage text has exactly
# one home. Duplicating it into a heredoc here would be a second source of truth
# that drifts the moment a flag is added -- which is how --with-mlp-sweep came
# to be undocumented in two of the four READMEs.
usage () {
  awk 'NR>1 && /^#/ {sub(/^#( |$)/, ""); print; next} NR>1 {exit}' "$0"
}

WITH_DOWNLOAD=0
CHECK_ONLY=0
WITH_LEGACY=0
SKIP_TUNING=0
SKIP_ABLATION=0
WITH_MLP_SWEEP=0
FROM_STEP=""
GRIDDED_TRIALS=15
for arg in "$@"; do
  case "$arg" in
    --with-download) WITH_DOWNLOAD=1 ;;
    --check)         CHECK_ONLY=1 ;;
    --with-legacy)   WITH_LEGACY=1 ;;
    --skip-tuning)   SKIP_TUNING=1 ;;
    --skip-ablation) SKIP_ABLATION=1 ;;
    --with-mlp-sweep) WITH_MLP_SWEEP=1 ;;
    --from=*)        FROM_STEP="${arg#*=}" ;;
    --trials=*)      GRIDDED_TRIALS="${arg#*=}" ;;
    -h|--help)       usage; exit 0 ;;
    *)
      echo "unknown option: $arg" >&2
      echo >&2
      usage >&2
      exit 2 ;;
  esac
done

# Both helpers redirect into "$LOG", so python's stdout is not a tty and CPython
# switches to block buffering. A long step then writes nothing to the log until
# it exits, which makes a running pipeline indistinguishable from a hung one --
# during the first full run the tuning step appeared frozen for three hours while
# it was in fact progressing normally. Unbuffering costs nothing here and makes
# `tail -f run_full_pipeline.log` actually track progress.
export PYTHONUNBUFFERED=1

say () { echo -e "$*" | tee -a "$LOG"; }

# --from support. Until a command mentioning FROM_STEP is seen, every step is
# announced and skipped; from that one onward STARTED stays 1 and the pipeline
# behaves exactly as usual. The gate lives in the helpers rather than at each
# call site so a step added later cannot forget to be resumable.
STARTED=1
[ -n "$FROM_STEP" ] && STARTED=0
gate () {
  [ "$STARTED" -eq 1 ] && return 0
  for tok in "$@"; do
    case "$tok" in
      *"$FROM_STEP"*) STARTED=1; return 0 ;;
    esac
  done
  return 1
}

# Failures are COUNTED HERE, not grepped out of the log afterwards.
#
# The log is append-only across a --from resume, and the old summary both counted
# every line containing "FAILED" and echoed the matching lines back into the log
# with `tee -a`. So each run re-counted the previous run's summary and re-echoed
# it: one real failure was reported as five, and the log ended up holding lines
# like "655:97:    FAILED" -- an echo of an echo of an echo. A counter cannot do
# that, and it also scopes the report to THIS run rather than to the file's
# entire history.
FAIL_COUNT=0
FAILED_STEPS=""
note_failure () {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  FAILED_STEPS="${FAILED_STEPS}
    $*"
}

run () {
  gate "$@" || { say "\n>>> $*\n    skipped (--from=$FROM_STEP)"; return 0; }
  say "\n>>> $*"
  if "$@" >>"$LOG" 2>&1; then
    say "    OK"
  else
    say "    FAILED (continuing)"
    note_failure "$*"
  fi
}
require () {
  gate "$@" || { say "\n>>> $* [required]\n    skipped (--from=$FROM_STEP)"; return 0; }
  say "\n>>> $* [required]"
  if "$@" >>"$LOG" 2>&1; then
    say "    OK"
  else
    note_failure "$* [required]"
    say "    FAILED - aborting, because later steps depend on this."
    say "    See $LOG for the error."
    exit 1
  fi
}

# Truncate only once the arguments have parsed, so `--help`, `-h` and a bad flag
# cannot destroy the log of a previous run just by being typed. `--check` does
# still reset it, because it writes its own preflight output there. A --from
# resume APPENDS instead: the whole point is to continue a run, and throwing away
# the record of the steps being skipped would make the log describe less than
# actually happened.
if [ -n "$FROM_STEP" ]; then
  say "\n=== resuming from $FROM_STEP ==="
else
  : > "$LOG"
fi

say "=== regeneration started: $(date) ==="

# A --from that matches nothing would skip every step and exit "all steps OK",
# which is the most misleading possible outcome. Check it against the file now.
if [ -n "$FROM_STEP" ] && ! grep -q -- "$FROM_STEP" "$0"; then
  say "\n--from=$FROM_STEP matches no step in this pipeline. Nothing has been run."
  exit 2
fi

# ---------------------------------------------------------------------------
# PREFLIGHT. Two files cannot be regenerated by any script in this repository.
# If either is missing the run cannot proceed, and finding that out now is much
# cheaper than finding out after several hours of model fitting.
# ---------------------------------------------------------------------------
say "\n--- preflight ---"
MISSING=0
need () {  # need <path> <how to obtain it>
  if [ -e "$1" ]; then
    say "  present : $1"
  else
    say "  MISSING : $1"
    say "            -> $2"
    MISSING=1
  fi
}

need "../Data/Ganga Basin Shapefile/Ganga_basin.shp" \
     "Basin boundary. NOT regenerable - restore from git or a backup."
need "../Data/Groundwater/gwl_india/Quality_controlled_groundwater_levels_over_India/Output/CGWB_India_filtered_Dug_wells_GWLs_ref_sy_2000_2022.csv" \
     "CGWB wells (Kuruva et al. 2025). NOT regenerable - re-download the 27 MB zip from https://doi.org/10.6084/m9.figshare.29293877 (v3) into Data/Groundwater/gwl_india/"

if [ "$WITH_DOWNLOAD" -eq 0 ]; then
  # Only the LEGACY basin-scale path reads this. It is checked under
  # --with-legacy rather than unconditionally, so a run that only wants the
  # gridded product -- the deliverable -- is not blocked by a file no gridded
  # step opens. See the note beside build_all_data.py below.
  if [ "$WITH_LEGACY" -eq 1 ]; then
    need "../Data/Outputs/Daily_GEE_GLDAS_V021.csv" \
         "Basin-mean predictors, LEGACY path only. Re-run with --with-download, or: python gee_download.py"
  fi
  need "../Data/Gridded/cubes/grace_cube.nc" \
       "GRACE cube. Re-run with --with-download, or: python gee_gridded_download.py && python build_cube.py"
  need "../Data/Gridded/cubes/era5_cube.nc" \
       "ERA5-Land cube. Re-run with --with-download, or: python gee_gridded_download.py && python build_cube.py"
  need "../Data/Gridded/cubes/grids_aux.nc" \
       "Grid geometry + mascon partition. Re-run with --with-download, or: python build_cube.py"
  need "../Data/Gridded/cubes/static_cube.nc" \
       "Static covariates. Re-run with --with-download, or: python gee_static_download.py && python build_cube.py"
fi

if [ "$MISSING" -ne 0 ]; then
  say "\nPreflight FAILED. Nothing has been run."
  exit 1
fi
say "  preflight OK"

if [ "$CHECK_ONLY" -eq 1 ]; then
  say "\n--check requested; stopping before any work."
  exit 0
fi

# ---------------------------------------------------------------------------
# 0. ACQUISITION (only with --with-download). Slow, hits Earth Engine.
# ---------------------------------------------------------------------------
if [ "$WITH_DOWNLOAD" -eq 1 ]; then
  say "\n--- acquisition (Earth Engine) ---"
  require python gee_download.py            # -> Data/Outputs/Daily_GEE_GLDAS_V0*.csv
  require python gee_gridded_download.py    # -> Data/Gridded/raw/  (~5 GB)
  require python gee_static_download.py     # -> Data/Gridded/static/ (covariates)
  require python build_cube.py              # -> Data/Gridded/cubes/ (incl. static_cube)
fi

# ---------------------------------------------------------------------------
# 1. INPUT TABLES. Everything below reads these.
# ---------------------------------------------------------------------------
say "\n--- inputs ---"

# All_Data.csv is built under --with-legacy, NOT here.
#
# It used to be a `require` in the main flow, which meant a run that wanted only
# the 0.1 degree product -- the deliverable, and the only thing the paper reports
# -- aborted at step 1 unless Daily_GEE_GLDAS_V021.csv was present, in order to
# build a table no gridded module opens. Verified: none of downscale_features,
# downscale_model, downscale_daily, downscale_uncertainty, downscale_holdouts or
# validate_wells reads All_Data.csv or gridded_config.PREDICTOR_TABLE; every
# consumer (run_analysis.py, tune_hyperparameters.py, analyze_results.py, the
# holdout_* modules) is legacy or uncalled.

# GRACE target, derived from the same gridded mascon product the 0.1 deg
# pipeline uses so both halves rest on one source. EVERYTHING depends on this.
require python export_basin_grace.py --compare

# Independent validation set (never a predictor).
require python wells_ingest.py

# ---------------------------------------------------------------------------
# 2. GRIDDED (spatial downscaling to 0.1 deg) -- THE PRODUCT
# ---------------------------------------------------------------------------
# This section, plus the daily disaggregation below it, produces the deliverable.
# The basin-scale analysis that used to run here is legacy and now sits behind
# --with-legacy at the end: it is a different model on basin-mean series, it
# shares no code path with the downscaling, and nothing here depends on it.
say "\n--- gridded (the product) ---"

# 2a. FEATURE ABLATION. Does the design matrix earn its size? Scored on the same
# grouped CV the tuner optimises, so the two tables are directly comparable.
#
# It exists because the memory block is far smaller than its column count: PCA on
# the 42 antecedent/API features puts 97% of their variance in 10 components,
# consecutive antecedent windows correlate up to 0.98, and `gwsa` is exactly
# `runoff_surface - runoff_surface_clim_mean` with both already present (its own
# `gwsa_clim_mean` is identically zero). Against ~19 independent mascons that
# redundancy is worth quantifying rather than assuming either way.
#
# `run`, and it changes nothing by itself: the ACTIVE feature set is fixed by
# gridded_config.PREDICTORS and downscale_features.ANTECEDENT_MONTHS /
# API_TAU_MONTHS. This step reports what those choices cost; changing them is a
# deliberate edit, not a side effect of a pipeline run.
if [ "$SKIP_ABLATION" -eq 0 ]; then
  run python downscale_ablation.py
else
  say "\n>>> feature ablation skipped (--skip-ablation)"
fi

# 2b. TUNING of the model that ACTUALLY BUILDS THE PRODUCT, scored on
# mascon-grouped CV of the anomaly -- the same criterion the product is judged
# by, and spatial rather than temporal because transfer across mascons is what
# the model is asked to do.
#
# Previously the pipeline tuned six BASIN-SCALE models on basin-mean series and
# fed the result to run_analysis.py, while downscale_model.py used hardcoded
# hyperparameters. The entire tuning budget went to models that do not produce
# the deliverable.
#
# `run`, not `require`: downscale_model.py falls back to its documented defaults
# when the JSON is absent, so a tuning failure degrades the run rather than
# invalidating it. Skipped entirely with --skip-tuning, which is the right
# choice when only the product needs regenerating.
# Free (reads one CSV column, builds nothing), so it runs before the expensive
# steps: a sweep measured on a different predictor set is worth knowing about
# now rather than after an hour of tuning. The sweep ITSELF runs after tuning --
# see below.
run python mlp_configuration_sweep.py --check-stale

GRIDDED_PARAMS="../Results/tuning/gridded_best_params.json"
# Derived from GRIDDED_PARAMS so the pair stays together if that path moves;
# tune_gridded.py --select writes the winner beside the JSON it ranked.
SELECTED_FILE="$(dirname "$GRIDDED_PARAMS")/selected_model.txt"
if [ "$SKIP_TUNING" -eq 0 ]; then
  # ALL candidate models are tuned, then the best is selected on the same
  # grouped CV. Tuning only one would leave the ensemble members below on
  # hand-set defaults while one member was optimised, which would put a
  # tuned-vs-untuned artefact into `sigma_within` -- a term that is supposed to
  # measure disagreement between model FAMILIES.
  #
  # Cost is models x trials x folds refits, so this is the expensive step: 80
  # fits per model at the default 15 trials / 5 folds. The MLP dominates at
  # ~100 s per fit on this dataset against tens of seconds for the trees, so
  # expect hours rather than minutes.
  # --trials=N lowers it; --skip-tuning reuses whatever is already in the JSON.
  run python tune_gridded.py --all --trials "$GRIDDED_TRIALS" --out "$GRIDDED_PARAMS"
else
  say "\n>>> tuning skipped (--skip-tuning); using whatever is in $GRIDDED_PARAMS"
fi

# The product is built with the SELECTED model, not a hardcoded one. Falls back
# to xgboost when no selection exists (e.g. --skip-tuning on a fresh checkout),
# and says which it used rather than choosing silently.
GRIDDED_MODEL="xgboost"
if [ -s "$SELECTED_FILE" ]; then
  GRIDDED_MODEL="$(tr -d '[:space:]' < "$SELECTED_FILE")"
  say "    selected model: $GRIDDED_MODEL (from $SELECTED_FILE)"
else
  say "    no selection file; defaulting to $GRIDDED_MODEL"
fi

# The MLP's configuration evidence. Opt-in, because re-deriving it every run is
# exactly the cost that fixing the configuration was meant to avoid.
#
# It runs HERE, after tuning, on purpose. Its tree reference uses each tree's
# ADOPTED configuration, read from the tuning JSON -- so the comparison is the
# MLP's best of 21 configurations against the trees at THEIR best. Run before
# tuning it would silently fall back to hand-set tree defaults, which flatters
# the network on a technicality and weakens the one claim the sweep exists to
# make: that the MLP's loss is not an artefact of how it was configured.
if [ "$WITH_MLP_SWEEP" -eq 1 ]; then
  run python mlp_configuration_sweep.py
fi

require python downscale_model.py --model "$GRIDDED_MODEL"   # spatial CV + 0.1 deg monthly product

# TreeSHAP attribution of the fitted model, grouped by feature FAMILY rather than
# by column -- 78 columns spread one physical signal across six antecedent
# windows and several APIs, so a per-column ranking understates it.
#
# `run`, not `require`: it is a diagnostic, and a missing joblib must not fail a
# run that has already produced the product. It reads the persisted model rather
# than refitting, so it must come after the step above.
#
# It explains the ANOMALY only. Level and trend are taken from GRACE per mascon
# and are never fitted, so nothing here speaks to the depletion trend.
#
# If selection ever picks the MLP this step fails its argument check and the run
# continues, which is the correct outcome: TreeSHAP explains trees, and there is
# no equivalent attribution for the network that would be comparable to it.
run python downscale_shap.py --model "$GRIDDED_MODEL"
run python downscale_uncertainty.py                 # per-pixel uncertainty ensemble

# 3. DAILY product. Disaggregates the monthly field using ERA5-Land fluxes and
#    states, de-meaned within each month so the monthly means -- and therefore
#    GRACE -- are preserved exactly. Deliberately not an ML step: with no daily
#    target, a daily model would be trained on an interpolation of the monthly
#    product and could only reproduce that assumption.
#
#    Note this makes monthly closure an ARITHMETIC IDENTITY, not a test: the
#    de-meaning guarantees it, and the residual printed (~5e-4 mm) is float
#    error. Evidence for the within-month shape is the agreement between the two
#    independent derivations (`daily_method_spread`), not a closure statistic.
require python downscale_daily.py

# 4. VALIDATION of the product.
#
# Two axes, measured separately because they are different claims:
#   downscale_model.py     SPATIAL transfer, leave-one-mascon-out (already run above)
#   downscale_holdouts.py  TEMPORAL transfer, holding out whole MONTHS
#
# The month holdouts report random / blocked / forward on one table. The random
# split is included but labelled OPTIMISTIC: monthly TWSA is autocorrelated, so
# a random split leaves each test month's neighbours in training. It is reported
# because the gap between it and `forward` quantifies that optimism, which is
# more convincing than asserting it -- and because it is the holdout the
# first-round reviewers saw.
run python downscale_holdouts.py --model "$GRIDDED_MODEL"
run python generate_gridded_maps.py --model "$GRIDDED_MODEL"  # per-pixel climatology, seasonal, trend, uncertainty

# Annual cycle of the basin, scored on LEAVE-ONE-MASCON-OUT predictions rather
# than on the published field. The product's basin mean is pinned to GRACE by
# mass conservation (r = 1.00000, RMSE 0.29 mm over the 227 observed months), so
# plotting the product against GRACE would draw one curve twice and read as
# perfect skill. Reads lomo_oof_<model>.csv, written by downscale_model.py.
run python downscale_annual_cycle.py --model "$GRIDDED_MODEL"
run python validate_wells.py                        # independent well comparison (quarterly, in mm)
# The well validation uses ONE source: the published, quality-controlled CGWB
# dataset of Kuruva et al. (2025, Sci Data 12, 1609), 656 dug wells in basin with
# a per-well Reference_Sy. A second in-house compilation was evaluated and not
# adopted; a peer-reviewed, openly archived source with per-well specific yield is
# reproducible independently of us, which an in-house one is not.

# Wells at three aggregation scales. Neither side of the well comparison is an
# observation of groundwater storage -- the model side subtracts modelled stores,
# the well side multiplies a measured level by an estimated specific yield -- so a
# single pooled number cannot say which estimate the disagreement belongs to.
# Aggregating averages away per-well Sy and point-to-pixel error, so basin/mascon/
# per-well read together separate scale-dependent skill from decomposition error.
run python validate_wells_scales.py

# ---------------------------------------------------------------------------
# 5. LEGACY BASIN-SCALE (only with --with-legacy)
# ---------------------------------------------------------------------------
# Superseded. Kept runnable so the earlier manuscript's figures can be
# reproduced, not because anything downstream needs it.
#
# What it is: a basin-MEAN temporal model on Data/All_Data.csv, where the whole
# basin takes one value per day. Its random and temporal holdouts describe that
# model, not the 0.1 deg downscaling, and its `monthly_seasonal_maps` coloured
# the entire basin polygon with a single value per month -- superseded per-pixel
# by generate_gridded_maps.py, so generate_monthly_maps.py is no longer called.
#
# Its temporal closure test also retires with it. That test was meaningful
# because the basin-scale daily series came from a model free to disagree with
# GRACE; the gridded daily field cannot disagree, by construction.
if [ "$WITH_LEGACY" -eq 1 ]; then
  say "\n--- legacy basin-scale (superseded) ---"

  # The basin-mean predictor table, built here because this is the only section
  # that reads it. Reconstructed from the GEE download so its provenance is in
  # code rather than an uncommitted manual step; verified rather than overwritten
  # when it already exists, which catches a stale table without discarding one a
  # run in progress may be reading.
  if [ ! -e "../Data/All_Data.csv" ]; then
    require python build_all_data.py --write
  else
    run python build_all_data.py --verify
  fi

  TUNED_PARAMS="../Results/tuning/best_params.json"
  run python tune_hyperparameters.py --models randomforest xgboost lightgbm \
          --trials 50 --splits 4 --out "$TUNED_PARAMS"
  run python tune_hyperparameters.py --models lstm bilstm bilstm_attention \
          --nn-trials 15 --splits 3 --out "$TUNED_PARAMS"
  run python tune_hyperparameters.py --summarize --out "$TUNED_PARAMS"
  run python run_analysis.py --analysis all --compare --tuned-params "$TUNED_PARAMS"
  run python analyze_results.py --holdout-dir ../Results/figures/temporal_holdout --n-boot 2000
  run python analyze_results.py --holdout-dir ../Results/figures/random_holdout --n-boot 2000
  run python analyze_results.py --leakage --models randomforest xgboost lightgbm
  run python temporal_closure_validation.py --predictions-dir ../Results/figures/temporal_holdout --n-boot 2000
fi

# ---------------------------------------------------------------------------
say "\n=== regeneration finished: $(date) ==="
if [ "$FAIL_COUNT" -gt 0 ]; then
  # Names the steps directly. Nothing is grepped back out of the log, so this
  # block cannot feed itself on the next resume.
  say "\n$FAIL_COUNT step(s) FAILED in this run - inspect $LOG before trusting any output:$FAILED_STEPS"
  exit 1
fi
say "all steps OK"
