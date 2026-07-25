#!/usr/bin/env bash
# Deterministic regeneration of every figure at 600 DPI with cleaned labels,
# plus all reviewer-requested statistical outputs. Seeds are fixed, so results
# are reproducible. Each step logs and continues on error.
set -u
cd "$(dirname "$0")"
LOG="regenerate_all.log"
echo "=== regeneration started: $(date) ===" | tee "$LOG"

run () { echo -e "\n>>> $*" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1 && echo "    OK" | tee -a "$LOG" || echo "    FAILED (continuing)" | tee -a "$LOG"; }

# 1. All three holdouts + cross-holdout comparison (regenerates performance,
#    comparison, overfitting, SHAP, inline figures; runners also emit stats CSVs
#    and the temporal closure test).
run python run_analysis.py --analysis all --compare

# 2. Monthly / seasonal basin-scale maps and cycles.
run python generate_monthly_maps.py

# 3. Statistics figures (metrics-with-CI, DM matrix) for each holdout.
run python analyze_results.py --holdout-dir ../Results/figures/temporal_holdout --n-boot 2000
run python analyze_results.py --holdout-dir ../Results/figures/random_holdout --n-boot 2000
run python analyze_results.py --holdout-dir ../Results/figures/spatial_holdout --n-boot 2000

# 4. Leakage diagnostic (random vs blocked vs purged CV).
run python analyze_results.py --leakage --models randomforest xgboost lightgbm

# 5. Temporal closure test figures (also run by the temporal holdout, repeated
#    here so it works even if run standalone).
run python temporal_closure_validation.py --predictions-dir ../Results/figures/temporal_holdout --n-boot 2000

echo -e "\n=== regeneration finished: $(date) ===" | tee -a "$LOG"
