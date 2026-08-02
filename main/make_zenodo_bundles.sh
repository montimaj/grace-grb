#!/usr/bin/env bash
# Assemble the Zenodo deposit: a handful of zips plus the loose product netCDFs.
#
# WHY ZIP AT ALL
# --------------
# Not for space. The netCDFs are already internally compressed (zlib level 4)
# and the COGs are DEFLATE, so zipping them returns about 1%. Zip is here for
# FILE COUNT: Zenodo caps a record at 100 files and this deposit contains ~28,000.
# Everything except the four product netCDFs is therefore bundled, and the
# netCDFs stay loose because they are what people actually come for -- burying
# the 31 MB monthly field inside an 838 MB archive would force a large download
# on anyone who wants the small file.
#
# -1 (fastest) rather than the default: the inputs are overwhelmingly
# already-compressed rasters, so higher levels spend a long time to save almost
# nothing.
#
# Output goes OUTSIDE the repository tree by default, because 13 GB of archives
# in the working copy is a git accident waiting to happen.

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="${1:-$ROOT/../grace-grb-zenodo}"
mkdir -p "$OUT"

echo "repo:   $ROOT"
echo "output: $OUT"
echo

# GLDAS is EXCLUDED from the deposit. It is not a predictor of the gridded
# product, is not used anywhere in the paper, and its only consumer is the
# optional `validate_wells.py --stores gldas` comparison. Dropping the raw tiles
# and gldas_cube.nc removed ~1.4 GB. The one file that stays is
# Daily_GEE_GLDAS_V021.csv. That file feeds All_Data.csv, which no gridded module
# reads -- build_all_data.py and its preflight check now live inside the
# --with-legacy block, so nothing on the path to the deliverable needs either.
# Both cases: zip's -x patterns are case-sensitive on Unix, and the paths are
# lowercase (Data/Gridded/raw/gldas/) while the basin-mean file is uppercase
# (Daily_GEE_GLDAS_V021.csv). Matching only one let that file through a rebuild
# that reported success.
# .geeup-state.json is geeup's upload resume state, written into the COG
# directory when an ingest runs. It is a transient artefact of the upload, not
# part of the product, and it must not travel in the archive.
EXCLUDE=(-x '*.DS_Store' '*gldas*' '*GLDAS*' '*.geeup-state.json')

bundle () {  # bundle <zipname> <path> [more paths...]
  local name="$1"; shift
  local target="$OUT/$name"
  if [ -e "$target" ]; then
    echo "  $name: exists, skipped"
    return 0
  fi
  local missing=0
  for p in "$@"; do [ -e "$p" ] || { echo "  $name: MISSING $p"; missing=1; }; done
  [ "$missing" -eq 1 ] && return 1
  echo "  $name ..."
  # Daily_GEE_GLDAS_V021.csv is re-added below: the blanket *gldas* exclusion
  # would otherwise drop a file the pipeline preflight requires.
  ( cd "$ROOT" && zip -r -q -1 "$target" "$@" "${EXCLUDE[@]}" ) \
    && printf '    %s\n' "$(du -h "$target" | cut -f1)"
}

echo "--- inputs ---"
bundle inputs_raw_gee.zip          "Data/Gridded/raw"
bundle inputs_static_covariates.zip "Data/Gridded/static"
bundle inputs_basin_shapefile.zip  "Data/Ganga Basin Shapefile"
bundle inputs_cgwb_wells.zip       "Data/Groundwater"

echo "--- intermediates ---"
bundle intermediates_cubes.zip     "Data/Gridded/cubes"
bundle intermediates_basin_series.zip "Data/Outputs"

echo "--- outputs ---"
bundle cogs_monthly.zip "Results/downscaling/cogs/twsa_0p1deg_monthly_twsa__sigma_total"
bundle cogs_daily.zip   "Results/downscaling/cogs/twsa_0p1deg_daily_twsa_flux__twsa_state__daily_method_spread"
# The trend field in both formats in one bundle. Small enough (0.5 MB) that
# separating them would only make a reader fetch twice; and unlike the three
# products, the trend is a derived diagnostic rather than something people come
# to the record specifically to download.
bundle trend_field.zip  "Results/downscaling/cogs/twsa_0p1deg_trend" \
                        "Results/downscaling/twsa_trend_significance.nc"

echo "--- evaluation and figures ---"
# Evaluation is every CSV/JSON the pipeline wrote, plus the tuning record and the
# per-figure caption files. Globs are evaluated from $ROOT by the subshell above,
# so they are passed through unexpanded here.
( cd "$ROOT" && zip -q -1 "$OUT/evaluation_tables.zip" \
    Results/downscaling/*.csv Results/downscaling/*.json Results/downscaling/CAPTIONS.md \
    Results/tuning/* 2>/dev/null ) && echo "  evaluation_tables.zip  $(du -h "$OUT/evaluation_tables.zip" | cut -f1)"
( cd "$ROOT" && zip -r -q -1 "$OUT/figures.zip" \
    Results/figures Results/downscaling/*.png figures/output 2>/dev/null ) \
    && echo "  figures.zip          $(du -h "$OUT/figures.zip" | cut -f1)"

echo "--- loose product files (not zipped, deliberately) ---"
for f in Results/downscaling/twsa_0p1deg_monthly_with_uncertainty.nc \
         Results/downscaling/twsa_0p1deg_daily.nc \
         Results/downscaling/twsa_0p1deg_monthly_xgboost.nc; do
  if [ -e "$f" ]; then
    cp -n "$f" "$OUT/" 2>/dev/null
    printf '  %-46s %s\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)"
  else
    echo "  MISSING $f"
  fi
done

cp -f DATA_README.md "$OUT/" 2>/dev/null && echo "  DATA_README.md"

echo
echo "=== deposit ==="
ls -1 "$OUT" | wc -l | sed 's/^/  files: /'
du -sh "$OUT" | cut -f1 | sed 's/^/  total: /'
