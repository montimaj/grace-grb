# `Results/` — what the pipeline writes

Everything under this directory is **output**. Nothing here is an input to
anything else in the repository, so deleting the whole tree costs a re-run, not
data. The pipeline's Python steps create their own directories as they write, so a directory that is absent means the step that owns it has not run.

Three documents sit behind this one, and this file points at them rather than
copying them — duplicated tables are how this project's docs drifted apart
before:

| for | read |
|---|---|
| what each product **contains** — grid, time, variables, size, caveats | [root README, *Final products*](../README.md#final-products). That section is the manifest. |
| what is fitted, what is constrained, what is validated and what is not | [METHODS.md](../METHODS.md) |
| which script writes which file, and in what order | [main/README.md](../main/README.md) |

> **What is actually on disk depends on how far the last run got.** The pipeline
> has two step classes: a `require` step aborts the run, a `run` step logs and
> continues. So a tree that is missing `Results/figures/` but has a complete
> `Results/downscaling/` is a normal outcome — the map and well steps are leaf
> steps and were allowed to fail or had not been reached. Read
> `main/run_full_pipeline.log` before concluding a file is missing because
> something is broken. A run in progress leaves the same signature.

---

## `downscaling/`

The products and the evidence for them, in one directory because they are read
as a set: a product file without its validation CSVs cannot be assessed.

**Data products.** Three netCDF files — `twsa_0p1deg_monthly_with_uncertainty.nc`
(the one to cite or distribute), `twsa_0p1deg_daily.nc`, and the single-model
intermediate `twsa_0p1deg_monthly_<model>.nc` that `downscale_model.py` writes
before the uncertainty ensemble supersedes it. Their dimensions, variables and
per-file caveats are in the [manifest](../README.md#final-products); they are not
repeated here. `generate_gridded_maps.py` adds a fourth,
`twsa_trend_significance.nc`, holding the per-pixel Mann-Kendall trend field.

**Evidence CSVs and JSON.** The manifest also enumerates the curated evidence set
— leave-one-mascon-out, month holdouts, gap recovery, well validation, ablation,
transfer error by mascon and season, and `summary_<model>.json`. A completed run
leaves more here than that table lists: `lomo_cv_ci_<model>.csv` (bootstrap
intervals on the LOMO folds), `holdouts_month_<model>_runs.csv` (the per-run
detail behind the three pooled holdout rows), and the SHAP outputs
`shap_families_<model>.csv`, `shap_columns_<model>.csv`,
`shap_summary_<model>.json`.

**Things in here that are not products or evidence.**

- `model_<model>.joblib` — the fitted model, with its feature names and the
  parameters it was given. The product is reproducible without it (refitting is
  deterministic given the seed and the same cube), but `downscale_shap.py` reads
  it rather than refitting, and it is the only way to inspect exactly what
  produced a shipped file.
- Three figures land here rather than in `figures/`, because each documents the
  CSV beside it: `Fig_temporal_closure.png` (`downscale_daily.py`),
  `well_metric_maps.png` and `well_scales_*.png` (`validate_wells_scales.py`).
  `well_metric_maps.png` maps MBE, MAE, RMSE, R², NSE and the month count at each
  of the 656 wells, panels (a)–(f). Its panel (a) supersedes the former
  standalone `well_residual_map.png`, which plotted that same quantity alone.
- `cogs/<collection>/` — only if `export_cogs.py` is run by hand. One GeoTIFF per
  time step plus `metadata.csv` and `upload.sh`. It is not in
  `run_full_pipeline.sh`; for the daily product it is ~9,500 rasters.

---

## `tuning/`

| file | written by | holds |
|---|---|---|
| `gridded_best_params.json` | `tune_gridded.py` | per model: the tuned parameters, the hand-set defaults' grouped-CV RMSE, the tuned RMSE, `adopted` (which of the two the model is ranked on), `fold_sd`, and the CV definition |
| `selected_model.txt` | `tune_gridded.py`, selection step | one line — the model the product is built from |
| `gridded_tuning_summary.csv` | `tune_gridded.py --summarize` | the JSON flattened to a table. **A default pipeline run does not write it**: the pipeline calls `--all`, never `--summarize`, so its absence means nobody asked for it |
| `mlp_configuration_sweep.csv` | `mlp_configuration_sweep.py` | every point of the MLP configuration sweep, not only the winner. Written only under `--with-mlp-sweep` (~1 h), because re-deriving a fixed configuration on every run is the cost that fixing it exists to avoid |

Under `--with-legacy`, `tune_hyperparameters.py` writes `best_params.json` and
`tuning_summary.{csv,md,png}` here too. Those are the superseded **basin-scale**
models and share no code path with the four gridded files above.

### The product model is selected, not hardcoded

All five candidates are tuned or scored on one mascon-grouped `GroupKFold(5)`,
ranked on their *adopted* configuration — tuned only where tuning beat the
hand-set defaults on those folds, so a model is never credited with a search that
made it worse — and the winner's name is written to `selected_model.txt`.
`run_full_pipeline.sh` reads that file and passes the name to every downstream
script. Tuning only one model would have left the uncertainty ensemble's other
members on hand-set defaults while one member was optimised, putting a
tuned-versus-untuned artefact into `sigma_within`, which is meant to measure
disagreement between model *families*.

The selector then asks whether the ranking means anything, and answers it with a
measured quantity rather than a threshold picked in advance: it compares the
spread between models against each model's own fold-to-fold standard deviation on
the same folds. From the run that wrote the current `selected_model.txt`
(`main/run_full_pipeline.log`):

```
  1. xgboost          76.3238 [tuned]  <- selected
  2. random_forest    76.6016 [defaults]  (+0.278, +0.36%)
  3. mlp              77.0871 [fixed]  (+0.763, +1.00%)
  4. lightgbm         77.2284 [tuned]  (+0.905, +1.19%)
  5. xgboost_rf       77.3988 [tuned]  (+1.075, +1.41%)

  spread across models: 1.075 mm (1.41% of the best)
  typical fold-to-fold sd within one model: 6.615 mm
  NOTE: the models differ by LESS than a single model varies across folds.
        The ranking is not resolvable on these data
```

So `xgboost` builds the product because something had to, not because it is
better. The five candidates are separated by about a sixth of the noise on any
one of their scores. Nothing downstream should be read as a claim that gradient
boosting won.

Two further caveats, both in [METHODS.md §8](../METHODS.md): selection uses the
same folds the tuning optimised, so the winner's CV score is optimistic by an
unknown amount — an honest selection would need a third CV layer over ~19 spatial
units, which the data cannot support. The absolute skill figure to quote is the
leave-one-mascon-out result in `downscaling/`, computed after selection on folds
selection never touched.

---

## `figures/`

| location | contents |
|---|---|
| `figures/` | `Fig_shap_families_<model>.png` — TreeSHAP grouped by feature family. `Fig_mlp_configuration_sweep.png` — the MLP sweep, redrawable from its CSV with `--plot-only` |
| `figures/gridded_maps/` | the per-pixel maps: monthly climatology, seasonal means, RMS, trend, uncertainty components, `Fig_mascon_skill_<model>.png`, and `Fig_method_comparison_<when>.png`, which puts the downscaled field, bilinear interpolation and the native mascons side by side for one month `<when>` |
| `figures/temporal_holdout/`, `random_holdout/`, `comparison/`, `leakage_diagnostic/`, `monthly_seasonal_maps/` | **legacy basin-scale**, written only under `--with-legacy`. One value per time step for the whole basin. Retained so the earlier manuscript's figures can be reproduced; nothing in the current method reads them |

---

## None of this is in git, and the products should not be

`.gitignore` ignores `*.nc` and `*.tif` repository-wide (lines 203-204, repeated
at 227-228). There is a `!Results/downscaling/**` negation at line 214 meant to
keep the small text evidence versionable, but the blanket rules come **after**
it and last match wins, so the netCDF and GeoTIFF products stay ignored. That is
deliberate: the daily file is roughly 800 MB and belongs in the Zenodo data
record, where it can carry a DOI and a checksum, not in a git history where every
regeneration would add another copy. Check any specific path rather than reasoning
about the rules:

```bash
git check-ignore -v Results/downscaling/twsa_0p1deg_daily.nc
```

CSV, JSON, TXT and PNG under `Results/` are **not** ignored and can be committed.
At present none of them are: `git ls-tree -r HEAD` lists nothing under `Results/`,
so a fresh clone has no `Results/` directory at all until the pipeline makes one.
The root README's disk-space table is the reference for what a full run costs.

---

## `<model>` in a filename

Filenames carrying `<model>` — `twsa_0p1deg_monthly_<model>.nc`,
`lomo_cv_<model>.csv`, `summary_<model>.json`, and the rest — track whichever
model selection picked, because `run_full_pipeline.sh` substitutes the contents
of `selected_model.txt`. `xgboost` appears throughout the documentation only
because it is the fallback the pipeline uses when that file is absent, as on a
fresh checkout or under `--skip-tuning`. A tree full of `*_xgboost.*` is not
evidence that xgboost was selected; `selected_model.txt` is.

**One file breaks the pattern.** `feature_ablation_xgboost.csv` is named
literally, not by substitution: `downscale_ablation.py` takes no `--model`, runs
before tuning, and scores every feature configuration with XGBoost's hand-set
defaults on purpose. Those defaults are the point — tuning is performed on the
full matrix, so tuned values would favour the full configuration over every trim
and bias the table toward the conclusion it exists to test. The `xgboost` in that
filename is therefore a fact about the ablation, not a record of the selection.

## Citing this data

Kaushik, P. R., Majumdar, S., Lenczuk, A., Sharma, Y. K., Banerjee, S., &
Thakur, P. K. (2026). _Downscaled GRACE terrestrial water storage anomalies
for the Ganga (Ganges) River Basin at 0.1°: Monthly and daily fields with
per-pixel uncertainty, 2000–2025_ [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.21745158 — concept DOI; cite the **version** DOI
in a paper.

Cite the paper alongside it: Kaushik et al. (2026), _Groundwater for Sustainable
Development_ (under review). Full entry in the [root README](../README.md).
