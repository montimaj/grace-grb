# Changelog

What changed between the original basin-scale study and the current 0.1° gridded
product, and why.

This file exists so [`README.md`](README.md) can describe what the project **is**
without also carrying what it **used to be**. Every note here was previously an
inline caveat in the README; none of it is new. Anything marked *retired* below
is kept in the tree only so the earlier manuscript's figures remain
reproducible — not because it is still in use.

Versions track the Zenodo record. The scientific method of record is
[`METHODS.md`](METHODS.md); this file records history, not method.

---

## [1.0.1] — 2026-09-11

Changes made while answering the second round of review for *Groundwater for
Sustainable Development*. No value in the product changed: the 0.1° monthly and
daily fields, the model and the uncertainty components are bit-for-bit those of
1.0.0. Two things changed. Numbers quoted in the paper became reproducible, which
added one result file, `Results/downscaling/trend_by_region.csv`, and three figure
scripts. And the deposit's own documentation was corrected against the archives:
most consequentially, a licence note that denied redistributing three
NonCommercial-restricted raster sets the record does in fact redistribute.

Version DOI <https://doi.org/10.5281/zenodo.22718970> —
concept DOI <https://doi.org/10.5281/zenodo.21745158>.

The R2 manuscript cites this release, not 1.0.0: the regional trend figures in
Section 5.5 are produced by `main/trend_regions.py`, which exists only from
1.0.1, so a reader who fetched 1.0.0 would not find the script that made them.

### Added

- **`main/trend_regions.py`.** The manuscript quoted a mean trend for "the
  Punjab–Haryana–western Uttar Pradesh plain" that no committed mask defined, so
  a reader holding the archived trend field could not reproduce it. The basin is
  now split at 27°N and 79.5°E into four quadrants, written down in one place,
  and `main()` asserts the partition — every one of the 9,538 tested cells falls
  in exactly one quadrant — before writing
  `Results/downscaling/trend_by_region.csv`. The text now quotes that file.
- **`figures/make_fig3_holdouts.py`** (Fig. 3), which re-runs the seeded month
  selection from `downscale_holdouts.py` and draws all three holdout designs
  against the observation record, answering a reviewer question about whether
  the blocked experiment covers the GRACE/GRACE-FO mission gap. It does not, and
  cannot; the figure shows why.
- **`figures/make_fig10_trend_regions.py`** (Fig. 10) and
  **`figures/make_fig5_lomo_metrics.py`** (Fig. 5). Fig. 10 takes its region
  masks from `trend_regions.region_mask()` rather than recomputing them, after
  an earlier version disagreed with the table by 40 pixels.

### Changed

- **Manuscript title**, which gained *Anomalies*: "... Disaggregation of GRACE
  Terrestrial Water Storage **Anomalies** over the Ganges River Basin". TWSA is
  the quantity the model actually produces, and the title now says so.
- Fig. 10 replaces the former trend map, carrying the same field with the
  quadrant boundaries drawn on it and a second panel showing the distribution of
  trends within each region.
- `figures/README.md` now states the Figure 1 attribution requirement directly,
  in place of the removed caption file.

### Removed

- **`.zenodo.json`.** It silently overrode `CITATION.cff`, which is now the only
  deposit metadata.
- **`figures/output/FIGURE_CAPTIONS.md`.** A second copy of the captions that
  drifted from the manuscript: it described Figure 2's colour key backwards, and
  that error reached the submitted paper. The attribution requirement it existed
  to record now lives in `figures/README.md`; the duplicated caption text is
  gone.

### Fixed

Documentation errors found by auditing the deposit against the archives
themselves rather than against the previous README.

- **The licence note denied a redistribution that is happening.**
  `DATA_README.md` stated that "no MERIT, HWSD or C3S raster is redistributed
  here". All three are: `inputs_static_covariates.zip` carries MERIT Hydro
  (`upa_log.tif`, `hnd.tif`), HWSD v2 (`awc.tif`, `root_depth.tif`) and the ESA
  C3S land cover (`crop_irrigated_*.tif`, `crop_rainfed_*.tif`), each under its
  own provider's terms rather than this record's CC-BY-4.0. That sentence was the
  one a NonCommercial-restricted user would have relied on. The true half of the
  claim is kept: none of their values can be recovered from the released
  products.
- **The third-party archives were misnamed in both directions.** `DATA_README.md`
  pointed at `inputs_raw_gee.zip` and `inputs_cgwb_wells.zip`; the first holds
  only ERA5-Land (CC-BY-4.0) and GRACE (public domain), and the second is
  CC-BY-4.0 and compatible with the record. Naming the raw archive encumbered
  5.97 GB with terms it does not carry, while the archive that does carry them
  went unnamed.
- **`Data/README.md` recorded the GLOBGM and CGWB licences as "not established"**
  and instructed that they be settled before publication — shipping inside the
  source archive, contradicting what the record itself asserts. Both now agree
  with `DATA_README.md`.
- **Deposit manifest.** The file counts for `evaluation_tables.zip` (23 → 24) and
  `figures.zip` (22 → 30) were stale for exactly the two archives this release
  rebuilds; `grace-grb-1.0.1.zip` and `trend_by_region.csv` had no manifest row
  at all; and a `covariate_gate_<model>.csv` row named a file that exists neither
  in the archive nor in the repository.
- **Dead pointers and pre-publication language** removed from `DATA_README.md`: a
  `TODO.md` reference to a file that ships in neither the deposit nor the source
  archive, and an instruction about settling licences "before this record is
  published", which does not belong on a permanent archive.
- `README.md` gained the four scripts this release adds, which its file tree had
  never listed, and six stale figure sizes in `figures/README.md` were corrected
  against the files on disk.

---

## [1.0.0] — 2026-08-01

Rebuilt as **spatial downscaling plus water-balance-guided temporal
disaggregation at 0.1°**. The earlier work reduced every input to one basin-mean
value per time step; this release is per-pixel.

Version DOI <https://doi.org/10.5281/zenodo.21745159> —
concept DOI <https://doi.org/10.5281/zenodo.21745158>.

### Added

- **0.1° monthly TWSA product**, Ganga basin, 2000–2025, with per-pixel
  uncertainty reported as four separate components rather than one number.
- **Daily product**, disaggregated under an exact monthly constraint, with
  nothing fitted at the daily scale.
- **Mass conservation.** Every mascon's area-weighted mean is forced back onto
  the observed value by a minimum-norm correction. Mascon-scale agreement is
  therefore arithmetic and is *not* evidence of skill — stated plainly in the
  README and on the graphical abstract so it cannot be mistaken for validation.
- **Leave-one-mascon-out spatial cross-validation**, 19 folds with a neighbour
  buffer, replacing the synthetic spatial holdout retired below.
- **Month holdouts**: random, blocked and forward (out-of-record).
- **Independent well validation** against 656 CGWB wells at three aggregation
  scales, reported as scale-dependent rather than as a single pooled number.
- **Earth Engine explorer** ([`gee/`](gee/)) — the product in a browser, no
  account required.
- **Archive metadata**: `CITATION.cff`, `DATA_README.md`.
- **`environment.yml`** — the pinned versions every result was produced under.
- **New figures**: study area, method workflow, and a graphical abstract built
  from the product rather than drawn around it.

### Changed

- **Manuscript title**, from *"Explainable AI-Based Temporal Downscaling of GRACE
  Terrestrial Water Storage in the Ganges River Basin"*.

  The original described the earlier basin-scale analysis, and its "temporal
  downscaling" would overstate the daily component of this work: the daily field
  is disaggregated under an exact monthly constraint, not downscaled.
  "Downscaling" now attaches to the spatial step, where a predictor–TWSA relation
  is fitted across mascons and interpreted with SHAP; the daily step is
  disaggregation, its within-month shape supplied by the ERA5-Land water balance
  with nothing fitted. See [METHODS.md](METHODS.md) for the distinction.

- **`GWSA` renamed to `runoff_anom`.** It contained no groundwater — it is
  de-meaned runoff — and is now carried under the honest name.

- **Units corrected.** All MAE/RMSE in the original were reported 10× too small:
  centimetres labelled millimetres.

### Removed

- **GLDAS predictors.** GLDAS 2.2 CLSM **assimilates GRACE**, so the model was
  partly predicting the target from the target. This invalidates the original
  "antecedent SMS ≈ 80% of explanatory power / storage memory" result, which
  rested on that field.

- **`TWS_JPL.xlsx`**, the earlier GRACE target. It had no recorded provenance,
  contained malformed month names (`'January '` ×20, `'july'` ×1) that silently
  dropped 8.9% of the GRACE record, stored centimetres while labelled
  millimetres, and correlated only r = 0.94 with a properly area-weighted basin
  mean of the product it claimed to be.

  `utils.read_grace_monthly()` still accepts the .xlsx layout — and now raises
  rather than dropping rows — so a restored copy would be read correctly, but no
  such file ships and `--compare` is therefore a no-op. The target everything
  uses is now `TWS_GRACE_GEE.csv`, generated by `export_basin_grace.py`.

### Retired

Kept in the tree only to reproduce the earlier manuscript's figures. Nothing in
the current method reads their output.

- **The synthetic spatial holdout.** It replicated one basin-mean series across
  fabricated locations with noise, leaked by construction, and scored R² ≈ 0.99
  for the quantity measured honestly at R² = 0.05. `holdout_spatial.py` is no
  longer reachable from `run_analysis.py`, and `--analysis spatial` exits with an
  explanation. Its replacement is `downscale_model.py --skip-product`.

- **`generate_monthly_maps.py`.** It coloured the whole basin polygon with a
  single value per month, which `generate_gridded_maps.py` supersedes per pixel.
  No pipeline calls it.

- **The covariate gate.** `downscale_covariate_gate.py` is a standalone
  diagnostic and no longer selects the feature set.

---

## [0.1.0] — the basin-scale study, 2025-06 to 2026-07

The original manuscript: a **basin-scale (spatially integrated) temporal**
downscaling. All inputs were basin-mean series, so those results are not
per-pixel and the corresponding "maps" are basin-scale summaries. The daily
target was a linear interpolation of monthly GRACE — no independent daily
observation — so daily consistency was assessed with a temporal closure test
(`temporal_closure_validation.py`).

That code is retained under `--with-legacy` and shares no code path with the
downscaling above.

### The superseded abstract

Recorded here because it is what the original submission claimed, and because
several of its claims no longer hold — see *Changed* and *Removed* above for
which, and why.

> Terrestrial water storage anomalies (TWSA) are a key indicator of hydrological
> variability in a basin and are widely used to assess groundwater sustainability
> and climate-driven changes in the water cycle. GRACE and GRACE-FO satellite
> gravimetry are an invaluable source of observations of TWSA, but due to their
> low spatial resolution and monthly time resolution they cannot be directly used
> to carry out high-resolution hydrological analysis and water resource
> management, especially in monsoon-controlled basins. This study fills this gap
> in the temporal downscaling and prediction of TWSA down to a daily resolution
> and physical interpretability of the Ganges River Basin. We develop an
> explainable artificial intelligence (XAI) model that combines GRACE-derived
> TWSA with daily hydroclimatic predictors, including precipitation,
> evapotranspiration (ET), soil moisture storage (SMS), surface runoff, and
> groundwater storage anomalies, evaluated with random and temporal holdout
> strategies and interpreted with SHapley Additive exPlanations (SHAP). The
> ensemble tree-based models (Random Forest and XGBoost) perform best under the
> random and temporal validation schemes and are statistically indistinguishable
> from each other, while outperforming the recurrent-network models. Because
> GRACE observes TWSA only monthly, daily-scale skill is assessed indirectly
> through a temporal closure test, in which the predicted daily TWSA is
> re-aggregated to monthly and compared against the original monthly GRACE; this
> confirms strong temporal self-consistency. SHAP analysis shows that antecedent
> SMS is the primary predictor of TWSA (about 80% of the total explanatory
> power), followed by lagged ET (about 10-15%), with relatively minor direct
> effects of precipitation, providing a mechanistic, storage-memory-based
> explanation of the reconstructed seasonal cycles. The framework enhances
> GRACE-based hydrological monitoring by providing interpretable TWSA prediction
> that can support groundwater assessment and climate-resilient water-resource
> management in data-sparse regions.
