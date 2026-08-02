# Downscaled GRACE TWSA, Ganga basin — 0.1°, monthly and daily, 2000–2025

README for the Zenodo record. **Code and data ship as one record**, so the
record-level licence (CC-BY-4.0) describes the DATA and the bundled source code
remains GPL-3.0-only under its own `LICENSE` — a Zenodo record carries only one
licence field, so the carve-out is stated rather than encoded (see *Licence*
below). Development history is at <https://github.com/montimaj/grace-grb>, where
`README.md` and `METHODS.md` are the method of record; this file does not restate
them.

Terrestrial water storage anomalies (TWSA) in **mm of equivalent water height**
on a 0.1° grid over the Ganga (Ganges) river basin. GRACE/GRACE-FO JPL RL06
mascons supply the mass constraint; ERA5-Land supplies the fine-scale
texture, which is **inferred, not observed**. Read *Before you use these files*
before doing anything with them.

---

## Licence

**CC-BY-4.0**, with the qualification below. Attribute as:

> Ganga basin 0.1° TWSA product (2000–2025), derived from GRACE/GRACE-FO JPL
> RL06 mascons and ERA5-Land. Kaushik, P. R. et al., 2026. DOI
> 10.5281/zenodo.21745158. CC-BY-4.0.

### Inputs that carry their own terms — read before redistributing

The fields in these files were produced by this work and are released CC-BY-4.0.
They were, however, produced *using* several third-party datasets, three of which
are more restrictively licensed than CC-BY-4.0. We name them explicitly rather
than let a downstream user assume the whole chain is permissive:

| input | what it contributed | its terms |
|---|---|---|
| **MERIT Hydro v1.0.1** (Yamazaki et al. 2019) | the `upa_log` and `hnd` covariates | **CC-BY-NC-4.0 or ODbL-1.0** (dual) |
| **HWSD v2** (FAO/IIASA) | the `awc` and `root_depth` covariates | **CC-BY-NC-SA-4.0** |
| **ESA C3S land cover (LCCS)** | the `crop_irrigated` and `crop_rainfed` covariates | educational and/or scientific use, credit required |

These are **model covariates**, not layers reproduced in the released files: no
MERIT, HWSD or C3S raster is redistributed here, and none of their values can be
recovered from the products. What the products carry is the influence of six of
seventy-eight predictor columns on a fitted anomaly field.

**What this notice does and does not do.** It discharges the attribution those
licences require, and it tells you what stands behind the product. It does not
purport to relicense anything: if your intended use is commercial, MERIT Hydro's
NonCommercial arm and HWSD's NonCommercial-ShareAlike terms may still reach you
through this product, and whether they do is genuinely unsettled — we found no
case law on whether a trained model's output is an ODbL "Derivative Database" or
a "Produced Work". **If you plan commercial use, check with the upstream
providers rather than relying on this record.** For academic and research use,
which is what these files are built for, no such question arises.

This is a statement of position, not legal advice.

---

## Before you use these files

Quoted **verbatim** from the repository `README.md`, because these are the
statements the product's honesty rests on and paraphrasing them softens them.

> **27.2% of the record is reconstruction, not observation.** 227 of 312 months
> are GRACE-observed; the other 85 have no mass constraint (daily: 2,586 of 9,497
> days). `grace_observed` is `0` there. Mask on it for observation-only use, and
> expect `sigma_gap` to dominate the error budget in those months.

> **Fine structure is inferred.** 19 independent mascons support 9,538 cells, so
> agreement with GRACE at mascon scale is imposed by mass conservation and is
> **not** evidence of skill. The evidence is `lomo_cv_xgboost.csv` and
> `well_validation_summary.csv`.

> **Daily variation is not observed at all.** GRACE is monthly and the wells are
> quarterly, so nothing validates the sub-monthly shape directly; the two routes
> agree to `daily_method_spread`, and both re-aggregate to the monthly field
> exactly (closure ≤ 0.0005 mm for `twsa_flux`, ≤ 0.0055 mm for `twsa_state`).
> `sigma_within` is likewise a **lower bound**, not a calibrated error.

> Mass conservation forces every mascon's area-weighted aggregate onto the
> observed value (residual < 1e-9 mm), so **agreement with GRACE at mascon scale
> is imposed by construction and is not evidence of skill**.

> Wells are quarterly, so they validate spatial pattern and seasonal amplitude,
> never daily structure.

Two notes on the quotations, not part of them: the evaluation filename carries the
**selected** model, so `lomo_cv_xgboost.csv` is `lomo_cv_<model>.csv` in general;
and the 2,586 reconstructed days are exactly the days belonging to the 85
unobserved months (the remaining 6,911 days sit inside observed months)

---

## Explore it without downloading

<https://grace-grb-ml.projects.earthengine.app/view/twsa-explorer> — an Earth Engine app showing monthly and seasonal means, per-pixel time series with their ±1σ band, which months GRACE actually measured, and the significance-masked trend. No account needed.

---

## Grid, projection and time

| | |
|---|---|
| CRS | **EPSG:4326** (geographic, WGS 84) |
| Resolution | **0.1°** (~11 km), the ERA5-Land native grid |
| Shape | **101 lat × 179 lon = 18,079 cells**, of which **9,538 are in-basin** |
| Cell centres | lat 21.5 → 31.5 °N, lon 73.4 → 91.2 °E |
| Cell edges | lat 21.45 → 31.55 °N, lon 73.35 → 91.25 °E |
| Units | **mm** of equivalent water height (mm w.e.) |
| Baseline | **JPL RL06 mascon convention, 2004.0–2010.0** |
| Outside the basin | `NaN` |
| Monthly time | 312 steps, 2000-01 → 2025-12 |
| Daily time | 9,497 steps, 2000-01-01 → **2025-12-31** |
| Time encoding | `days since 1970-01-01`, standard calendar |

The daily record runs to **2025-12-31**, the last day of the monthly record, and
every one of the 312 months carries its full complement of days (verified against
the calendar, leap years included).

The 0.1° grid is snapped to ERA5-Land's native lattice (cell edges on the ×.05
line), which is why the edges are at 21.45/31.55 rather than 21.5/31.5.

---

## Files

Products are loose netCDFs so they can be downloaded individually; the small
files are bundled because Zenodo caps a record at **100 files** regardless of
size.

**Loose — the products.**

| file | dims (time × lat × lon) | variables | size |
|---|---|---|---|
| `twsa_0p1deg_monthly_with_uncertainty.nc` — **the file to use** | 312 × 101 × 179 | `twsa`, `sigma_total`, `sigma_grace`, `sigma_transfer`, `sigma_within`, `sigma_gap`, `grace_observed` | ~31 MB |
| `twsa_0p1deg_daily.nc` | 9497 × 101 × 179 | `twsa_flux`, `twsa_state`, `daily_method_spread`, `grace_observed` | ~799 MB |
| `twsa_0p1deg_monthly_<model>.nc` | 312 × 101 × 179 | `twsa`, `grace_observed` | ~10 MB |

**Bundled — everything else.** Zipping here is a file-count device, not a
compression one: the netCDFs are already internally compressed (`zlib` level 4)
and the COGs are DEFLATE, so the archives are close to the size of their
contents.

| archive | contents | files inside |
|---|---|---|
| `cogs_monthly.zip` | the monthly product as cloud-optimised GeoTIFFs, two bands (`twsa`, `sigma_total`) | 312 rasters + `metadata.csv` + `upload.sh` |
| `cogs_daily.zip` | the daily product as COGs, three bands (`twsa_flux`, `twsa_state`, `daily_method_spread`) | 9,497 rasters + `metadata.csv` + `upload.sh` |
| `trend_field.zip` | the per-pixel trend in both formats: `twsa_trend_significance.nc` and a 9-band COG (`sen_slope`, `ols_slope`, `p_value`, `z_score`, `kendall_tau`, `variance_factor`, `significant`, `significant_fdr`, `tested`) | 4 |
| `evaluation_tables.zip` | every validation table and the tuning record — see *Evaluation files* below. The fitted model (`model_<model>.joblib`) and the output-tree README ship with the code archive, not here | 23 |
| `figures.zip` | per-pixel maps, diagnostic figures, and the manuscript figures | 22 |
| `inputs_raw_gee.zip` | the raw Earth Engine downloads the ERA5-Land and GRACE cubes are built from; GLDAS tiles excluded | 4,680 |
| `inputs_static_covariates.zip` | the nine static covariate rasters | 53 |
| `inputs_basin_shapefile.zip` | the Ganga basin boundary | 8 |
| `inputs_cgwb_wells.zip` | the CGWB well dataset used for independent validation | 27 |
| `intermediates_cubes.zip` | the derived GRACE/ERA5/static netCDF cubes the model reads; the GLDAS cube is excluded | 4 |
| `intermediates_basin_series.zip` | basin-mean predictor series (legacy basin-scale path); GLDAS-derived files excluded | 228 |

The single-model monthly file is the intermediate the pipeline writes before the
uncertainty ensemble; the uncertainty file supersedes it and is what should be
cited or redistributed. `<model>` is whichever candidate won cross-validation in
the run that produced this release — it is **not** fixed to XGBoost.

**The two `inputs_*` archives that are third-party data** —
`inputs_raw_gee.zip` and `inputs_cgwb_wells.zip` — are redistributed under the
terms of their own providers, not under this record's CC-BY-4.0. See *Third-party
inputs* below before reusing them; the products and evaluation tables carry no such
restriction.

Sizes above are for the reference run; check the record's own file listing for
this release's exact bytes.

### Variables — monthly product

| variable | units | meaning |
|---|---|---|
| `twsa` | mm | Downscaled TWSA, ensemble mean, mass-conserved to GRACE |
| `sigma_total` | mm | Combined 1σ uncertainty, the quadrature sum of the terms below |
| `sigma_grace` | mm | GRACE mascon measurement uncertainty (observational) |
| `sigma_transfer` | mm | Leave-one-mascon-out transfer RMSE, by mascon and season |
| `sigma_within` | mm | Ensemble spread after mass conservation (within-mascon) |
| `sigma_gap` | mm | Extra uncertainty where GRACE did not observe |
| `grace_observed` | flag, per time step | `0` where GRACE made no observation that month |

### Variables — daily product

| variable | units | meaning |
|---|---|---|
| `twsa_flux` | mm | **Primary.** Within-month shape from the running sum of P − ET − Qs − Qsb |
| `twsa_state` | mm | Independent route: ERA5-Land soil water layers 1–4 plus snow water equivalent |
| `daily_method_spread` | mm | Half the absolute difference between the two routes |
| `grace_observed` | flag, per time step | `0` where GRACE did not observe the month this day belongs to |

Both daily routes are **de-meaned per pixel per month**, so the monthly means are
untouched and either route re-aggregates to the monthly product exactly. The two
therefore differ only in within-month shape, which is what
`daily_method_spread` measures — and it is the only uncertainty available on the
sub-monthly shape, because no daily observation exists to compare against.

---

## The uncertainty decomposition

Five `sigma_*` variables ship: **four components and their quadrature sum**,
kept separate because they mean different things and only some can be measured
against an observation.

| term | measurable? | measured how |
|---|---|---|
| `sigma_grace` | yes — observational | the mascon product's own uncertainty band |
| `sigma_transfer` | yes — held out | leave-one-mascon-out RMSE by mascon and calendar month |
| `sigma_gap` | yes — synthetic blackouts | contiguous 11-month blocks of *observed* months hidden, refitted, and scored against the withheld truth |
| `sigma_within` | **lower bound only** | spread across the four-member tree ensemble, after conservation |
| `sigma_total` | — | √(Σ of the above, squared) |

`sigma_within` deserves its caveat and the netCDF carries it as a variable
attribute. Because conservation forces every member to reproduce GRACE at mascon
scale, the remaining spread is disagreement about *within-mascon* distribution —
exactly the structure GRACE cannot see, so nothing can calibrate it. Members
sharing predictors and training data agree more than they are jointly correct.
Read it as "how much do defensible methods disagree", not "how wrong is this".

`sigma_transfer` is de-meaned per mascon-month, because conservation removes that
component exactly and leaving it in would double-count.

**A sixth term, `sigma_seed`, may be absent.** It is the within-family spread
across random seeds, and it is written **only** when the run used more than one
seed; a single-seed run omits the variable rather than shipping zeros, so its
absence means *not estimated*, not *estimated to be nil*. Check the file: if
`sigma_seed` is missing, the global attribute `sigma_seed_status` says so, and
`sigma_total` is then missing its within-family stochastic term.

```bash
ncdump -h twsa_0p1deg_monthly_with_uncertainty.nc | grep -E 'sigma_seed|ensemble_seeds'
```

---

## Evaluation files (`evaluation_tables.zip`)

These decide whether the product is trustworthy; they are outputs in their own
right, not scratch. Full descriptions are in the repository `README.md`.

| file | what it answers |
|---|---|
| `well_validation_summary.csv` | downscaled vs bilinear against 656 CGWB wells — **the only non-circular test** |
| `well_validation_by_scale.csv` | the same wells scored per-well, per-mascon and basin-wide; the scale changes the verdict |
| `well_validation_matches.csv` | the per-observation matches behind the summary |
| `lomo_cv_<model>.csv` | leave-one-mascon-out skill, one row per mascon: transfer across **space** |
| `holdouts_month_<model>.csv` | month holdouts, `random` / `blocked` / `forward`: transfer across **time** |
| `gap_recovery_rmse_by_depth.csv` | reconstruction error vs months into a blackout, by regime |
| `transfer_rmse_by_mascon_season.csv` | spatial-transfer error by mascon × calendar month (feeds `sigma_transfer`) |
| `temporal_holdout_<model>.csv` | forward-block error vs months beyond the training record |
| `feature_ablation_xgboost.csv` (always this name — the ablation is scored with xgboost defaults regardless of which model is selected) | whether the design matrix earns its size |
| `covariate_gate_<model>.csv` | each covariate's individual held-out skill (retired gate) |
| `summary_<model>.json` | pooled CV metrics, conservation residuals, per-month `grace_observed`, provenance |

The `random` month holdout is reported but is labelled optimistic in its own
output: monthly TWSA is autocorrelated, so a random split leaves each held-out
month's neighbours in training. Read the gap between `random` and `forward`, and
quote `forward`.

---

## Provenance

Every netCDF carries `provenance_*` global attributes — Python, numpy, pandas,
scikit-learn, xgboost, lightgbm, netCDF4, optuna and scipy versions, the random
seed, and the git commit — so a file states the environment it came from even
when separated from the repository. Gradient-boosting results are version
sensitive, which is why this is recorded rather than assumed.

```bash
ncdump -h twsa_0p1deg_monthly_with_uncertainty.nc | grep provenance_
```

`twsa_0p1deg_monthly_with_uncertainty.nc` carries `caveat`, `baseline` and `created_by` global
attributes. Read them before use.

---

## Reading the files

```python
import xarray as xr

ds = xr.open_dataset("twsa_0p1deg_monthly_with_uncertainty.nc")

# Observation-constrained months only. The other 85 are reconstructions:
# extrapolated per-mascon level and trend plus a modelled seasonal anomaly,
# with no mass constraint.
observed = ds.sel(time=ds.grace_observed == 1)

# A per-pixel series without its band overstates confidence, worst in the
# reconstructed months where sigma_gap dominates.
ts = ds[["twsa", "sigma_total"]].sel(lat=26.5, lon=82.0, method="nearest")
```

---

## Third-party inputs

The product is derived from the datasets below. Licences are as recorded in
`TODO.md` D1 and in `main/gridded_config.py`; the ones marked **not established**
were not recorded anywhere in the repository and must be settled before this
record is published.

This table and the *Required attribution* block below must also be pasted into
the Zenodo record description. A reader who downloads a single netCDF never opens
this file, and the Copernicus and C3S terms bind them anyway.

| input | used for | source identifier | licence |
|---|---|---|---|
| **ERA5-Land daily aggregates** | **every predictor of the gridded model** (`sms`, `et`, `ppt`, `runoff_surface`, and the derived `runoff_anom`), and the whole daily disaggregation | `ECMWF/ERA5_LAND/DAILY_AGGR` | CC-BY-4.0 (since 2 July 2025) |
| **GRACE/GRACE-FO JPL RL06 mascons, CRI-filtered** | the target and the mass constraint | `NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI` | NASA, public domain |
| **GLDAS 2.1 NOAH** | **not a predictor of the gridded product**, not used in the paper, and **not included in this record at all**. It fed only the superseded basin-scale path, whose inputs now sit behind `--with-legacy` | `NASA/GLDAS/V021/NOAH/G025/T3H` | NASA, public domain |
| **MERIT Hydro v1.0.1** | `upa_log` (log upstream area), `hnd` (height above nearest drainage) | `MERIT/Hydro/v1_0_1` | **CC-BY-NC-4.0 or ODbL-1.0** — see *Licence*: used as a covariate, not redistributed |
| **HWSD v2** | `awc`, `root_depth` | `projects/sat-io/open-datasets/FAO/HWSD_V2_SMU` | CC-BY-NC-SA-4.0 |
| **ESA C3S land cover (LCCS)** | `crop_irrigated`, `crop_rainfed` | `projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS` | educational and/or scientific purposes only, credit required |
| **GLOBGM steady-state water table depth** | `wtd` | `projects/sat-io/open-datasets/GLOBGM/STEADY-STATE/globgm-wtd-ss` | **GPL-3.0** — a copyleft licence, see the note below |
| **NASADEM** | `elevation`, `elevation_std` | `NASA/NASADEM_HGT/001` | NASA, public domain |
| **CGWB groundwater levels** (Kuruva et al. 2025) | independent validation only; not a model input | figshare `10.6084/m9.figshare.29293877` | **CC-BY-4.0** — compatible with this record; attribution required |
| **Ganga basin boundary shapefile** | the basin mask | provenance not recorded | CC-BY-4.0, asserted by the authors |

`wtd` is a **model output** (PCR-GLOBWB driven), not an observation, and carries
its own structural error. GLOBGM is validated against GRACE rather than
assimilating it, so it is independent in the sense that matters here.

**GLOBGM is GPL-3.0, which is a copyleft SOFTWARE licence applied to a data
product, and that is worth stating rather than glossing.** Two consequences
follow. First, `wtd.tif` is redistributed verbatim inside
`inputs_static_covariates.zip`, so that copy travels under GPL-3.0 and not under
this record's CC-BY-4.0 — the same carve-out already stated for the bundled
source code. Second, whether GPL-3.0 propagates from a raster used as 1 of 78
predictor columns into a fitted model's output is not settled; the GPL is written
for programs, and a downscaled TWSA field is neither a copy of `wtd` nor a work
that links against it, and no value of `wtd` can be recovered from the products.
We attribute it and flag the question rather than assert an answer. If your use is
commercial, take this up with the GLOBGM authors rather than relying on this
record.

**The CGWB well dataset (Kuruva et al. 2025) is CC-BY-4.0**, which is compatible
with this record: `inputs_cgwb_wells.zip` may be redistributed here provided the
attribution below is carried. It is used for independent validation only and is
never a model input.

GLDAS **2.1 NOAH** is used, not 2.2 CLSM: 2.2 assimilates GRACE, so its storage
states contain the signal this product predicts. Using them would make the model
partly predict GRACE from GRACE.

---

## Required attribution

**Copernicus — reproduce this verbatim, disclaimer included**, when using these
data. It covers both the ERA5-Land inputs and the C3S land-cover covariates:

> Contains modified Copernicus Climate Change Service information 2026. Neither
> the European Commission nor ECMWF is responsible for any use that may be made
> of the Copernicus information or data it contains.

**GRACE/GRACE-FO.** GRACE/GRACE-FO mascon data are available at
<https://grace.jpl.nasa.gov>, supported by the NASA MEaSUREs Program. Cite the
mascon product itself:

> Wiese, D. N., Yuan, D.-N., Boening, C., Landerer, F. W., & Watkins, M. M.
> (2023). *JPL GRACE and GRACE-FO Mascon Ocean, Ice, and Hydrology Equivalent
> Water Height, CRI Filtered.* PO.DAAC, CA, USA.
> <https://doi.org/10.5067/TEMSC-3JC634>

The release/version string is deliberately not quoted here: take it from the
PO.DAAC landing page. The copy actually used was the Earth Engine mirror
`NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI`.

**ESA C3S land cover** requires credit and permits educational and scientific use
only. **MERIT Hydro** and **HWSD v2** carry the attribution terms of whichever
licence arm is elected above.

**CGWB wells**, used for validation only:

> Kuruva et al. (2025). *Scientific Data* **12**, 1609.
> <https://doi.org/10.6084/m9.figshare.29293877>

---

## How to cite

> Kaushik, P. R., Majumdar, S., Lenczuk, A., Sharma, Y. K., Banerjee, S., &
> Thakur, P. K. (2026). *Downscaled GRACE terrestrial water storage
> anomalies for the Ganga (Ganges) River Basin at 0.1°: Monthly and daily fields
> with per-pixel uncertainty, 2000–2025* (Version 1.0.0) [Data set].
> Zenodo. <https://doi.org/10.5281/zenodo.21745159>

Cite the **version** DOI above: it pins v1.0.0, the exact artefact these files
belong to. **All versions:** <https://doi.org/10.5281/zenodo.21745158> — the concept DOI,
which always resolves to the most recent release.

**On the basin boundary.** It is released CC-BY-4.0 with the rest of the record.
Its provenance is not recorded anywhere in the repository, and the attribute
schema (`MRBID`, `RIVERBA`, `SUM_SUB`) resembles the GRDC Major River Basins
layer without establishing it. If it is later shown to derive from a third-party
product with its own terms, this row is the one to revisit — and the safe
substitute is already available, since every number in this release comes from
the rasterised `basin_frac` mask rather than from the polygon itself.

Code and data ship as one record, so there is no separate code DOI. Associated
paper: TODO-PAPER-DOI.

---

## Known limitations

Restated from the repository `METHODS.md` §10, because a data file separated from
its repository must still state them.

- **Fine structure is inferred, not observed.** 19 independent observations
  support 9,538 cells.
- **Mascon-scale agreement with GRACE is imposed**, so it is not evidence of skill.
- **Trend skill at unseen mascons is low** (R² ≈ 0.11) because the trend is
  groundwater abstraction and no reanalysis simulates pumping. The product takes
  level and trend from GRACE for that reason, which also means reconstruction
  quality across a GRACE blackout depends on the trend staying linear across it.
- **Human water use is represented only by a proxy.** Irrigated cropland fraction
  gives *where* irrigated agriculture occurs, not the withdrawal; it is annual,
  frozen after 2022, and cannot reach the trend the model does not fit.
  **Reservoir operation has no proxy at all.** Any statement linking irrigation to
  depletion is therefore spatial **association**, not attribution.
- **Sub-monthly variation is inferred from ERA5-Land**; nothing validates it
  directly.
- **27.2% of 2000–2025 is reconstruction** — 85 of 312 months carry no GRACE
  observation (the GRACE record runs 2002-04 to 2024-09). Every month carries a
  `grace_observed` flag, and the reconstructed ones an inflated `sigma_gap`.
- **C3S land cover ends 2022** and is held constant thereafter, recorded as
  `frozen_after`.
- **`sigma_seed` may be unmeasured**, in which case `sigma_total` is missing its
  within-family stochastic term. See *The uncertainty decomposition*.
- **The well comparison depends on the scale it is made at.** The downscaled
  field beats bilinear interpolation at basin scale (r 0.814 vs 0.742, RMSE 57.0
  vs 73.1 mm), is mixed per mascon (better correlation in 6 of 10), and **does
  not beat it at the point scale** (median well r 0.430 vs 0.484, RMSE 253.1 vs
  242.2 mm). Do not quote a single well-test number for this product; read the
  level you actually need from `well_validation_by_scale.csv`.
