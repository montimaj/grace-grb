# Methods

**Spatial Downscaling and Temporal Disaggregation of GRACE Terrestrial Water Storage over the Ganges River Basin**

From ~3° monthly GRACE mascons to a **0.1° (~10 km) daily** product spanning
2000–2025.

## Terminology, used strictly

The two halves of this work are different operations and the paper distinguishes
them, because conflating them overstates what the daily field contains.

**Downscaling** infers information not present in the coarse field: a
relationship is fitted at one scale and applied at a finer one, so the output
carries content the input did not.

**Disaggregation** distributes a coarse-scale value across finer units under a
constraint. The pattern is supplied externally, nothing is fitted, and the coarse
aggregate is preserved exactly.

| Component | Operation |
|---|---|
| **Spatial, 0.1°** | *downscaling* (a predictor→TWSA relation is learned across mascons and applied at 0.1°) **followed by** *disaggregation* (mass conservation redistributes so every mascon mean is preserved) |
| **Temporal, daily** | *disaggregation only* — no relation is fitted, the monthly mean is preserved exactly, and the within-month shape comes from the ERA5-Land water balance |

So the daily field is **not** downscaled. There is no learned transfer function at
the daily scale and no information beyond what ERA5-Land already contains; the
monthly total is distributed across days in proportion to a physically derived
shape. Describing it as "daily downscaling" would claim skill the method does not
have.

---

## 1. What GRACE can and cannot constrain

This governs every later choice, so it comes first.

JPL RL06 mascons are **3° equal-area cells** served on a 0.5° grid. Measured over
the Ganga basin: **364 grid cells carry only ~20 distinct values**. The 0.5°
spacing is a representation, not a resolution.

A 0.1° product therefore has ~9,100 in-basin cells against ~20 independent
observations — a **~450× expansion of spatial degrees of freedom**. It follows
that:

- **GRACE supplies the mass constraint.** ERA5-Land supplies all fine-scale
  texture. Nothing at 10 km is observed by gravimetry.
- **Agreement with GRACE is not evidence of skill**, because §6 forces it by
  construction. The evidence is the spatial holdout (§8) and the independent well
  comparison (§9).

The mascon partition is recovered exactly rather than assumed: 0.5° cells whose
*entire time series* is byte-identical belong to one 3° mascon. That yields 35
mascons in the domain, 19 with substantial in-basin area.

---

## 2. Data

| Role | Source | Period |
|---|---|---|
| **Target** | `NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI`, 3° native on a 0.5° grid | 2002-04 → 2024-09 |
| **Predictors** | ERA5-Land (`ECMWF/ERA5_LAND/DAILY_AGGR`), 0.1° | 1950 → present |
| **Covariates** | GLOBGM, HWSD v2, NASADEM, MERIT Hydro, C3S land cover | static / annual |
| **Validation** | CGWB wells (Kuruva et al., *Sci Data* 12, 1609, 2025) | 2000–2022, quarterly |

### GLDAS 2.2 CLSM is excluded, and this matters

The original analysis drew soil moisture and ET from GLDAS 2.2 CLSM. That product
(`…/V022/CLSM/G025/**DA1D**`, "DA" = Data Assimilation) **assimilates GRACE
terrestrial water storage**. Its `TWS`/`GWS` bands *are* GRACE, and assimilation
updates the model's water-storage states, so its soil moisture carries the GRACE
increment and its ET is a flux computed from GRACE-updated states.

Using it meant **predicting the target from the target**. All predictors are now
ERA5-Land, which has no GRACE anywhere in its production chain, spans 1950–present
(enabling the 2000 start), and is native to the 0.1° product grid — so predictors
need no regridding at all.

ERA5-Land soil water is volumetric (m³/m³) and is converted to millimetres by
layer thickness (×70 / 210 / 720 / 1890 for layers 1–4).

### Units

The JPL mascon `lwe_thickness` is **centimetres**. The ×10 conversion to
millimetres happens exactly once, in `utils.read_grace_monthly()`.

---

## 3. Grids and regridding

Three native lattices, each downloaded on its own grid so Earth Engine never
resamples on our behalf:

| Grid | Resolution | Transform origin |
|---|---|---|
| ERA5-Land | 0.1° | (−180.05, 90.05) |
| GLDAS | 0.25° | (−180.0, 90.0) |
| GRACE | 0.5° | (−180.0, 90.0) |

**The lattices are not nested.** ERA5-Land cell *edges* fall on x.x5 while GRACE
edges fall on multiples of 0.5, so **36.7% of fine cells straddle a mascon
boundary**. Assigning each fine cell wholly to the mascon containing its centre
would misallocate that area, and the "conserved" product would not actually
reproduce GRACE.

All aggregation therefore uses an **exact area-overlap operator**: axis-aligned
lon/lat boxes make the 2-D overlap a product of two 1-D overlaps, with spherical
areas (`R²·Δlon·[sin φ_n − sin φ_s]`, not planar). Verified: column sums reproduce
exact cell areas to 8.5×10⁻¹⁴ relative error.

---

## 4. Target decomposition

Leave-one-mascon-out (§8) shows the predictor set transfers the **seasonal cycle**
across space well and the **trend** hardly at all — yet the trend carries ~53% of
TWSA variance in this basin.

That is expected, not a modelling failure: the Ganga's secular decline is
**groundwater abstraction**, and no reanalysis simulates pumping. No predictor
encodes it.

Fitting the raw series therefore spends model capacity on an unlearnable component
and contaminates the learnable one. So each mascon's TWSA is split

```
TWSA(m,t) = level(m) + trend(m)·t  +  anomaly(m,t)
            └── taken from GRACE ──┘   └─ fitted ─┘
```

and the model fits only the anomaly. Level and trend come from the observation.

---

## 5. Features

Built for every predictor on the target grid:

| Family | Detail |
|---|---|
| Contemporaneous | the five predictors (`sms`, `et`, `ppt`, `runoff_surface`, `gwsa`) |
| Antecedent | backward rolling means at 1, 2, 3, 6, 12, 24 months, shifted one step |
| Context | Gaussian neighbourhood means at 1° and 2° |
| Climatology | per-pixel long-term mean and standard deviation |
| **API** | exponential, τ = 3, 12, 60, 120 months, for `ppt`, `et` and the water balance |
| Seasonal | sin/cos harmonics, orders 1 and 2 |
| Covariates | see §5.2 |

### 5.1 Antecedent Precipitation Index

Boxcar antecedent means weight a window equally then cut off; storage memory
decays exponentially and never fully ends:

```
API_t = exp(−1/τ)·API_{t−1} + P_t
```

The 60- and 120-month terms are deliberately longer than any boxcar window,
because the model's central weakness is that it cannot represent an accumulation
over two decades — which is what the secular trend is. The **water balance**
(P − ET − Q) gets the same treatment; a damped integral of the flux imbalance is
the closest available storage-memory term. A plain cumulative sum would integrate
ERA5-Land bias without bound over 26 years; finite τ damps it.

The seed window scales with τ: seeding a 120-month filter from a single month left
a spin-up artefact of ~134 mm a decade later, sitting exactly across the
backward-extrapolation period.

### 5.2 Covariates, and how they are admitted

Nine static or annual covariates on the 0.1° grid: water table depth (GLOBGM
steady state), AWC and root depth (HWSD v2), elevation and sub-grid relief
(NASADEM), log upstream drainage area and height above nearest drainage (MERIT
Hydro), and irrigated / rainfed cropland fraction (C3S).

**Categorical land cover is one-hot masked at its native 309 m before
aggregation.** Requesting the class field at 10 km makes Earth Engine average
class *codes* through its overview pyramids: measured over this basin that
returned values like 13, 15, 18, 23 — none in the LCCS legend, all looking like
ordinary data.

Covariates are not simply added. Each is judged by
`downscale_covariate_gate.py` on whether it improves **held-out** skill, because
with ~19 independent mascons a covariate correlated with the depletion trend can
manufacture apparent skill. Feature attribution cannot detect this: raw `lat`
ranked 6th of 62 features by gain while being pure memorisation of the mascon
partition, and removing it changed held-out skill by 0.006.

Coordinates (`lat`, `lon`) are excluded for that reason — during training the
target is constant within a mascon, so optimal coordinate splits fall exactly on
mascon boundaries, and the model prints them into the product as seams.

---

## 6. Prediction and mass conservation

The model is trained on the 0.5° GRACE grid (predictors vary cell to cell; the
target is the parent mascon's value, repeated), then applied to 0.1° predictors.

Chunking the prediction is a memory measure only: `predict_stack` takes an
explicit `reference_months`, so the API, climatology and antecedent terms are
always computed over the **full record** regardless of chunk size. Rebuilding them
per chunk shifted the field by rms 48 mm (max 252 mm) against a field σ of 120 mm.

### Conservation

Each mascon's area-weighted aggregate is forced onto the observed GRACE value.
A correction of the form `c = B·x` is sought, where `B` are Gaussian bumps on a
lattice **much finer than the mascons**, making

```
A·x = r ⊙ row_sum,     A = W·B
```

underdetermined. Its **minimum-norm solution** is the least-energy — hence
smoothest — correction that still reproduces every mascon mean exactly.

This matters: with one bump per mascon the system is fully determined and adopts
whatever shape closes it, printing 3° seams (gradient 12% higher along mascon
edges). The over-complete basis reduced that to 4% while keeping closure exact to
~10⁻¹¹ mm.

A mascon must contribute ≥2000 km² of in-basin area to act as a constraint. One
mascon contributes a single 0.1° cell of 0.41 km²; forcing the full 3° mean onto
it moved 4,326 of 9,538 cells by >10 mm — and the conservation residual stayed at
~10⁻¹² mm either way, so the diagnostic could not see it.

---

## 7. Daily disaggregation (not downscaling)

```
twsa_daily(p,d) = twsa_monthly(p, m) + δ(p,d),    mean of δ over month m = 0
```

De-meaning within each month per pixel leaves monthly means untouched, so the
daily field re-aggregates to the monthly product — and through it to GRACE —
exactly. Conservation is structural, not iterative.

**No daily ML model is used, deliberately.** GRACE is monthly, so no daily target
exists; training one requires interpolating the monthly product to daily, and the
model then learns to reproduce that assumption. Its apparent "daily skill" would
measure fit to an interpolation.

Two independent routes, both genuinely daily:

| Variable | Derivation | Weakness |
|---|---|---|
| `twsa_flux` | running sum of P − ET − Q_s − Q_sb within month | integrates ERA5-Land bias (bounded by restarting monthly) |
| `twsa_state` | ERA5-Land soil water layers 1–4 + SWE | sees only water ERA5-Land models — no groundwater |

They share a driving dataset but not a derivation. `daily_method_spread` records
half their absolute difference. Since **no daily observation exists** (GRACE
monthly, wells quarterly), that disagreement is the only available uncertainty on
the sub-monthly shape.

---

## 8. Validation

### Spatial: leave-one-mascon-out

Each of the 19 in-basin mascons is held out in turn, **together with its spatial
neighbours** — JPL's CRI filtering correlates adjacent mascons, so a neighbour
left in training is a partial answer key. A fold trains on roughly 12–13 mascons.

This replaces a synthetic spatial holdout that replicated one basin-mean series
across fabricated locations with added noise. That test leaked by construction and
scored R² ≈ 0.99 for a quantity honestly measured at **0.05**.

### Temporal: three regimes, measured separately

GRACE runs 2002-04 → 2024-09 while the product spans 2000–2025, so months outside
that window are reconstructions — but not all reconstructions are alike:

| Regime | Period | Anchoring |
|---|---|---|
| `interior` | GRACE/GRACE-FO gap | bracketed on both sides; trend interpolated |
| `forward` | 2024-10 → 2025-12 | no closing anchor; trend extrapolated |
| `backward` | 2000-01 → 2002-03 | extrapolated backwards |

`gap_recovery_error()` manufactures blackouts by hiding contiguous blocks of
*observed* months and refitting, giving an error-vs-horizon curve **per regime**.
Reusing the interior curve for out-of-record months would understate them badly.

### Independent: CGWB wells

656 quality-controlled dug wells inside the basin, converted to storage via
`ΔGWS = −S_y·Δh` on the **same 2004.0–2010.0 baseline** as the mascons. Compared
against the product's groundwater component (TWSA minus modelled root-zone soil
moisture, snow and canopy), and scored head-to-head against bilinear interpolation
of GRACE.

This is the only test using an observation the downscaling never saw. Wells are
quarterly, so they validate spatial pattern and seasonal amplitude — never daily
structure. Dug wells only: bore and tube wells may screen confined units whose
storativity is orders of magnitude from specific yield.

---

## 9. Uncertainty

Four terms, reported separately as well as combined, because they mean different
things and only some are measurable against an observation:

| Term | Meaning | Measurable? |
|---|---|---|
| `sigma_grace` | mascon measurement error, from the product's own band | yes — observational |
| `sigma_transfer` | leave-one-mascon-out RMSE by mascon and season | yes — held-out |
| `sigma_gap` | reconstruction error by regime and depth into the gap | yes — synthetic blackouts |
| `sigma_within` | ensemble spread **after** conservation | **lower bound only** |

`sigma_within` deserves its caveat. Because conservation forces every member to
reproduce GRACE at mascon scale, the remaining spread is purely disagreement about
*within-mascon* distribution — the structure GRACE cannot see. Nothing can
calibrate it, since no observation of within-mascon structure exists. Read it as
"how much do defensible methods disagree", not "how wrong is this".

`sigma_transfer` is de-meaned per mascon-month, because conservation removes that
component exactly and leaving it in would double-count.

---

## 10. Known limitations

- **Fine structure is inferred, not observed.** ~20 independent observations
  support ~9,100 cells.
- **Mascon-scale agreement with GRACE is imposed**, so it is not evidence of skill.
- **Trend skill at unseen mascons is low** (R² ≈ 0.11) because the trend is
  abstraction. The product takes level and trend from GRACE for that reason.
- **Sub-monthly variation is inferred from ERA5-Land**; nothing validates it
  directly.
- **~40% of 2000–2025 is reconstruction.** Every month carries a `grace_observed`
  flag and an inflated `sigma_gap`.
- **C3S land cover ends 2022** and is held constant thereafter, recorded as
  `frozen_after`. MODIS was not spliced in: a sensor change at 2022 would land
  inside the forward-extrapolation period, where the forward holdout could not
  separate covariate discontinuity from extrapolation error.
- **`GWSA` is a misnomer.** It is de-meaned surface runoff and contains no
  groundwater. Retained under its original name for continuity; on the 0.1° grid
  the per-pixel offset makes it a genuine runoff anomaly rather than a duplicate
  of `runoff`.

---

## 11. Why there are two pipelines

The repository contains two distinct analyses, and conflating them causes
confusion.

| | Basin-scale | Gridded |
|---|---|---|
| Spatial unit | one basin mean | 0.1° pixels |
| Target | monthly GRACE **interpolated to daily** | observed **monthly** GRACE |
| Daily signal | learned by the model | physically disaggregated, not fitted |
| Models | 6, incl. recurrent networks | tree ensembles |
| Spatial validation | none possible (one series) | leave-one-mascon-out |

The basin-scale pipeline is the earlier manuscript. Its central limitation is
structural: with a single spatially-integrated series there is nothing to hold
out spatially, and its daily target is an interpolation, so daily skill cannot be
validated from the data. Both limitations are documented in that work.

It is retained for reproducibility, not as the basis for any daily or per-pixel
claim. Where the two disagree, the gridded result supersedes.

---

## References

- Kuruva, S.K., Suryawanshi, M.R., Shakya, A. et al. Quality controlled, reliable
  groundwater level data with corresponding specific yield over India.
  *Sci Data* **12**, 1609 (2025). doi:10.1038/s41597-025-05899-5
- Watkins, M.M. et al. Improved methods for observing Earth's time variable mass
  distribution with GRACE using spherical cap mascons. *JGR Solid Earth* (2015).
- Muñoz-Sabater, J. et al. ERA5-Land: a state-of-the-art global reanalysis dataset
  for land applications. *ESSD* **13**, 4349–4383 (2021).
- Verkaik, J. et al. GLOBGM v1.0: a parallel implementation of a 30 arcsec global
  groundwater model. *GMD* (2024).
- Yamazaki, D. et al. MERIT Hydro: a high-resolution global hydrography map.
  *WRR* **55**, 5053–5073 (2019).
