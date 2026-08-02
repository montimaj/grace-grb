# Methods

**Explainable AI-Based Spatial Downscaling and Water Balance-Guided Temporal
Disaggregation of GRACE Terrestrial Water Storage over the Ganges River Basin**

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

A 0.1° product therefore has 9,538 in-basin cells against ~20 independent
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
edges fall on multiples of 0.5, so **36.7% of fine cells straddle a 0.5° grid-cell
boundary** and **7.1% straddle an actual 3° mascon boundary** (both measured on
the real grids). The mascon figure is the smaller of the two because a mascon is
six 0.5° cells across, but it is the one that matters for conservation:
assigning each straddling cell wholly to the mascon containing its centre would
misallocate that area, and the "conserved" product would not actually reproduce
GRACE.

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
**groundwater abstraction**, and no reanalysis simulates pumping.

The covariate set does carry a **proxy** — irrigated cropland fraction (§5.2) —
but it cannot close this gap, for reasons worth separating:

- it is **area, not volume**: the fraction of a pixel under irrigation does not
  give the withdrawal, which varies with cropping intensity, conveyance loss and
  the surface/groundwater mix;
- it is **annual and ends in 2022**, so it carries none of the strongly seasonal
  *rabi* pumping cycle;
- and structurally it cannot act where abstraction does, because the model fits
  only the **anomaly** while abstraction expresses itself in the **trend**.

So no predictor encodes abstraction in the component being fitted, which is
exactly why the decomposition below takes level and trend from the observation.

Fitting the raw series therefore spends model capacity on an unlearnable component
and contaminates the learnable one. So each mascon's TWSA is split

```
TWSA(m,t) = level(m) + trend(m)·t  +  anomaly(m,t)
            └── taken from GRACE ──┘   └─ fitted ─┘
```

and the model fits only the anomaly. Level and trend come from the observation.
The fit is one ordinary least-squares line per mascon, over that mascon's own
observed months, requiring at least 24 of them.

### The trend is not merely hard to learn — it is unrepresentable

The R² = 0.11 above says the predictors transfer the trend badly. The structural
reason is sharper: **the design matrix spans no monotonic direction in time.**
Its only time-varying terms are seasonal harmonics (`sin1`, `cos1`, `sin2`,
`cos2`), which are periodic by construction, together with antecedent means,
APIs and climatologies, all of which are bounded functions of bounded inputs.
There is no year, no month index, no elapsed-time column — deliberately, since
raw `lat`/`lon` were already removed for letting trees memorise the mascon
partition, and a raw time index invites the same failure along the time axis.

A model fitted to raw TWSA therefore has no mechanism to produce a secular
decline anywhere, in-sample or out. It would not merely extrapolate the trend
badly; it could not represent it at all.

**This is what makes the out-of-record months possible.** Of the 85
reconstructed months, 42 lie outside the GRACE record entirely — 27 before
2002-04 and 15 after 2024-09 — and for those the level and trend are a linear
extrapolation of a line fitted to observations, not a model output. Tree
ensembles predict leaf means and so cannot leave the range of their training
target under any configuration; the analytic background can. The decomposition
converts an impossible prediction into an explicit, stated assumption, and that
assumption is what `sigma_gap` (§9) prices.

Two honest limits on that argument. First, the benefit is about
*representability*, not extrapolation distance: tested on a synthetic series
with a time index supplied, a raw-fitted tree is no worse than the decomposed
model over 12–42 month horizons and only falls behind beyond about six years.
The case here does not rest on horizon length. Second, a straight line per
mascon over 2002–2024 assumes the depletion rate did not change within the
record; a change point would be absorbed imperfectly and its remainder would
land in the anomaly, where the model would try to fit it.

---

## 5. Features

Built for every predictor on the target grid:

| Family | Detail |
|---|---|
| Contemporaneous | four ERA5-Land predictors (`sms`, `et`, `ppt`, `runoff_surface`) plus the derived `runoff_anom` |
| Antecedent | backward rolling means at 1, 2, 3, 6, 12, 24 months, shifted one step |
| Context | Gaussian neighbourhood means at 1° and 2° |
| Climatology | per-pixel long-term mean and standard deviation. **Not built for `runoff_anom`**: it is defined by subtracting exactly that mean, so its climatological mean is identically zero and its standard deviation duplicates `runoff_surface`'s (`downscale_features.NO_CLIMATOLOGY`) |
| **API** | exponential, τ = 3, 12, 60, 120 months, for `ppt`, `et` and the water balance |
| Seasonal | sin/cos harmonics, orders 1 and 2 |
| Covariates | see §5.2 |

**78 features in total**, on 83,309 observed (cell, month) samples across 19
mascons. `downscale_ablation.py` measures whether that size earns itself; the
answer is that only one trim in the ladder — dropping `runoff_anom` — moves the
score by more than the repeat-to-repeat spread.

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

### 5.2 Covariates

Nine covariates on the 0.1° grid, **all of them used, with no selection step**.

Seven are static: water table depth (GLOBGM steady state), AWC and root depth
(HWSD v2), elevation and sub-grid relief (NASADEM), log upstream drainage area
and height above nearest drainage (MERIT Hydro). **Two are annual**, not static:
irrigated and rainfed cropland fraction (C3S land cover), one value per year
2000–2022, broadcast month-to-year and held constant after 2022.

**Categorical land cover is one-hot masked at its native 309 m before
aggregation.** Requesting the class field at 10 km makes Earth Engine average
class *codes* through its overview pyramids: measured over this basin that
returned values like 13, 15, 18, 23 — none in the LCCS legend, all looking like
ordinary data.

**A selection gate was tried and removed.** `downscale_covariate_gate.py` judged
each covariate on whether it improved leave-one-mascon-out skill, on the
reasoning that with ~19 independent mascons a covariate correlated with the
depletion trend can manufacture apparent skill. It admitted none of the nine. It
was removed because its criterion does not match what these fields do:

- The gate had two criteria, anomaly R² > 0.005 **or** trend R² > 0.02. Measured,
  trend R² is ~0 for the baseline *and* every covariate (5.3×10⁻⁵ to 2.9×10⁻⁴),
  so the second clause was **inert** and selection ran entirely on
  seasonal-anomaly skill.
- But the model fits only the **anomaly**; level and trend come from GRACE per
  mascon (§4, §6). Irrigated cropland acts on the depletion **trend** — the
  component the gate does not score and the model does not predict. Judged on
  seasonal anomalies it scored +0.0009 and was dropped, which is a statement
  about the criterion, not about irrigation.
- The leakage worry that motivated the gate is defused by that same
  decomposition: a covariate correlated with depletion cannot inject a false
  trend into the product, because the trend does not come from the model.

Including all nine costs **+0.0016** anomaly R² against the baseline — slightly
positive. The measurements are retained in
`Results/downscaling/covariate_gate_joint_sets_xgboost.csv`, and the gate script
is kept as a standalone diagnostic that no longer controls the feature set.

The concern that motivated the gate is real but is handled directly instead, by
excluding features that memorise training geometry. Feature attribution cannot
detect those: raw `lat` ranked 6th of 62 features by gain (measured on the build of the time; the matrix is now 78) while being pure
memorisation of the mascon partition, and removing it changed held-out skill by
0.006.

Coordinates (`lat`, `lon`) are excluded for that reason — during training the
target is constant within a mascon, so optimal coordinate splits fall exactly on
mascon boundaries, and the model prints them into the product as seams.

**What the gate measured, for the record.** On anomaly R² against baseline: best
single covariate `hnd` +0.0041; the three strongest together +0.0041 (so `wtd`
and `elevation_std` add −4×10⁻⁷ on top of `hnd` — they are mutually redundant);
all nine together +0.0016. The nine individual gains scatter with sd 0.0025 about
a mean of +0.0015, so none separates from the pack. Read correctly this says the
covariates neither help nor hurt *seasonal anomaly* skill at 0.1° with ~20
independent constraints — which is the expected result for fields that are
constant or slowly varying in time, and is not a reason to exclude them.

> The greedy `--cumulative` mode could not have reached a different answer: it
> admits a covariate only if it clears the bar **alone** first, so with none
> clearing, the search never starts. The joint sets above were therefore
> evaluated directly rather than by greedy search.

---

## 6. Prediction and mass conservation

The model is trained on the 0.5° GRACE grid (predictors vary cell to cell; the
target is the parent mascon's value, repeated), then applied to 0.1° predictors.

**The deployed model is fitted on all observed data.** Cross-validation exists to
*estimate* skill; its fold models are scored and discarded. After validation a
**fresh** model is fitted on every observed mascon-month and that is what
generates the product, so nothing held out in §8 reaches the shipped field. The
fitted estimator is saved to `Results/downscaling/model_<name>.joblib` alongside
its feature names — refitting is deterministic given the seed, but SHAP or
applying the model to a new period should not require reproducing a run.

**Five candidate models.** Four tree ensembles — RandomForest, XGBoost,
LightGBM, and XGBoost's random-forest mode (`XGBRFRegressor`, which bags like
RandomForest but with XGBoost's split finding and L2 regularisation, its
`learning_rate` fixed at 1.0 by definition and its column sampling per *node*) —
plus **one neural model, a multilayer perceptron**, so the comparison is not four
variations on a single idea. The MLP is a **comparison model only**: the
uncertainty ensemble (§9) remains the four tree members, because `sigma_within`
is a band shipped in the product and a structurally dissimilar member would
widen it without a reader being able to tell whether that reflected genuine
uncertainty about within-mascon structure or an architecture less suited to this
design. That is revisited if the comparison shows it performing comparably.

The MLP is fitted as a pipeline of median imputation → standardisation → network.
Both steps are necessary rather than stylistic: the design matrix carries 44,040
NaNs across 4,404 rows from the edges of the antecedent and API windows, which
the trees absorb natively and a network cannot, and feature standard deviations
span ~15 orders of magnitude (climatological means in mm beside cropland
fractions in [0,1]), so an unscaled network would be driven entirely by the
largest-variance columns. Reporting that an unprepared MLP lost would have
measured the preparation, not the model.

### The MLP's configuration is fixed, and the fixing is the evidence

The four tree models are tuned by Optuna on every run (§8). The MLP is **not**.
It is scored, ranked and released alongside them, but its configuration comes
from a sweep run once — `mlp_configuration_sweep.py`, released as
`Results/tuning/mlp_configuration_sweep.csv` and Fig. `Fig_mlp_configuration_sweep`.

The reason is not cost, though the cost is real: searching the MLP every run is
roughly sixteen times a single scored fit and would dominate the pipeline. The
reason is that a per-run search would be the *weaker* disclosure. A reader given
fifteen TPE trials learns the winner and nothing about the surface; a reader
given the sweep can see the whole response and judge whether the adopted setting
sits somewhere sensible. That matters here specifically because the neural model
loses, and the standing objection to any such result is that the network was
never given a fair configuration.

The sweep answers that objection on both axes that matter, under the same
mascon-grouped cross-validation the models are ranked by:

- **Width is not the constraint; depth is harmful.** Across an 8× range of
  hidden units (32 to 256) the score moves by 1.2% (79.04–79.95 mm), while
  adding a second hidden layer costs 8.9% (best two-layer 86.05 against best
  one-layer 79.04). The adopted single small layer is therefore the MLP's *best*
  case, not a handicap — the opposite of the usual under-sized-baseline
  criticism — and it is why the previous default of two layers, 128 and 64, was
  replaced: at 86.05 it was among the worst configurations tested.

  Width and depth are reported separately because a single "capacity" figure
  averages a flat axis with a steep one and is true of neither. (128,) is
  nominally the best single configuration at 79.04 against 79.13; we keep (64,)
  because 0.12% is inside the flat band and (64,) is the architecture the
  learning-rate axis was swept at, so the adopted configuration lies on both
  swept lines rather than in an unmeasured corner.
- **Learning rate is the axis that matters, and its optimum is interior.** A
  sweep whose best point sits at a grid boundary has found a stopping place, not
  an optimum, so the range was extended downward until the curve turned.
- **Regularisation is not load-bearing where it is adopted.** At the adopted
  learning rate of 3e-4, `alpha` moves the score by 0.05% across four orders of
  magnitude (79.10–79.15 mm). It only acquires any leverage at learning rates
  that are already worse — 2.2% at 1e-3 — which is a statement about those
  learning rates, not about regularisation.

None of this rescues the MLP: the best of 21 configurations, 79.04 mm, still
trails both tree models on identical folds (XGBoost 76.85, Random Forest 77.48).

That the MLP loses is the expected outcome for tabular data rather than a surprise
(Grinsztajn et al., 2022), and we report it as a confirmation of a known result
on a new dataset, not as a discovery. It is also why we do not describe the
trees as *superior* — see §8 on what the ranking can and cannot support.

Two limits worth stating. The sweep varies one factor at a time about a
reference point rather than searching jointly, so an interaction between
capacity and learning rate away from the swept lines would not be seen.

And the protocols are not identical: the trees get a per-run search, the MLP a
fixed configuration. That asymmetry favours the trees, which is to say it
favours our own conclusion, so it should be weighed sceptically rather than
waved through. What limits it is the size of the effect. Tuning moved the tree
models by under 1% on this problem, and the gap between the best MLP and the
best tree is larger than that, so an equal search is unlikely to reverse the
ordering. Unlikely is not impossible, and this is one reason §8 reports the
ranking with its resolvable margin instead of claiming superiority.

### Attribution, and what it can be asked

`downscale_shap.py` runs TreeSHAP on the fitted model and reports importance **by
feature family** as well as by column. Grouping is the honest unit: 78 columns
spread one physical signal across six antecedent windows and several APIs, so a
per-column ranking splits a single signal many ways and understates it.

Three limits, all of which bound what may be claimed from it:

- **It explains the anomaly only.** Level and trend are taken from GRACE per
  mascon and are never fitted (§4), so nothing in an attribution speaks to the
  depletion trend — which is the part a groundwater reader most wants explained.
- **Trees only.** There is no comparable attribution for the MLP, so if selection
  ever picked the network this step would not run.
- **Attributions are unreliable under strong collinearity**, and adjacent
  antecedent windows here correlate up to 0.98. Family grouping mitigates this;
  it does not remove it.

The reviewed manuscript's "antecedent SMS ≈ 80% of explanatory power" result is
**not** carried forward. It rested on GLDAS 2.2 CLSM soil moisture, which
assimilates GRACE, and on a degenerate `GWSA` column. Both are gone, and the
attribution is re-derived from scratch on the ERA5-Land matrix. Throughout, the
language is association, not causation.

### Why not recurrent or convolutional networks

The reviewed manuscript compared six models including LSTM, BiLSTM and
BiLSTM+Attention. They are **not** carried into the gridded comparison, and the
reason is structural rather than a preference.

**Recurrent networks have no sequence to consume here.** Temporal memory lives in
the *features* (§5): antecedent means at 1–24 months, exponential APIs at
τ = 3–120 months, seasonal harmonics. Each training row is an independent
(cell, month) vector, not a time series. An LSTM given these rows degenerates to
an expensive MLP with no recurrence to exploit — which is precisely why the
comparison includes the MLP instead, at a fraction of the cost and with the same
function class. Using a recurrent network *properly* would mean feeding per-pixel
sequences, a different model fitted against ~19 independent spatial units (§1).

**A convolutional network cannot be trained on this problem as posed.** CNNs are
the standard tool for super-resolution, but super-resolution is normally trained
on coarse→fine *pairs*, and **no fine-scale observation of TWSA exists**. The
alternative framing — patch of predictors → mascon value — fails for three
reasons:

1. The target is **constant within a mascon**, so a per-pixel output is fitted
   against something that does not vary at pixel scale. The network would learn
   to reproduce the mascon mean, which the trees already do with far fewer
   parameters.
2. ~19 independent spatial units cannot constrain a network with thousands of
   weights.
3. It would **reintroduce the leakage this study removed**. Raw `lat`/`lon` were
   excluded because trees memorised the mascon partition and printed 3° seams
   into the field (§5.2). A CNN over spatial patches has implicit access to
   position through patch content and could reproduce that failure more
   effectively and less visibly.

The exclusions are therefore about what this design can identify, not about which
architectures are fashionable. Any future spatiotemporal network here would need
a fine-scale target that GRACE does not supply.

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

### The reported skill is not nested, and is therefore slightly optimistic

Hyperparameters are chosen by `tune_gridded.py` on `GroupKFold(5)` over **the
same mascons** the leave-one-mascon-out below is computed on. So each LOMO fold's
"held-out" mascon was visible to the tuner through its own folds, and the
reported skill carries a hyperparameter-selection bias.

The same applies to the choice of model. The four tree candidates are tuned and
the MLP carries a fixed configuration (§6); all five are then ranked on that
grouped CV, and the winner builds the product — so the selection
sees the same folds the tuning did, and the winner's CV score is optimistic by an
unknown amount on top of the hyperparameter bias above. Two things limit the
damage. Each model is ranked on its *adopted* configuration (tuned only where
tuning beat the hand-set defaults on those folds), so a model is never credited
with a search that made it worse. And the LOMO skill reported below is computed
after selection, on folds selection never optimised, so the headline numbers are
not the selection score. What the ranking supports is the *ordering* of the
candidates, not the winner's absolute value.

An honest selection would need a third cross-validation layer outside the
tuning's, over ~19 spatial units — not enough independent groups to split three
ways.

The spread between candidates is in practice small, and rather than judge that
against a threshold chosen in advance we compare it to a **measured** quantity:
each model's own fold-to-fold standard deviation on the same folds. If the models
differ by less than a single model differs from itself across folds, the ranking
is not resolvable on these data, and the selector says so. That is the yardstick
R3-38's request for uncertainty ranges actually calls for — a fixed percentage
cannot supply it, because it does not know how noisy this dataset's folds are.

We therefore report the ordering together with that margin, and do not describe
the winning family as superior.

Removing it would require nesting the tuning inside each of the 19 LOMO folds —
19× the tuning cost, which is not affordable and is rarely done. The bias should
be small here: the search is 15 trials over a modest space, and the tuned
configuration is kept only when it beats the hand-set defaults on the same folds
(`"adopted"` in `gridded_best_params.json` records which was used). It is stated
rather than corrected.

Note this is a *new* caveat. Before the gridded tuner existed the model used
fixed hand-set hyperparameters, which have no selection bias at all — so if
tuning turns out not to help, reverting to the defaults also removes this.

### Per-pixel trend significance

A trend map drawn from a seasonal, strongly autocorrelated monthly series with no
significance layer would overstate what is resolved, so `trend_map()` no longer
paints slope alone. `stats_utils` supplies Mann-Kendall with the tie-corrected
variance, Sen's slope as the robust slope estimate, and a Hamed-Rao variance
inflation for autocorrelation — plain Mann-Kendall assumes independence and would
report far too many significant pixels here. The series is deseasonalised by the
calendar-month median first. Non-significant pixels are stippled rather than
silently painted, and the slope, *p*-value and significance mask are written to
`Results/downscaling/twsa_trend_significance.nc` rather than discarded.

Testing ~9,538 pixels at p < 0.05 would yield ~477 false positives by chance, so
a Benjamini-Hochberg correction is available and its use is recorded with the
output.

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

#### Pooled month holdouts, and the leakage ladder

The regime curves above are error *versus depth into a gap*, because that is what
`sigma_gap` needs. `downscale_holdouts.py` reports the same refits pooled, which
is the form a holdout table takes, and adds a third scheme for comparison:

| scheme | months held out | reading |
|---|---|---|
| `random` | drawn at random | **optimistic — do not quote as skill** |
| `blocked` | contiguous, interior | honest: interior gap-filling |
| `forward` | the last ones, chronological | honest: out-of-record extrapolation |

All three hold out whole **months**, never individual pixels: every pixel of a
month shares that month's GRACE constraint, so a pixel-level split would place
the target itself on both sides.

**The background is refitted inside every split.** The target is an anomaly
about a per-mascon level+trend line, and that line has to be fitted on
something. Fitting it once on the whole record before splitting — the obvious
way to write it, and the way this repository wrote it until the audit below —
lets each held-out month help build the background that is then subtracted from
it before scoring. Measured on the forward split, the fitted background moved by
27.5 mm on average and up to 76 mm across the held-out window, and per-mascon
trends by up to 4.7 mm yr⁻¹.

The scheme it damaged most was the one advertised as honest: `forward` was
extrapolating a trend fitted on the future it claimed to be extrapolating into.
Refitting the line on the training months alone and extrapolating it forward
moves that split from **RMSE 77.2 mm, R² 0.764 to RMSE 91.1 mm, R² 0.730** — the
leak was worth 13.9 mm, about 18 % of the reported error. The corrected figures
are the ones quoted throughout; `downscale_holdouts.py` still computes the old
ones into `*_leaky` columns, because the size of the gap is a result rather than
something to fix quietly.

The same defect sat in `gap_recovery_error`, which manufactures synthetic
blackouts and whose RMSE-by-depth curve *is* `sigma_gap` — so it was narrowing
the error bar on precisely the ~44 months that carry no observation. Refitting
per gap widens the forward-blackout curve at every depth, pooled **74.9 → 86.6
mm**:

| month into the blackout | 1 | 3 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|
| leak-free (mm) | 73.2 | 81.3 | 81.3 | 81.3 | 95.3 | 107.8 |
| all-months background (mm) | 68.8 | 77.3 | 56.4 | 49.5 | 89.6 | 103.5 |

The old curve did not merely sit lower, it sagged in the middle — depths 5 to 8
scored *better* than depth 1, which is not how extrapolation behaves and was the
tell. Those months are the furthest from a training edge and the most reliant on
the fitted line, so they gained the most from a line that had seen them.

Leave-one-mascon-out and the published fields are unaffected. There the level
and trend come from the enclosing mascon's own GRACE record, which is available
at prediction time — that is the design, not a leak. Only splits that withhold
*months* ask the line to reach somewhere it has no data, and only there does the
provenance of the line matter.

The `random` scheme is reported despite being optimistic. Monthly TWSA is smooth
and strongly autocorrelated, so a random split leaves each held-out month's
immediate neighbours in training and the model can interpolate between them
without using the predictors — on the basin-scale series the same design leaked
at R² = 0.96 even after grouping by interpolation segment. The number is
published because **the gap between `random` and `forward` measures that
optimism**, and a measurement is a stronger statement than a refusal to compute.
Quote `forward` for the extrapolated months and `blocked` for the interior gaps.

### Independent: CGWB wells

656 quality-controlled dug wells inside the basin, converted to storage via
`ΔGWS = −S_y·Δh` on the **same 2004.0–2010.0 baseline** as the mascons. Compared
against the product's groundwater component — TWSA minus the modelled
non-groundwater stores — and scored head-to-head against bilinear interpolation
of GRACE.

This is the only test using an observation the downscaling never saw. Wells are
quarterly, so they validate spatial pattern and seasonal amplitude — never daily
structure. Dug wells only: bore and tube wells may screen confined units whose
storativity is orders of magnitude from specific yield.

#### Which stores are subtracted

Two definitions are selectable with `--stores`:

| | fields | grid |
|---|---|---|
| **`era5`** (default) | soil layers 1–3 (0–100 cm) + snow water equivalent | **0.1°, native** |
| `gldas` | GLDAS 2.1 NOAH root-zone + snow + canopy | 0.25°, upsampled |

`era5` is the default because it is native to the product's own 0.1° grid:
subtracting a 0.25° field inside a comparison made at 0.1° puts a resolution
mismatch into the groundwater residual before the well test ever runs. Its cost
is that predictors and subtracted stores then share a model, which `gldas`
avoids — GLDAS 2.1 NOAH is independent of the ERA5-Land predictors and has no
groundwater store of its own, so it cannot double-count the quantity under test.
(2.2 CLSM is excluded entirely — see §2.)

Every number reported in this section uses the default, `era5`. The two store
definitions are alternatives, not an experiment: no cross-store comparison is
run or released.

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

`sigma_within` is the spread across a **four-member ensemble** — two boosting
implementations (XGBoost, LightGBM) and two bagging ones (RandomForest, XGBoost's
random-forest mode), so both tree families are represented rather than one being
outnumbered 2:1.

It deserves its caveat. Because conservation forces every member to reproduce
GRACE at mascon scale, the remaining spread is purely disagreement about
*within-mascon* distribution — the structure GRACE cannot see. Nothing can
calibrate it, since no observation of within-mascon structure exists. Read it as
"how much do defensible methods disagree", not "how wrong is this".

Widening the ensemble does not repair that. `XGBRFRegressor` bags much as
RandomForest does, differing in split finding and regularisation rather than in
principle, and all four share predictors and training data — so they remain
correlated and still agree more than they are jointly correct. The fourth member
makes the standard deviation less jumpy (four values rather than three) and
balances the families; it does not turn a lower bound into a calibrated error.

`sigma_transfer` is de-meaned per mascon-month, because conservation removes that
component exactly and leaving it in would double-count.

---

## 10. Known limitations

- **Fine structure is inferred, not observed.** ~20 independent observations
  support 9,538 cells.
- **Mascon-scale agreement with GRACE is imposed**, so it is not evidence of skill.
- **Trend skill at unseen mascons is low** (R² ≈ 0.11) because the trend is
  abstraction. The product takes level and trend from GRACE for that reason.
- **Human water use is represented only by a proxy.** Irrigated cropland fraction
  gives *where* irrigated agriculture occurs, not the withdrawal; it is annual,
  frozen after 2022, and cannot reach the trend the model does not fit (§4).
  **Reservoir operation has no proxy at all.** Any statement linking irrigation
  to depletion is therefore spatial **association**, not attribution.
- **Sub-monthly variation is inferred from ERA5-Land**; nothing validates it
  directly.
- **27.2% of 2000–2025 is reconstruction** — 85 of 312 months carry no GRACE
  observation (the record runs 2002-04 to 2024-09). Every month carries a
  `grace_observed` flag, and the reconstructed ones an inflated `sigma_gap`.
- **C3S land cover ends 2022** and is held constant thereafter, recorded as
  `frozen_after`. MODIS was not spliced in: a sensor change at 2022 would land
  inside the forward-extrapolation period, where the forward holdout could not
  separate covariate discontinuity from extrapolation error.
- **`GWSA` is renamed `runoff_anom`, and kept.** It was never groundwater — it
  is `runoff_surface` minus that field's own per-pixel long-term mean. The name
  was the problem, not the field: carrying a predictor called GWSA through a
  paper in a groundwater journal invites precisely the overstatement the
  reviewers flagged, and renaming costs nothing. The legacy basin-scale path
  still uses the uppercase `GWSA` column and is untouched.

  **It is kept because it is the only feature-set decision this data can
  resolve.** Measured on the 78-feature build (3 repeats, mascon-grouped CV),
  dropping it costs **+0.75 mm** against a repeat-to-repeat spread of 0.55 mm,
  and it loses in **every one of the three repeats**. Every other trim —
  `thin_ante` +0.38, `lean` +0.36, `minimal` +0.32 — lands *inside* that spread
  and is indistinguishable from doing nothing. Discarding the one measurable
  effect to solve a naming problem that a rename already solves would have been
  the wrong trade.

  The comparison is **paired**: every configuration sees the same fold partition
  in each repeat, so the differences above are differences on identical splits
  rather than a contest between two means with overlapping ranges. At three
  repeats the accompanying *p*-values have almost no power and the
  repeats-won column is the honest one; `downscale_ablation.py` prints both and
  says so.

  What it buys is a **centring** effect rather than new information. A tree
  splits on one feature at a time and cannot form `runoff − clim_mean` itself,
  and a per-pixel anomaly lets one global split threshold mean the same thing in
  every pixel — which is worth something to a model fitted on 19 mascons and
  applied to 9,538 cells.

  **Two of its columns are no longer built at all.** `runoff_anom_clim_mean` is
  identically zero by construction (the climatology block subtracts the same
  long-term mean that defines the field) and `runoff_anom_clim_std` is
  bit-for-bit `runoff_surface_clim_std`. The duplicate mattered beyond waste:
  two identical columns split SHAP credit, so the shared feature read as about
  half as important as it is. The design matrix is therefore **78 features**,
  not the 80 it carried when those columns were present.

---

## 11. One method, and what it replaced

**The method of this work is the gridded one described in §1–§10.** Its outputs —
the 0.1° monthly product and the daily disaggregation of it — are the
deliverables. Everything in the repository either feeds those two or validates
them.

An earlier basin-scale analysis exists and is superseded. It is retained behind
`./run_full_pipeline.sh --with-legacy` so the earlier manuscript's figures remain
reproducible, and for no other purpose: nothing in the current method reads its
outputs.

| | Basin-scale (superseded) | Gridded (this work) |
|---|---|---|
| Spatial unit | one basin mean | 0.1° pixels (9,538 in basin) |
| Target | monthly GRACE **interpolated to daily** | observed **monthly** GRACE |
| Daily signal | learned by the model | physically disaggregated, not fitted |
| Spatial validation | none possible (one series) | leave-one-mascon-out |
| Temporal validation | random + temporal holdout | random / blocked / forward month holdouts |
| Uncertainty | none per-pixel | five per-pixel terms (§9) |

The replacement is not a refinement of the same idea; it removes two structural
limits rather than reducing them. With a single spatially-integrated series there
is **nothing to hold out spatially**, so the earlier work could make no
verifiable statement about spatial pattern. And its daily target was an
interpolation of monthly GRACE, so its daily skill could not be validated from
the data at all — the temporal closure test it used compared a model against the
observations that generated its own target.

The gridded framework answers both. Spatial transfer is measured directly by
leave-one-mascon-out over 19 independent mascons (§8), and the daily field is
disaggregated under an exact monthly constraint rather than fitted, so it makes
no daily claim that requires daily validation. In place of a skill number it
carries an honest uncertainty: five per-pixel terms including `sigma_gap` for
reconstructed months and `sigma_transfer` from the spatial holdout.

That is the substance of "a more robust framework": the same physical question,
asked in a form where spatial error can be quantified per pixel instead of
assumed away.

### What the earlier holdouts were, and their replacements

The random and temporal holdouts of the earlier work were **basin-mean**
analyses. They shared no code path with the downscaling and described a
different model. Their gridded equivalents are:

| earlier (basin-scale) | now (gridded) |
|---|---|
| random holdout | `downscale_holdouts.py --model xgboost`, scheme `random` — reported but labelled OPTIMISTIC |
| temporal holdout | schemes `blocked` (interior gap-filling) and `forward` (out-of-record extrapolation) |
| — | leave-one-mascon-out spatial CV, which had no basin-scale equivalent |
| temporal closure test | retired **as a test**: monthly closure is an arithmetic identity for the disaggregated field (§7), so it cannot fail. `Fig_temporal_closure.png` still documents that the identity holds, and its caption says so — the two must not be read as the same thing. The basin-scale version could fail because it scored a model free to disagree with GRACE; this one scores a field de-meaned within each month, which cannot |
| `monthly_seasonal_maps` | `Fig_monthly_climatology_0p1deg.png`, `Fig_seasonal_mean_0p1deg.png`, `Fig_trend_0p1deg.png` — per pixel rather than one value per basin |

The random split is retained and published because the gap between it and
`forward` **measures** the optimism that a smooth autocorrelated series
introduces, which is a stronger statement than declining to compute it. It must
not be quoted as the model's skill.

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
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. Why do tree-based models still
  outperform deep learning on tabular data? *36th Conference on Neural
  Information Processing Systems (NeurIPS 2022) Track on Datasets and
  Benchmarks*, 1–48 (2022).
