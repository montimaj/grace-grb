## v1.0.1 — reproducibility release for the published paper

The code and data behind the paper as published in *Groundwater for Sustainable Development*.

**No value in the product changed.** The 0.1° monthly and daily TWSA fields, the model and the four uncertainty components are bit-for-bit those of v1.0.0. If you use the data, there is nothing to re-download. Three things did change: numbers quoted in the paper became reproducible from committed code, the figures were brought into line with the published paper, and the deposit's own documentation was corrected against the archives — including a licence note that was wrong in a way that mattered.

**Paper:** Kaushik, P. R., Majumdar, S., Lenczuk, A., Sharma, Y. K., Banerjee, S., & Thakur, P. K. (2026). Explainable AI-Based Spatial Downscaling and Water Balance-Guided Temporal Disaggregation of GRACE Terrestrial Water Storage Anomalies over the Ganges River Basin. *Groundwater for Sustainable Development*, 101688. https://doi.org/10.1016/j.gsd.2026.101688

**Data and code, this version:** https://doi.org/10.5281/zenodo.22718970
**All versions (concept DOI):** https://doi.org/10.5281/zenodo.21745158

> **The paper cites this release, not v1.0.0.** The regional trend figures in Section 5.5 come from `main/trend_regions.py`, which exists only from v1.0.1 — a reader who fetched v1.0.0 would not find the script that produced them.

### Added

- **`main/trend_regions.py`** — the region definitions behind the trend numbers. The manuscript quoted a mean trend for "the Punjab–Haryana–western Uttar Pradesh plain" that no committed mask defined, so nobody holding the archived trend field could reproduce it. The basin is now split at 27°N and 79.5°E into four quadrants, stated once, and `main()` *asserts* the partition — each of the 9,538 tested cells falls in exactly one quadrant — before writing `Results/downscaling/trend_by_region.csv`. The text quotes that file, not the other way round, and `main/run_full_pipeline.sh` now runs the script.
- **`figures/make_fig3_holdouts.py`** (Fig. 3) — re-runs the seeded month selection from `downscale_holdouts.py` and draws all three holdout designs against the observation record. It answers a reviewer question directly: the blocked experiment does **not** cover the GRACE/GRACE-FO mission gap, and cannot — a month must carry an observation before it can be withheld and scored.
- **`figures/make_fig5_lomo_metrics.py`** (Fig. 5) — RMSE, R², NSE and mean bias error for all 19 leave-one-mascon-out folds, one colour bar per metric. Mean bias error is recomputed from the out-of-fold predictions, and the script refuses to draw unless recomputing RMSE the same way reproduces the stored value.
- **`figures/make_fig6_seasonal_cycle.py`** (Fig. 6) — stacks the annual-cycle and seasonal-mean images into the single file the journal takes. It pastes the rendered PNGs rather than redrawing them, so Fig. 6 stays exactly what the pipeline writes.
- **`figures/make_fig10_trend_regions.py`** (Fig. 10) — takes its masks from `trend_regions.region_mask()` rather than recomputing them, after an earlier version disagreed with the table by 40 pixels.

### Changed

- **Manuscript title** gained *Anomalies*: "… Disaggregation of GRACE Terrestrial Water Storage **Anomalies** over the Ganges River Basin". TWSA is the quantity the model produces, and the title now says so.
- **Paper citation** now points to the published article in every README, in `METHODS.md` and in `DATA_README.md`.
- **`CITATION.cff`** describes the record as a data set rather than software, carries v1.0.1 with its version DOI alongside the concept DOI, and lists the paper under `references`. The data set remains the primary citation.
- **The Earth Engine explorer** links the v1.0.1 record.
- **Fig. 10 replaces the former trend map** — the same field, with the quadrant boundaries drawn on it and a second panel showing the distribution of trends within each region.
- **No titles on the manuscript figures.** The plotting code no longer draws a title above Figs. 4, 6, 7, 8 and 9; the caption carries it, as it already did for Figs. 3, 5 and 10. Fig. 8's panels are now labelled (a)–(c) in bold, like the other multi-panel figures. Only the drawing changed; every plotted value is the same.
- **Graphical abstract** — the arrow between the claim and the map panels is gone; it overlapped the TWSA panel title.
- **Badges** at the top of `README.md` for the paper, the Zenodo record, both licences, the Python version and the Earth Engine app.
- **Zenodo preview** — the record carries `Graphical_Abstract_preview.png`, the graphical abstract reduced to 750 × 300 px and set as its preview image; the full-resolution original stays in `figures.zip`.
- `figures/README.md` states the Figure 1 attribution requirement directly and documents each new figure script.

### Removed

- **`.zenodo.json`** — it silently overrode `CITATION.cff`, which is now the only deposit metadata.
- **`figures/output/FIGURE_CAPTIONS.md`** — a second copy of the figure captions that drifted from the manuscript. It described Figure 2's colour key backwards, and that error reached the submitted paper. The requirement it existed to record now lives in `figures/README.md`; the duplicated caption text is gone.

### Fixed

Documentation errors found by auditing the deposit against the archives themselves rather than against the previous README.

- **The licence note denied a redistribution that is happening.** `DATA_README.md` said "no MERIT, HWSD or C3S raster is redistributed here". All three are: `inputs_static_covariates.zip` carries MERIT Hydro (`upa_log.tif`, `hnd.tif`), HWSD v2 (`awc.tif`, `root_depth.tif`) and the ESA C3S land cover (`crop_irrigated_*.tif`, `crop_rainfed_*.tif`), each under its provider's terms rather than the record's CC-BY-4.0. That was the sentence a NonCommercial-restricted user would have relied on. The true half is kept: none of their values can be recovered from the released products.
- **The third-party archives were misnamed in both directions.** The docs pointed at `inputs_raw_gee.zip` (ERA5-Land and GRACE only — CC-BY-4.0 and public domain) and `inputs_cgwb_wells.zip` (CC-BY-4.0, compatible). Naming the raw archive encumbered 5.97 GB with terms it does not carry, while the archive that does carry them went unnamed.
- **`Data/README.md` recorded the GLOBGM and CGWB licences as "not established"** and told the reader they must be settled before publication — shipping inside the source archive and contradicting what the record asserts. Both now agree.
- **Deposit manifest.** File counts for `evaluation_tables.zip` (23 → 24) and `figures.zip` (22 → 31) were stale for exactly the two archives this release rebuilds; `grace-grb-1.0.1.zip` and `trend_by_region.csv` had no manifest row; and a `covariate_gate_<model>.csv` row named a file that exists nowhere.
- **Stale pointers and pre-publication language** removed: a `TODO.md` reference to a file that ships in neither the deposit nor the source archive, and an instruction about settling licences "before this record is published", which does not belong on a permanent archive.
- `README.md` gained the five scripts this release adds, which its file tree had never listed; six stale figure sizes in `figures/README.md` were corrected against the files on disk.
- **`README.md`'s project tree** described `DATA_README.md` as the README of a separate data record, when code and data ship as one, and said `run_full_pipeline.sh` regenerates everything, when the `figures/` scripts run separately. `figures/README.md` opened by listing only Figures 1 and 2 and the graphical abstract; it now lists every figure it draws.

### Upgrading

Nothing to do if you use the data — the released fields are unchanged from v1.0.0. If you cite this work, move to the v1.0.1 version DOI above, and cite the paper by its published reference. If you rely on the licence statements for the archived inputs, **re-read `DATA_README.md`**: the terms on the static-covariate rasters are as they always were, but this release is the first to state them correctly.

**Full changelog:** https://github.com/montimaj/grace-grb/blob/main/CHANGELOG.md
**Compare:** https://github.com/montimaj/grace-grb/compare/v1.0.0...v1.0.1
