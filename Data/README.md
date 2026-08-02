# `Data/` — what is here, where it came from, what rebuilds it

This directory holds the pipeline's inputs and its intermediates. It does not
hold the products: those are written to `../Results/downscaling/`.

The scientific method is in [`../METHODS.md`](../METHODS.md), the module
reference in [`../main/README.md`](../main/README.md), and the front door is
[`../README.md`](../README.md). This file describes the directory and nothing
else — the numbers and the method live in those three, and duplicating them here
is how they drifted apart before.

Two facts govern everything below.

**Almost all of it is regenerable.** A full run leaves 12 GB here. 120 MB of
that is in git. The rest is downloaded from Earth Engine and assembled by code,
and the `.gitignore` rules that exclude it are deliberate — see
*Redistribution*, because the reason is not only size.

**Two inputs are not regenerable by any script in this repository**: the basin
polygon and the CGWB well archive. `main/run_full_pipeline.sh` checks for both
before it runs anything and aborts if either is absent, because a missing
polygon changes the domain silently and a missing well archive removes the only
non-circular validation the project has.

---

## At a glance

Sizes are `du -h` on the working tree after a full run.

| path | size | in git? | produced by |
|---|---|---|---|
| `Ganga Basin Shapefile/` | 7.9 MB | tracked | third party; provenance not recorded |
| `Groundwater/gwl_india/` | 110 MB | tracked | Kuruva et al. (2025), downloaded by hand |
| `All_Data.csv` | 980 kB | **untracked** | `python build_all_data.py --write` |
| `TWS_GRACE_GEE.csv` | 12 kB | **untracked** | `python export_basin_grace.py` |
| `Outputs/` | 40 MB | gitignored | `python gee_download.py` |
| `Gridded/` | 12 GB | gitignored | four scripts, see below |
| `Literature/` | 41 MB | gitignored | reference PDFs; no code reads them |

The 110 MB well archive is 92% of everything git carries under `Data/`.

`All_Data.csv` and `TWS_GRACE_GEE.csv` are the awkward pair: neither matches a
`.gitignore` rule, and neither has ever been committed (`git log --all` on both
paths is empty). A clean clone therefore has no predictor table and no GRACE
target. That is intended — both were once shipped as opaque hand-made files,
both are now derived from code, and shipping a copy alongside the generator
invites the two to disagree. `run_full_pipeline.sh` writes `All_Data.csv` if it
is absent and only *verifies* it if it is present, so a run already in progress
is never overwritten under itself.

---

## Tracked, and not reproducible from code

### `Ganga Basin Shapefile/`

One polygon, WGS 84 geographic (`GEOGCS["GCS_WGS_1984", ...]`), read by
`gridded_config.BASIN_SHAPEFILE`. Every basin mask, area weight and mascon
weight in the project derives from it.

Its `.dbf` carries a single record with the fields `OBJECTI`, `MRBID`,
`CONTINE`, `SEA`, `OCEAN`, `SUM_SUB`, `Shp_Lng`, `Shap_Ar`, `RIVERBA`, holding
`MRBID = 2306`, `RIVERBA = GANGES`, `SEA = Bay of Bengal`, and
`SUM_SUB = 1.00656e+06`. That schema matches the GRDC *Major River Basins of the
World* layer, which **suggests** the source but does not establish it: no
provenance is recorded anywhere in the repository, and if the layer is instead
India-WRIS or CGWB it may carry GODL-India terms. Treat the licence as unsettled
(see the table at the end).

`Ganga_Basin.jpg` is 7.7 MB of the 7.9 MB and is a picture of the basin. No code
reads it.

### `Groundwater/gwl_india/`

A verbatim copy of the quality-controlled CGWB groundwater-level archive of
**Kuruva et al. (2025), *Scientific Data* 12, 1609**, figshare
[`10.6084/m9.figshare.29293877`](https://doi.org/10.6084/m9.figshare.29293877),
unzipped in place. It is validation data only and is never a model input.

`main/wells_ingest.py` reads three files from `.../Output/`:

| file | role |
|---|---|
| `CGWB_India_filtered_Dug_wells_GWLs_ref_sy_2000_2022.csv` | primary — dug wells in unconfined aquifers, so specific yield applies |
| `CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv` | all well types, for a sensitivity check |
| `Filtered_GWLs_2000_2024/CGWB_filtered_wells_2000_2024.csv` | the long-record subset |

The primary file is India-wide and wide-format: 2,580 wells, one column per
quarterly reading (`Jan-00`, `May-00`, `Aug-00`, `Nov-00`, … `Nov-22`), plus a
per-well `Reference_Sy`. Filtering to the basin and to wells with enough baseline
and total observations leaves **656**, which is what `Gridded/wells/` contains.
The readings are quarterly, which is why the wells can test spatial pattern and
seasonal amplitude and can say nothing at all about the daily product.

Two files in this tree are **not** in git:
`Hydrogeological_map/Hydrogeological_map.tif` (caught by the blanket `*.tif`
rule) and `Code/.ipynb_checkpoints/`. The raster's absence is harmless —
`wells_ingest.py` defines `HYDROGEO_MAP` and never uses it. `ReadMe.docx` *is*
tracked despite the blanket `*.docx` rule, because it was committed before that
rule existed and `.gitignore` only governs untracked paths.

Re-download from figshare (27 MB zip, v3) if the tree is missing; nothing here
regenerates it.

---

## Generated

### `All_Data.csv` — the legacy basin-scale predictor table

9,497 daily rows, 2000-01-01 to 2025-12-31, six columns: `Date`, `SMS`, `ET`,
`rainfall`, `runoff`, `GWSA`. Read via `gridded_config.read_predictor_table()`.

```bash
python build_all_data.py --write     # build
python build_all_data.py --verify    # check an existing copy against the GEE download
```

This table feeds the **legacy basin-scale** path, not the 0.1° downscaling. Two
of its column names are misnomers kept for continuity with the manuscript, and
both are documented at length in `main/build_all_data.py` and
[`../main/README.md`](../main/README.md): `SMS` is a shallow layer rather than
root-zone storage, and `GWSA` contains no groundwater. Read that docstring before
using either column.

### `TWS_GRACE_GEE.csv` — the GRACE target

227 monthly rows, 2002-04-01 to 2024-09-01, columns `Date`, `Year`, `Month`,
`TWS` (mm), `n_cells`. `n_cells` is 433 in every row: the count of 0.5° GRACE
cells with non-zero in-basin weight, constant because the mask is static.

```bash
python export_basin_grace.py --compare   # -> Data/TWS_GRACE_GEE.csv
```

Only observed months appear — the 85 unobserved months of the 312-month product
window are **absent rows**, not NaN rows. Derived from the same gridded mascon
cube the 0.1° pipeline uses, so both halves of the project rest on one GRACE
source.

> **Provenance.** An earlier `TWS_JPL.xlsx` was removed from the repository: no
> recorded provenance, malformed month names that silently dropped part of the
> record, centimetres labelled as millimetres, and only r = 0.94 against a
> properly area-weighted mean of the product it claimed to be. The full account
> is in [`../README.md`](../README.md#project-structure) and in the
> `main/export_basin_grace.py` docstring. No such file ships, so
> `export_basin_grace.py --compare` has nothing to compare against and is a
> no-op.

### `Outputs/` — basin-mean CSVs (gitignored: `Outputs/`)

```bash
python gee_download.py
```

| file | contents |
|---|---|
| `Daily_GEE_GLDAS_V021.csv` | 9,497 rows, 13 basin-mean daily columns; the input to `build_all_data.py` |
| `Monthly_GRACE.csv` | 227 rows, basin-mean `lwe_thickness` |
| `temp/GLDAS_V021/` | 9,497 per-day CSVs, one Earth Engine request each |
| `temp/GRACE/` | 227 per-month CSVs |

`temp/` is 38 MB of the 40 MB and exists so an interrupted download resumes
instead of restarting; the two top-level CSVs are just its concatenation. The
`V021` in the filename records that `gee_download.py` is pinned to GLDAS 2.1
NOAH. It is a filename, not a dependency — every column that reaches
`All_Data.csv` is ERA5-Land.

### `Gridded/` — 12 GB, gitignored (`Data/Gridded/`)

| subdirectory | size | contents | rebuilt by |
|---|---|---|---|
| `raw/` | 6.4 GB | 8,112 GeoTIFFs: 26 variables × 312 months, each on its **native** lattice | `python gee_gridded_download.py` |
| `static/` | 2.0 MB | 53 GeoTIFFs on the 0.1° grid: 7 static covariates + 2 annual ones × 23 years (2000–2022) | `python gee_static_download.py` |
| `cubes/` | 5.3 GB | the five netCDF cubes below | `python build_cube.py` |
| `wells/` | 3.0 MB | `well_meta.csv` (656 wells), `well_gws_anomaly.csv` (54,102 quarterly readings, mm) | `python wells_ingest.py` |

`raw/` splits into `era5/` (13 variables), `gldas/` (11) and `grace/` (2), one
subdirectory per variable and one `<var>_YYYY-MM.tif` per month. Nothing is
resampled at download time — Earth Engine reprojecting on our behalf is exactly
the failure mode `gridded_config.py` exists to prevent.

| cube | size | shape | notes |
|---|---|---|---|
| `era5_cube.nc` | 4.5 GB | 9,497 × 101 × 179 | daily, 0.1°, 2000-01-01 to 2025-12-31 |
| `gldas_cube.nc` | 730 MB | 9,497 × 42 × 74 | same days at 0.25°; **not** a predictor of the gridded product |
| `grace_cube.nc` | 164 kB | 312 × 22 × 39 | 0.5°, monthly, `tws` and `tws_uncertainty`, cm → mm applied |
| `static_cube.nc` | 960 kB | 101 × 179 (× 23 years) | the 9 covariates, `frozen_after` recorded on the annual ones |
| `grids_aux.nc` | 52 kB | — | cell areas, `basin_frac_*`, parent-cell indices, and the mascon partition (`n_mascons = 35` in the domain) |

`grids_aux.nc` is the piece to understand: the JPL 3° mascon partition is not
distributed as a label field, so `build_cube.py` recovers it by grouping 0.5°
cells whose time series are byte-identical. Everything that conserves mass reads
that grouping.

`gldas_cube.nc` is downloaded and assembled but is used only by the superseded
basin-scale path — no GLDAS field appears in `downscale_features.py`. GLDAS 2.1
NOAH is used rather than 2.2 CLSM because 2.2 assimilates GRACE, which would make
the model partly predict the target from the target.

---

## Redistribution

`Data/Gridded/` being gitignored is usually described as a size decision. It is
also a licensing one. `static/` holds regridded derivatives of MERIT Hydro, HWSD
v2 and ESA C3S land cover, and those three carry NonCommercial or research-only
terms. The released products can state that no MERIT, HWSD or C3S raster is
redistributed **because these files never leave this machine**. If you add
`Data/Gridded/` to a release archive, that statement stops being true.

The same applies to `raw/era5/`, which is 5.6 GB of ERA5-Land: the licence
permits it, but there is no reason to re-host what Copernicus already serves.

## Third-party sources and licences

Reproduced from `DATA_README.md` (the Zenodo data-record README), which carries
the fuller discussion of the NonCommercial conflict and what it does and does not
reach. Entries marked **not established** are recorded nowhere in the repository
and must be settled before anything is published.

| input | lands in | source identifier | licence |
|---|---|---|---|
| ERA5-Land daily aggregates | `Gridded/raw/era5/`, `era5_cube.nc`, `Outputs/`, `All_Data.csv` | `ECMWF/ERA5_LAND/DAILY_AGGR` | CC-BY-4.0 (since 2 July 2025) |
| GRACE/GRACE-FO JPL RL06 mascons, CRI-filtered | `Gridded/raw/grace/`, `grace_cube.nc`, `TWS_GRACE_GEE.csv` | `NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI` | NASA, public domain |
| GLDAS 2.1 NOAH | `Gridded/raw/gldas/`, `gldas_cube.nc`, `Outputs/` | `NASA/GLDAS/V021/NOAH/G025/T3H` | NASA, public domain |
| MERIT Hydro v1.0.1 | `Gridded/static/upa_log.tif`, `hnd.tif` | `MERIT/Hydro/v1_0_1` | **CC-BY-NC-4.0 or ODbL-1.0** (dual) |
| HWSD v2 | `Gridded/static/awc.tif`, `root_depth.tif` | `projects/sat-io/open-datasets/FAO/HWSD_V2_SMU` | **CC-BY-NC-SA-4.0** |
| ESA C3S land cover (LCCS) | `Gridded/static/crop_irrigated_*.tif`, `crop_rainfed_*.tif` | `projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS` | **educational and/or scientific use only**, credit required |
| GLOBGM steady-state water table depth | `Gridded/static/wtd.tif` | `projects/sat-io/open-datasets/GLOBGM/STEADY-STATE/globgm-wtd-ss` | **not established** |
| NASADEM | `Gridded/static/elevation.tif`, `elevation_std.tif` | `NASA/NASADEM_HGT/001` | NASA, public domain |
| CGWB groundwater levels (Kuruva et al. 2025) | `Groundwater/`, `Gridded/wells/` | figshare `10.6084/m9.figshare.29293877` | **not established** — the *Sci Data* article being CC-BY does not settle the dataset's licence, and figshare returned 403 when checked |
| Ganga basin boundary | `Ganga Basin Shapefile/` | provenance not recorded | CC-BY-4.0, asserted by the authors |

ERA5-Land and the C3S land cover both require the Copernicus attribution
statement to be reproduced verbatim, disclaimer included; it is in
`DATA_README.md`.

## Citing this data

Kaushik, P. R., Majumdar, S., Lenczuk, A., Sharma, Y. K., Banerjee, S., &
Thakur, P. K. (2026). _Downscaled GRACE terrestrial water storage anomalies
for the Ganga (Ganges) River Basin at 0.1°: Monthly and daily fields with
per-pixel uncertainty, 2000–2025_ [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.21745158 — concept DOI; cite the **version** DOI
in a paper.

Cite the paper alongside it: Kaushik et al. (2026), _Groundwater for Sustainable
Development_ (under review). Full entry in the [root README](../README.md).
