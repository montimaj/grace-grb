# `figures/` — manuscript figures

Figures 1 and 2 of the revised manuscript and the graphical abstract, and the
code that draws them. All three are rewrites rather than restyles: the submitted
versions showed a variable the paper does not use, a method the paper no longer
uses, and clip art.

This file documents the figures only. The scientific method is in
[METHODS.md](../METHODS.md), the code reference is in
[main/README.md](../main/README.md), and products, installation and how to run
the pipeline are in the [root README](../README.md). Nothing here restates them —
duplication across these files is how the documentation drifted before.

Terminology follows METHODS.md strictly: the spatial step is **downscaling**, the
daily step is **disaggregation**. The figures use the words that way and so does
this file.

**Drawn text on all three plates says "Ganges"**, matching the manuscript title;
prose and code here say "Ganga". See the note at the top of the
[root README](../README.md) for why, and for the one string in this directory
that must never be renamed — `rivname == 'Ganga'` in `make_fig1_study_area.py`,
which is a value in the Government of India river network, not a word.

---

## The boundary requirement

**India is drawn from a Survey of India consistent outline** —
`data/india-soi.geojson`, which carries its own `Source` attribute ("Survey of
India State Map, Datameet"). Its extents are **37.08 °N** and **97.42 °E**, so
Jammu and Kashmir including Aksai Chin, and Arunachal Pradesh, are shown as
Indian territory.

Natural Earth and most international basemaps draw the Line of Control instead.
They are **not acceptable** for this paper, which has an Indian institutional
co-author. The failure mode is silent: substituting one would redraw India's
northern and eastern borders and nothing downstream would complain, because
nothing downstream reads the boundary for anything but drawing.

`fetch_data.py` therefore checks the outline's northern and eastern extents and
exits non-zero on a file that fails. Two things that check does not do:

* It is a bounding-box test against 36.5 °N and 96.5 °E. It catches the common
  substitution, because an outline drawn to the Line of Control stops well short
  of both thresholds, but it cannot verify any other part of the line.
* If `geopandas` is not importable it prints a notice, skips the check and exits
  **0**. Outside the `grace-grb` environment the check is not a guarantee.

Nothing outside India is drawn as a political entity. The basin crosses into
Nepal, Bangladesh and China; those are labelled in text rather than outlined, so
the plate asserts no boundary it cannot source.

---

## The figures

### Figure 1 — study area (`make_fig1_study_area.py`)

Three panels: (a) the basin over HydroSHEDS terrain with the river network, the
named main channel and the **656** CGWB validation wells; (b) the basin's
position within India; (c) the **19** GRACE mascons drawn over the **9,538**-cell
0.1° target grid.

Panel (c) is the point of the figure. It states the resolution gap the whole
project works against — 19 coarse observations, 9,538 cells asked for — which is
also the reason fine structure in the product is inferred rather than observed.

Twenty-two mascons touch the basin at all; the figure draws the 19 with basin
fraction above 0.5, because those are the ones leave-one-mascon-out actually
holds out. Drawing all 22 would put a number on the plate that no table in the
paper supports.

**What it replaced.** A land-cover map of the basin with wells scattered on top.
Land cover was never a predictor in the reviewed study and is not one now, so the
figure spent its whole area on a variable the paper does not use while the
constraint that governs the work went unshown. The old palette (yellow markers on
orange land) also failed in greyscale and under common colour-vision
deficiencies.

### Figure 2 — method workflow (`make_fig2_workflow.py`)

Six lanes from inputs to evaluation, drawn with orthogonal connectors and one
pastel fill per category. It carries four things the prose has to be able to
point at: only the anomaly is fitted (level and trend come from GRACE); mass
conservation is imposed rather than learned, so mascon-scale agreement is not
evidence of skill; the daily field is disaggregated with nothing fitted at the
daily scale; and the model comparison is a real comparison of five candidates
under one tuning protocol.

**What it replaced.** A diagram of a superseded method. Every box in the
submitted version is now wrong — GLDAS predictors (dropped: GLDAS 2.2 CLSM
assimilates GRACE, so the model was partly predicting the target from the
target), "GWSA" as a predictor (it contains no groundwater and is now carried as
`runoff_anom`), daily-to-monthly aggregation, a leave-one-**well**-out spatial
holdout, and the recurrent networks that are not carried forward. It showed no
spatial downscaling, no mass conservation, no uncertainty decomposition and no
daily disaggregation. There was nothing in it to restyle.

The plate deliberately names feature *families* rather than window lengths or a
column count. The design matrix changed twice during the revision; the numbers
belong in METHODS.md and the released ablation table, where a reader can check
them against something.

### Figure 3 — temporal holdout design (`make_fig3_holdouts.py`)

The GRACE/GRACE-FO observation record and the three temporal validation designs
drawn against one axis. It exists because a reviewer asked which months each
scheme withholds, and in particular whether the blocked experiment covers the
GRACE/GRACE-FO mission gap. It cannot, and the figure shows why: a month must
carry an observation before it can be withheld and scored, the gap contains
none, and the blocked blocks therefore straddle it rather than cover it.

The months are not transcribed from the paper. The script re-runs the same
seeded selection as `main/downscale_holdouts.py` against the `grace_observed`
flag in the released product, so the figure cannot drift from the experiment it
describes. The outage count, the 27 months before the record and the 15 after
are all counted from that flag at draw time, which is what lets the caption
state that they and the in-record outages sum to the 85 reconstructed months.

### Figure 10 — trend field and regional structure (`make_fig10_trend_regions.py`)

Two panels: (a) the Theil–Sen trend masked to significance, with the four
quadrants that partition the basin drawn on it; (b) the distribution of
significant trends inside each. It replaced an earlier single-panel trend map
whose north-western mean rested on a region that had never been written down.

Non-significant pixels are **dotted, not blanked**, reusing `_stipple()` from
`main/generate_gridded_maps.py` so the dots match the other trend maps exactly.
Blanking them would say "no data" where the truth is "a trend was estimated
here but the evidence is weak", and it erases the basin outline wherever weak
pixels reach the edge.

Region masks come from `main/trend_regions.region_mask()` rather than being
re-derived here. That is not tidiness: an earlier version computed its own masks
and silently disagreed with the summary table by 40 pixels, because one used
half-open bounds and the other inclusive ones.

### Figure 5 — leave-one-mascon-out metrics (`make_fig5_lomo_metrics.py`)

RMSE, R², NSE and mean bias error for all 19 cross-validation folds, each patch
one GRACE mascon coloured by the score a model achieved there when that mascon
and its neighbours were withheld.

It exists because the Results section promised "the complete fold-wise spatial
distribution of RMSE, R², NSE and bias" and pointed at `mascon_metric_maps.png`,
which is the **well** comparison redrawn at mascon scale and covers only the 10
of 19 mascons holding enough CGWB wells to score. Both plates are kept — they
answer different questions — and the supplement now labels each for what it is.

Bias is not in `lomo_cv_<model>.csv`; its PBIAS column is empty, because a
percentage bias on a near-zero-mean anomaly is unstable. Mean bias error is
therefore recomputed per mascon from `lomo_oof_<model>.csv`, and the script
refuses to draw the figure unless recomputing RMSE from those same rows
reproduces the stored value. It currently matches to 0.0000 mm across all 19.

Each panel carries its own colour bar: two metrics are millimetres and two are
dimensionless, so the shared scale used by `panel_figure` would leave three of
the four unreadable. NSE is clipped at −1 with an arrow on the bar — two folds
sit near −5 and −9, and unclipped they flatten the other seventeen into one
colour.

### Figure 6 — seasonal cycle (`make_fig6_seasonal_cycle.py`)

The two plates the manuscript sets one above the other under a single caption,
stacked into the one file the journal takes: the basin-mean annual cycle from
`main/downscale_annual_cycle.py` above, the 0.1° seasonal means from
`main/generate_gridded_maps.py` below. Nothing is redrawn — the script pastes the
rendered PNGs, resampling the lower plate by 4% to the upper one's width so the
two share edges — which keeps Fig. 6 pixel for pixel what the pipeline writes.
It reads `Results/`, not downloads, and refuses to run if either plate is missing
or the two are at different DPI.

### Graphical abstract (`make_graphical_abstract.py`)

Built from the product rather than drawn around it. A claim and four held-out
numbers on the left; on the right a 2 × 2 of real data for one month — what GRACE
sees (19 mascons), what the product resolves it to (9,538 cells), the per-pixel
σ, and the bias against wells the model never saw. The two TWSA panels are on one
symmetric scale with a bar each, and all four panels share identical map limits;
the GRACE grid is wider than the ERA5-Land grid, so without that the basin is
drawn at two different sizes in the two panels the reader is asked to compare.

The month must be one GRACE actually observed, or the left panel is not an
observation. The script refuses a reconstructed month rather than drawing one:

```bash
python figures/make_graphical_abstract.py --month 2016-06   # default
```

**What it replaced.** A satellite, a brain, a cartoon India and an invented bar
chart. For a data paper that is the wrong artefact — the one useful thing a
graphical abstract can do here is show the actual transformation, in real data,
at both scales, on one colour scale.

#### Elsevier's format rules are a hard constraint

ScienceDirect renders the abstract into a **500 × 200 pixel** window and requires
at least 1328 × 531 px at 300 dpi in that same 2.5:1 ratio. The canvas is
**11 × 4.4 in at 300 dpi = 3300 × 1320 px** — deliberately *smaller* than the
12.5 × 5.0 in it was first drawn at, because at a fixed point size a smaller
canvas means larger type once the whole thing is scaled into that window.

Elsevier permits Times, Arial, Courier or Symbol. matplotlib's DejaVu Sans
default is none of them, and `font.family` alone is not enough: anything inside
`$...$` is drawn by **mathtext**, which has its own font set and also defaults to
DejaVu. One subscript was enough to embed `DejaVuSans-Oblique` in the submitted
PDF next to Arial. The `mathtext.*` rcParams at the top of the script are what
fix that. Verify with:

```bash
python -c "import re; d=open('figures/output/Graphical_Abstract.pdf','rb').read(); \
print(sorted(set(re.findall(rb'/BaseFont /([A-Za-z0-9+\-]+)', d))))"
# expect ArialMT and Arial-BoldMT, and nothing else
```

#### Nothing may overlap, and that is checked

Every element sits in an explicit rectangle in the `LAYOUT` block, so
`constrained_layout` is not available as a guard. `check_layout` is the
substitute: after writing the files it re-measures the rendered bounding box of
every text, axes and colourbar — tick labels and axis labels included, since
those are the parts that actually collide — and fails on any intersection or
anything off-canvas. It runs by default and sets the exit code; `--no-check`
skips it.

This matters because the coordinates are hand-tuned and drift the moment a label
changes length. The previous version had the shared colourbar sitting on top of
the body text and the headline running under a panel title, which is exactly what
the check now catches.

#### The 500 × 200 preview is not the deliverable

`make_graphical_abstract.py` also writes
`output/Graphical_Abstract_preview_500x200.png`, a Lanczos downsample to the size
ScienceDirect actually shows. The only way to know whether a font is large enough
is to look at it at the size the reader gets.

It carries no dpi tag, because at thumbnail size only the pixel count means
anything, and it is gitignored — the one exception to the rule that
`figures/output/` is tracked. **Submit `Graphical_Abstract.png` or the PDF, never
this file.** It needs Pillow; if Pillow is not importable the script prints a
notice, skips the preview and still writes both real outputs.

At thumbnail size the headline, the four numbers, their labels, the panel titles
and the footnote all hold up. Getting there meant cutting the plate back: the
per-fact method captions and the redundant mascons-to-cells sentence were dropped,
the footnote was reduced to one line, and nothing is now set below 10 pt. That is
the reason the content is four panels rather than six.

### Output

| File | Size | Notes |
|---|---|---|
| `output/Fig1_study_area.pdf` | 5.2 MB | 183 mm wide (double column) |
| `output/Fig1_study_area.png` | 3.7 MB | 600 dpi |
| `output/Fig2_workflow.pdf` | 34 KB | 240 mm wide |
| `output/Fig2_workflow.png` | 810 KB | 600 dpi |
| `output/Fig3_holdouts.pdf` | 54 KB | 185 mm wide |
| `output/Fig3_holdouts.png` | 322 KB | 600 dpi |
| `output/Fig10_trend_regions.pdf` | 180 KB | 188 mm wide |
| `output/Fig10_trend_regions.png` | 817 KB | 600 dpi |
| `output/Fig5_lomo_metrics.pdf` | 339 KB | 185 mm wide |
| `output/Fig5_lomo_metrics.png` | 657 KB | 600 dpi |
| `output/Fig6_seasonal_cycle.png` | 2.0 MB | 600 dpi, 11.1 × 9.7 in; PNG only, as it is stacked from two rendered PNGs |
| `output/Graphical_Abstract.pdf` | 315 KB | 11 × 4.4 in, Arial only |
| `output/Graphical_Abstract.png` | 503 KB | 3300 × 1320 px at 300 dpi — **this is the file to submit** |
| `output/Graphical_Abstract_preview_500x200.png` | 75 KB | gitignored; legibility proof, not for submission |

Fig 1's PDF is large because the terrain and hillshade are embedded as rasters;
only the text and vector overlays are vector. Fig 2 is drawn entirely in vectors,
which is the whole difference between 35 KB and 5.4 MB. The graphical abstract
sits between the two — four rasterised map panels, everything else vector. All
PDFs use Type 42 fonts so the text stays selectable and editable at the journal.

Fig 2 is sized at 240 mm rather than a single- or double-column width on purpose:
the type was set first (7 pt detail, 8 pt headings) and the plate made large
enough to hold it, because a reviewer meets it on screen. It will need scaling to
the journal's column width at typesetting.

---

## Regenerating

```bash
conda activate grace-grb            # see ../environment.yml
python figures/fetch_data.py        # once per clone: ~285 MB downloaded, ~845 MB on disk after extraction
python figures/make_fig1_study_area.py
python figures/make_fig2_workflow.py
python figures/make_graphical_abstract.py
python figures/make_fig3_holdouts.py        # Fig. 3
python figures/make_fig10_trend_regions.py  # Fig. 10
python figures/make_fig5_lomo_metrics.py    # Fig. 5
python figures/make_fig6_seasonal_cycle.py  # Fig. 6
```

Figures 3, 5, 6 and 10 read results rather than downloads, so they run only after
`main/run_full_pipeline.sh` has produced
`Results/downscaling/twsa_0p1deg_monthly_xgboost.nc`,
`twsa_trend_significance.nc` and `trend_by_region.csv` — and, for Fig. 6, the
two plates it stacks. None invents a number: `make_fig3_holdouts.py` re-runs the
seeded month selection from `downscale_holdouts.py` against the released
observation flag, and
`make_fig10_trend_regions.py` calls `trend_regions.summarise()` and
`region_mask()` directly, and `make_fig5_lomo_metrics.py` checks its recomputed
bias against the stored RMSE before drawing, so the figures, Supplementary Table
S9 and the manuscript cannot disagree.

The scripts resolve every path relative to their own location, so the working
directory does not matter.

`fetch_data.py` skips anything already downloaded and re-extracts nothing, so
re-running it is cheap. It fetches three things:

| Source | Why this one |
|---|---|
| Survey of India outline | See [the boundary requirement](#the-boundary-requirement). |
| HydroRIVERS v1.0 (Asia) | Transboundary. The Indian national network stops at the Bangladesh border, which left the Ganga trunk ending at 88.16 °E and the entire delta with no drainage. |
| HydroSHEDS 15″ void-filled DEM (Asia) | ~460 m. The 0.1° `elevation` covariate is already in the cubes, but at ~11 km per cell the Himalayan front rendered as visible blocks, which reads as a fault rather than as relief. |

**Figure 2 needs nothing but matplotlib.** It draws no data; it can be
regenerated from a clean clone with no downloads at all.

**Figure 1 additionally needs pipeline outputs**, which are gitignored and are
not what `fetch_data.py` restores:

| Input | Produced by |
|---|---|
| `Data/Gridded/cubes/grids_aux.nc`, `static_cube.nc` | `main/build_cube.py` |
| `Data/Gridded/wells/well_meta.csv` | `main/wells_ingest.py` |
| `Data/Ganga Basin Shapefile/Ganga_basin.shp` | tracked in git |

**The graphical abstract needs no basemaps at all** — `fetch_data.py` is
irrelevant to it — but it is the most data-dependent of the three, because every
number and every panel on it is read from the pipeline rather than typed. The
mascon, cell and well counts are counted from the arrays for that reason: a
hand-typed "9,538 cells" goes stale the first time the grid or the basin mask
changes.

| Input | Produced by |
|---|---|
| `Results/downscaling/twsa_0p1deg_monthly_with_uncertainty.nc` | `main/downscale_uncertainty.py` |
| `Results/downscaling/summary_xgboost.json`, `holdouts_month_xgboost.csv` | `main/downscale_model.py`, `main/downscale_holdouts.py` |
| `Results/downscaling/well_validation_by_scale.csv`, `well_metrics_per_well.csv` | `main/validate_wells_scales.py` |
| `Data/Gridded/cubes/grids_aux.nc` | `main/build_cube.py` |
| `Data/Ganga Basin Shapefile/Ganga_basin.shp` | tracked in git |

## The one input that cannot be downloaded

`data/rivers/Rivers.shp` — the Government of India river network from
data.gov.in. It is used for exactly one thing: it carries a `rivname` attribute,
which is the only way to identify the Ganga main channel *by name*. Tracing it
through HydroRIVERS does not work, and the script's docstring explains why —
walking upstream by contributing area follows the Yamuna and then the Chambal and
ends in Rajasthan, because "main stem" is a naming convention rather than a
network property.

data.gov.in serves the file behind a session, so it has to be fetched by hand
from <https://www.data.gov.in/resource/shapefile-rivers> and unzipped into
`figures/data/rivers/`.

`fetch_data.py` prints a notice when it is absent but does **not** fail. Note
that `make_fig1_study_area.py` then reads the shapefile unconditionally and will
raise `DataSourceError` on the missing file — its docstring's claim that the
figure still renders without the red main channel does not hold as the code
currently stands. Either fetch the file or guard the read.

---

## What is tracked and what is not

The scripts and the rendered figures are not
ignored: they are the deliverable, and a reviewer or a co-author should get them
from a clone without downloading a gigabyte first.

The one exception is `output/Graphical_Abstract_preview_500x200.png`, which is
gitignored deliberately. It is a thumbnail regenerated on every run, it carries
no information the full-size PNG does not, and committing it would put a file in
the tree that looks submittable and is not. Its rule sits *after* the
`!figures/output/*.png` re-inclusion, because last match wins.

`figures/data/` is gitignored. It is **845 MB** of third-party basemaps that are
better versioned by their publishers than by us, and `fetch_data.py` restores all
but one of them.

Both rules need care in `.gitignore`, because blanket `*.pdf` and `*.png` rules
appear earlier in the file and last match wins. The re-inclusions for
`figures/output/` are placed after them for that reason. If a figure stops
appearing in `git status`, check with:

```bash
git check-ignore -v figures/output/Fig1_study_area.pdf
```

Read that output carefully: `-v` prints the matching rule even when the match is
a **negation**, and it exits 0 either way. A printed line therefore does not mean
"ignored" — check whether the rule it names starts with `!`. For a plain
yes/no, use the exit code of `git check-ignore -q <path>` instead.

---

## Captions

The captions live in the manuscript and only there. What this file has to record
is narrower, and it is the thing a reader of the repository alone cannot see:
**Figure 1 is deliberately incomplete.** Four attributions were moved off the
plate to keep it readable, they are not recoverable from the image, and they are
not optional:

* HydroSHEDS, for the terrain and river network — HydroRIVERS v1.0 and the
  15-arcsecond void-filled DEM (Lehner & Grill, 2013);
* the Government of India river network (data.gov.in), for the reach named
  "Ganga" that the main channel follows;
* Kuruva et al. (2025), for the 656 CGWB monitoring wells;
* the **Survey of India** outline, which is the statement that the boundary
  shown is the official one — see [the boundary requirement](#the-boundary-requirement).

No other plate carries a third-party attribution: Figure 2 is drawn from nothing
external, and Figures 3, 5, 6 and 10 are drawn from the archived result files.

A second copy of the captions used to live at `output/FIGURE_CAPTIONS.md`. It was
removed because it drifted: it described Figure 2's colour key backwards — saying
filled boxes were the fitted steps, when the filled boxes are the constraints and
are precisely the steps where nothing is fitted — and that error reached the
submitted manuscript before it was caught. The requirement is worth recording;
a duplicate of the captions is not.
