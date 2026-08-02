# `gee/` — the Earth Engine app

**Published: <https://grace-grb-ml.projects.earthengine.app/view/twsa-explorer>**

`twsa_explorer.js` is an Earth Engine Apps script. Paste it into the
[Code Editor](https://code.earthengine.google.com/), run it, then **Apps →
Publish**. Viewers do **not** need an Earth Engine account — Google's
documentation is explicit that a published App is public — so that is not a
reason to prefer or avoid this architecture.

---

## The assets it reads

| collection | asset id | images | bands |
|---|---|---:|---|
| monthly | `projects/grace-grb-ml/assets/twsa_0p1deg_monthly_twsa__sigma_total` | 312 | `b1` = `twsa`, `b2` = `sigma_total` |
| daily | `projects/grace-grb-ml/assets/twsa_0p1deg_daily_twsa_flux__twsa_state__daily_method_spread` | 9,497 | `b1` = `twsa_flux`, `b2` = `twsa_state`, `b3` = `daily_method_spread` |
| trend | `projects/grace-grb-ml/assets/twsa_0p1deg_trend` | 1 | `b1`…`b9` = `sen_slope`, `ols_slope`, `p_value`, `z_score`, `kendall_tau`, `variance_factor`, `significant`, `significant_fdr`, `tested` |
| boundary | `projects/grace-grb-ml/assets/grb-boundary` | — | `FeatureCollection`, not imagery — the basin outline drawn over every layer |

The trend is a one-image collection because that is what geeup uploads into; the
app reads it as `ee.Image(ee.ImageCollection(TREND).first())`. It is written by
`export_cogs.py --which trend`, which is a separate code path from the two
time-series exports because the field is static — `(lat, lon)`, no time axis.

All three are written by `main/export_cogs.py` and ingested by the `upload.sh`
it generates beside each COG directory.

### Band names are `b1, b2, …`, and that is not fixable through geeup

geeup 2.0.0 builds its ingestion manifest as
`{name, pyramidingPolicy, tilesets, properties}` — there is **no `bands` key**
(`geeup/batch_uploader.py`), so nothing in a metadata column or a CLI flag can
set band ids, and Earth Engine falls back on file order. This was confirmed
against a real ingest, not assumed.

The true order travels in each image's `bands` property (`"twsa,sigma_total"`),
and `twsa_explorer.js` renames on read. **Anything else consuming these
collections must do the same**, or it will silently shade TWSA by σ.

If band names ever matter enough to fix: submit `ee.data.startIngestion`
directly with a `bands` array of `{id, tilesetId, tilesetBandIndex}` alongside
the usual payload. That was built and verified — the ingest returned
`['twsa', 'sigma_total']` — but it needs a GCS staging bucket and its own
numeric-property coercion, which geeup does for free, so it was not adopted.

### Properties that survive, and are load-bearing

| property | type | use |
|---|---|---|
| `grace_observed` | int, `0` or `1` | `.filter(ee.Filter.eq('grace_observed', 1))` recovers the 227 observed months. **Verified: the int form works, the string form matches nothing.** |
| `bands` | string | the band-order mapping above |
| `units`, `baseline` | string | `mm`, `2004.0-2010.0` |

---

## What the app must keep doing

These are not styling preferences. Each exists because the app would
misrepresent the product without it, and each is easy to lose in a refactor.

1. **`grace_observed` is surfaced.** 85 of 312 months (27.2%) carry no GRACE
   observation. The month selector states which regime is displayed, and the
   per-pixel chart draws reconstructed months as a separate dashed series.
   Plotting them undifferentiated presents inference as measurement.
2. **Uncertainty is surfaced.** Every series carries its ±1σ envelope from
   `sigma_total`, and σ is available as its own map layer. A per-pixel series
   without it overstates confidence, worst in the reconstructed months where
   `sigma_gap` dominates.
3. **The daily collection is not charted.** Nothing observational validates the
   sub-monthly shape — GRACE is monthly, the wells quarterly — and it
   re-aggregates to the monthly field exactly by construction, which is
   arithmetic rather than evidence. A per-pixel chart over 9,497 images is also
   the pathological Earth Engine query: interactive timeouts on every click.
4. **The trend is masked to `significant_fdr`, and the unmasked slope is not
   offered as a one-click alternative.** Two caveats travel with it in the panel,
   both from the source file's own metadata: the p-value is not a calibrated
   error rate — on simulated AR(1) series of this length the
   serial-dependence-corrected test still rejected a true null **18.8%** of the
   time at a nominal 5% — and FDR treats 9,538 pixels as 9,538 tests when GRACE
   resolves 19 independent mascons, so the mask is optimistic even so. The scale
   is kept symmetric about zero although the data are not (−84 to +12 mm/yr), so
   that zero stays neutral rather than sitting two thirds along the ramp.

The panel also states, non-collapsibly, that fine structure is inferred (19
mascons → 9,538 cells) and that mascon-scale agreement with GRACE is imposed by
mass conservation rather than earned.

### Period selection

The map shows the **mean over a year range × a month set**, not a single indexed
image. An earlier version exposed a raw 0–311 slider, which told a user nothing:
"month 300" is not a date.

The month presets are copied from `generate_gridded_maps.SEASONS` — Winter (DJF),
Pre-monsoon (MAM), Monsoon (JJAS), Post-monsoon (ON) — so the app and the paper's
seasonal maps cannot disagree about what "monsoon" means. `Custom months…`
reveals twelve checkboxes for anything else. A single month of a single year is
still reachable by narrowing both controls, so nothing was lost.

`month` is set as a property once, up front, because a non-contiguous set like
DJF cannot be expressed with `ee.Filter.calendarRange` and needs
`ee.Filter.inList`.

### Opacity

Two sliders, one per layer, not one global control — the trend is meant to be
read *over* the field, and a single slider cannot fade one against the other.
Both default to 0.85 rather than 1 so the terrain basemap stays faintly visible:
a 0.1° field with no landmarks under it is hard to place on the ground.

**Whenever a mean is displayed, the panel reports how many of its contributing
months are reconstructions**, as a count and a percentage. This is constraint 1
applied to aggregates: a "monsoon mean" drawn mostly from unobserved months is a
different claim from one drawn from observed months, and the map itself cannot
show the difference.

---

## Outstanding

- [x] ~~Make all four assets public~~ — done, 1 Aug. The app and every snippet
      below now work for an anonymous visitor.

      `grb-boundary` belongs in this set and is easy to miss, because it is a
      FeatureCollection rather than imagery: were it left private, the raster
      layers would render for an anonymous visitor while the basin outline
      silently did not, which reads as a drawing bug rather than a permissions
      one. If that symptom ever appears, re-check the ACLs first:

          import ee; ee.Initialize(project='grace-grb-ml')
          for a in ('twsa_0p1deg_monthly_twsa__sigma_total',
                    'twsa_0p1deg_daily_twsa_flux__twsa_state__daily_method_spread',
                    'twsa_0p1deg_trend',
                    'grb-boundary'):
              print(a, ee.data.getAssetAcl('projects/grace-grb-ml/assets/' + a)
                        .get('all_users_can_read'))

- [x] ~~Publish and record the App URL~~ — done; recorded here, in the root
      `README.md` and in the Zenodo record description.
- [ ] The daily collection was still ingesting when this was written; confirm
      9,497 images before relying on the daily map layer.

---

## Local checks

There is no Earth Engine emulator, so this file cannot be run outside the Code
Editor and is not exercised by `run_full_pipeline.sh`. What *can* be checked
locally is that the assets it names exist and carry the structure it assumes:

```bash
python - <<'PY'
import sys; sys.path.insert(0, 'main')
import ee, gridded_config as cfg
cfg.ee_initialize()
c = ee.ImageCollection('projects/grace-grb-ml/assets/twsa_0p1deg_monthly_twsa__sigma_total')
img = ee.Image(c.first())
print('images  :', c.size().getInfo())
print('bands   :', [b['id'] for b in img.getInfo()['bands']])
print('observed:', c.filter(ee.Filter.eq('grace_observed', 1)).size().getInfo())
PY
```

Expected: 312 images, `['b1', 'b2']`, 227 observed.
