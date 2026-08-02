/**
 * GRACE-GRB Explorer — 0.1° downscaled TWSA over the Ganga basin, 2000–2025.
 *
 * Paste into the Earth Engine Code Editor and Apps → Publish. Viewers do NOT
 * need an Earth Engine account; Google's docs are explicit about that, so it is
 * not a reason to choose one architecture over another.
 *
 * WHY THE UI LOOKS LIKE THIS
 * --------------------------
 * Four constraints drove the design, and each exists because a plausible-looking
 * app would misrepresent the product without it.
 *
 *  1. `grace_observed` IS SURFACED. 85 of the 312 months (27.2%) carry no GRACE
 *     observation and are reconstructions. Plotting them undifferentiated beside
 *     observed months would present inference as measurement, so the series
 *     draws them as a separate dashed series, and the period selector reports how
 *     many months of the displayed mean are reconstructions.
 *  2. UNCERTAINTY IS SURFACED. Every series carries its ±1σ band from
 *     `sigma_total`. A per-pixel series without it overstates confidence, worst
 *     in exactly the reconstructed months where `sigma_gap` dominates.
 *  3. THE DAILY COLLECTION IS NOT CHARTED. Nothing validates the sub-monthly
 *     shape — GRACE is monthly, the wells quarterly — so a daily series would
 *     invite reading structure that no observation supports. It is also the
 *     query Earth Engine handles worst: one request per image over 9,497 images.
 *     Monthly and annual series are offered; the daily field is in the archive.
 *  4. THE TREND IS MASKED TO ITS SIGNIFICANCE LAYER, and the unmasked field is
 *     not offered as a one-click alternative. A per-pixel trend map without it
 *     would be a new integrity gap of exactly the kind the other three prevent.
 *     Both caveats travel with it: the p-value is not a calibrated error rate,
 *     and FDR over 9,538 pixels is optimistic when GRACE resolves 19 mascons.
 *
 * BAND NAMES
 * ----------
 * The assets carry `b1, b2, ...`, not variable names: geeup's ingestion manifest
 * has no band-id field, so Earth Engine numbers them in file order. The true
 * order is in each image's `bands` property. Everything below renames on read,
 * which is the only reason the rest of this file can be legible.
 */

// ---------------------------------------------------------------- config
var PROJECT = 'projects/grace-grb-ml/assets';
var MONTHLY = PROJECT + '/twsa_0p1deg_monthly_twsa__sigma_total';
var DAILY   = PROJECT + '/twsa_0p1deg_daily_twsa_flux__twsa_state__daily_method_spread';

// Band order as written by export_cogs.py, recorded in each image's `bands`
// property and lost on ingest. Renaming here rather than at each use site.
var TREND   = PROJECT + '/twsa_0p1deg_trend';
var BASIN   = PROJECT + '/grb-boundary';

var MONTHLY_BANDS = ['twsa', 'sigma_total'];
var DAILY_BANDS   = ['twsa_flux', 'twsa_state', 'daily_method_spread'];
var TREND_BANDS   = ['sen_slope', 'ols_slope', 'p_value', 'z_score',
                     'kendall_tau', 'variance_factor', 'significant',
                     'significant_fdr', 'tested'];

var INK = '#1a1a1a', MUTED = '#7a7a7a';
// Diverging, two hues about a neutral midpoint: TWSA is signed and zero means
// "at the 2004–2010 baseline", which must read as neutral rather than as low.
//
// RED IS NEGATIVE, BLUE IS POSITIVE -- water lost is red, water gained is blue.
// This is matplotlib's RdBu, which `generate_gridded_maps.DIVERGING` uses for
// every signed field in the paper; the app had it reversed, which would have put
// the same basin in opposite colours in the figures and in the viewer.
var DIVERGING = ['#b2182b', '#ef8a62', '#fddbc7', '#f7f7f7',
                 '#d1e5f0', '#67a9cf', '#2166ac'];
// Single hue, light to dark, for a magnitude that cannot be negative -- and
// deliberately a different hue family from DIVERGING so an uncertainty map is
// never mistaken for an anomaly map at a glance. Sampled from matplotlib's
// `Purples` at seven stops, which is `generate_gridded_maps.SEQUENTIAL` and
// what `validate_wells_scales.py` and the graphical abstract also use; the app
// previously carried a pink PuRd ramp, so the same sigma field appeared in two
// different colours in the paper and in the viewer.
var SIGMA_PALETTE = ['#fcfbfd', '#e8e7f2', '#c6c7e1', '#9e9ac8',
                     '#796eb2', '#5b3495', '#3f007d'];

// Display limits live apart from the palettes, because a palette is a design
// decision and a limit is a reading decision -- and the limits are the reader's
// to change (see "Colour scale" below). DEFAULT_SCALE is the reset target and
// is never mutated.
//
// These are the PUBLICATION limits — the fixed, symmetric ones the paper's
// figures use — and they are what the app OPENS on, so a visitor comparing the
// viewer against a printed figure is on the same scale without touching a
// control. "Fit to data" departs from them; "Publication scale" comes back.
// Zero stays on the neutral colour either way, because divergingFor recentres
// the palette rather than relying on the limits being symmetric.
var DEFAULT_SCALE = {
  twsa:        {min: -300, max: 300},
  sigma_total: {min: 0,    max: 150},
  trend:       {min: -40,  max: 40}
};

// Round outward to 1, 2 or 5 times a power of ten, so a fitted bar end reads as
// a number a person would have chosen. (Math.log10 is absent from the Code
// Editor's engine; Math.log/LN10 is the portable form.)
function niceUp(v) {
  if (!(v > 0)) return 1;
  var p = Math.pow(10, Math.floor(Math.log(v) / Math.LN10));
  var n = v / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}

// How a measured range becomes a colour scale. `lo`/`hi` are the 2nd and 98th
// PERCENTILES, not the extremes: a min/max fit is one outlier pixel away from a
// scale nobody wants. TWSA's true extremes are about -812..+498, but almost the
// whole basin sits within a third of that, so fitting them washes the map to
// near-neutral everywhere and hides the pattern the map exists to show.
//
// The fit is ASYMMETRIC, snapped outward to a tidy step of about a twentieth of
// the range. This basin is heavily one-sided -- the 2-98% band of the mean is
// roughly -449..+85 mm -- so a symmetric fit would spend five sixths of the
// ramp on values that do not occur. Zero is kept on the neutral colour by
// recentring the PALETTE instead (see divergingFor), which is the part that
// actually controls where neutral lands.
function fitScale(key, lo, hi) {
  if (key === 'sigma_total') lo = 0;
  var step = niceUp((hi - lo) / 20) || 1;
  return {min: key === 'sigma_total' ? 0 : tidy(Math.floor(lo / step) * step),
          max: tidy(Math.ceil(hi / step) * step)};
}

// Kill the float noise that floor/ceil * step leaves behind, so a bar end reads
// "1.5" rather than "1.5000000000000002".
function tidy(v) {
  return Math.abs(v) < 1e-9 ? 0 : parseFloat(v.toPrecision(12));
}

// ---- recentring a diverging palette on zero ------------------------------
// Earth Engine maps min..max linearly onto the palette, so the neutral middle
// colour sits at (min+max)/2 -- which for an asymmetric range is NOT zero. On
// [-450, +100] the neutral white would land on -175 mm: substantial depletion
// painted as "at baseline", and zero painted blue. That is the single most
// misleading thing this app could do with a signed field.
//
// The fix is to hand Earth Engine the SLICE of the diverging ramp that the
// range actually spans. Treat the full ramp as covering [-m, +m] where
// m = max(|min|, |max|); the visible window is then the sub-interval
// [(min+m)/2m, (max+m)/2m] of it, resampled. Zero lands on the ramp's own
// midpoint by construction, whatever the limits are.
function hexToRgb(h) {
  return [parseInt(h.substring(1, 3), 16),
          parseInt(h.substring(3, 5), 16),
          parseInt(h.substring(5, 7), 16)];
}
function rgbToHex(c) {
  var s = '';
  for (var i = 0; i < 3; i++) {
    var v = Math.max(0, Math.min(255, Math.round(c[i]))).toString(16);
    s += v.length < 2 ? '0' + v : v;
  }
  return '#' + s;
}
function rampAt(stops, t) {
  t = Math.max(0, Math.min(1, t));
  var x = t * (stops.length - 1);
  var i = Math.min(Math.floor(x), stops.length - 2);
  var f = x - i;
  var a = hexToRgb(stops[i]), b = hexToRgb(stops[i + 1]);
  return rgbToHex([a[0] + (b[0] - a[0]) * f,
                   a[1] + (b[1] - a[1]) * f,
                   a[2] + (b[2] - a[2]) * f]);
}
function divergingFor(lo, hi) {
  var m = Math.max(Math.abs(lo), Math.abs(hi));
  if (!(m > 0)) return DIVERGING;
  var t0 = (lo + m) / (2 * m), t1 = (hi + m) / (2 * m);
  var out = [], N = 25;      // enough stops that the slice still reads smooth
  for (var i = 0; i < N; i++) {
    out.push(rampAt(DIVERGING, t0 + (t1 - t0) * i / (N - 1)));
  }
  return out;
}

// Built on demand rather than stored, so a layer and its colour bar cannot be
// drawn from different numbers.
function fieldVis() {
  var s = state.scale[state.band];
  return {min: s.min, max: s.max,
          palette: state.band === 'twsa' ? divergingFor(s.min, s.max)
                                         : SIGMA_PALETTE};
}
function trendVis() {
  var s = state.scale.trend;
  return {min: s.min, max: s.max, palette: divergingFor(s.min, s.max)};
}

var monthly = ee.ImageCollection(MONTHLY)
    .map(function (img) { return img.rename(MONTHLY_BANDS); })
    .sort('system:time_start');

// ---------------------------------------------------------------- map
var map = ui.Map();
map.setOptions('SATELLITE');
map.setControlVisibility({layerList: false, drawingToolsControl: false});
map.style().set('cursor', 'crosshair');
map.setCenter(82.0, 26.0, 6);

var first = ee.Image(monthly.first());
var basin = first.select('twsa').mask();   // the product is NaN outside the basin

// ---------------------------------------------------------------- panels
function heading(text, size) {
  return ui.Label(text, {fontWeight: 'bold', fontSize: size || '15px',
                         color: INK, margin: '4px 0 2px 0'});
}
function note(text) {
  return ui.Label(text, {fontSize: '11px', color: MUTED, margin: '2px 0 6px 0'});
}

var left = ui.Panel({style: {width: '330px', padding: '8px'}});

left.add(heading('GRACE-GRB Explorer', '18px'));
left.add(note('Downscaled GRACE/GRACE-FO terrestrial water storage anomaly, ' +
              '0.1°, Ganga River Basin, 2000–2025. Values are mm of equivalent ' +
              'water height relative to the JPL 2004.0–2010.0 baseline.'));

// --- the caveat that must not be collapsible -----------------------------
left.add(heading('Read this first', '13px'));
left.add(ui.Label(
    'Fine structure is INFERRED, not observed. GRACE resolves 19 independent ' +
    'mascons over this basin; this product has 9,538 cells. Agreement with ' +
    'GRACE at mascon scale is imposed by mass conservation and is not evidence ' +
    'of skill.',
    {fontSize: '11px', color: INK, margin: '2px 0 4px 0'}));
left.add(ui.Label(
    '85 of 312 months (27.2%) have no GRACE observation and are ' +
    'reconstructions. They are drawn separately in the chart below.',
    {fontSize: '11px', color: INK, margin: '0 0 8px 0'}));

// --- period selection -----------------------------------------------------
// A raw 0..311 index told a user nothing: "month 300" is not a date. The
// controls are a YEAR RANGE and a MONTH SET, and the map shows the mean over
// their intersection. One month of one year is still reachable -- set the range
// to a single year and pick one month -- so nothing is lost by dropping the
// index slider.
//
// The season presets are `generate_gridded_maps.SEASONS`, copied rather than
// reinvented so the app and the paper's seasonal maps cannot disagree about
// what "monsoon" means.
var SEASONS = {
  'All months':          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  'Winter (DJF)':        [12, 1, 2],
  'Pre-monsoon (MAM)':   [3, 4, 5],
  'Monsoon (JJAS)':      [6, 7, 8, 9],
  'Post-monsoon (ON)':   [10, 11]
};
var MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// The season the app opens on. Named once because four things have to agree
// about it -- the state, the dropdown, the twelve custom checkboxes and the
// period label -- and a default that disagrees with its own controls is worse
// than the wrong default.
var DEFAULT_SEASON = 'Post-monsoon (ON)';

// `month` is added once here so a non-contiguous set like DJF can be selected
// with inList; ee.Filter.calendarRange only expresses contiguous ranges.
var withMonth = monthly.map(function (img) {
  return img.set('month', ee.Date(img.get('system:time_start')).get('month'));
});

var state = {years: [2000, 2025], months: SEASONS[DEFAULT_SEASON],
             band: 'twsa', season: DEFAULT_SEASON,
             opacity: 0.85, trendOpacity: 0.85};

// Deep copy, so "Publication scale" always has a pristine DEFAULT_SCALE to
// restore from and no edit can corrupt the reset target.
//
// autofit starts FALSE: the app opens on the publication limits, so what a
// visitor sees first is the same scale as the printed figures. Fitting is opt
// in, via the button. The trade is that the default scale clips -- the 2-98%
// band of the mean reaches -449 mm against a -300 bar -- which is why the
// measured range and its clip note are shown under every bar regardless.
state.scale = {};
state.autofit = {};
for (var scaleKey in DEFAULT_SCALE) {
  state.scale[scaleKey] = {min: DEFAULT_SCALE[scaleKey].min,
                           max: DEFAULT_SCALE[scaleKey].max};
  state.autofit[scaleKey] = false;
}

left.add(heading('Period', '13px'));

var yearSlider = ui.Slider({min: 2000, max: 2025, value: 2000, step: 1,
                            style: {width: '140px'}});
var yearSlider2 = ui.Slider({min: 2000, max: 2025, value: 2025, step: 1,
                             style: {width: '140px'}});
left.add(ui.Label('Years', {fontSize: '11px', color: MUTED, margin: '2px 0 0 0'}));
left.add(ui.Panel([yearSlider, yearSlider2],
                  ui.Panel.Layout.flow('horizontal')));

var seasonSelect = ui.Select({
  items: Object.keys(SEASONS).concat(['Custom months…']),
  value: DEFAULT_SEASON,
  style: {width: '300px'}
});
left.add(ui.Label('Months', {fontSize: '11px', color: MUTED, margin: '4px 0 0 0'}));
left.add(seasonSelect);

// The twelve checkboxes stay hidden until "Custom months…" is chosen: they are
// the escape hatch, not the primary control.
var monthChecks = [];
var customRow1 = ui.Panel([], ui.Panel.Layout.flow('horizontal'));
var customRow2 = ui.Panel([], ui.Panel.Layout.flow('horizontal'));
MONTH_ABBR.forEach(function (name, i) {
  // Seeded from the default season, not all-true. Only seasonSelect.onChange
  // syncs these, and it does not fire at startup -- so a hard-coded `true` left
  // twelve ticked boxes behind a map showing two months, and the first click on
  // any of them would have silently widened the selection to all twelve.
  var cb = ui.Checkbox(name, SEASONS[DEFAULT_SEASON].indexOf(i + 1) !== -1,
                       null, false, {margin: '1px 4px 1px 0'});
  cb.onChange(function () {
    var picked = [];
    monthChecks.forEach(function (c, j) { if (c.getValue()) picked.push(j + 1); });
    state.months = picked;
    refresh();
  });
  monthChecks.push(cb);
  (i < 6 ? customRow1 : customRow2).add(cb);
});
var customPanel = ui.Panel([customRow1, customRow2]);
customPanel.style().set('shown', false);
left.add(customPanel);

var periodLabel = ui.Label('', {fontSize: '12px', color: INK, margin: '4px 0 0 0'});
var regimeLabel = ui.Label('', {fontSize: '11px', margin: '0 0 6px 0'});
left.add(periodLabel);
left.add(regimeLabel);

function refresh() {
  var y0 = Math.min(yearSlider.getValue(), yearSlider2.getValue());
  var y1 = Math.max(yearSlider.getValue(), yearSlider2.getValue());
  state.years = [y0, y1];

  if (state.months.length === 0) {
    periodLabel.setValue('No months selected');
    regimeLabel.setValue('');
    return;
  }

  var sel = withMonth
      .filter(ee.Filter.calendarRange(y0, y1, 'year'))
      .filter(ee.Filter.inList('month', state.months));

  // The mean over the selection, not a single image. Masked to the basin so the
  // NaN surround does not paint.
  var vis = fieldVis();
  var shown = sel.select(state.band).mean().updateMask(basin);
  map.layers().set(0, ui.Map.Layer(
      shown, vis,
      state.band === 'twsa' ? 'Mean TWSA' : 'Mean σ_total',
      true, state.opacity));

  // The bar is redrawn from the SAME `vis` the layer just used. Without this the
  // layer repainted on new limits while its tick labels kept the old ones, which
  // is the one failure a legend must never have.
  syncLegend();

  // The range is of THIS selection's mean, not of the whole record: averaging
  // 300 months flattens the extremes, so a fixed record-wide figure would
  // overstate what is actually on screen.
  fieldLegend.showRange(shown, vis, state.band, function () {
    fieldScale.sync();
    refresh();      // safe: the autofit flag is already cleared, so this ends here
  });

  var n = sel.size();
  var nObs = sel.filter(ee.Filter.eq('grace_observed', 1)).size();
  // How many contributing months are reconstructions is not a footnote when a
  // mean is displayed: a "monsoon mean" drawn mostly from unobserved months is a
  // different claim from one drawn from observed ones, and the map cannot show
  // the difference.
  ee.List([n, nObs]).evaluate(function (v) {
    var total = v[0], obs = v[1], recon = total - obs;
    periodLabel.setValue(state.season + ', ' + y0 + '–' + y1 +
                         '  ·  mean of ' + total + ' month' +
                         (total === 1 ? '' : 's'));
    if (total === 0) {
      regimeLabel.setValue('Nothing in this selection');
      regimeLabel.style().set('color', MUTED);
    } else if (recon === 0) {
      regimeLabel.setValue('All ' + total + ' GRACE-observed — mass-constrained');
      regimeLabel.style().set('color', '#2f7d63');
    } else {
      regimeLabel.setValue(recon + ' of ' + total + ' are RECONSTRUCTED ' +
                           '(' + Math.round(100 * recon / total) + '%) — ' +
                           'no GRACE observation, no mass constraint');
      regimeLabel.style().set('color', '#b0524d');
    }
  });
}

seasonSelect.onChange(function (label) {
  state.season = label;
  var custom = label === 'Custom months…';
  customPanel.style().set('shown', custom);
  if (!custom) {
    state.months = SEASONS[label];
    monthChecks.forEach(function (c, j) {
      c.setValue(state.months.indexOf(j + 1) !== -1, false);
    });
  }
  refresh();
});
yearSlider.onChange(refresh);
yearSlider2.onChange(refresh);

// --- layer toggles -------------------------------------------------------
left.add(heading('Layers', '13px'));
var sigmaCheck = ui.Checkbox('Show total uncertainty (σ_total) instead', false);
sigmaCheck.onChange(function (checked) {
  state.band = checked ? 'sigma_total' : 'twsa';
  refresh();             // repaints the layer AND its bar, palette included
  fieldScale.sync();     // the two bands keep separate limits
});
left.add(sigmaCheck);

// Opacity, per layer rather than one global control: the trend is meant to be
// read OVER the field, and a single slider cannot fade one against the other.
// Defaults sit at 0.85 rather than 1 so the terrain basemap stays faintly
// visible -- a 0.1 degree field with no landmarks under it is hard to place.
function opacityRow(label, initial, apply) {
  // The slider runs in WHOLE PERCENT, not 0..1. ui.Slider prints its own value,
  // and 0.85 has no exact binary representation, so a 0..1 slider opened
  // displaying "0.8500000000000001". Integers have no such problem, and the
  // conversion to the 0..1 that setOpacity wants happens here instead.
  var s = ui.Slider({min: 0, max: 100, value: Math.round(initial * 100),
                     step: 5, style: {width: '150px'}});
  s.onChange(function (pct) { apply(pct / 100); });
  return ui.Panel(
      [ui.Label(label, {fontSize: '11px', color: MUTED, margin: '6px 4px 0 0'}), s],
      ui.Panel.Layout.flow('horizontal'));
}

left.add(opacityRow('Field opacity %', state.opacity, function (v) {
  state.opacity = v;
  if (map.layers().length() > 0) map.layers().get(0).setOpacity(v);
}));

// --- trend, masked to significance ---------------------------------------
// Constraint 4. The mask is applied by DEFAULT and the unmasked field is not
// offered as a one-click alternative: a per-pixel trend map without its
// significance layer is precisely the over-reading the rest of this app exists
// to prevent.
var trendImg = ee.Image(ee.ImageCollection(TREND).first()).rename(TREND_BANDS);
var trendCheck = ui.Checkbox('Show 2000–2025 trend (FDR-significant pixels)', false);

// One draw path, shared by the checkbox and the scale boxes, so a rescale
// cannot repaint the layer while leaving the bar on the old numbers.
function drawTrend() {
  if (trendCheck.getValue()) {
    map.layers().set(1, ui.Map.Layer(
        trendImg.select('sen_slope')
                .updateMask(trendImg.select('significant_fdr'))
                .updateMask(basin),
        trendVis(), 'Theil-Sen slope (mm/yr), FDR-significant',
        true, state.trendOpacity));
  } else if (map.layers().length() > 1) {
    // get(1) is undefined if the layer was never added, and remove(undefined)
    // throws; setting an empty layer is the safe way to clear a slot.
    map.layers().set(1, ui.Map.Layer(ee.Image().mask(0), {}, 'none', false));
  }
  syncLegend();          // the trend bar appears and leaves with its layer
}
trendCheck.onChange(drawTrend);
left.add(trendCheck);
left.add(opacityRow('Trend opacity %', state.trendOpacity, function (v) {
  state.trendOpacity = v;
  if (map.layers().length() > 1) map.layers().get(1).setOpacity(v);
}));

// --- colour scale, editable ----------------------------------------------
// The app opens on the publication limits, because a fixed scale is what makes
// two periods -- or the viewer and a printed figure -- comparable at a glance,
// and that is the commoner need. A fitted scale shows more of the pattern in
// any one view but moves under the reader, so it is offered rather than
// imposed: "Fit to data" fits whatever is currently drawn to its 2-98%
// percentile range, "Publication scale" comes back, and the boxes take any
// limits at all.
left.add(heading('Colour scale', '13px'));

function scaleRow(label, get, redraw) {
  var minBox = ui.Textbox({value: '', style: {width: '58px', margin: '4px 2px 0 0'}});
  var maxBox = ui.Textbox({value: '', style: {width: '58px', margin: '4px 0 0 0'}});
  function commit() {
    var a = parseFloat(minBox.getValue()), b = parseFloat(maxBox.getValue());
    var s = get();
    // Reject rather than render nonsense: an inverted or zero-width scale
    // paints the whole basin one colour, which looks exactly like missing data.
    if (isNaN(a) || isNaN(b) || b <= a) {
      minBox.setValue('' + s.min, false);
      maxBox.setValue('' + s.max, false);
      return;
    }
    s.min = a;
    s.max = b;
    redraw();
  }
  minBox.onChange(commit);
  maxBox.onChange(commit);
  return {
    panel: ui.Panel([ui.Label(label, {fontSize: '11px', color: MUTED,
                                      margin: '8px 4px 0 0'}), minBox, maxBox],
                    ui.Panel.Layout.flow('horizontal')),
    // Called whenever the numbers change from outside the boxes -- an auto-fit,
    // a reset, or switching band. `false` suppresses onChange so writing a
    // value back cannot re-enter commit().
    sync: function () {
      var s = get();
      minBox.setValue('' + s.min, false);
      maxBox.setValue('' + s.max, false);
    }
  };
}

var fieldScale = scaleRow('Field  min / max',
                          function () { return state.scale[state.band]; },
                          refresh);
var trendScale = scaleRow('Trend  min / max',
                          function () { return state.scale.trend; },
                          drawTrend);
left.add(fieldScale.panel);
left.add(trendScale.panel);
// Populate immediately with the fallback limits. The first measurement replaces
// them a moment later, but a box that opens empty looks broken, and if the
// measurement fails these are what the map is actually drawn from anyway.
fieldScale.sync();
trendScale.sync();

function applyScales() {
  fieldScale.sync();
  trendScale.sync();
  refresh();
  // The trend image never changes, so its range is only re-queried when a fit
  // has actually been asked for; otherwise redrawing is enough.
  if (state.autofit.trend) measureTrend(); else drawTrend();
}

left.add(ui.Panel([
  ui.Button({label: 'Fit to data', style: {margin: '6px 4px 0 0'},
             onClick: function () {
               // Re-arm both, then let the next measurement land.
               state.autofit[state.band] = true;
               state.autofit.trend = true;
               applyScales();
             }}),
  ui.Button({label: 'Publication scale', style: {margin: '6px 0 0 0'},
             onClick: function () {
               for (var k in DEFAULT_SCALE) {
                 state.autofit[k] = false;      // an explicit choice, not a default
                 state.scale[k].min = DEFAULT_SCALE[k].min;
                 state.scale[k].max = DEFAULT_SCALE[k].max;
               }
               applyScales();
             }})
], ui.Panel.Layout.flow('horizontal')));

left.add(note('Opens on the fixed limits used by the figures in the paper, so ' +
              'the viewer and a printed figure can be read against each other. ' +
              'Those limits clip: the line under each colour bar gives the ' +
              'measured range and says so. "Fit to data" instead fits the ' +
              '2nd–98th percentile of what is drawn, so one outlier pixel ' +
              'cannot set the range. On a signed field the palette is ' +
              'recentred to whatever limits are in force, so the neutral ' +
              'colour always means zero — "at the 2004–2010 baseline" — even ' +
              'when the range is lopsided.'));
left.add(note('Theil-Sen slope of the deseasonalised monthly anomaly, masked to ' +
              'pixels passing Benjamini-Hochberg FDR at 0.05. Two caveats it ' +
              'would be misleading to omit. The p-value is NOT a calibrated ' +
              'error rate: on simulated AR(1) series of this length the ' +
              'serial-dependence-corrected test still rejected a true null ' +
              '18.8% of the time at a nominal 5%, so read the field as an ' +
              'ordering of pixels by strength of evidence. And FDR treats 9,538 ' +
              'pixels as 9,538 tests when GRACE resolves 19 independent mascons, ' +
              'so the mask is optimistic even so.'));
left.add(note('The trend is inherited from GRACE by mass conservation. Its ' +
              'large-scale pattern is observed; the within-mascon detail comes ' +
              'from the smooth background interpolation, not from data.'));

left.add(note('σ_total combines GRACE measurement error, spatial-transfer ' +
              'error, reconstruction error and ensemble spread. The last term ' +
              'is a LOWER BOUND: no observation of within-mascon structure ' +
              'exists to calibrate it against.'));

// --- basin outline --------------------------------------------------------
// Drawn from the uploaded shapefile rather than from the product's own mask.
// The mask edge is the 0.1 degree lattice stepping around the catchment; the
// polygon is the catchment. On a satellite basemap the difference is what tells
// a viewer whether a cell near the rim is inside the basin or just inside its
// bounding raster.
var basinFC = ee.FeatureCollection(BASIN);
var basinOutline = ee.Image().byte()
    .paint({featureCollection: basinFC, color: 1, width: 2});

var basinCheck = ui.Checkbox('Show basin boundary', true);
function drawBasin() {
  map.layers().set(2, ui.Map.Layer(basinOutline, {palette: ['#1a1a1a']},
                                   'Basin boundary', basinCheck.getValue()));
}
basinCheck.onChange(drawBasin);
left.add(basinCheck);

// --- the chart -----------------------------------------------------------
left.add(heading('Per-pixel time series', '13px'));
var chartHint = ui.Label('Click anywhere in the basin for its monthly series ' +
                         'with the ±1σ band, which months GRACE measured, and ' +
                         'annual means.',
                         {fontSize: '11px', color: MUTED});
left.add(chartHint);
var chartPanel = ui.Panel();
left.add(chartPanel);

map.onClick(function (coords) {
  chartPanel.clear();
  chartPanel.add(ui.Label('Building series…', {fontSize: '11px', color: MUTED}));

  var pt = ee.Geometry.Point([coords.lon, coords.lat]);
  var where = coords.lon.toFixed(2) + '°E, ' + coords.lat.toFixed(2) + '°N';

  // Guard: outside the basin every band is masked, so `twsa` comes back null and
  // twsa.subtract(sig) throws rather than returning nothing. Check once, on the
  // client, before building anything.
  first.select('twsa').reduceRegion({
    reducer: ee.Reducer.first(), geometry: pt, scale: 11132, maxPixels: 4
  }).get('twsa').evaluate(function (v) {
    if (v === null || v === undefined) {
      chartPanel.clear();
      chartPanel.add(ui.Label('Outside the basin — no data at ' + where + '.',
                              {fontSize: '11px', color: '#b0524d'}));
      return;
    }
    drawSeries(pt, where);
  });
});

function drawSeries(pt, where) {
  // ---- monthly ----------------------------------------------------------
  // EVERY charted column is numeric. An earlier version split the line into an
  // `observed` and a `reconstructed` series padded with nulls, which Google
  // Charts rejects outright -- "All series on a given axis must be of the same
  // data type" -- because a column whose first value is null has no inferable
  // type, and January 2000 is a reconstruction. The observed/reconstructed
  // distinction moved to the availability strip below, which is a clearer idiom
  // anyway: one line for the value, one strip for what backs it.
  var fcM = monthly.map(function (img) {
    var v = img.select(['twsa', 'sigma_total'])
               .reduceRegion({reducer: ee.Reducer.first(), geometry: pt,
                              scale: 11132, maxPixels: 4});
    var twsa = ee.Number(v.get('twsa'));
    var sig = ee.Number(v.get('sigma_total'));
    return ee.Feature(null, {
      't': img.date().format('YYYY-MM'),
      'lower': twsa.subtract(sig),
      'TWSA': twsa,
      'upper': twsa.add(sig),
      // 0/1, never null, so the strip below shares a numeric axis.
      'GRACE observed': ee.Number(img.get('grace_observed'))
    });
  }).filter(ee.Filter.notNull(['TWSA']));

  var monthlyChart = ui.Chart.feature.byFeature(
        fcM, 't', ['lower', 'TWSA', 'upper'])
      .setChartType('LineChart')
      .setOptions({
        title: 'Monthly · ' + where,
        titleTextStyle: {fontSize: 12, bold: true},
        hAxis: {title: '', showTextEvery: 24, textStyle: {fontSize: 9}},
        vAxis: {title: 'TWSA (mm)', titleTextStyle: {fontSize: 11},
                textStyle: {fontSize: 10}},
        series: {
          0: {color: '#b9c4cc', lineWidth: 1, pointSize: 0},   // lower
          1: {color: '#1a1a1a', lineWidth: 2, pointSize: 0},   // TWSA
          2: {color: '#b9c4cc', lineWidth: 1, pointSize: 0}    // upper
        },
        legend: {position: 'top', textStyle: {fontSize: 10}},
        chartArea: {width: '80%', height: '62%'}, height: 230
      });

  // A data-availability strip on the same x. Reading "is this month measured?"
  // off a bar is easier than off a dashed segment, and it cannot collide with
  // the value axis.
  var availChart = ui.Chart.feature.byFeature(fcM, 't', ['GRACE observed'])
      .setChartType('ColumnChart')
      .setOptions({
        title: 'GRACE observation (1 = measured, 0 = reconstructed)',
        titleTextStyle: {fontSize: 10, bold: false, italic: false},
        hAxis: {title: '', showTextEvery: 24, textStyle: {fontSize: 9}},
        vAxis: {textPosition: 'none', gridlines: {count: 0},
                viewWindow: {min: 0, max: 1}},
        series: {0: {color: '#2f7d63'}},
        bar: {groupWidth: '100%'},
        legend: {position: 'none'},
        chartArea: {width: '80%', height: '55%'}, height: 90
      });

  // ---- annual means ------------------------------------------------------
  // Aggregated from the monthly field, so a year containing reconstructed
  // months inherits them. The count is charted rather than hidden.
  var years = ee.List.sequence(2000, 2025);
  var fcY = ee.FeatureCollection(years.map(function (y) {
    var sel = monthly.filter(ee.Filter.calendarRange(y, y, 'year'));
    var v = sel.select('twsa').mean()
               .reduceRegion({reducer: ee.Reducer.first(), geometry: pt,
                              scale: 11132, maxPixels: 4});
    var nRec = sel.size().subtract(
        sel.filter(ee.Filter.eq('grace_observed', 1)).size());
    return ee.Feature(null, {
      't': ee.Number(y).format('%d'),
      'annual mean': v.get('twsa'),
      'reconstructed months': nRec
    });
  })).filter(ee.Filter.notNull(['annual mean']));

  var annualChart = ui.Chart.feature.byFeature(
        fcY, 't', ['annual mean', 'reconstructed months'])
      .setChartType('ColumnChart')
      .setOptions({
        title: 'Annual mean · ' + where,
        titleTextStyle: {fontSize: 12, bold: true},
        hAxis: {title: '', textStyle: {fontSize: 9}, slantedText: true,
                slantedTextAngle: 60},
        // Two axes here are not the dual-axis anti-pattern: the second series is
        // a COUNT of reconstructed months drawn as a thin context line, not a
        // second measure of the same quantity on a rescaled y.
        vAxes: {0: {title: 'TWSA (mm)', titleTextStyle: {fontSize: 11}},
                1: {title: 'recon.', viewWindow: {min: 0, max: 12}}},
        series: {0: {targetAxisIndex: 0, color: '#2166ac'},
                 1: {targetAxisIndex: 1, color: '#e6cfcc', type: 'line',
                     lineWidth: 1, pointSize: 0}},
        legend: {position: 'top', textStyle: {fontSize: 10}},
        chartArea: {width: '75%', height: '55%'}, height: 200
      });

  chartPanel.clear();
  chartPanel.add(monthlyChart);
  chartPanel.add(availChart);
  chartPanel.add(annualChart);
  chartPanel.add(ui.Label(
      'Grey envelope is ±1σ_total. The green strip marks months GRACE actually ' +
      'measured; gaps in it are reconstructions carrying no mass constraint. ' +
      'The thin line on the annual chart counts reconstructed months in each ' +
      'year, so a low bar drawn from mostly-reconstructed months is visible as ' +
      'such.',
      {fontSize: '10px', color: MUTED, margin: '0 0 6px 0'}));
}

// --- daily, deliberately not charted -------------------------------------
left.add(heading('Daily field', '13px'));
left.add(note('A 0.1° daily disaggregation exists (9,497 days). It is NOT ' +
              'charted here: the within-month shape comes from the ERA5-Land ' +
              'water balance and nothing observational validates it — GRACE is ' +
              'monthly, the validation wells quarterly. It re-aggregates to the ' +
              'monthly field exactly by construction, which is arithmetic ' +
              'rather than evidence. Use the archived netCDF if you need it.'));

// --- provenance ----------------------------------------------------------
left.add(heading('Source', '13px'));
left.add(ui.Label('Data record (Zenodo, v1.0.0)',
                  {fontSize: '11px', margin: '1px 0'},
                  'https://doi.org/10.5281/zenodo.21745159'));
left.add(ui.Label('Code and methods (GitHub)',
                  {fontSize: '11px', margin: '1px 0'},
                  'https://github.com/montimaj/grace-grb'));
left.add(note('Contains modified Copernicus Climate Change Service information ' +
              '2026. GRACE/GRACE-FO mascons courtesy NASA JPL.'));

// ---------------------------------------------------------------- legend
// A legend that does not track the layer beneath it is worse than no legend: it
// puts numbers on a colour that does not carry them. Up to two layers are on
// screen at once and no two of the three possible fields share a scale --
//
//   field  twsa         +/-300 mm     diverging
//   field  sigma_total     0-150 mm   sequential, a DIFFERENT palette
//   trend  sen_slope     +/-40 mm/yr  diverging, and per YEAR
//
// A single static TWSA bar therefore showed the wrong palette entirely for the
// sigma layer and rescaled the trend by 7.5x while changing its units. The bar
// is built once and re-parameterised instead, and the trend carries its own
// block that appears and disappears with its layer.
function makeLegendBlock() {
  var title = ui.Label('', {fontWeight: 'bold', fontSize: '11px',
                            margin: '0 0 2px 0'});
  // A 0..1 longitude ramp stretched across the strip is the standard idiom for
  // a continuous legend; the palette does the rest.
  var bar = ui.Thumbnail({
    image: ee.Image.pixelLonLat().select('longitude'),
    params: {bbox: [0, 0, 1, 0.1], dimensions: '160x12',
             format: 'png', min: 0, max: 1, palette: DIVERGING},
    style: {stretch: 'horizontal', margin: '0'}
  });
  var lo = ui.Label('', {fontSize: '10px', margin: '0'});
  var mid = ui.Label('', {fontSize: '10px', margin: '0 0 0 34px'});
  var hi = ui.Label('', {fontSize: '10px', margin: '0 0 0 34px'});
  // The bar ends are DISPLAY limits, not data limits, and every one of the
  // three fields overshoots them: the trend runs to -84 mm/yr against a bar
  // that stops at -40, so the whole depleting core of the basin is painted the
  // single darkest red. Without this line a viewer reads that plateau as the
  // maximum rather than as saturation.
  var range = ui.Label('', {fontSize: '9px', color: MUTED, margin: '1px 0 0 0'});
  var panel = ui.Panel({
    widgets: [title, bar,
              ui.Panel({widgets: [lo, mid, hi],
                        layout: ui.Panel.Layout.flow('horizontal')}),
              range],
    style: {margin: '0 0 4px 0'}
  });

  function fmt(v) {
    var a = Math.abs(v);
    return (v > 0 ? '+' : '') + (a < 10 ? v.toFixed(1) : v.toFixed(0));
  }

  // A slider drag fires several of these and they need not return in order, so
  // each request carries a ticket and a stale reply is dropped rather than
  // overwriting a newer one.
  var seq = 0;
  // The measured range is cached, because editing the scale changes only
  // whether the data is clipped, never what the data is. Re-querying Earth
  // Engine on every keystroke in the min box would be a round trip for an
  // answer we already hold.
  var lastLo = null, lastHi = null;

  function renderRange(vis) {
    if (lastLo === null) return;
    var clipped = lastLo < vis.min || lastHi > vis.max;
    range.setValue('data ' + fmt(lastLo) + ' to ' + fmt(lastHi) +
                   (clipped ? '  ·  beyond the bar is clipped' : ''));
  }

  return {
    panel: panel,
    update: function (name, vis, unit) {
      title.setValue(name);
      bar.setParams({bbox: [0, 0, 1, 0.1], dimensions: '160x12',
                     format: 'png', min: 0, max: 1, palette: vis.palette});
      lo.setValue('' + vis.min);
      mid.setValue(unit);
      // '+' only on a signed scale. "+150" on a magnitude reads as a typo.
      hi.setValue((vis.min < 0 ? '+' : '') + vis.max);
      renderRange(vis);          // the clip note follows the new limits
    },
    // `img` must be single-band; it is renamed so the reducer's output keys are
    // the same whichever field is on screen. `key` names the scale to auto-fit
    // on first measurement; `onFit` redraws once that has happened.
    showRange: function (img, vis, key, onFit) {
      var mine = ++seq;
      range.setValue('measuring range…');
      // bounds(), not the basin polygon: every image passed here is already
      // updateMask'd to the basin, so masked pixels are excluded from the
      // reducer whatever the region is. A rectangle costs far less to
      // rasterise than a catchment outline and gives the identical answer.
      // Percentiles AND extremes in one request. The scale is fitted to the
      // 2-98% range so a single outlier pixel cannot set it; the extremes are
      // what the clip note reports, because they are what tells a reader the
      // bar is saturated.
      img.rename('v').reduceRegion({
        reducer: ee.Reducer.percentile([2, 98])
                   .combine({reducer2: ee.Reducer.minMax(), sharedInputs: true}),
        geometry: basinFC.geometry().bounds(),
        scale: 11132, maxPixels: 1e8, bestEffort: true
      }).evaluate(function (r) {
        if (mine !== seq) return;                  // overtaken by a newer request
        if (!r || r.v_min === null || r.v_min === undefined) {
          lastLo = lastHi = null;
          range.setValue('range unavailable here');
          return;
        }
        lastLo = r.v_min;
        lastHi = r.v_max;
        // Auto-fit happens ONCE per scale, never on every period change: a bar
        // that re-fits itself as the selection moves makes two periods
        // impossible to compare, which is the one thing a fixed scale is for.
        if (key && state.autofit[key]) {
          state.autofit[key] = false;
          // Fall back to the extremes if a percentile came back empty, which
          // happens when the selection leaves too few unmasked pixels.
          var flo = (r.v_p2 === null || r.v_p2 === undefined) ? lastLo : r.v_p2;
          var fhi = (r.v_p98 === null || r.v_p98 === undefined) ? lastHi : r.v_p98;
          var f = fitScale(key, flo, fhi);
          state.scale[key].min = f.min;
          state.scale[key].max = f.max;
          if (onFit) { onFit(); return; }          // the redraw re-renders this
        }
        renderRange(vis);
      });
    }
  };
}

var fieldLegend = makeLegendBlock();
var trendLegend = makeLegendBlock();
trendLegend.update('Trend (FDR-significant)', trendVis(), 'mm/yr');

// The trend image is static, so its range is measured once. It is the range of
// the MASKED field — what the layer actually draws — rather than of every pixel
// in the trend image, so a pixel that fails FDR cannot widen the bar.
function measureTrend() {
  trendLegend.showRange(
      trendImg.select('sen_slope')
              .updateMask(trendImg.select('significant_fdr'))
              .updateMask(basin),
      trendVis(), 'trend',
      function () { trendScale.sync(); drawTrend(); });
}
measureTrend();

map.add(ui.Panel({
  widgets: [fieldLegend.panel, trendLegend.panel],
  style: {position: 'bottom-right', padding: '6px',
          backgroundColor: 'rgba(255,255,255,0.85)'}
}));

// Called by the sigma and trend toggles, which are defined further up the file.
// A function declaration hoists, so those handlers can reach it even though it
// is written here, beside the widgets it drives.
function syncLegend() {
  var isTwsa = state.band === 'twsa';
  fieldLegend.update(isTwsa ? 'TWSA' : 'Total uncertainty (σ)',
                     fieldVis(), 'mm');
  trendLegend.update('Trend (FDR-significant)', trendVis(), 'mm/yr');
  trendLegend.panel.style().set('shown', trendCheck.getValue());
}
syncLegend();

// ---------------------------------------------------------------- layout
ui.root.clear();
ui.root.add(ui.SplitPanel({firstPanel: left, secondPanel: map}));

// Layer slots are fixed: 0 field, 1 trend, 2 basin outline. They are filled in
// order at startup because ui.Map.Layers.set(i) needs every lower index to
// exist -- setting slot 2 first throws.
refresh();                                   // slot 0
map.layers().set(1, ui.Map.Layer(ee.Image().mask(0), {}, 'trend', false));
drawBasin();                                 // slot 2
