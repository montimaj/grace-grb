"""
Well comparison at THREE aggregation scales: per-well, per-mascon, whole-basin.

Why scale matters here
----------------------
Neither side of this comparison is an observation of groundwater storage. Both
are an observation passed through an estimate:

    model side   GWS = TWSA_downscaled - (modelled soil moisture + snow)
                 TWSA is GRACE-anchored, but the subtracted stores are pure model
                 output.

    well side    GWS = -Sy * dh * 1000
                 The water LEVEL is measured; Sy is a single reference estimate
                 per well, and treating a point level as areal storage over a
                 ~120 km^2 pixel is an assumption, not a measurement.

The two estimate errors are independent, so the disagreement contains both and a
single pooled number cannot say which is responsible. Aggregating separates them.

What each scale isolates
------------------------
It is tempting to say mass conservation makes the basin and mascon comparisons
independent of the downscaling. It does NOT, for two reasons, and the measured
series prove it -- downscaled and bilinear diverge visibly at basin scale.

  1. Conservation pins the AREA-WEIGHTED mascon mean. These aggregates average
     the pixels WELLS OCCUPY, which are a biased, non-uniform sample of each
     mascon -- concentrated in the populated plains. A well-weighted subset mean
     is not the area-weighted mean, so within-mascon structure survives every
     level of aggregation here.
  2. Bilinear interpolation is not mass-conserved at all, so it is not pinned to
     anything.

What actually changes across the three scales is how much POINT-SCALE error is
averaged away:

    BASIN     ~627 wells per month. Per-well Sy error and point-to-pixel
              mismatch are largely averaged out, leaving basin-scale timing.
    MASCON    16-142 wells per scored unit. Intermediate.
    WELL      One well, one pixel. Sy error and point-to-pixel mismatch are at
              their largest and dominate the residual.

So a method that is genuinely better at large scales but noisier at points will
win at BASIN and lose at WELL. That is a real and reportable property, not a
contradiction -- and it is invisible in a single pooled number, which is the
argument for computing all three.

What remains common to every scale is the DECOMPOSITION: the subtracted stores,
Sy, and the point-to-areal assumption. Poor agreement at all three indicts that
rather than the downscaling.

Sampling is controlled, not assumed
-----------------------------------
The model series is averaged over THE SAME PIXELS the wells occupy, in the same
months -- never over the full mascon. Comparing a whole-mascon model mean against
a well-network mean would confound downscaling error with the fact that wells sit
in the populated plains and not on the Himalayan margin.

Composition is also controlled. Monthly well counts range from 8 to 645 across
the record; a month carrying 8 wells is not comparable to one carrying 645, and
averaging them into one series manufactures variance that is pure sampling.
`--min-wells` drops the thin months.

Seven of the 19 in-basin mascons contain no CGWB well at all. Of the twelve that
do, mascon 6 has a single well and mascon 32 has nine -- both below the ten-well
default -- so the mascon panels cover 10 of 19. That is a statement about where
the CGWB network is, not about where the product works, and `mascon_metric_maps`
draws the other nine in flat grey so the two cannot be confused.

Usage
-----
    python validate_wells_scales.py
    python validate_wells_scales.py --min-wells 20 --stores gldas
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import downscale_model as M
import figure_captions
import validate_wells as V
from plot_style import DPI, SCI_AQUA, SCI_BLUE, SCI_GRID, SCI_INK, SCI_MUTED, SCI_ORANGE

# Fixed series order, never cycled. Colour is backed by line style and marker so
# identity survives greyscale printing and any colour-vision deficiency.
SERIES = [
    ('gws_anomaly_mm', 'Wells (observed level x Sy)', SCI_BLUE, '-', 'o'),
    ('gws_downscaled', 'Downscaled 0.1 deg', SCI_ORANGE, '--', 's'),
    ('gws_bilinear', 'Bilinear interpolation', SCI_AQUA, ':', '^'),
]
VALUE_COLS = [s[0] for s in SERIES]


def aggregate(obs: pd.DataFrame, level: str, min_wells: int) -> pd.DataFrame:
    """
    Mean series per unit per month, with the well count that produced each point.

    `level` is 'basin', 'mascon' or 'well'. Averaging plain anomalies is safe
    because every series is already centred on the same 2004-2010 baseline.
    """
    d = obs.copy()
    d['date'] = pd.to_datetime(d['date'])
    d['month'] = d['date'].values.astype('datetime64[M]')

    if level == 'basin':
        keys = ['month']
        d['unit'] = 'basin'
    elif level == 'mascon':
        keys = ['unit', 'month']
        d['unit'] = d['mascon'].astype(int).astype(str)
    elif level == 'well':
        keys = ['unit', 'month']
        d['unit'] = d['well_id'].astype(str)
    else:
        raise ValueError(f"level must be basin/mascon/well, got {level!r}")

    if level == 'basin':
        g = d.groupby('month')
    else:
        g = d.groupby(['unit', 'month'])

    out = g.agg(**{c: (c, 'mean') for c in VALUE_COLS},
                n_wells=('well_id', 'nunique')).reset_index()
    if level == 'basin':
        out.insert(0, 'unit', 'basin')
    # A well is its own unit, so the count is always 1 and the threshold would
    # remove everything.
    if level != 'well':
        out = out[out.n_wells >= min_wells]
    return out.sort_values(['unit', 'month']).reset_index(drop=True)


def _metrics(obs_v: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(obs_v) & np.isfinite(pred)
    if ok.sum() < 6 or np.std(obs_v[ok]) == 0 or np.std(pred[ok]) == 0:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': int(ok.sum())}
    o, p = obs_v[ok], pred[ok]
    return {'r': float(np.corrcoef(o, p)[0, 1]),
            'RMSE': float(np.sqrt(np.mean((p - o) ** 2))),
            'bias': float(np.mean(p - o)),
            'n': int(ok.sum())}


def score(series: pd.DataFrame, level: str, min_months: int = 8) -> pd.DataFrame:
    """Skill of each model series against the wells, per unit."""
    rows = []
    for unit, g in series.groupby('unit'):
        if len(g) < min_months:
            continue
        o = g.gws_anomaly_mm.to_numpy()
        row = {'level': level, 'unit': unit, 'n_months': len(g),
               'n_wells': int(g.n_wells.max())}
        for col, label in (('gws_downscaled', 'downscaled'),
                           ('gws_bilinear', 'bilinear')):
            m = _metrics(o, g[col].to_numpy())
            row.update({f'{label}_{k}': v for k, v in m.items() if k != 'n'})
        row['gap_r'] = row['downscaled_r'] - row['bilinear_r']
        row['gap_RMSE'] = row['downscaled_RMSE'] - row['bilinear_RMSE']
        rows.append(row)
    return pd.DataFrame(rows)


def _style_axis(ax):
    ax.grid(True, color=SCI_GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SCI_MUTED)
    ax.tick_params(colors=SCI_MUTED, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(SCI_INK)


def plot_basin(series: pd.DataFrame, sc: pd.DataFrame, out_dir: str) -> str:
    """Basin-mean time series. One y-axis; never two."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for col, label, colour, ls, mk in SERIES:
        ax.plot(series.month, series[col], ls, color=colour, lw=2.0,
                marker=mk, ms=4, mew=0, label=label, alpha=0.95)
    ax.axhline(0, color=SCI_MUTED, lw=1)
    _style_axis(ax)
    ax.set_ylabel('Groundwater storage anomaly (mm)', color=SCI_INK)
    ax.set_xlabel('')
    r = sc.iloc[0] if len(sc) else None
    sub = ('' if r is None else
           f'   downscaled r = {r.downscaled_r:.2f}, '
           f'bilinear r = {r.bilinear_r:.2f}, over {int(r.n_months)} months')
    ax.set_title('Basin-mean groundwater anomaly: wells vs product' + sub,
                 fontweight='bold', color=SCI_INK, loc='left')
    ax.legend(frameon=False, ncol=3, fontsize=9, loc='upper right')
    fig.tight_layout()
    p = os.path.join(out_dir, 'well_scales_basin.png')
    figure_captions.record(
        p,
        'Averaged over the pixels wells occupy, so per-well specific-yield '
        'error and point-to-pixel mismatch are largely averaged out. Well '
        'pixels are a biased sample of each mascon, so within-mascon structure '
        'survives aggregation and the two products are not forced to agree.')
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


def plot_mascons(series: pd.DataFrame, sc: pd.DataFrame, out_dir: str) -> Optional[str]:
    """Small multiples, one panel per mascon, shared y so panels are comparable."""
    import matplotlib.pyplot as plt

    units = list(sc.sort_values('n_wells', ascending=False).unit)
    if not units:
        return None
    ncol = 3
    nrow = int(np.ceil(len(units) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.5 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, unit in zip(axes, units):
        g = series[series.unit == unit]
        for col, _label, colour, ls, _mk in SERIES:
            ax.plot(g.month, g[col], ls, color=colour, lw=1.5, alpha=0.95)
        ax.axhline(0, color=SCI_MUTED, lw=0.8)
        _style_axis(ax)
        row = sc[sc.unit == unit].iloc[0]
        ax.set_title(f'mascon {unit}  ({int(row.n_wells)} wells)  '
                     f'r={row.downscaled_r:.2f}/{row.bilinear_r:.2f}',
                     fontsize=9, color=SCI_INK, loc='left')
    for ax in axes[len(units):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2, label=lab)
               for _c, lab, c, ls, _m in SERIES]
    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=9,
               loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Groundwater anomaly by mascon: wells vs product  '
                 '(r shown as downscaled/bilinear)',
                 fontweight='bold', color=SCI_INK)
    fig.supylabel('GWS anomaly (mm)', color=SCI_INK, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, 'well_scales_mascon.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


def plot_well_distribution(sc: pd.DataFrame, out_dir: str) -> Optional[str]:
    """
    Per-well correlations as paired distributions.

    656 panels would be unreadable, and a mean would hide that the question is
    which method wins MORE OFTEN, not by how much on average.
    """
    import matplotlib.pyplot as plt

    if sc.empty:
        return None
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[1.3, 1])
    bins = np.linspace(-1, 1, 41)
    ax.hist(sc.downscaled_r.dropna(), bins=bins, color=SCI_ORANGE, alpha=0.75,
            label='Downscaled 0.1 deg')
    ax.hist(sc.bilinear_r.dropna(), bins=bins, color=SCI_AQUA, alpha=0.6,
            label='Bilinear interpolation')
    for v, c in ((sc.downscaled_r.median(), SCI_ORANGE),
                 (sc.bilinear_r.median(), SCI_AQUA)):
        ax.axvline(v, color=c, lw=2, ls='--')
    _style_axis(ax)
    ax.set_xlabel('Per-well correlation with observed GWS', color=SCI_INK)
    ax.set_ylabel('Wells', color=SCI_INK)
    ax.set_title('Per-well temporal correlation (dashed = median)',
                 fontweight='bold', color=SCI_INK, loc='left')
    ax.legend(frameon=False, fontsize=9)

    gap = sc.gap_r.dropna()
    ax2.hist(gap, bins=np.linspace(-1, 1, 41),
             color=np.where(gap.median() <= 0, SCI_ORANGE, SCI_AQUA).item()
             if len(gap) else SCI_ORANGE, alpha=0.8)
    ax2.axvline(0, color=SCI_INK, lw=1.5)
    ax2.axvline(gap.median(), color=SCI_MUTED, lw=2, ls='--')
    _style_axis(ax2)
    ax2.set_xlabel('downscaled r  -  bilinear r', color=SCI_INK)
    ax2.set_ylabel('Wells', color=SCI_INK)
    won = float((gap > 0).mean()) if len(gap) else np.nan
    ax2.set_title(f'Downscaling wins at {won:.0%} of wells '
                  f'(median {gap.median():+.3f})',
                  fontweight='bold', color=SCI_INK, loc='left')
    fig.tight_layout()
    p = os.path.join(out_dir, 'well_scales_per_well.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


def per_well_metrics(obs: pd.DataFrame) -> pd.DataFrame:
    """
    All five skill metrics at every well, plus its location and sample count.

    Scored through `downscale_model.metrics` so a well is judged by exactly the
    same definitions as a mascon or a holdout month -- in particular R2 is the
    SQUARED PEARSON correlation, not sklearn's r2_score. That distinction is the
    only reason it is worth mapping R2 and NSE side by side: sklearn's r2_score
    is algebraically identical to NSE, so the two panels would be one panel drawn
    twice. Squared Pearson asks whether the product tracks the SHAPE of the well
    series; NSE asks whether it also gets the magnitude and the offset right. A
    well can score well on the first and badly on the second, and where that
    happens is diagnostic rather than decorative.

    MBE is `downscaled - well`, the same sign convention as the residual map:
    positive means the product reads high. It is reported instead of PBIAS
    because both series are anomalies centred on 2004-2010, so the denominator
    PBIAS divides by is a mean near zero and the percentage is meaningless.
    """
    rows = []
    for well, g in obs.groupby('well_id'):
        o = g.gws_anomaly_mm.to_numpy(dtype='float64')
        p = g.gws_downscaled.to_numpy(dtype='float64')
        m = M.metrics(o, p)
        ok = np.isfinite(o) & np.isfinite(p)
        rows.append({'well_id': well,
                     'lat': float(g.lat.iloc[0]), 'lon': float(g.lon.iloc[0]),
                     'n_months': int(ok.sum()),
                     'RMSE': m['RMSE'], 'MAE': m['MAE'],
                     'MBE': float(np.mean(p[ok] - o[ok])) if ok.any() else np.nan,
                     'R2': m['R2'], 'NSE': m['NSE']})
    return pd.DataFrame(rows)


# Panel order groups the metrics by the question they answer: how far off and in
# which direction (MBE), how big the error typically is (MAE) and how much of it
# is concentrated in outliers (RMSE), then whether the shape is right (R2) and
# whether shape, magnitude and offset are all right together (NSE). The last
# panel is sampling context rather than skill: how many months stand behind each
# of the other five.
#
# Metric names are written out in full to match the other map figures in the
# project, which spell theirs out too.
#
# `div` picks a diverging ramp with a neutral midpoint, used only where the value
# has a meaningful zero to diverge about: MBE at no bias, NSE at "no better than
# the well's own mean". Magnitudes get one sequential hue, light to dark. R2 is
# bounded on [0, 1] with no interesting midpoint, so it is sequential too.
# `zero_based` starts the ramp at 0 where 0 is the ideal (an error of none);
# where it is not, the ramp starts at the data minimum so the range is not spent
# on empty space. `clip` marks the panels whose limits are robust rather than
# full-range -- the month count is bounded and well behaved, so it is drawn whole.
_METRIC_PANELS = [
    dict(key='MBE',      label='Mean bias error (mm)',
         note='positive = product reads high', div=True,
         zero_based=False, clip=True, integer=False),
    dict(key='MAE',      label='Mean absolute error (mm)',
         note='typical size of the miss', div=False,
         zero_based=True, clip=True, integer=False),
    dict(key='RMSE',     label='Root mean square error (mm)',
         note='outlier-sensitive', div=False,
         zero_based=True, clip=True, integer=False),
    dict(key='R2',       label='R² (squared Pearson)',
         note='shape agreement only, ignores bias', div=False,
         zero_based=False, clip=True, integer=False),
    dict(key='NSE',      label='Nash–Sutcliffe efficiency',
         note='shape, magnitude and bias together', div=True,
         zero_based=False, clip=True, integer=False),
    dict(key='n_months', label='Months compared',
         note='observations behind each well above', div=False,
         zero_based=False, clip=False, integer=True),
]


def _scale_for(spec: Dict, v: np.ndarray):
    """
    Colour norm, ramp and clip arrows for one metric panel.

    Shared by the per-well and per-mascon figures so the two are read on the same
    rules. If the mascon map used its own limits, a reader comparing the pair
    would be reading a change of scale as a change of skill.
    """
    from matplotlib.colors import TwoSlopeNorm, Normalize

    if spec['div']:
        # Symmetric about zero so the neutral colour always means "no bias" /
        # "no skill", never an arbitrary midpoint of the data.
        lim = float(np.nanpercentile(np.abs(v), 90)) or 1.0
        extend = 'both'
        if spec['key'] == 'NSE':
            # NSE below -1 is uninformatively bad, so clip there and flag it.
            # The top needs no arrow: NSE is bounded above by 1 exactly, and an
            # arrow would imply units that cannot exist.
            lim, extend = 1.0, 'min'
        return TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), 'RdBu_r', extend

    lo = 0.0 if spec['zero_based'] else float(np.nanmin(v))
    hi = float(np.nanpercentile(v, 95)) if spec['clip'] else float(np.nanmax(v))
    return (Normalize(vmin=lo, vmax=hi), 'Purples',
            'max' if np.nanmax(v) > hi else 'neither')


def plot_metric_maps(obs: pd.DataFrame, out_dir: str) -> Optional[str]:
    """
    The five skill metrics, each mapped at the wells that produced them.

    Same drawing rules as `plot_residual_map`: points only, never a filled
    surface, because a metric exists only where a well exists.

    COLOUR LIMITS ARE ROBUST, AND THAT IS A JUDGEMENT WORTH STATING. Per-well
    RMSE spans 75 to 1987 mm and NSE reaches -419, so limits taken from the
    extremes would collapse every panel to a single colour and hide all of the
    spatial structure. Each panel is therefore clipped to a percentile range and
    the colourbar carries an arrow on any clipped end, so a saturated point reads
    as "at least this bad" rather than "this bad". The counts printed under the
    NSE panel are computed on the UNCLIPPED values.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.ticker import MaxNLocator
    import gridded_config as cfg

    per_well = per_well_metrics(obs)
    if per_well.empty:
        return None

    # Narrow rather than wide. The basin is ~1.6:1 and geopandas fixes the
    # aspect, so in a broad figure each map is limited by its row height and
    # leaves most of its column empty. Sizing the width down until the maps are
    # limited by their column instead is what fills the panels.
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.0), layout='constrained')
    flat = axes.ravel()
    try:
        basin = cfg.load_basin()
    except Exception:  # noqa: BLE001 - the points are the content, not the outline
        basin = None

    for letter, ax, spec in zip('abcdef', flat, _METRIC_PANELS):
        v = per_well[spec['key']].to_numpy(dtype='float64')
        if basin is not None:
            basin.boundary.plot(ax=ax, color=SCI_INK, linewidth=0.6, zorder=1)

        norm, cmap, extend = _scale_for(spec, v)

        sc = ax.scatter(per_well.lon, per_well.lat, c=v, cmap=cmap, norm=norm,
                        s=13, edgecolor=SCI_INK, linewidth=0.15, zorder=3)
        cax = ax.inset_axes([1.02, 0.0, 0.022, 1.0], transform=ax.transAxes)
        cb = fig.colorbar(sc, cax=cax, extend=extend)
        cb.ax.tick_params(labelsize=7, colors=SCI_INK)
        if spec['integer']:
            # A count of months has no half-values; the default locator was
            # ticking 47.5 and 52.5, which name months that cannot be counted.
            cb.locator = MaxNLocator(integer=True)
            cb.update_ticks()
        _style_axis(ax)
        ax.tick_params(labelsize=7)
        # Every panel carries its own axis labels rather than only the outer
        # ones: these panels get lifted into slides and talks one at a time, and
        # a lone map with bare numbers on both axes is ambiguous.
        ax.set_xlabel('Longitude (°E)', color=SCI_INK, fontsize=8)
        ax.set_ylabel('Latitude (°N)', color=SCI_INK, fontsize=8)
        ax.set_title(f'({letter}) {spec["label"]}', fontweight='bold',
                     color=SCI_INK, loc='left', fontsize=10, pad=14)
        ax.text(0.0, 1.015, spec['note'], transform=ax.transAxes,
                fontsize=7.5, color=SCI_MUTED, ha='left', va='bottom')

    fig.suptitle(f'Product skill against {len(per_well)} CGWB dug wells, well by well',
                 fontweight='bold', color=SCI_INK, fontsize=13, x=0.006, ha='left')
    p = os.path.join(out_dir, 'well_metric_maps.png')
    figure_captions.record(
        p,
        'Points, not a filled field: a metric exists only where a well exists, '
        'and interpolating between wells would manufacture a surface the data '
        'does not contain. Panels (a)–(e) use robust colour limits because the '
        'per-well spread is heavy-tailed — 95th percentile for the magnitudes, '
        '±90th of |MBE| for the bias, ±1 for NSE; arrowed colourbar ends mark '
        'clipping and every quoted number uses unclipped values. Panel (f) is '
        'drawn full-range. R² is the squared Pearson correlation, not '
        "sklearn's r2_score, which is algebraically identical to NSE; R² ≥ NSE "
        'is therefore an identity rather than a result, and the gap between (d) '
        'and (e) is exactly the amplitude-and-bias error in units of each '
        "well's own variability.")
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    per_well.to_csv(os.path.join(out_dir, 'well_metrics_per_well.csv'), index=False)
    _report_metric_summary(per_well)
    return p


# Same five metrics, plus the sampling context that matters AT THIS SCALE. For a
# single well that is how many months it was compared over; for a mascon it is
# how many wells were averaged into it, which is what controls how much
# point-scale error the aggregation removes.
_MASCON_PANELS = _METRIC_PANELS[:5] + [
    dict(key='n_wells', label='Wells averaged', div=False,
         note='how much point error is averaged out',
         zero_based=False, clip=False, integer=True),
]


def per_mascon_metrics(obs: pd.DataFrame, min_wells: int = 10,
                       min_months: int = 8) -> pd.DataFrame:
    """
    The same five metrics, computed on each mascon's MEAN series.

    Not an average of per-well scores -- the wells are averaged first, month by
    month, and the metric is computed on that series. The difference is the whole
    point of the figure: per-well specific-yield error and point-to-pixel
    mismatch are largely independent between wells, so averaging cancels them,
    and what survives is error the downscaling is actually responsible for.

    Mascons carrying fewer than `min_wells` wells in a month lose that month, and
    mascons left with fewer than `min_months` months are dropped entirely. Nine
    of nineteen fall out at the defaults. That is a statement about where the
    CGWB network is, not about where the product works, and the figure marks
    those mascons as unscored rather than leaving them to read as bad.
    """
    series = aggregate(obs, 'mascon', min_wells)
    rows = []
    for unit, g in series.groupby('unit'):
        if len(g) < min_months:
            continue
        o = g.gws_anomaly_mm.to_numpy(dtype='float64')
        pr = g.gws_downscaled.to_numpy(dtype='float64')
        m = M.metrics(o, pr)
        ok = np.isfinite(o) & np.isfinite(pr)
        rows.append({'mascon': int(unit), 'n_months': int(len(g)),
                     'n_wells': int(g.n_wells.max()),
                     'RMSE': m['RMSE'], 'MAE': m['MAE'],
                     'MBE': float(np.mean(pr[ok] - o[ok])) if ok.any() else np.nan,
                     'R2': m['R2'], 'NSE': m['NSE']})
    return pd.DataFrame(rows).sort_values('mascon').reset_index(drop=True)


def _mascon_field(values: Dict[int, float], aux: Dict) -> np.ndarray:
    """
    Broadcast one value per mascon onto the 0.1 degree grid, clipped to the basin.

    The same construction as the leave-one-mascon-out skill map, deliberately:
    the two figures answer neighbouring questions and should not differ in how
    they are drawn. Painted on the FINE grid rather than the 0.5 degree GRACE
    grid so the outline is the basin rather than the ragged set of coarse cells
    the mascons happen to touch. Nothing is interpolated -- a mascon's value is
    constant over every fine cell it owns.
    """
    mid = np.asarray(aux['mascon_id'], dtype='float64')
    parent = np.asarray(aux['parent_era5_to_grace'])
    flat = mid.ravel()
    fine = np.where((parent >= 0) & (parent < flat.size),
                    flat[np.clip(parent, 0, flat.size - 1)], np.nan)
    out = np.full(fine.shape, np.nan)
    for m, val in values.items():
        out[fine == m] = val
    out[~(np.asarray(aux['basin_frac_era5']) > 0)] = np.nan
    return out


def plot_mascon_metric_maps(obs: pd.DataFrame, out_dir: str,
                            min_wells: int = 10) -> Optional[str]:
    """
    The per-well metric maps, redrawn at mascon scale.

    Read as a pair with `well_metric_maps.png`. Same metrics, same colour rules,
    same layout; the only change is the unit the metric is computed on. Where a
    mascon panel is markedly better than the wells inside it, the difference is
    point-scale error that averaging removed -- not a different model.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import downscale_features as F
    import gridded_config as cfg

    per_mascon = per_mascon_metrics(obs, min_wells)
    if per_mascon.empty:
        return None

    aux = F.load_aux()
    g = cfg.build_grids()['era5']
    lon, lat = g.lon_centers(), g.lat_centers()
    ext = [lon[0], lon[-1], min(lat[0], lat[-1]), max(lat[0], lat[-1])]
    flip = lat[0] < lat[-1]

    # In-basin mascons with no score: drawn in flat grey so "not measured here"
    # cannot be misread as "measured and bad".
    # In-basin mascons only, at basin_frac > 0.5 -- the same 19 that
    # leave-one-mascon-out holds out and that Figure 1 draws. Counting every id
    # present in the raster instead gave "10 of 35": the mascon grid spans a
    # bounding box, so most of those 35 are nowhere near this basin, and the
    # denominator implied a coverage problem twice as bad as the real one.
    # Twenty-two mascons touch the basin at all; the extra three are rim slivers
    # that are never cross-validated.
    scored = set(per_mascon.mascon.astype(int))
    mid_g = np.asarray(aux['mascon_id'])
    bfrac_g = np.asarray(aux['basin_frac_grace'])
    all_mascons = {int(m) for m in np.unique(mid_g[(mid_g >= 0) & (bfrac_g > 0.5)])}
    unscored = _mascon_field({m: 1.0 for m in all_mascons - scored}, aux)

    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.0), layout='constrained')
    try:
        basin = cfg.load_basin()
    except Exception:  # noqa: BLE001 - the patches are the content
        basin = None

    for letter, ax, spec in zip('abcdef', axes.ravel(), _MASCON_PANELS):
        v = per_mascon[spec['key']].to_numpy(dtype='float64')
        field = _mascon_field(dict(zip(per_mascon.mascon.astype(int), v)), aux)
        norm, cmap, extend = _scale_for(spec, v)

        shown = (lambda a: a[::-1] if flip else a)
        # Solid light grey, not a faint wash. At alpha 0.18 this was
        # indistinguishable from the white page, so an unscored mascon read as
        # empty space -- precisely the confusion the underlay exists to prevent.
        ax.imshow(shown(np.where(np.isfinite(unscored), 0.22, np.nan)),
                  extent=ext, origin='upper', cmap='Greys', vmin=0, vmax=1,
                  zorder=1, interpolation='nearest')
        im = ax.imshow(shown(field), extent=ext, origin='upper', cmap=cmap,
                       norm=norm, zorder=2, interpolation='nearest')
        if basin is not None:
            basin.boundary.plot(ax=ax, color=SCI_INK, linewidth=0.6, zorder=3)

        cax = ax.inset_axes([1.02, 0.0, 0.022, 1.0], transform=ax.transAxes)
        cb = fig.colorbar(im, cax=cax, extend=extend)
        cb.ax.tick_params(labelsize=7, colors=SCI_INK)
        if spec['integer']:
            cb.locator = MaxNLocator(integer=True)
            cb.update_ticks()
        _style_axis(ax)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Longitude (°E)', color=SCI_INK, fontsize=8)
        ax.set_ylabel('Latitude (°N)', color=SCI_INK, fontsize=8)
        ax.set_title(f'({letter}) {spec["label"]}', fontweight='bold',
                     color=SCI_INK, loc='left', fontsize=10, pad=14)
        ax.text(0.0, 1.015, spec['note'], transform=ax.transAxes,
                fontsize=7.5, color=SCI_MUTED, ha='left', va='bottom')

    n_un = len(all_mascons - scored)
    fig.suptitle(f'Product skill against CGWB wells, aggregated to '
                 f'{len(per_mascon)} of {len(all_mascons)} GRACE mascons '
                 f'(grey: too few wells to score)',
                 fontweight='bold', color=SCI_INK, fontsize=13, x=0.006, ha='left')

    p = os.path.join(out_dir, 'mascon_metric_maps.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    per_mascon.to_csv(os.path.join(out_dir, 'well_metrics_per_mascon.csv'),
                      index=False)
    figure_captions.record(
        p,
        'The per-well metric maps of well_metric_maps.png, recomputed on each '
        "mascon's MEAN well series rather than well by well — the wells are "
        'averaged first, month by month, and the metric is taken on that series. '
        'Per-well specific-yield error and point-to-pixel mismatch are largely '
        'independent between wells, so averaging cancels them and what survives '
        'is error the downscaling is responsible for; the two figures should be '
        f'read as a pair. {n_un} of {len(all_mascons)} mascons carry too few '
        f'wells to score at the {min_wells}-well threshold and are drawn flat '
        'grey — that marks where the CGWB network is, not where the product '
        'fails. Colour rules, ramps and clipping are identical to the per-well '
        'figure so a change between them is a change in skill and not in scale. '
        'Values are constant within a mascon; nothing is interpolated, and the '
        'basin outline rather than the coarse mascon footprint is used only so '
        'the plate matches the other maps.')
    _report_mascon_summary(per_mascon, len(all_mascons))
    return p


def _report_mascon_summary(d: pd.DataFrame, n_total: int) -> None:
    """Numbers for the text; the figure carries no caption."""
    pos = int((d.NSE > 0).sum())
    print(f'\n  per-mascon skill ({len(d)} of {n_total} mascons scored)')
    print(f'    median  R2 {d.R2.median():.2f} | RMSE {d.RMSE.median():.0f} mm | '
          f'MAE {d.MAE.median():.0f} mm | MBE {d.MBE.median():+.0f} mm | '
          f'NSE {d.NSE.median():.2f}')
    print(f'    NSE > 0 at {pos}/{len(d)} mascons ({100 * pos / len(d):.0f}%)')
    worst = d.loc[d.RMSE.idxmax()]
    print(f'    worst: mascon {int(worst.mascon)} RMSE {worst.RMSE:.0f} mm on '
          f'{int(worst.n_wells)} wells; best: mascon '
          f'{int(d.loc[d.RMSE.idxmin()].mascon)} RMSE {d.RMSE.min():.0f} mm')


def _report_metric_summary(per_well: pd.DataFrame) -> None:
    """
    Print the readings the figure used to carry, for the manuscript to quote.

    These belong in the text, not lettered onto a panel, but they should not
    simply disappear either -- each is a number a reader of the maps would
    otherwise have to eyeball off a colour ramp.
    """
    n = len(per_well)
    pos = int((per_well.NSE > 0).sum())
    nw = per_well[(per_well.lat > 27) & (per_well.lon < 79)]
    cp = per_well[(per_well.lat < 26) & per_well.lon.between(77, 84)]
    print(f'\n  per-well skill ({n} wells, '
          f'{int(per_well.n_months.min())}-{int(per_well.n_months.max())} months each)')
    print(f'    median  R2 {per_well.R2.median():.2f} | '
          f'RMSE {per_well.RMSE.median():.0f} mm | '
          f'MAE {per_well.MAE.median():.0f} mm | '
          f'MBE {per_well.MBE.median():+.0f} mm | '
          f'NSE {per_well.NSE.median():.2f}')
    print(f'    NSE > 0 at {pos}/{n} wells ({100 * pos / n:.0f}%); at the rest the '
          f"product tracks a well less closely than that well's own mean would")
    # R2 - NSE is identically (sd_pred/sd_obs - r)^2 + (bias/sd_obs)^2, a sum of
    # squares, so R2 >= NSE always holds and the GAP is the amplitude-and-bias
    # error expressed in units of the well's own variability. Quoting the gap is
    # meaningful; quoting "R2 beats NSE" as a finding is not.
    print(f'    median R2 - NSE gap {(per_well.R2 - per_well.NSE).median():.2f} '
          f"(amplitude+bias error, in units of the well's own sd)")
    print(f'    bias is spatially organised, so the median MBE understates it: '
          f'northwest {int((nw.MBE < 0).sum())}/{len(nw)} wells low '
          f'({nw.MBE.median():+.0f} mm), central plain '
          f'{100 * (cp.MBE > 0).mean():.0f}% high ({cp.MBE.median():+.0f} mm)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--product', default=None)
    ap.add_argument('--stores', default=V.DEFAULT_STORES, choices=sorted(V.STORE_SETS))
    ap.add_argument('--min-wells', type=int, default=10,
                    help='Minimum wells contributing to an aggregated month.')
    ap.add_argument('--min-months', type=int, default=8)
    ap.add_argument('--matches', default=None,
                    help='Reuse an existing well_validation_matches.csv instead '
                         'of re-matching against the product.')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or M.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print('Well comparison at three scales: basin, mascon, per-well\n')
    if args.matches:
        obs = pd.read_csv(args.matches)
        print(f'  reusing {args.matches}')
    else:
        obs, info = V.match_wells(args.product, verbose=True, stores=args.stores)
        print(f'  product: {os.path.basename(str(info["product"]))} | '
              f'stores: {args.stores}')
    print(f'  {len(obs):,} well-months, {obs.well_id.nunique()} wells, '
          f'{obs.mascon.nunique()} mascons\n')

    tables: List[pd.DataFrame] = []
    figures = []
    for level in ('basin', 'mascon', 'well'):
        s = aggregate(obs, level, args.min_wells)
        sc = score(s, level, args.min_months)
        tables.append(sc)
        if sc.empty:
            print(f'{level:7s} no unit met the thresholds')
            continue
        print(f'{level.upper()}  ({len(sc)} unit(s), '
              f'{"per-well series" if level == "well" else f"min {args.min_wells} wells/month"})')
        print(f'   median r   downscaled {sc.downscaled_r.median():.3f}   '
              f'bilinear {sc.bilinear_r.median():.3f}   '
              f'gap {sc.gap_r.median():+.3f}')
        print(f'   median RMSE downscaled {sc.downscaled_RMSE.median():7.1f}   '
              f'bilinear {sc.bilinear_RMSE.median():7.1f}   '
              f'gap {sc.gap_RMSE.median():+.1f} mm')
        if level == 'basin':
            figures.append(plot_basin(s, sc, out_dir))
        elif level == 'mascon':
            figures.append(plot_mascons(s, sc, out_dir))
        else:
            figures.append(plot_well_distribution(sc, out_dir))
        print()

    # The per-well residual map is panel (a) of this figure, so it is no longer
    # drawn on its own -- one figure, six metrics, rather than a standalone map
    # of a quantity the panel grid already carries.
    mmap = plot_metric_maps(obs, out_dir)
    if mmap:
        figures.append(mmap)
    # The same metrics at mascon scale. Read as a pair with the per-well figure:
    # the gap between them is point-scale error that aggregation removes.
    smap = plot_mascon_metric_maps(obs, out_dir, args.min_wells)
    if smap:
        figures.append(smap)

    allt = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    p = os.path.join(out_dir, 'well_validation_by_scale.csv')
    allt.to_csv(p, index=False)
    caps = figure_captions.write_index(
        out_dir, title='Figure captions — well comparison')
    print('written:')
    for f in [p] + [f for f in figures if f]:
        print(f'  {f}')
    if caps:
        print(f'  {caps}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
