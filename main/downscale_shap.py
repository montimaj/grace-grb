"""
TreeSHAP attribution for the gridded downscaling model, grouped by feature family.

WHAT IS BEING EXPLAINED, AND WHAT IS NOT
----------------------------------------
`downscale_model.decompose_target` splits each mascon's GRACE series into a
level-plus-linear-trend background and an anomaly about it. The MODEL IS FITTED
ON THE ANOMALY ONLY; the level and the trend are taken from GRACE per mascon and
never pass through the model at all. Everything in this file therefore describes
the anomaly, in millimetres of anomaly, and nothing else.

Three consequences, and they are not pedantic:

* These numbers say NOTHING about the secular trend. Leave-one-mascon-out trend
  R^2 is ~0.11 (see `decompose_target`), and the trend is not modelled anyway.
* They say NOTHING about total storage. Total storage in the shipped product is
  background + anomaly + the mass-conservation correction, and two of those three
  terms come from GRACE rather than from the predictors.
* No statement about groundwater depletion can be supported from this file. The
  Ganga's decline lives entirely in the background term. A feature that ranks
  highly here ranks highly for the SEASONAL-TO-INTERANNUAL anomaly.

The attribution is also IN-SAMPLE. `downscale_model` fits the deployed model on
every observed row, so this describes what the fitted function does on the data it
saw. It is a description of the model, not a measure of skill; skill is the
leave-one-mascon-out holdout and the CGWB well comparison.

WHY THIS IS A NEW MODULE AND NOT A PORT
---------------------------------------
The basin-scale path (`utils.compute_shap_values`, used by holdout_random /
holdout_spatial / holdout_temporal) reported that antecedent soil moisture
carried about 80% of the explanatory power, and read that as storage memory.
Neither leg of that result survives:

* the soil moisture was GLDAS 2.2 CLSM, which ASSIMILATES GRACE. A large share
  attributed to it is partly the target predicting itself.
* the `GWSA` column of that design matrix was degenerate.

The predictor set is now ERA5-Land (`gridded_config.PREDICTORS`), the column is
renamed `runoff_anom`, and the withdrawn figure is not evidence for anything on
this matrix. It is quoted in the output only so a reader comparing against the
earlier manuscript sees explicitly that it has been dropped, not silently
restated.

GROUPING, AND WHY A PER-COLUMN RANKING IS THE WRONG UNIT
--------------------------------------------------------
The 78 columns are not 78 signals. Each dynamic predictor contributes a
same-month column, six antecedent boxcar means, two neighbourhood context means,
and (for `ppt`/`et`) four exponential APIs and two climatology columns. A tree
splits on one column at a time, so one physical signal has its credit divided
across up to fifteen highly correlated columns, and a per-column table then reads
as if that signal were minor.

Grouping is done on the SIGNED values, per row, before taking the absolute value:

    family contribution to row i  =  sum over the family's columns of phi_ij

which is exact, because SHAP values are additive per row. Summing per-column
mean|phi| instead would double-count credit that two collinear columns pass back
and forth with opposite signs. The module reports both totals; the gap between
them is printed as `internal cancellation` and is itself a collinearity readout.

Two partitions are reported, plus the raw per-column table:

* by SOURCE FIELD - soil moisture, ET, precipitation, surface runoff, the runoff
  anomaly, the water-balance API, five groups of static covariates (one per
  source dataset in `gridded_config.STATIC_VARS`), and the seasonal harmonics.
* by TERM TYPE - same month, antecedent boxcar, exponential API, neighbourhood
  context, per-pixel climatology, static, seasonal. This is the partition that
  addresses a memory claim, because "memory" is antecedent plus API and cuts
  across every source field.

THE MAIN CAVEAT: COLLINEARITY
-----------------------------
TreeSHAP in its default `tree_path_dependent` mode splits credit between
correlated columns according to how the trees happened to be built, not according
to any property of the data. This matrix is severely collinear -- consecutive
antecedent windows correlate 0.59-0.98, the 12- and 24-month windows sit at
0.96-0.98, and PCA on the memory block reaches 97.2% of variance in 10 components
(measured in `downscale_ablation`). `runoff_anom` is `runoff_surface` minus its
own per-pixel climatological mean to a residual of 1.7e-6, with both parents in
the matrix.

So: the ORDER of families that differ by a few percent is not identifiable, and
no ranking printed here should be read as though it were. Grouping repairs part
of the problem (it reunites the columns of one field) and cannot repair the rest
(fields are correlated with each other). The bootstrap intervals cover mascon
sampling, not this; a family whose interval excludes another's is still not
separated from it if the two are collinear. This is stated in the printed output
as well as here, because it is the binding limitation on every number produced.

COST, MEASURED
--------------
TreeSHAP is O(trees x leaves x depth^2) per row, so the affordable sample depends
entirely on the model. Measured on this machine on a synthetic matrix of the
production shape (83,309 x 78), fitting the documented `DEFAULT_PARAMS`:

    xgboost        600 trees, max_depth 8       1.8 ms/row    -> ~2.5 min for all
    random_forest  300 trees, min_samples_leaf 5,
                   ~8,400 leaves, depth 27-38   647 ms/row    -> ~15 h for all

i.e. an unpruned forest is ~350x more expensive per row than a depth-capped GBM.
Synthetic Gaussian columns grow deeper trees than the real matrix does, so the
forest figure is an upper bound, but the order of magnitude is structural and not
an artefact.

Rather than hard-code a sample size that is right for one model and absurd for the
other, the module TIMES A PILOT at run time, prints the measured rate, and takes
the whole observed set if it fits inside `--max-minutes`. If it does not, rows are
sampled stratified by mascon and the reduction is stated. Whether the sample was
large enough is then MEASURED, not asserted: `split-half agreement` recomputes the
family shares on two disjoint halves of the sampled rows and reports the largest
disagreement in percentage points. Read that number before reading any ranking.

LANGUAGE
--------
Reviewers R3-37 and R3-55 asked for association language throughout. Findings are
emitted through `_finding`, which refuses to print a sentence containing a causal
verb (see `CAUSAL_TERMS`). SHAP measures how a fitted function's output moves with
its inputs. It does not identify mechanism, and on a matrix this collinear it does
not even identify which of two correlated columns the function is using.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import downscale_features as F
import downscale_model as M
import figure_captions
import gridded_config as cfg
from plot_style import (DPI, SCI_BLUE, SCI_COLORS, SCI_GRID, SCI_INK, SCI_MUTED,
                        pretty_model)

RESULTS_DIR = M.RESULTS_DIR
FIG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'Results', 'figures'))

# Repeated verbatim in the console output, the CSVs and the figure. A reader who
# meets one of these tables detached from the others must still be told what the
# model was fitted on.
SCOPE_NOTE = ('Attribution is for the mascon-scale TWSA ANOMALY only. Level and '
              'trend are taken from GRACE per mascon and never enter the model, '
              'so nothing here bears on the depletion trend or on total storage.')

COLLINEARITY_NOTE = ('TreeSHAP splits credit between correlated columns by how the '
                     'trees were built, not by any property of the data. Antecedent '
                     'windows in this matrix correlate up to 0.98, so the ordering of '
                     'families separated by a few percent is not identifiable.')

# Quoted so that a reader comparing against the reviewed manuscript sees the claim
# withdrawn rather than quietly absent. NOT run through `_finding` -- it is a
# quotation of a superseded result and reproducing its wording is the point.
WITHDRAWN_CLAIM = (
    'WITHDRAWN, quoted not endorsed -- the reviewed basin-scale manuscript reported: '
    '"antecedent SMS is the primary predictor, about 80% of the total explanatory '
    'power, providing a mechanistic storage-memory explanation". That rested on GLDAS '
    '2.2 CLSM soil moisture, which assimilates GRACE, and on a degenerate GWSA column. '
    'Both are gone from this design matrix and the figure does not carry over.')

# Refused inside `_finding`. Word-boundary matched, so "context" survives
# "contexts" but "control" does not survive "controls".
CAUSAL_TERMS: Tuple[str, ...] = (
    'drive', 'drives', 'driven', 'driver', 'drivers', 'driving',
    'control', 'controls', 'controlled', 'controlling',
    'cause', 'causes', 'caused', 'causing', 'causal',
    'explain', 'explains', 'explained', 'explaining', 'explanatory',
    'determine', 'determines', 'determined', 'determinant',
    'govern', 'governs', 'governed', 'mechanism', 'mechanistic',
    'influence', 'influences', 'influenced', 'impact', 'impacts',
    'responsible', 'because', 'due to', 'leads to', 'gives rise to',
)
_CAUSAL_RE = re.compile(
    r'\b(' + '|'.join(t.replace(' ', r'\s+') for t in CAUSAL_TERMS) + r')\b',
    re.IGNORECASE)

# Static covariates grouped by SOURCE DATASET. Each group shares one `asset` in
# gridded_config.STATIC_VARS; `_check_static_groups` verifies that at run time
# rather than trusting this table to stay in step. Grouping is by source because
# that is the unit that shares a resolution, a production chain and an error
# structure -- elevation and its sub-grid s.d. are one measurement, not two
# independent covariates.
STATIC_FAMILIES: Dict[str, Tuple[str, ...]] = {
    'Water table depth': ('wtd',),
    'Soil hydraulic properties': ('awc', 'root_depth'),
    'Terrain': ('elevation', 'elevation_std'),
    'Drainage position': ('upa_log', 'hnd'),
    'Cropland fraction': ('crop_irrigated', 'crop_rainfed'),
}

SEASON_COLUMNS: Tuple[str, ...] = ('sin1', 'cos1', 'sin2', 'cos2')

_FIELD_LABEL: Dict[str, str] = {
    'sms': 'Soil moisture',
    'et': 'Evapotranspiration',
    'ppt': 'Precipitation',
    'runoff_surface': 'Surface runoff',
    'runoff_anom': 'Surface runoff anomaly',
    'wb': 'Water balance (P minus ET minus Q)',
}

_STATIC_LABEL: Dict[str, str] = {
    'wtd': 'Water table depth',
    'awc': 'Available water capacity',
    'root_depth': 'Obstacle-to-roots depth',
    'elevation': 'Elevation',
    'elevation_std': 'Sub-grid elevation s.d.',
    'upa_log': 'Log upstream drainage area',
    'hnd': 'Height above nearest drainage',
    'crop_irrigated': 'Irrigated cropland fraction',
    'crop_rainfed': 'Rainfed cropland fraction',
}

_TERM_LABEL: Dict[str, str] = {
    'same_month': 'Same month',
    'antecedent': 'Antecedent means',
    'api': 'Exponential memory (API)',
    'context': 'Neighbourhood context',
    'climatology': 'Per-pixel climatology',
    'static': 'Static and land cover',
    'season': 'Seasonal harmonics',
    'unclassified': 'Unclassified',
}

# Term types that are memory: a weighted average of the predictor over months
# BEFORE the target month. Both are shifted by one step in downscale_features, so
# neither can contain its own month.
MEMORY_TERMS: Tuple[str, ...] = ('antecedent', 'api')

# Column-name suffixes, mirroring the construction in
# downscale_features.build_features. Anything a dynamic field's suffix fails to
# match falls to `unclassified` and is named in a warning; a silent catch-all
# bucket is exactly how a grouping ends up understating a signal.
_SUFFIX_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r'^$', 'same_month'),
    (r'^_ante(?P<p>\d+)m$', 'antecedent'),
    (r'^_api(?P<p>\d+)m$', 'api'),
    (r'^_ctx(?P<p>[\d.]+)d$', 'context'),
    (r'^_clim_(?P<p>mean|std)$', 'climatology'),
)


# --------------------------------------------------------------------------
# Wording
# --------------------------------------------------------------------------

def _finding(text: str) -> str:
    """
    Return `text`, or raise if it makes a causal claim.

    SHAP reports how a fitted function's output co-varies with its inputs. On a
    matrix where antecedent windows correlate at 0.98 it does not even identify
    WHICH of two columns the function is using, let alone why. Every sentence this
    module emits as a finding passes through here so that the constraint is
    enforced rather than merely intended.
    """
    hit = _CAUSAL_RE.search(text)
    if hit:
        raise ValueError(
            f'causal wording {hit.group(0)!r} in a SHAP finding: {text!r}. '
            'Use association language (associated with, attributed share, '
            'co-varies with) -- see CAUSAL_TERMS and reviewers R3-37, R3-55.')
    return text


# --------------------------------------------------------------------------
# Column parsing and grouping
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Column:
    """One design-matrix column, resolved to its source field and term type."""

    name: str
    field: str          # 'sms' ... 'wb', a covariate name, 'season', 'unclassified'
    term: str           # key into _TERM_LABEL
    parameter: str      # window / tau / radius / 'mean' / 'std', '' where absent
    field_family: str   # display label for the source-field partition
    term_family: str    # display label for the term-type partition
    display: str        # human-readable column name, no underscores


def _check_static_groups(verbose: bool = False) -> None:
    """
    Verify STATIC_FAMILIES really does group covariates by source dataset.

    The grouping is written out by hand above because a family named after an
    Earth Engine asset path is unreadable, but hand-written tables go stale. This
    reads `gridded_config.STATIC_VARS` and warns if a family has come to span two
    source assets, or if an active covariate has appeared that no family claims.
    """
    for label, members in STATIC_FAMILIES.items():
        assets = {cfg.STATIC_BY_NAME[m].asset
                  for m in members if m in cfg.STATIC_BY_NAME}
        if len(assets) > 1:
            print(f'  WARNING: static family {label!r} now spans {len(assets)} '
                  f'source assets {sorted(assets)}; it is no longer one dataset '
                  'and the grouping overstates it')
    claimed = {m for members in STATIC_FAMILIES.values() for m in members}
    orphan = [c for c in cfg.resolve_active_covariates() if c not in claimed]
    if orphan:
        print(f'  WARNING: covariates {orphan} are in no STATIC_FAMILIES group; '
              'each is treated as its own family, which will understate it if it '
              'shares a source dataset with another covariate')
    if verbose:
        print(f'  static families: {len(STATIC_FAMILIES)} groups over '
              f'{len(claimed)} covariates')


def _static_family(name: str) -> str:
    for label, members in STATIC_FAMILIES.items():
        if name in members:
            return label
    return _STATIC_LABEL.get(name, name.replace('_', ' '))


def _dynamic_fields() -> List[str]:
    """
    Source-field prefixes, longest first.

    Longest-first matters: a `split('_')[0]` would turn `runoff_surface_ante6m`
    into the field `runoff`, merging it with `runoff_anom` and inventing a family
    that does not exist. Two of the five predictors carry an underscore.
    """
    fields = list(cfg.PREDICTORS)
    if F.WATER_BALANCE_API:
        fields.append('wb')
    return sorted(set(fields), key=len, reverse=True)


def _pretty_column(field: str, term: str, parameter: str, name: str) -> str:
    """Display name for one column. No underscores, per the figure convention."""
    if term == 'static':
        return _STATIC_LABEL.get(name, name.replace('_', ' '))
    if term == 'season':
        fn = 'sine' if name.startswith('sin') else 'cosine'
        return f'Seasonal {fn}, {name[-1]} per year'
    label = _FIELD_LABEL.get(field, field.replace('_', ' '))
    if term == 'same_month':
        return f'{label}, same month'
    if term == 'antecedent':
        return f'{label}, antecedent {parameter} mo'
    if term == 'api':
        return f'{label}, API {parameter} mo'
    if term == 'context':
        return f'{label}, {parameter} deg context'
    if term == 'climatology':
        return f'{label}, climatological {"mean" if parameter == "mean" else "s.d."}'
    return name.replace('_', ' ')


def parse_column(name: str) -> Column:
    """
    Resolve one column name to (source field, term type).

    Order is exact matches first, prefix matching last, because `elevation` is a
    prefix of `elevation_std` and a prefix rule applied first would swallow it.
    """
    if name in SEASON_COLUMNS:
        return Column(name, 'season', 'season', '', 'Seasonal harmonics',
                      _TERM_LABEL['season'],
                      _pretty_column('season', 'season', '', name))
    if name in cfg.STATIC_BY_NAME:
        return Column(name, name, 'static', '', _static_family(name),
                      _TERM_LABEL['static'],
                      _pretty_column(name, 'static', '', name))

    for field in _dynamic_fields():
        if not name.startswith(field):
            continue
        suffix = name[len(field):]
        for pattern, term in _SUFFIX_PATTERNS:
            hit = re.match(pattern, suffix)
            if hit is None:
                continue
            parameter = hit.groupdict().get('p') or ''
            return Column(name, field, term, parameter,
                          _FIELD_LABEL.get(field, field.replace('_', ' ')),
                          _TERM_LABEL[term],
                          _pretty_column(field, term, parameter, name))

    return Column(name, 'unclassified', 'unclassified', '', 'Unclassified',
                  _TERM_LABEL['unclassified'], name.replace('_', ' '))


def column_table(feature_names: Sequence[str], verbose: bool = True) -> pd.DataFrame:
    """
    One row per design-matrix column, with both family assignments.

    Term-family labels are completed here rather than hard-coded, so the printed
    "Antecedent means (1-24 mo)" reflects `downscale_features.ANTECEDENT_MONTHS`
    as it actually is and cannot go stale against it.
    """
    cols = [parse_column(n) for n in feature_names]
    tab = pd.DataFrame([c.__dict__ for c in cols])

    for term in ('antecedent', 'api', 'context'):
        sel = tab.term == term
        if not sel.any():
            continue
        vals = sorted({float(p) for p in tab.loc[sel, 'parameter'] if p})
        if not vals:
            continue
        unit = 'deg' if term == 'context' else 'mo'
        span = (f'{vals[0]:g}' if len(vals) == 1
                else f'{vals[0]:g} to {vals[-1]:g}')
        tab.loc[sel, 'term_family'] = f'{_TERM_LABEL[term]} ({span} {unit})'

    bad = tab.loc[tab.field == 'unclassified', 'name'].tolist()
    if bad:
        print(f'  WARNING: {len(bad)} column(s) matched no family rule and are '
              f'grouped as "Unclassified": {bad}. Family shares below are wrong '
              'in proportion to whatever these carry -- fix the rules in '
              '_SUFFIX_PATTERNS or STATIC_FAMILIES before quoting a share.')
    if verbose:
        print(f'  {len(tab)} columns -> {tab.field_family.nunique()} source-field '
              f'families, {tab.term_family.nunique()} term-type families')
        for fam, grp in tab.groupby('field_family', sort=True):
            print(f'    {fam:36s} {len(grp):3d} columns')
    return tab


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def stratified_sample(mascon: np.ndarray, n_target: int, seed: int = 20,
                      min_per_mascon: int = 200) -> np.ndarray:
    """
    Row indices sampled proportionally within each mascon.

    Mascons are the unit of independence here and they are very unequal -- the
    largest carries 9,534 observed rows and the smallest 227 (measured on the
    current cube). A plain uniform draw would leave the small mascons with a
    handful of rows each and make the mascon bootstrap below meaningless, so every
    mascon keeps at least `min_per_mascon` rows or all of them, whichever is fewer.

    That floor deliberately OVER-represents small mascons relative to the training
    distribution. The alternative under-represents them to the point of dropping
    them, which is worse for an interval whose whole purpose is to price mascon
    sampling.

    Where the budget is too small to honour the floor everywhere the allocation
    degrades towards equal shares and says so, and every mascon keeps at least one
    row even if that overshoots `n_target`. A mascon dropped entirely would be
    dropped from the bootstrap too, silently narrowing the interval.
    """
    rng = np.random.default_rng(seed)
    ids, counts = np.unique(mascon, return_counts=True)
    total = int(counts.sum())
    if n_target <= 0 or n_target >= total:
        return np.arange(total)

    floor = np.minimum(min_per_mascon, counts)
    if n_target < int(floor.sum()):
        print(f'  note: {n_target:,} rows cannot give all {len(ids)} mascons '
              f'{min_per_mascon} each; the allocation falls back towards equal '
              'shares, so small mascons are over-represented relative to training '
              'by more than the floor alone would do')

    alloc = np.maximum(np.rint(counts * n_target / total).astype(int), floor)
    alloc = np.minimum(alloc, counts)

    # Trim back to the target, taking only from mascons above the floor and in
    # proportion to how far above it they sit.
    excess = int(alloc.sum()) - n_target
    if excess > 0:
        room = np.maximum(alloc - floor, 0)
        if room.sum() > 0:
            cut = np.floor(room * excess / room.sum()).astype(int)
            alloc -= np.minimum(cut, room)
        while int(alloc.sum()) > n_target and alloc.max() > 1:
            alloc[int(np.argmax(alloc))] -= 1

    picks = [rng.choice(np.flatnonzero(mascon == m), size=int(k), replace=False)
             for m, k in zip(ids, alloc) if k > 0]
    return np.sort(np.concatenate(picks))


def shap_rate(explainer, X: np.ndarray, seed: int = 20) -> float:
    """
    Seconds per row for this explainer and this model, measured not assumed.

    Two stages: a handful of rows to find the order of magnitude, then a larger
    batch only if the first was fast enough that a larger one is cheap. TreeSHAP
    cost per row spans nearly three orders of magnitude between a depth-8 GBM and
    an unpruned forest (see the module docstring), so a fixed pilot size is either
    uselessly noisy or unaffordable depending on which model was loaded.
    """
    rng = np.random.default_rng(seed)
    pilot = X[rng.choice(len(X), size=min(8, len(X)), replace=False)]
    t0 = time.time()
    explainer.shap_values(pilot, check_additivity=False)
    rate = (time.time() - t0) / len(pilot)

    if rate * 256 < 10.0 and len(X) >= 256:
        pilot = X[rng.choice(len(X), size=256, replace=False)]
        t0 = time.time()
        explainer.shap_values(pilot, check_additivity=False)
        rate = (time.time() - t0) / len(pilot)
    return rate


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def family_matrix(shap_values: np.ndarray, table: pd.DataFrame,
                  key: str) -> Tuple[List[str], np.ndarray]:
    """
    Per-row family contributions, summed SIGNED before any absolute value.

    SHAP values are additive within a row -- they sum to the prediction minus the
    base value -- so the sum over a family's columns is exactly that family's
    contribution to that row's prediction. Taking |.| per column first and summing
    afterwards is NOT the family's contribution: it counts credit twice whenever
    two collinear columns carry it with opposite signs, which on this matrix is
    the normal case rather than the exception.

    Returns (family labels, matrix of shape (n_rows, n_families)).
    """
    labels = sorted(table[key].unique())
    out = np.empty((shap_values.shape[0], len(labels)), dtype='float64')
    for j, lab in enumerate(labels):
        cols = np.flatnonzero((table[key] == lab).to_numpy())
        out[:, j] = shap_values[:, cols].sum(axis=1)
    return labels, out


def _per_mascon_means(G: np.ndarray, mascon: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Row counts and mean |contribution| per mascon, the bootstrap's raw material."""
    ids = np.unique(mascon)
    counts = np.empty(len(ids), dtype='float64')
    means = np.empty((len(ids), G.shape[1]), dtype='float64')
    for i, m in enumerate(ids):
        sel = mascon == m
        counts[i] = sel.sum()
        means[i] = np.abs(G[sel]).mean(axis=0)
    return counts, means


def share_ci(G: np.ndarray, mascon: np.ndarray, n_boot: int = 2000,
             seed: int = 20) -> pd.DataFrame:
    """
    Family shares of total mean absolute attribution, with a mascon bootstrap CI.

    The resampling unit is the MASCON, not the row, for the same reason
    `downscale_model` gives: ~83,000 rows come from 19 independent 3-degree
    mascons, and a row-wise interval would count pixels as evidence and come back
    far too narrow. These intervals are wide. That width is what GRACE resolving
    about twenty numbers over this basin actually buys.

    Because whole mascons are drawn, the pooled mean over a resample is the
    count-weighted mean of the per-mascon means, so the bootstrap is a small
    matrix product rather than 2,000 passes over the SHAP array.

    The interval prices MASCON SAMPLING ONLY. It does not price the collinearity
    in the design matrix, which is the larger source of uncertainty on any
    ordering and which no resampling scheme can price.
    """
    counts, means = _per_mascon_means(G, mascon)
    point = (counts[:, None] * means).sum(axis=0) / counts.sum()
    shares = 100.0 * point / point.sum()

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    w = counts[draws]                                    # (n_boot, n_mascons)
    pooled = np.einsum('bk,bkf->bf', w, means[draws]) / w.sum(axis=1)[:, None]
    boot = 100.0 * pooled / pooled.sum(axis=1)[:, None]

    return pd.DataFrame({
        'mean_abs_shap_mm': point,
        'share_pct': shares,
        'share_ci_lo_pct': np.percentile(boot, 2.5, axis=0),
        'share_ci_hi_pct': np.percentile(boot, 97.5, axis=0),
    })


def split_half_shares(G: np.ndarray, mascon: np.ndarray,
                      seed: int = 20) -> Tuple[np.ndarray, float]:
    """
    Largest disagreement in family share between two disjoint halves of the rows.

    This is the answer to "was the sample big enough", and it is measured rather
    than argued. Rows are halved WITHIN each mascon so both halves see the same
    mascons; what is left varying is row sampling alone, which is exactly the term
    a smaller sample inflates. Mascon sampling is priced separately by `share_ci`.

    Returns (per-family absolute difference in percentage points, the maximum).
    """
    rng = np.random.default_rng(seed)
    left = np.zeros(len(mascon), dtype=bool)
    for m in np.unique(mascon):
        idx = np.flatnonzero(mascon == m)
        left[rng.permutation(idx)[:len(idx) // 2]] = True

    def shares(sel: np.ndarray) -> np.ndarray:
        v = np.abs(G[sel]).mean(axis=0)
        return 100.0 * v / v.sum()

    if left.sum() == 0 or (~left).sum() == 0:
        return np.zeros(G.shape[1]), float('nan')
    d = np.abs(shares(left) - shares(~left))
    return d, float(d.max())


def correlation_diagnostics(X: np.ndarray, table: pd.DataFrame,
                            key: str) -> pd.DataFrame:
    """
    Worst pairwise |Pearson r| inside each family, on the rows actually explained.

    Quoted from this matrix rather than from `downscale_ablation`'s published
    numbers so the caveat describes the columns that were explained, not a
    different build. A family whose members correlate at 0.98 has a credit split
    among them that TreeSHAP cannot resolve; the family TOTAL is still meaningful,
    which is the argument for grouping in the first place.

    Rows carrying any NaN are dropped first: `np.corrcoef` propagates a single NaN
    across the whole matrix, which would silently return 0.0 everywhere and hide
    the very collinearity this exists to report.
    """
    complete = np.isfinite(X).all(axis=1)
    if complete.sum() < 3:
        print('  WARNING: fewer than 3 complete rows; within-family correlations '
              'could not be computed and the collinearity column is empty')
        R = np.zeros((X.shape[1], X.shape[1]))
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            R = np.abs(np.corrcoef(X[complete], rowvar=False))
        # A constant column has no correlation defined. Zero is the right filler:
        # it cannot be collinear with anything, and it is reported as such.
        R[~np.isfinite(R)] = 0.0
    np.fill_diagonal(R, 0.0)

    rows = []
    for lab in sorted(table[key].unique()):
        cols = np.flatnonzero((table[key] == lab).to_numpy())
        sub = R[np.ix_(cols, cols)]
        rows.append({key: lab, 'n_columns': len(cols),
                     'max_within_family_abs_r': float(sub.max()) if len(cols) > 1
                     else float('nan')})
    return pd.DataFrame(rows)


def column_worst_correlate(X: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """
    Each column's strongest correlate anywhere in the matrix, and which one it is.

    The per-column table needs this next to it or it will be over-read. A column
    at 0.98 with a neighbour did not out-rank that neighbour on any property of
    the data -- the trees chose one of the pair, and TreeSHAP then reports the
    choice the trees made.
    """
    complete = np.isfinite(X).all(axis=1)
    if complete.sum() < 3:
        return pd.DataFrame({'name': list(names),
                             'max_abs_r_any_column': np.nan,
                             'most_correlated_with': ''})
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.abs(np.corrcoef(X[complete], rowvar=False))
    R[~np.isfinite(R)] = 0.0
    np.fill_diagonal(R, 0.0)
    j = R.argmax(axis=1)
    return pd.DataFrame({'name': list(names),
                         'max_abs_r_any_column': R[np.arange(len(names)), j],
                         'most_correlated_with': [names[k] for k in j]})


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def plot(field: pd.DataFrame, term: pd.DataFrame, columns: pd.DataFrame,
         model_name: str, out_path: str, top_n: int = 15) -> None:
    """Two family partitions plus the per-column table they are meant to replace."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.4, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.62)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    def bars(ax, frame, label_col, colours, title):
        d = frame.sort_values('share_pct').reset_index(drop=True)
        y = np.arange(len(d))
        lo = np.clip(d.share_pct - d.share_ci_lo_pct, 0, None)
        hi = np.clip(d.share_ci_hi_pct - d.share_pct, 0, None)
        ax.barh(y, d.share_pct, color=colours, height=0.72, zorder=3)
        ax.errorbar(d.share_pct, y, xerr=np.vstack([lo, hi]), fmt='none',
                    ecolor=SCI_INK, elinewidth=1.0, capsize=2.5, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(d[label_col], fontsize=7.5, color=SCI_INK)
        ax.set_xlabel('Share of total mean absolute attribution (%)', color=SCI_INK,
                      fontsize=8.5)
        ax.set_title(title, fontsize=9.5, color=SCI_INK, loc='left')
        return d

    bars(ax_f, field, 'family', SCI_BLUE,
         'a. By source field, signed contributions summed within family')
    bars(ax_t, term, 'family', SCI_BLUE,
         'b. By term type, the partition a memory claim needs')

    # Panel (c) is coloured by source field so the credit-splitting is visible:
    # one field's columns appear in one colour, repeatedly, down the ranking.
    top = columns.nlargest(top_n, 'share_pct')
    order = list(top.groupby('field_family').share_pct.sum()
                 .sort_values(ascending=False).index)
    cmap = {fam: SCI_COLORS[i % len(SCI_COLORS)] for i, fam in enumerate(order)}
    bars(ax_c, top, 'display', [cmap[f] for f in
                                top.sort_values('share_pct').field_family],
         f'c. Top {min(top_n, len(top))} individual columns, for comparison -- '
         'one field is split across several bars')
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[f]) for f in order]
    ax_c.legend(handles, order, frameon=False, fontsize=7.5, loc='lower right')

    for ax in (ax_f, ax_t, ax_c):
        ax.grid(True, axis='x', color=SCI_GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(SCI_MUTED)
        ax.tick_params(colors=SCI_MUTED, labelsize=7.5)

    fig.suptitle(f'TWSA anomaly attribution, {pretty_model(model_name)} '
                 '(TreeSHAP, association only)',
                 fontsize=11, color=SCI_INK, x=0.012, ha='left', y=0.997)
    # Caption recorded, not drawn -- see figure_captions.
    figure_captions.record(
        out_path,
        SCOPE_NOTE + ' ' + COLLINEARITY_NOTE + ' Whiskers are 95% intervals '
        'from a bootstrap over mascons; they price mascon sampling only, not '
        'the collinearity.')
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  written: {out_path}')


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_fitted(model_name: str, path: Optional[str] = None) -> Dict[str, object]:
    """
    The model `downscale_model` persisted, with its feature names.

    Read rather than refitted. A refit here would be a DIFFERENT model from the one
    that produced the shipped product -- same seed and same cube, but nothing
    guarantees the cube has not moved since -- and an attribution for a model that
    was never shipped is worse than none.
    """
    import joblib

    p = path or os.path.join(RESULTS_DIR, f'model_{model_name}.joblib')
    if not os.path.exists(p):
        raise SystemExit(
            f'{p} not found. Run:  python downscale_model.py --model {model_name}\n'
            'This module never refits; it explains the model that produced the '
            'shipped product.')
    return joblib.load(p)


def tree_explainer(model):
    """
    A TreeExplainer, or a refusal naming the reason.

    The MLP is a Pipeline (impute -> scale -> MLPRegressor) and has no trees to
    walk. KernelExplainer would run on it, but it is a sampling approximation with
    a wholly different bias and its numbers would not be comparable to the tree
    models' in the same table, so it is not offered here as though it were.
    """
    import shap

    if hasattr(model, 'named_steps'):
        raise SystemExit(
            f'{type(model).__name__} is a Pipeline, not a tree ensemble -- TreeSHAP '
            'does not apply. Only the tree models (random_forest, xgboost, '
            'lightgbm, xgboost_rf) can be explained by this module.')
    return shap.TreeExplainer(model)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='xgboost',
                    choices=['random_forest', 'xgboost', 'lightgbm', 'xgboost_rf'],
                    help='Which persisted model to explain. `mlp` is absent on '
                         'purpose: it is a Pipeline with no trees to walk.')
    ap.add_argument('--n-sample', type=int, default=0,
                    help='Rows to explain. 0 (default) means every observed row, '
                         'reduced only if the measured TreeSHAP rate says that '
                         'would exceed --max-minutes.')
    ap.add_argument('--max-minutes', type=float, default=20.0,
                    help='Wall-clock budget for the SHAP pass. The sample is cut to '
                         'fit it and the cut is reported, so the module never hangs '
                         'on a deep forest. 20 min takes every observed row for '
                         'xgboost but only a few thousand for random_forest, which '
                         'was timed at 647 ms/row on a synthetic matrix of the '
                         'production shape -- raise this if that model is selected.')
    ap.add_argument('--min-per-mascon', type=int, default=200,
                    help='Floor on rows kept per mascon when sampling, so the mascon '
                         'bootstrap still has something to resample in the small ones '
                         '(the smallest carries 227 observed rows).')
    ap.add_argument('--n-boot', type=int, default=2000,
                    help='Mascon cluster-bootstrap resamples for the share CIs.')
    ap.add_argument('--top-n', type=int, default=15,
                    help='Individual columns drawn in panel (c).')
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--model-path', default=None,
                    help='Explain a joblib bundle from somewhere other than '
                         'Results/downscaling, e.g. an archived run. Its stored '
                         'feature names must still match the current design matrix.')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--fig-dir', default=None)
    ap.add_argument('--no-figure', action='store_true')
    args = ap.parse_args()

    out_dir = args.out_dir or RESULTS_DIR
    fig_dir = args.fig_dir or FIG_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f'TreeSHAP attribution for the gridded model ({pretty_model(args.model)})')
    print(f'  {SCOPE_NOTE}\n')

    print('1. Loading the fitted model')
    bundle = load_fitted(args.model, path=args.model_path)
    model = bundle['model']
    saved_names = list(bundle.get('feature_names') or [])
    n_train = bundle.get('n_train')
    print(f'  {type(model).__name__}, fitted on '
          f'{f"{n_train:,}" if isinstance(n_train, int) else "an unrecorded number of"} '
          f'observed rows, {len(saved_names)} features')

    print('\n2. Rebuilding the design matrix')
    _check_static_groups()
    fs = F.build_features()
    _, anom, _ = M.decompose_target(fs)
    obs = fs.observed

    # A design matrix that has moved since the model was fitted would silently
    # mis-label every attribution, because the model indexes columns by POSITION.
    if saved_names and saved_names != list(fs.X.columns):
        missing = [c for c in saved_names if c not in set(fs.X.columns)]
        added = [c for c in fs.X.columns if c not in set(saved_names)]
        raise SystemExit(
            'the design matrix has changed since the model was fitted, so column '
            f'positions no longer line up.\n  in the model, not in the matrix: {missing}\n'
            f'  in the matrix, not in the model: {added}\n'
            f'Refit:  python downscale_model.py --model {args.model}')

    X_all = fs.X.to_numpy(dtype='float32')[obs]
    mascon_all = fs.mascon[obs]
    y_all = anom[obs]
    table = column_table(list(fs.X.columns))

    # Measured on the current cube: the 4,404 rows carrying NaN (the first 12
    # months, where the 24-month antecedent window is not yet filled) are all
    # UNOBSERVED, so restricting to observed rows already removes them. Checked
    # rather than assumed, because it is a property of the GRACE gap pattern and
    # not something the feature builder guarantees.
    n_nan = int((~np.isfinite(X_all)).any(axis=1).sum())
    if n_nan:
        print(f'  {n_nan:,} of {len(X_all):,} explained rows carry a NaN from an '
              'unfilled antecedent or API window; TreeSHAP routes them down the '
              "trees' default branches, so their attribution reflects that default "
              'rather than a measured predictor value')

    print('\n3. Sizing the SHAP pass against a measured rate')
    explainer = tree_explainer(model)
    rate = shap_rate(explainer, X_all, seed=args.seed)
    full_min = rate * len(X_all) / 60.0
    print(f'  measured {1000 * rate:.3g} ms/row -> {full_min:.2f} min for all '
          f'{len(X_all):,} observed rows')

    n_target = args.n_sample if args.n_sample > 0 else len(X_all)
    budget_rows = max(1, int(args.max_minutes * 60.0 / max(rate, 1e-9)))
    if n_target > budget_rows:
        print(f'  cut to {budget_rows:,} rows to fit the {args.max_minutes:g} min '
              'budget. Read the split-half agreement below before quoting any '
              'ordering from a sample this size.')
        n_target = budget_rows

    idx = stratified_sample(mascon_all, n_target, seed=args.seed,
                            min_per_mascon=args.min_per_mascon)
    X = X_all[idx]
    mascon = mascon_all[idx]
    t_idx = fs.time_index[obs][idx]
    sampled = len(idx) < len(X_all)
    print(f'  explaining {len(idx):,} rows ({100 * len(idx) / len(X_all):.1f}% of '
          f'observed) across {len(np.unique(mascon))} mascons and '
          f'{len(np.unique(t_idx))} of {len(fs.dates)} months')

    print('\n4. TreeSHAP')
    t0 = time.time()
    phi = explainer.shap_values(X, check_additivity=False)
    if isinstance(phi, list):                    # multi-output shape, not expected here
        phi = phi[0]
    phi = np.asarray(phi, dtype='float64')
    print(f'  {phi.shape[0]:,} x {phi.shape[1]} in {time.time() - t0:.0f} s')

    # Additivity is checked here rather than left to shap's own assertion so that
    # the residual is REPORTED as a number. shap raises an opaque error and loses
    # the run; a residual that is small relative to the anomaly is a pass worth
    # printing, and one that is not is a fact worth having in the log.
    base = float(np.ravel(explainer.expected_value)[0])
    resid = np.abs(phi.sum(axis=1) + base - model.predict(X))
    sd = float(np.std(y_all))
    print(f'  base value {base:+.2f} mm | additivity residual max '
          f'{resid.max():.3g} mm ({100 * resid.max() / sd:.2g}% of the '
          f'{sd:.1f} mm anomaly s.d.)')
    if resid.max() > 0.01 * sd:
        print('  WARNING: additivity residual exceeds 1% of the anomaly s.d.; the '
              'attribution does not reconstruct the prediction and the shares below '
              'are not trustworthy')

    print('\n5. Grouping')
    frames: Dict[str, pd.DataFrame] = {}
    totals: Dict[str, float] = {}
    stability: Dict[str, float] = {}
    for partition, key in (('field', 'field_family'), ('term', 'term_family'),
                           ('column', 'name')):
        labels, G = family_matrix(phi, table, key)
        stats = share_ci(G, mascon, n_boot=args.n_boot, seed=args.seed)
        stats.insert(0, 'family', labels)
        stats['n_columns'] = [int((table[key] == lab).sum()) for lab in labels]

        # Two totals per family: the additive one (sum signed, then |.|) and what a
        # per-column rollup would have given. Their gap is credit that collinear
        # columns inside the family pass back and forth.
        naive = np.abs(phi).mean(axis=0)
        stats['sum_of_column_mean_abs_mm'] = [
            float(naive[np.flatnonzero((table[key] == lab).to_numpy())].sum())
            for lab in labels]
        # |sum of phi| <= sum of |phi| always, so cancellation cannot really be
        # negative; a tiny negative is float noise on a single-column family where
        # the two totals are the same number computed two ways. A family the model
        # never split on has 0/0 and no cancellation to report.
        with np.errstate(invalid='ignore', divide='ignore'):
            stats['internal_cancellation_pct'] = np.clip(100.0 * (
                1.0 - stats.mean_abs_shap_mm / stats.sum_of_column_mean_abs_mm),
                0.0, None)
        stats.loc[stats.sum_of_column_mean_abs_mm <= 0,
                  'internal_cancellation_pct'] = np.nan

        stats = stats.merge(correlation_diagnostics(X, table, key)
                            .rename(columns={key: 'family'}, errors='ignore')
                            .drop(columns=['n_columns']), on='family', how='left')
        _, worst = split_half_shares(G, mascon, seed=args.seed)
        stats['split_half_max_disagreement_pct'] = worst
        stability[partition] = worst
        totals[partition] = float(stats.mean_abs_shap_mm.sum())
        frames[partition] = stats.sort_values('share_pct', ascending=False)

    print('  total mean absolute attribution by partition (mm of anomaly):')
    for partition in ('column', 'field', 'term'):
        n = len(frames[partition])
        drop = 100.0 * (1.0 - totals[partition] / totals['column'])
        note = ('reference' if partition == 'column'
                else f'{drop:.0f}% of the per-column total cancels within families')
        print(f'    {partition:7s} {n:3d} groups  {totals[partition]:7.2f}   {note}')
    print('  Partitions have different totals BY CONSTRUCTION, so a per-column share '
          'and a family share are not on the same scale and must not be compared.')
    print(f'  split-half agreement (row-sampling error): field '
          f'{stability["field"]:.2f} pp, term {stability["term"]:.2f} pp, column '
          f'{stability["column"]:.2f} pp -- the largest change in any share between '
          'two disjoint halves of the explained rows')

    for partition, title in (('field', 'By source field'), ('term', 'By term type')):
        print(f'\n  {title}')
        print(f'    {"family":44s} share  [95% CI mascon]     mm  cols  '
              'worst within-family |r|  internal cancellation')
        for _, r in frames[partition].iterrows():
            r_txt = ('   -' if not np.isfinite(r.max_within_family_abs_r)
                     else f'{r.max_within_family_abs_r:.2f}')
            c_txt = ('   -' if not np.isfinite(r.internal_cancellation_pct)
                     else f'{r.internal_cancellation_pct:3.0f}%')
            print(f'    {r.family:44s} {r.share_pct:5.1f}%  '
                  f'[{r.share_ci_lo_pct:4.1f}, {r.share_ci_hi_pct:5.1f}]  '
                  f'{r.mean_abs_shap_mm:6.2f}  {int(r.n_columns):4d}  '
                  f'{r_txt:>23s}  {c_txt:>21s}')

    print(f'\n  Top {args.top_n} individual columns (different denominator, see above)')
    cols = (frames['column']
            .drop(columns=['max_within_family_abs_r'])   # meaningless for one column
            .merge(table[['name', 'display', 'field_family', 'term_family']],
                   left_on='family', right_on='name', how='left')
            .merge(column_worst_correlate(X, list(fs.X.columns)), on='name',
                   how='left'))
    print(f'    {"column":44s} share  [95% CI mascon]     mm  strongest correlate')
    for _, r in cols.nlargest(args.top_n, 'share_pct').iterrows():
        print(f'    {r.display:44s} {r.share_pct:5.1f}%  '
              f'[{r.share_ci_lo_pct:4.1f}, {r.share_ci_hi_pct:5.1f}]  '
              f'{r.mean_abs_shap_mm:6.2f}  |r| {r.max_abs_r_any_column:.2f} with '
              f'{r.most_correlated_with}')

    print('\n6. Reading')
    print(f'  {COLLINEARITY_NOTE}')
    fld = frames['field']
    top = fld.iloc[0]
    print('  ' + _finding(
        f'The largest attributed share is {top.family}: {top.share_pct:.1f}% of total '
        f'mean absolute attribution (95% CI {top.share_ci_lo_pct:.1f}-'
        f'{top.share_ci_hi_pct:.1f}, mascon bootstrap), over {int(top.n_columns)} '
        'columns. This is an association between the fitted function and those '
        'columns, on the training rows, for the anomaly alone.'))
    if len(fld) > 1:
        second = fld.iloc[1]
        overlap = top.share_ci_lo_pct <= second.share_ci_hi_pct
        print('  ' + _finding(
            f'The second is {second.family} at {second.share_pct:.1f}% '
            f'(95% CI {second.share_ci_lo_pct:.1f}-{second.share_ci_hi_pct:.1f}); '
            + ('the two intervals overlap, so they are not separated even before '
               'collinearity is considered.' if overlap else
               'the intervals do not overlap, but collinearity between the two '
               'fields is not priced by this interval and may still account for the '
               'gap.')))

    mem = frames['term']
    mem_sel = mem.family.str.startswith(tuple(_TERM_LABEL[t] for t in MEMORY_TERMS))
    if mem_sel.any():
        worst_r = mem.loc[mem_sel, 'max_within_family_abs_r'].max(skipna=True)
        print('  ' + _finding(
            f'Memory terms (antecedent means and exponential APIs together) carry '
            f'{mem.loc[mem_sel, "share_pct"].sum():.1f}% of the attributed total, '
            f'spread over {int(mem.loc[mem_sel, "n_columns"].sum())} columns. '
            + ('Their members correlate up to '
               f'{worst_r:.2f}, so how that share divides among individual windows '
               'is not identifiable.' if np.isfinite(worst_r) else
               'Each memory family holds a single column, so no within-family '
               'correlation was computable.')))

    sms = fld[fld.family == _FIELD_LABEL['sms']]
    if len(sms):
        s = sms.iloc[0]
        print('  ' + _finding(
            f'Soil moisture, all {int(s.n_columns)} of its columns taken together, is '
            f'at {s.share_pct:.1f}% (95% CI {s.share_ci_lo_pct:.1f}-'
            f'{s.share_ci_hi_pct:.1f}). ERA5-Land soil moisture is a free-running '
            'land-surface state and does not assimilate GRACE, so unlike the '
            'withdrawn basin-scale figure this share is not the target re-entering '
            'through a predictor.'))
    print(f'  {WITHDRAWN_CLAIM}')

    print('\n7. Writing')
    stamp = dict(model=args.model, target='mascon_anomaly', units='mm',
                 n_rows_explained=len(idx), n_rows_observed=len(X_all),
                 sampled=bool(sampled), n_mascons=int(len(np.unique(mascon))),
                 n_features=int(phi.shape[1]),
                 predictors=','.join(cfg.PREDICTORS),
                 base_value_mm=base, scope=SCOPE_NOTE, caveat=COLLINEARITY_NOTE)

    fam_out = pd.concat([frames['field'].assign(grouping='source_field'),
                         frames['term'].assign(grouping='term_type')],
                        ignore_index=True)
    for k, v in stamp.items():
        fam_out[k] = v
        cols[k] = v
    fam_path = os.path.join(out_dir, f'shap_families_{args.model}.csv')
    col_path = os.path.join(out_dir, f'shap_columns_{args.model}.csv')
    fam_out.to_csv(fam_path, index=False)
    cols.drop(columns=['family']).to_csv(col_path, index=False)
    print(f'  written: {fam_path}')
    print(f'  written: {col_path}')

    # Reusing downscale_model's numpy JSON encoder rather than writing a second
    # one: the failure it exists for -- a numpy scalar aborting the write after
    # every expensive result is already computed -- is the same failure here.
    with open(os.path.join(out_dir, f'shap_summary_{args.model}.json'), 'w') as fh:
        json.dump({**stamp,
                   'shap_seconds_per_row': rate,
                   'additivity_residual_max_mm': float(resid.max()),
                   'split_half_max_disagreement_pct': stability,
                   'total_mean_abs_by_partition_mm': totals,
                   'withdrawn_claim': WITHDRAWN_CLAIM,
                   'provenance': M.provenance(args.seed)},
                  fh, indent=2, default=M._json_safe)

    if not args.no_figure:
        plot(frames['field'], frames['term'], cols, args.model,
             os.path.join(fig_dir, f'Fig_shap_families_{args.model}.png'),
             top_n=args.top_n)
    caps = figure_captions.write_index(
        fig_dir, title='Figure captions — SHAP attribution')
    if caps:
        print(f'  captions -> {caps}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
