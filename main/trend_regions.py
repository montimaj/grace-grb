"""
Regional summaries of the Theil-Sen trend field.

WHY THIS MODULE EXISTS
----------------------
The manuscript states that depletion is concentrated in the north-west, and
quotes a mean trend for "the Punjab-Haryana-western Uttar Pradesh plain". That
number was computed once, by hand, from a region that existed only in the
author's head: no bounding box, no state polygon and no mask defining it was
ever committed, so a reader holding the archived trend field could not
reproduce it, and neither could we.

This module fixes that by writing the region definitions down. Each region is a
plain latitude/longitude box, stated in one place, applied to the archived
`twsa_trend_significance.nc`, and written out as a table alongside the other
result files. The numbers quoted in the text are then whatever this script
prints -- not the other way round. A box was chosen over an administrative
polygon because the basin mask, the trend field and the 0.1 degree grid are all
that the archive contains; adding a state shapefile would introduce a
dependency the released record cannot satisfy.

WHAT THE REGIONS MEAN
---------------------
The basin is split at 27 N and 79.5 E into four quadrants, so the regions
PARTITION it: every tested cell falls in exactly one of them, and the parts
therefore account for the whole rather than sampling the interesting corner and
leaving the rest unexamined. `main()` asserts that partition rather than
trusting it. The dividing lines are round numbers chosen to separate the
north-western depletion zone from the Himalayan front to its east and from the
plateau to its south, not tuned to any result.

`northwest_plain_lowland` is the one exception: a nested subset of the
north-western quadrant that drops the strip above 30 N, so the question "is the
north-western signal just the Himalayan front?" can be answered directly. It
overlaps `northwest_plain` by construction and is excluded from the partition
check.

WHAT IS SUMMARISED
------------------
Only cells that pass Benjamini-Hochberg false discovery rate control are
averaged, because that is the population the manuscript sentence refers to. The
all-tested mean is reported next to it so the difference between "mean trend
over significant cells" and "mean trend over the basin" can never again be
confused -- they differ by nearly 3 mm/yr basin-wide, and the earlier text
called the former a basin mean.

USAGE
-----
    python trend_regions.py                 # writes the CSV and prints it
    python trend_regions.py --no-write      # print only
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import gridded_config as cfg
    RESULTS_DIR = cfg.GRIDDED_RESULTS_DIR
except Exception:                                             # pragma: no cover
    RESULTS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Results', 'downscaling')

TREND_FILE = 'twsa_trend_significance.nc'
OUT_FILE = 'trend_by_region.csv'

# name -> (lat_min, lat_max, lon_min, lon_max), degrees north and east.
# Bounds are half-open, [min, max), so the four quadrants tile without overlap.
# The outer edges deliberately sit outside the grid; the `tested` mask does the
# clipping to the basin, so no cell is lost to a boundary chosen by hand.
SPLIT_LAT, SPLIT_LON = 27.0, 79.5

REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    'basin':                              (-90.0, 90.0, -180.0, 180.0),
    # Punjab, Haryana, western Uttar Pradesh -- the depletion zone.
    'northwest_plain':                    (SPLIT_LAT, 90.0, -180.0, SPLIT_LON),
    # Uttarakhand Himalaya, Nepal, the northern Bihar plain.
    'northern_plain_himalayan_front':     (SPLIT_LAT, 90.0, SPLIT_LON, 180.0),
    # The Rajasthan and Madhya Pradesh plateau, Chambal and Betwa headwaters.
    'southern_plateau':                   (-90.0, SPLIT_LAT, -180.0, SPLIT_LON),
    # Eastern Uttar Pradesh, Bihar, West Bengal and the delta.
    'eastern_plain_delta':                (-90.0, SPLIT_LAT, SPLIT_LON, 180.0),
}

# Nested inside northwest_plain, so NOT part of the partition.
SUBSETS: Dict[str, Tuple[float, float, float, float]] = {
    'northwest_plain_lowland': (SPLIT_LAT, 30.0, -180.0, SPLIT_LON),
}

PARTITION = [k for k in REGIONS if k != 'basin']


def load_trend(results_dir: str = RESULTS_DIR):
    """Return (lat, lon, sen_slope, significant_fdr, tested) from the archive."""
    import netCDF4 as nc
    path = os.path.join(results_dir, TREND_FILE)
    with nc.Dataset(path) as ds:
        lat = np.asarray(ds['lat'][:], float)
        lon = np.asarray(ds['lon'][:], float)
        slope = np.asarray(ds['sen_slope'][:], float)
        sig = np.asarray(ds['significant_fdr'][:]).astype(bool)
        tested = np.asarray(ds['tested'][:]).astype(bool)
    return lat, lon, slope, sig, tested


def region_mask(name: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Boolean grid mask for a named region or subset.

    The single definition of what a region covers. `summarise`, the partition
    check and the supplementary figure all call this, so a bound can never be
    half-open in one place and inclusive in another -- which is exactly how the
    figure and the table first came to disagree.
    """
    la0, la1, lo0, lo1 = {**REGIONS, **SUBSETS}[name]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return ((lat_grid >= la0) & (lat_grid < la1)
            & (lon_grid >= lo0) & (lon_grid < lo1))


def summarise(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """One row per region: counts and trend statistics, in mm/yr."""
    lat, lon, slope, sig, tested = load_trend(results_dir)

    rows = []
    for name, (la0, la1, lo0, lo1) in {**REGIONS, **SUBSETS}.items():
        box = region_mask(name, lat, lon)
        in_box = box & tested
        signif = box & sig
        if not signif.any():
            continue
        s = slope[signif]
        rows.append({
            'region': name,
            'lat_min': la0 if name != 'basin' else np.nan,
            'lat_max': la1 if name != 'basin' else np.nan,
            'lon_min': lo0 if name != 'basin' else np.nan,
            'lon_max': lo1 if name != 'basin' else np.nan,
            'n_tested': int(in_box.sum()),
            'n_significant': int(signif.sum()),
            'mean_significant_mm_yr': float(s.mean()),
            'median_significant_mm_yr': float(np.median(s)),
            'min_significant_mm_yr': float(s.min()),
            'pct_negative_significant': float(100.0 * (s < 0).mean()),
            'mean_all_tested_mm_yr': float(slope[in_box].mean()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', default=RESULTS_DIR)
    ap.add_argument('--no-write', action='store_true',
                    help='print the table without writing the CSV')
    args = ap.parse_args()

    df = summarise(args.results_dir)

    # The partition is a claim about the data, so check it rather than assert it
    # in a docstring: every tested cell must fall in exactly one quadrant.
    lat, lon, _, _, tested = load_trend(args.results_dir)
    cover = np.zeros_like(tested, dtype=int)
    for name in PARTITION:
        cover += (region_mask(name, lat, lon) & tested).astype(int)
    if not (cover[tested] == 1).all():
        raise SystemExit('regions do not partition the tested cells: '
                         f'{int((cover[tested] != 1).sum())} cell(s) covered '
                         'zero or multiple times')
    print(f'partition checked: {int(tested.sum()):,} tested cells, '
          'each in exactly one quadrant\n')

    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(df.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    if not args.no_write:
        out = os.path.join(args.results_dir, OUT_FILE)
        df.to_csv(out, index=False)
        print(f'\nwritten: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
