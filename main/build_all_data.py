"""
Rebuild `Data/All_Data.csv` from the Earth Engine download.

Closes a provenance gap. The predictor table was read by eight modules and written
by none: the step that produced it was never committed, so the input every
basin-scale result rests on could not be regenerated or audited. Its columns had
to be reverse-engineered by testing them for linear identity against the GEE
downloads.

It was originally a hand-made `All_Data.xlsx`. Now that this module regenerates
it, the one reason to keep the Excel format -- comparing against the shipped copy
-- is gone, and the output is CSV: no undeclared openpyxl dependency, ~40x faster
to read, and greppable. `--target` still accepts a .xlsx path, so a legacy
spreadsheet can be verified against.

The chain is:

    gee_download.py  ->  Data/Outputs/Daily_GEE_GLDAS_V021.csv  ->  [this]  ->  All_Data.csv

The original spreadsheet was reverse-engineered against the GLDAS-sourced columns
of that CSV, reproducing all five to better than 1e-13 (8,401 rows, 2002-01-01 to
2024-12-31):

    All_Data column | column it was traced to | max abs difference
    ----------------|-------------------------|-------------------
    SMS             | ssm_gldas_mean_mm       | 9.8e-15
    ET              | et_gldas_mean_mm        | 1.0e-14
    rainfall        | ppt_era5_mean_mm        | 9.8e-14
    runoff          | runoff_era5_mean_mm     | 8.5e-14
    GWSA            | runoff - mean(runoff)   | 1.8e-15

That table records provenance, not current behaviour. COLUMN_MAP below now draws
SMS and ET from ERA5-Land instead, because GLDAS 2.2 CLSM assimilates GRACE, so
`--verify` against a shipped GLDAS-era All_Data.xlsx will report DIFF on those two
columns by design. rainfall, runoff and GWSA still reproduce exactly.

Two column names are misnomers, preserved here only for continuity with the
manuscript. Both are documented in main/README.md:

  SMS   is a SHALLOW layer, not root-zone storage. It was GLDAS `SoilMoist_S_tavg`
        (~0-2 cm, basin mean 4.58 mm) and is now ERA5-Land
        `volumetric_soil_water_layer_1` (0-7 cm, basin mean 18.4 mm). Root-zone
        storage is ~281 mm, available as `rzsm_era5_mean_mm`. Conclusions phrased
        as "storage memory" describe a near-surface layer either way.

  GWSA  contains NO groundwater. It is surface runoff minus its own temporal
        mean, so in this basin-mean setting it is perfectly collinear with
        `runoff` and any SHAP attribution between the two is arbitrary.

Why the source file is named V021
--------------------------------
`gee_download.py` is pinned to GLDAS V021 NOAH, so the CSV it writes is named
Daily_GEE_GLDAS_V021.csv. That is a filename, not a data dependency: every column
this module reads is ERA5-Land (see COLUMN_MAP) and no GLDAS band reaches
All_Data.csv.

V022 CLSM is not an alternative -- it assimilates GRACE, which is the target.

The earlier objection to V021 was its ET: V021 NOAH is 3-hourly, and this repo
once summed the eight sub-daily images before multiplying by 86400, overstating
daily ET eightfold. That was a bug in the conversion, not in V021.
`gee_download.py` now takes the daily mean rate, which is exact for both the
3-hourly V021 and the daily V022.

Usage
-----
    python build_all_data.py --verify        # compare against the existing file
    python build_all_data.py --write         # regenerate it
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from gridded_config import DATA_DIR, PREDICTOR_TABLE, read_predictor_table

SOURCE_CSV = os.path.join(DATA_DIR, 'Outputs', 'Daily_GEE_GLDAS_V021.csv')
TARGET = PREDICTOR_TABLE

# All_Data column -> source column in the GEE daily CSV.
# All from ERA5-Land. GLDAS 2.2 CLSM was dropped on discovering it assimilates
# GRACE: its soil moisture carried the assimilation increment and its TWS/GWS
# bands are GRACE itself, so two of the five predictors were derived from the
# target. ERA5-Land is a single homogeneous source, 1950-present, GRACE-free.
# The `_mean` infix is not decoration: Earth Engine's region reduction appends
# the reducer name to the band name, so a band renamed `sm1_era5_mm` in
# gee_download.py lands in the CSV as `sm1_era5_mean_mm`. Use the CSV names.
COLUMN_MAP = {
    'SMS': 'sm1_era5_mean_mm',   # 0-7 cm soil water storage (volumetric x 70 mm)
    'ET': 'et_era5_mean_mm',     # total evaporation, sign-flipped to positive
    'rainfall': 'ppt_era5_mean_mm',
    'runoff': 'runoff_era5_mean_mm',
}


def build(source_csv: str = SOURCE_CSV) -> pd.DataFrame:
    """Reconstruct the predictor table from the Earth Engine daily download."""
    if not os.path.exists(source_csv):
        raise FileNotFoundError(
            f'{source_csv} not found. Regenerate it with:\n'
            f'    python gee_download.py        # writes Daily_GEE_GLDAS_V021.csv\n'
            f'It resumes from Data/Outputs/temp/, so retrying an interrupted '
            f'download only fetches the days still missing.')

    src = pd.read_csv(source_csv, parse_dates=['Date']).sort_values('Date')
    missing = [c for c in COLUMN_MAP.values() if c not in src.columns]
    if missing:
        raise ValueError(f'{source_csv} is missing expected columns: {missing}')

    out = pd.DataFrame({'Date': src['Date'].to_numpy()})
    for name, col in COLUMN_MAP.items():
        out[name] = src[col].to_numpy()

    # GWSA is DERIVED, not downloaded: surface runoff minus its own mean.
    out['GWSA'] = out['runoff'] - out['runoff'].mean()
    return out.reset_index(drop=True)


def verify(built: pd.DataFrame, target: str = TARGET) -> bool:
    """Compare a rebuild against an existing table, column by column.

    `target` may be the current CSV or a legacy .xlsx -- passing the old
    spreadsheet is how the CSV migration was checked.
    """
    if not os.path.exists(target):
        print(f'  {target} does not exist - nothing to compare against')
        return False

    old = read_predictor_table(target)
    merged = old.merge(built, on='Date', how='outer', suffixes=('_old', '_new'))
    print(f'  existing {len(old)} rows | rebuilt {len(built)} rows | '
          f'union {len(merged)} rows')

    ok = True
    for name in list(COLUMN_MAP) + ['GWSA']:
        a, b = merged[f'{name}_old'], merged[f'{name}_new']
        both = a.notna() & b.notna()
        diff = float(np.abs(a[both] - b[both]).max()) if both.any() else np.nan
        # The existing file carries NaNs the rebuild fills (GWSA's 366-day tail)
        # and vice versa (SMS/ET before GLDAS V022 begins).
        only_old, only_new = int((a.notna() & ~b.notna()).sum()), int((~a.notna() & b.notna()).sum())
        # A pure constant offset is not a mismatch. GWSA is runoff minus a mean;
        # the existing file de-meaned over the 8,035 rows it happened to contain
        # while a rebuild de-means over all 8,401 -- a fixed 6.96e-03 shift. Tree
        # splits are invariant to a constant offset in a feature, and
        # utils.load_and_preprocess_data recomputes GWSA from runoff anyway, so
        # it changes no result. Report it as OFFSET, not DIFF.
        d_ = (a[both] - b[both]) if both.any() else None
        spread = float(np.abs(d_ - d_.mean()).max()) if d_ is not None else np.nan
        if np.isnan(diff) or diff < 1e-9:
            flag = 'OK  '
        elif spread < 1e-9:
            flag = 'OFFS'
        else:
            flag = 'DIFF'
            ok = False
        note = f'   constant offset {float(d_.mean()):+.3e}' if flag == 'OFFS' else ''
        print(f'  {flag} {name:9s} max|diff| {diff:.2e} over {int(both.sum()):5d} shared rows'
              f'  (+{only_new} filled by rebuild, {only_old} only in existing){note}')
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default=SOURCE_CSV)
    ap.add_argument('--target', default=TARGET,
                    help='table to write or compare against; a .xlsx path is '
                         'read as the legacy format')
    ap.add_argument('--write', action='store_true',
                    help='overwrite the target table (verify first)')
    ap.add_argument('--verify', action='store_true',
                    help='compare a rebuild against the existing file')
    args = ap.parse_args()

    built = build(args.source)
    print(f'rebuilt {len(built)} rows, {built.Date.min():%Y-%m-%d} to '
          f'{built.Date.max():%Y-%m-%d}')

    if args.verify or not args.write:
        identical = verify(built, args.target)
        print(f'\n  {"reproduces the existing file (any OFFS rows are immaterial)" if identical else "DIFFERS from the existing file"}')

    if args.write:
        backup = args.target + '.bak'
        if os.path.exists(args.target) and not os.path.exists(backup):
            # Copy rather than read-and-rewrite: the backup is meant to preserve
            # exactly what was there, including a legacy .xlsx.
            shutil.copy2(args.target, backup)
            print(f'  backed up existing file to {backup}')
        built.to_csv(args.target, index=False)
        print(f'  written: {args.target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
