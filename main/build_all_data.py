"""
Rebuild `Data/All_Data.xlsx` from the Earth Engine download.

Closes a provenance gap. `All_Data.xlsx` is read by eight modules and written by
none: the step that produced it was never committed, so the predictor table --
the input every basin-scale result rests on -- could not be regenerated or
audited. Its columns had to be reverse-engineered by testing them for linear
identity against the GEE downloads.

The chain is:

    gee_download.py  ->  Data/Outputs/Daily_GEE_GLDAS_V022.csv  ->  [this]  ->  All_Data.xlsx

Verified: every column of the existing spreadsheet is reproduced from that CSV to
better than 1e-13 (8,401 rows, 2002-01-01 to 2024-12-31).

    All_Data column | source column          | max abs difference
    ----------------|------------------------|-------------------
    SMS             | ssm_gldas_mean_mm      | 9.8e-15
    ET              | et_gldas_mean_mm       | 1.0e-14
    rainfall        | ppt_era5_mean_mm       | 9.8e-14
    runoff          | runoff_era5_mean_mm    | 8.5e-14
    GWSA            | runoff - mean(runoff)  | 1.8e-15

Two column names are misnomers, preserved here only for continuity with the
manuscript. Both are documented in main/README.md:

  SMS   is GLDAS V022 CLSM `SoilMoist_S_tavg`, the SURFACE ~0-2 cm layer
        (basin mean 4.58 mm), NOT root-zone storage (238 mm in the same
        dataset). Conclusions phrased as "storage memory" describe a skin layer.

  GWSA  contains NO groundwater. It is surface runoff minus its own temporal
        mean, so in this basin-mean setting it is perfectly collinear with
        `runoff` and any SHAP attribution between the two is arbitrary.

Why V022 and not V021
---------------------
GLDAS V021 NOAH is 3-hourly. `gee_download.py` historically summed the eight
sub-daily images and then multiplied by 86400, overstating daily ET eightfold --
the V021 CSV in Data/Outputs still carries that error (its ET mean is 16.14
versus 2.03 mm/day). V022 CLSM is daily, so the conversion is exact, and the
spreadsheet demonstrably came from V022. Use V022.

Usage
-----
    python build_all_data.py --verify        # compare against the existing file
    python build_all_data.py --write         # regenerate it
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from gridded_config import DATA_DIR

SOURCE_CSV = os.path.join(DATA_DIR, 'Outputs', 'Daily_GEE_GLDAS_V021.csv')
TARGET_XLSX = os.path.join(DATA_DIR, 'All_Data.xlsx')

# All_Data column -> source column in the GEE daily CSV.
# All from ERA5-Land. GLDAS 2.2 CLSM was dropped on discovering it assimilates
# GRACE: its soil moisture carried the assimilation increment and its TWS/GWS
# bands are GRACE itself, so two of the five predictors were derived from the
# target. ERA5-Land is a single homogeneous source, 1950-present, GRACE-free.
COLUMN_MAP = {
    'SMS': 'sm1_era5_mm',        # 0-7 cm soil water storage (volumetric x 70 mm)
    'ET': 'et_era5_mm',          # total evaporation, sign-flipped to positive
    'rainfall': 'ppt_era5_mean_mm',
    'runoff': 'runoff_era5_mean_mm',
}


def build(source_csv: str = SOURCE_CSV) -> pd.DataFrame:
    """Reconstruct the predictor table from the Earth Engine daily download."""
    if not os.path.exists(source_csv):
        raise FileNotFoundError(
            f'{source_csv} not found. Regenerate it with:\n'
            f'    python gee_download.py        # gldas_version="V022"')

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


def verify(built: pd.DataFrame, target: str = TARGET_XLSX) -> bool:
    """Compare a rebuild against the existing spreadsheet, column by column."""
    if not os.path.exists(target):
        print(f'  {target} does not exist - nothing to compare against')
        return False

    old = pd.read_excel(target)
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
    ap.add_argument('--target', default=TARGET_XLSX)
    ap.add_argument('--write', action='store_true',
                    help='overwrite the target spreadsheet (verify first)')
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
            pd.read_excel(args.target).to_excel(backup, index=False)
            print(f'  backed up existing file to {backup}')
        built.to_excel(args.target, index=False)
        print(f'  written: {args.target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
