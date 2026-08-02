"""
Fetch the third-party basemaps the manuscript figures draw on.

`figures/data/` is ~845 MB and is not in git. This script restores it. Run it
once from a clean clone before `make_fig1_study_area.py`.

WHAT IS FETCHED, AND WHY EACH ONE
---------------------------------
* **Survey of India national outline** — the boundary shown for India. It extends
  to 37.08 N and 97.42 E, so Jammu and Kashmir including Aksai Chin, and
  Arunachal Pradesh, are Indian territory. This is not interchangeable with
  Natural Earth or any other international basemap: those draw the Line of
  Control, which is not acceptable in a paper with an Indian institutional
  co-author. The script CHECKS those two extents and refuses a file that fails.
* **HydroRIVERS v1.0 (Asia)** — the river network. Transboundary, which the
  Indian national network is not: with the latter the Ganga stopped dead at
  88.16 E where it crosses into Bangladesh and the entire delta carried no
  drainage.
* **HydroSHEDS 15 arcsecond void-filled DEM (Asia)** — the terrain background.
  The 0.1 degree `elevation` covariate already in the cubes is ~11 km per cell
  and rendered the Himalayan front as visible blocks.

ONE FILE THIS SCRIPT CANNOT FETCH
---------------------------------
`data/rivers/Rivers.shp`, the Government of India river network from
data.gov.in, which carries the `rivname` attribute used to identify the Ganga
main channel by NAME. data.gov.in serves it behind a session, so it has to be
downloaded by hand:

    https://www.data.gov.in/resource/shapefile-rivers

and unzipped to `figures/data/rivers/`. Without it Figure 1 still renders; the
Ganga simply is not picked out in red.
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

# A plain urllib request is rejected by data.hydrosheds.org with HTTP 403.
UA = {'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                     'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 '
                     'Safari/537.36')}

SOURCES = [
    dict(name='Survey of India outline',
         url=('https://raw.githubusercontent.com/yashveeeeeeer/india-geodata/'
              'main/data/administrative/country/india-soi.geojson'),
         dest='india-soi.geojson', unzip=None),
    dict(name='HydroRIVERS v1.0 (Asia)',
         url='https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip',
         dest='HydroRIVERS_as.zip', unzip='hydrorivers'),
    dict(name='HydroSHEDS 15" DEM (Asia)',
         url='https://data.hydrosheds.org/file/hydrosheds-v1-dem/hyd_as_dem_15s.zip',
         dest='hyd_as_dem_15s.zip', unzip='dem'),
]


def fetch(src) -> None:
    out = os.path.join(DATA, src['dest'])
    if os.path.exists(out):
        print(f"  {src['name']}: already present ({os.path.getsize(out)/1e6:.0f} MB)")
    else:
        print(f"  {src['name']}: downloading …", flush=True)
        req = urllib.request.Request(src['url'], headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r, open(out, 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"    {os.path.getsize(out)/1e6:.0f} MB")
    if src['unzip']:
        target = os.path.join(DATA, src['unzip'])
        if not os.path.isdir(target) or not os.listdir(target):
            with zipfile.ZipFile(out) as z:
                z.extractall(target)
            print(f"    extracted -> {src['unzip']}/")


def check_soi() -> int:
    """
    Refuse a boundary that is not Survey of India consistent.

    This is a correctness check, not a convenience one. Substituting Natural
    Earth here would silently redraw India's northern and eastern borders, and
    nothing downstream would complain -- the figure would simply be wrong in the
    one way this project cannot afford.
    """
    try:
        import geopandas as gpd
    except ImportError:
        print('  geopandas missing; skipping the boundary check')
        return 0
    path = os.path.join(DATA, 'india-soi.geojson')
    if not os.path.exists(path):
        return 1
    b = gpd.read_file(path).total_bounds
    ok_n, ok_e = b[3] > 36.5, b[2] > 96.5
    print(f"\n  boundary extents: north {b[3]:.2f} N, east {b[2]:.2f} E")
    print(f"    Jammu and Kashmir incl. Aksai Chin : {'yes' if ok_n else 'NO'}")
    print(f"    Arunachal Pradesh                  : {'yes' if ok_e else 'NO'}")
    if not (ok_n and ok_e):
        print('\n  REFUSED: this outline is not Survey of India consistent.')
        return 1
    return 0


def main() -> int:
    os.makedirs(DATA, exist_ok=True)
    print('Fetching figure basemaps into figures/data/\n')
    for src in SOURCES:
        fetch(src)
    rc = check_soi()

    manual = os.path.join(DATA, 'rivers', 'Rivers.shp')
    if not os.path.exists(manual):
        print('\n  NOTE: data/rivers/Rivers.shp is absent. It cannot be fetched'
              '\n  programmatically -- download "Shapefile of Rivers" from'
              '\n  https://www.data.gov.in/resource/shapefile-rivers and unzip it'
              '\n  to figures/data/rivers/. Figure 1 renders without it, but the'
              '\n  Ganga main channel will not be drawn in red.')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
