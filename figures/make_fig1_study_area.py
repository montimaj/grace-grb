"""
Figure 1 — study area, for the revised manuscript.

WHAT THIS REPLACES, AND WHY
---------------------------
The submitted Figure 1 was a land-cover map of the basin with the wells scattered
on top. Land cover was never a predictor in the reviewed study and is not one
now, so the figure spent its whole area on a variable the paper does not use,
while the fact that actually governs the work went unshown: GRACE resolves ~20
independent 3-degree mascons over this basin, and the product asks for 9,538
cells at 0.1 degrees. Panel (c) states that directly.

It also used a saturated categorical palette (yellow markers on orange land) that
does not survive greyscale or common colour-vision deficiencies.

GEOPOLITICAL BOUNDARIES
-----------------------
India is drawn from a Survey of India-consistent outline: it extends to 37.08 N
and 97.42 E, so Jammu and Kashmir (including Aksai Chin) and Arunachal Pradesh
are shown as Indian territory. International datasets such as Natural Earth draw
the Line of Control instead and are NOT acceptable for a paper with an Indian
institutional co-author. The file is recorded in `data/india-soi.geojson` with
its own `Source` attribute; do not substitute another basemap without checking
its northern and eastern extents.

Nothing outside India is drawn as a political entity: the basin crosses into
Nepal, Bangladesh and China, and those neighbours are labelled in text rather
than outlined, so the figure asserts no boundary it cannot source.

OUTPUT
------
`output/Fig1_study_area.{pdf,png}` — PDF for the journal (vector text), PNG at
600 dpi for review copies. Double-column width, 183 mm.
"""

from __future__ import annotations

import os
import sys

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402
import matplotlib.patheffects as pe                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'main'))

OUT = os.path.join(HERE, 'output')
INDIA = os.path.join(HERE, 'data', 'india-soi.geojson')
BASIN = os.path.join(REPO, 'Data', 'Ganga Basin Shapefile', 'Ganga_basin.shp')
CUBES = os.path.join(REPO, 'Data', 'Gridded', 'cubes')
WELLS = os.path.join(REPO, 'Data', 'Gridded', 'wells', 'well_meta.csv')
RIVERS = os.path.join(HERE, 'data', 'hydrorivers',
                      'HydroRIVERS_v10_as_shp', 'HydroRIVERS_v10_as.shp')
DEM = os.path.join(HERE, 'data', 'dem', 'hyd_as_dem_15s.tif')
RIVERS_IN = os.path.join(HERE, 'data', 'rivers', 'Rivers.shp')   # named, India only

BOUNDS = (73.35, 21.45, 91.25, 31.55)   # the map window, also the DEM read window

MM = 1 / 25.4
FIG_W = 183 * MM          # Nature double column
INK = '#111111'
MUTED = '#6f6f6f'
WELL = '#0b5394'          # deep blue: reads on tan terrain and in greyscale
MASCON = '#c1272d'        # red, used ONLY for the mascon tessellation
RIVER = '#3d7fb5'
GANGA = '#c1272d'      # the named main channel, the one river worth naming

# Hypsometric ramp. Desaturated on purpose: the terrain is context, and the
# wells and mascon lines must sit on top of it without competing.
TERRAIN = LinearSegmentedColormap.from_list('hypso', [
    '#dbe6cf', '#c6d8b4', '#dfe0b4', '#e6d5ab', '#d9bd93',
    '#c6a17c', '#b08a6e', '#a68b7d', '#c2b6ad', '#efefef'])


def load():
    import netCDF4
    aux = netCDF4.Dataset(os.path.join(CUBES, 'grids_aux.nc'))
    sta = netCDF4.Dataset(os.path.join(CUBES, 'static_cube.nc'))
    d = dict(
        lat=np.asarray(aux['lat_era5'][:]), lon=np.asarray(aux['lon_era5'][:]),
        bfrac=np.asarray(aux['basin_frac_era5'][:]),
        parent=np.asarray(aux['parent_era5_to_grace'][:]),
        lat_g=np.asarray(aux['lat_grace'][:]), lon_g=np.asarray(aux['lon_grace'][:]),
        mascon=np.asarray(aux['mascon_id'][:]),
        bfrac_g=np.asarray(aux['basin_frac_grace'][:]),
        elev=np.asarray(sta['elevation'][:]), upa=np.asarray(sta['upa_log'][:]),
    )
    aux.close(); sta.close()
    basin = gpd.read_file(BASIN).to_crs(4326)
    # The shapefile carries the Sundarbans distributary network at full detail:
    # 14,018 vertices, most of them in the delta. At the line weight this figure
    # uses they collapse into a black smudge that reads as an error rather than
    # as a delta, so the OUTLINE is simplified for DRAWING ONLY.
    #
    # 0.03 degrees is ~3 km, well below anything separable at 183 mm across. It
    # cuts the outline to 1,073 vertices and changes the enclosed area by 0.13%,
    # measured. Every NUMBER in this figure comes from the unmodified basin_frac
    # grid, never from this geometry, so the simplification cannot reach them.
    d['basin'] = basin
    d['basin_draw'] = basin.copy()
    d['basin_draw']['geometry'] = basin.geometry.simplify(0.03, preserve_topology=True)
    d['india'] = gpd.read_file(INDIA).to_crs(4326)

    # Rivers from HydroRIVERS v1.0 (HydroSHEDS), clipped to the basin.
    #
    # This replaced the Government of India network, which was the obvious first
    # choice but covers Indian territory only: the Ganga trunk simply stopped at
    # 88.16 E where it crosses into Bangladesh, and the entire delta -- a fifth
    # of the basin -- carried no drainage at all. HydroRIVERS is transboundary,
    # so the trunk now runs to the Bay of Bengal.
    #
    # An earlier attempt derived rivers by thresholding the 0.1-degree
    # upstream-area covariate. At ~11 km per cell a channel is sub-pixel and came
    # out as a staircase; smoothing it enough to look like a river turned the
    # confluences into lakes. A surveyed network is the right source for a line.
    riv = gpd.read_file(RIVERS, bbox=tuple(basin.total_bounds))
    d['rivers'] = gpd.clip(riv, basin.geometry.iloc[0])
    d['ganga'] = _ganga_main_channel(riv, basin)

    d['wells'] = pd.read_csv(WELLS)

    # Terrain from the HydroSHEDS 15 arc-second void-filled DEM (~460 m),
    # windowed to the basin and masked to it.
    #
    # The 0.1-degree `elevation` covariate was the obvious source -- it is
    # already in the cubes -- but at ~11 km per cell the Himalayan front came out
    # as visible blocks. A relief background exists to be read at a glance, and
    # blocky relief reads as a rendering fault. 463 m per cell is 24x finer and
    # the same provenance as the river network, so one attribution covers both.
    import rasterio
    from rasterio import features as rfeat
    from rasterio.windows import from_bounds
    with rasterio.open(DEM) as src:
        win = from_bounds(*BOUNDS, src.transform)
        dem = src.read(1, window=win).astype('float32')
        tr = src.window_transform(win)
    dem[(dem > 9000) | (dem < -500)] = np.nan          # 32767 voids and sea fill
    mask = rfeat.rasterize([(basin.geometry.iloc[0], 1)], out_shape=dem.shape,
                           transform=tr, fill=0, dtype='uint8').astype(bool)
    d['dem'] = np.where(mask, dem, np.nan)
    d['dem_extent'] = (tr.c, tr.c + tr.a * dem.shape[1],
                       tr.f + tr.e * dem.shape[0], tr.f)
    d['dem_res_m'] = abs(tr.a) * 111000.0
    return d


def _ganga_main_channel(hydro, basin):
    """
    The Ganga main channel: the NAMED reach, extended to the sea.

    Tracing it automatically through HydroRIVERS does not work, and it is worth
    saying why rather than leaving a reader to wonder. Walking upstream from the
    mouth and taking the largest contributing area at each confluence follows the
    Yamuna at Allahabad and then the Chambal, ending in Rajasthan (75.6 E,
    22.5 N) after 3,077 km -- hydrologically the largest flow path, but not the
    channel anyone calls the Ganga. Tracing by discharge instead broke after
    132 km. "Main stem" is a naming convention, not a network property.

    So the named channel comes from the Government of India network
    (data.gov.in), which carries `rivname`. That dataset stops at the Bangladesh
    border, so the last stretch to the Bay of Bengal is completed by walking
    DOWNSTREAM through HydroRIVERS from the named channel's end. Downstream is
    unambiguous -- each reach has exactly one NEXT_DOWN -- so this half of the
    line involves no choice at all.
    """
    named = gpd.read_file(RIVERS_IN).to_crs(4326)
    named = named[named.rivname == 'Ganga']
    named = gpd.clip(named, basin.geometry.iloc[0])
    if named.empty:
        return named

    # downstream end of the named channel = its easternmost point
    pts = [c for g in named.geometry
           for part in (g.geoms if g.geom_type == 'MultiLineString' else [g])
           for c in part.coords]
    tail = max(pts, key=lambda c: c[0])

    h = hydro[hydro.intersects(basin.geometry.iloc[0])].copy()
    by_id = h.set_index('HYRIV_ID')
    from shapely.geometry import Point
    start = int(h.iloc[h.geometry.distance(Point(tail)).values.argmin()].HYRIV_ID)

    chain, cur, seen = [start], start, {start}
    while True:
        nd = int(by_id.at[cur, 'NEXT_DOWN'])
        if nd == 0 or nd not in by_id.index or nd in seen:
            break
        chain.append(nd); seen.add(nd); cur = nd
    tail_geom = h[h.HYRIV_ID.isin(chain)]
    return gpd.GeoDataFrame(
        geometry=list(named.geometry) + list(tail_geom.geometry), crs=4326)


def _extent(lat, lon):
    """imshow extent for cell-centre coordinate vectors."""
    dx = float(abs(lon[1] - lon[0])) / 2.0
    dy = float(abs(lat[1] - lat[0])) / 2.0
    return (lon.min() - dx, lon.max() + dx, lat.min() - dy, lat.max() + dy)


def _oriented(a, lat):
    """imshow wants row 0 at the top; the cubes may be stored either way."""
    return a[::-1] if lat[0] < lat[-1] else a


def mascon_polygons(d):
    """
    Dissolve the mascon id raster into polygons.

    Drawn from the 0.5-degree GRACE lattice rather than the 0.1-degree parent
    map because that is the grid the values are actually served on; the polygons
    that come out are the 3-degree mascons, six 0.5-degree cells across.
    """
    from rasterio import features, transform
    lon, lat = d['lon_g'], d['lat_g']
    res_x = float(abs(lon[1] - lon[0]))
    res_y = float(abs(lat[1] - lat[0]))
    top = lat.max() + res_y / 2.0
    tr = transform.from_origin(lon.min() - res_x / 2.0, top, res_x, res_y)

    # basin_frac > 0.5, which is the set leave-one-mascon-out actually holds out
    # and the 19 the paper reports. Twenty-two mascons touch the basin at all;
    # the extra three are slivers along the rim and are not cross-validated, so
    # drawing them here would put a number in the figure that no table supports.
    m = _oriented(d['mascon'], lat).astype('int32')
    bf = _oriented(d['bfrac_g'], lat)
    m = np.where(bf > 0.5, m, -1)

    geoms, vals = [], []
    for geom, val in features.shapes(m, mask=(m >= 0), transform=tr):
        geoms.append(geom); vals.append(int(val))
    g = gpd.GeoDataFrame({'mascon': vals},
                         geometry=gpd.GeoSeries.from_wkt(
                             [__import__('shapely').geometry.shape(x).wkt for x in geoms]),
                         crs=4326)
    return g.dissolve(by='mascon').reset_index()


def scalebar(ax, lon0, lat0, km=200, n=2):
    """
    Uptick scale bar: a baseline with ticks rising at each division.

    The bar length is computed GEODESICALLY at the bar's own latitude rather
    than assumed from a degree-to-kilometre constant. A degree of longitude is
    ~12% shorter at the basin's northern edge than at its southern one, so a bar
    drawn with a fixed conversion is wrong almost everywhere on this map.
    """
    from pyproj import Geod
    g = Geod(ellps='WGS84')
    lon1, _, _ = g.fwd(lon0, lat0, 90, km * 1000)
    dx = (lon1 - lon0) / n
    up = 0.16                      # tick height, in degrees of latitude

    ax.plot([lon0, lon0 + n * dx], [lat0, lat0], color=INK, lw=1.0,
            zorder=13, solid_capstyle='butt')
    for i in range(n + 1):
        x = lon0 + i * dx
        ax.plot([x, x], [lat0, lat0 + up], color=INK, lw=1.0, zorder=13,
                solid_capstyle='butt')
        lab = f'{int(km * i / n)}' + (' km' if i == n else '')
        ax.text(x, lat0 + up + 0.06, lab, ha='center', va='bottom',
                fontsize=7, color=INK, zorder=13)


def north_arrow(ax, x, y):
    ax.annotate('N', xy=(x, y), xytext=(x, y - 0.85), ha='center', va='top',
                fontsize=9.5, fontweight='bold', color=INK, zorder=13,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    d = load()
    lat, lon = d['lat'], d['lon']
    ext = _extent(lat, lon)

    inb = d['bfrac'] > 0
    elev = np.where(inb, d['elev'], np.nan)
    upa = np.where(inb, d['upa'], np.nan)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.62))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.35, 1.0], height_ratios=[1, 1],
                          wspace=0.10, hspace=0.16,
                          left=0.055, right=0.985, top=0.965, bottom=0.135)
    ax = fig.add_subplot(gs[:, 0])       # (a) the basin
    axb = fig.add_subplot(gs[0, 1])      # (b) India locator
    axc = fig.add_subplot(gs[1, 1])      # (c) the resolution gap

    # ---------------------------------------------------------------- (a)
    dem, dext, res_m = d['dem'], d['dem_extent'], d['dem_res_m']
    ls = LightSource(azdeg=315, altdeg=45)
    z = np.where(np.isfinite(dem), dem, 0.0)
    shade = ls.hillshade(z, vert_exag=12, dx=res_m, dy=res_m)
    shade = np.where(np.isfinite(dem), shade, np.nan)
    ax.imshow(dem, extent=dext, origin='upper', cmap=TERRAIN, vmin=0,
              vmax=float(np.nanpercentile(dem, 99.5)), zorder=2,
              interpolation='bilinear')
    ax.imshow(shade, extent=dext, origin='upper', cmap='gray', alpha=0.30,
              zorder=3, interpolation='bilinear')

    # Two tiers by long-term average discharge, which is what makes a river read
    # as a trunk or a tributary. Measured on the clipped network: 978 reaches
    # carry >= 1000 m3/s and the largest is 34,917 m3/s.
    riv = d['rivers']
    major = riv[riv.DIS_AV_CMS >= 1000.0]
    minor = riv[(riv.DIS_AV_CMS >= 100.0) & (riv.DIS_AV_CMS < 1000.0)]
    minor.plot(ax=ax, color=RIVER, linewidth=0.35, alpha=0.65, zorder=4)
    major.plot(ax=ax, color=RIVER, linewidth=1.0, alpha=0.92, zorder=4)
    if len(d['ganga']):
        d['ganga'].plot(ax=ax, color=GANGA, linewidth=1.5, alpha=0.95, zorder=5)

    d['basin_draw'].boundary.plot(ax=ax, color=INK, linewidth=1.0, zorder=7)
    # The international boundary is CASED and drawn last.
    #
    # Measured, 15.9% of the basin outline lies within 2 km of the India
    # boundary: through Nepal and along the lower Ganga the basin divide and the
    # border are the same line. Drawn underneath, the boundary simply disappeared
    # beneath the heavier basin outline for a sixth of its length, and a reader
    # would have concluded the basin does not cross into Nepal at all.
    #
    # A white casing under a dark dash is the usual way to show two coincident
    # linear features: the casing interrupts the basin line, and the dashes sit
    # in the gap, so both are legible exactly where they coincide.
    d['india'].boundary.plot(ax=ax, color='white', linewidth=1.25, alpha=0.85,
                             zorder=8)
    d['india'].boundary.plot(ax=ax, color='#333333', linewidth=0.6,
                             linestyle=(0, (3.0, 2.0)), zorder=9)

    w = d['wells']
    ax.scatter(w.lon, w.lat, s=7.5, c=WELL, edgecolors='white', linewidths=0.28,
               zorder=10, label=f'CGWB wells (n = {len(w)})')

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(float(np.mean(lat)))))
    ax.set_xlabel('Longitude (°E)', color=INK)
    ax.set_ylabel('Latitude (°N)', color=INK)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for s in ax.spines.values():
        s.set_color(MUTED)

    # INDIA sits OUTSIDE the basin, on the plain background south of it. Inside,
    # it competed with the terrain shading and the well markers and was the
    # least legible label on the map despite naming the largest territory.
    for name, (tx, ty) in {'INDIA': (80.6, 22.35), 'NEPAL': (83.4, 28.55),
                           'BANGLADESH': (89.55, 22.75), 'CHINA': (85.0, 31.05)}.items():
        ax.text(tx, ty, name, fontsize=8.0, color='#3a3a3a', zorder=9,
                ha='center', style='italic', fontweight='bold')

    scalebar(ax, 82.6, 22.02, km=400, n=2)
    north_arrow(ax, 90.4, 30.9)
    # Lower left: the only sizeable area of the frame the basin does not enter.
    # Upper left sat on the basin's northwestern boundary and on the wells there.
    handles = [
        Line2D([], [], marker='o', linestyle='none', markersize=3.4,
               markerfacecolor=WELL, markeredgecolor='white', markeredgewidth=0.3,
               label=f'CGWB wells (n = {len(w)})'),
        # Plate text says "Ganges" to match the paper title. The rivname VALUE
        # matched in _ganga_main_channel stays 'Ganga' -- that is what the
        # Government of India river network calls it, and renaming the string
        # would silently drop the red channel.
        Line2D([], [], color=GANGA, lw=1.6, label='Ganges main channel'),
        Line2D([], [], color=RIVER, lw=1.3, label='River, ≥ 1000 m³ s⁻¹'),
        Line2D([], [], color=RIVER, lw=0.5, alpha=0.8, label='River, 100–1000 m³ s⁻¹'),
        Line2D([], [], color='#333333', lw=0.8, linestyle=(0, (3.0, 2.0)),
               label='International boundary'),
        Line2D([], [], color=INK, lw=1.2, label='Ganges River basin boundary'),
    ]
    # The legend sits OUTSIDE the map, in a single row beneath it.
    #
    # Every in-map position was occupied: upper left sat on the basin's
    # northwestern boundary and its wells, lower left needed the source credit
    # stacked on top of it, and lower right is the Bangladesh delta. Moving it
    # out returns the whole frame to the data and removes the last thing
    # covering any of it. Unframed, because a box outside the map is a border
    # around nothing.
    # Two columns, not four: a single row spanned the whole figure width and ran
    # underneath panel (c)'s caption. Anchored to panel (a) alone.
    leg = ax.legend(handles=handles, loc='upper center', frameon=False,
                    fontsize=7.2, ncol=3, columnspacing=1.7,
                    handletextpad=0.55, handlelength=1.9,
                    bbox_to_anchor=(0.5, -0.105))
    leg.set_zorder(20)
    ax.set_title('a', loc='left', fontsize=11, fontweight='bold', color=INK)

    # ---------------------------------------------------------------- (b)
    d['india'].plot(ax=axb, facecolor='#f0f0ee', edgecolor='#555555',
                    linewidth=0.5, zorder=2)
    d['basin_draw'].plot(ax=axb, facecolor=WELL, alpha=0.72, edgecolor=INK,
                         linewidth=0.4, zorder=3)
    axb.set_aspect(1 / np.cos(np.deg2rad(22.0)))
    axb.set_xticks([]); axb.set_yticks([])
    for s in axb.spines.values():
        s.set_color(MUTED)
    axb.set_title('b', loc='left', fontsize=11, fontweight='bold', color=INK)
    axb.text(0.5, -0.055, 'Ganges basin within India',
             transform=axb.transAxes, ha='center', va='top', fontsize=7.0,
             color=MUTED)

    # ---------------------------------------------------------------- (c)
    mas = mascon_polygons(d)
    n_cells = int(inb.sum())
    axc.imshow(np.where(inb, 1.0, np.nan)[::-1] if lat[0] < lat[-1]
               else np.where(inb, 1.0, np.nan),
               extent=ext, origin='upper', cmap=LinearSegmentedColormap.from_list(
                   'g', ['#e9eef2', '#e9eef2']), zorder=2)
    # the 0.1 degree lattice, drawn every 5th line so it reads as a mesh
    for x in lon[::5]:
        axc.plot([x, x], [ext[2], ext[3]], color='#b9c4cc', lw=0.16, zorder=3)
    for y in lat[::5]:
        axc.plot([ext[0], ext[1]], [y, y], color='#b9c4cc', lw=0.16, zorder=3)
    mas.boundary.plot(ax=axc, color=MASCON, linewidth=0.85, zorder=5)
    d['basin_draw'].boundary.plot(ax=axc, color=INK, linewidth=0.7, zorder=6)

    # Number every mascon with its ACTUAL id, not a 1..19 renumbering.
    #
    # These ids key `lomo_cv_<model>.csv`, the per-mascon skill map and the
    # transfer-error table, so a reader holding a per-mascon time series or a row
    # of held-out metrics has no way to find the place it refers to unless the
    # same ids appear on the map. They are not contiguous -- 3, 4, 6, 7, ... --
    # because they index the full JPL mascon set, of which this basin contains
    # 19 at basin_frac > 0.5.
    #
    # `representative_point` rather than `centroid`: several of these polygons
    # are concave where they follow the basin edge, and a centroid can fall
    # outside its own mascon, which would put a label in a neighbour.
    for _, row in mas.iterrows():
        pt = row.geometry.representative_point()
        axc.text(pt.x, pt.y, str(int(row.mascon)), ha='center', va='center',
                 fontsize=5.0, fontweight='bold', color=MASCON, zorder=7,
                 path_effects=[pe.withStroke(linewidth=1.3, foreground='white')])

    axc.set_xlim(ext[0], ext[1]); axc.set_ylim(ext[2], ext[3])
    axc.set_aspect(1 / np.cos(np.deg2rad(float(np.mean(lat)))))
    axc.set_xticks([]); axc.set_yticks([])
    for s in axc.spines.values():
        s.set_color(MUTED)
    axc.set_title('c', loc='left', fontsize=11, fontweight='bold', color=INK)
    axc.text(0.5, -0.055,
             f'{len(mas)} GRACE mascons (3°, red, numbered) over the\n'
             f'{n_cells:,}-cell 0.1° target grid (grey)',
             transform=axc.transAxes, ha='center', va='top', fontsize=7.0,
             color=MUTED)

    for ext_ in ('pdf', 'png'):
        path = os.path.join(OUT, f'Fig1_study_area.{ext_}')
        fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f'written: {path}')
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
