"""
Figure 6 -- the seasonal cycle, as the one file the journal takes.

WHY THIS SCRIPT EXISTS
----------------------
Fig. 6 is two plates drawn by two different pipeline steps: the basin-mean
annual cycle by `main/downscale_annual_cycle.py`, and the 0.1 degree seasonal
means by `main/generate_gridded_maps.py`. The manuscript places them one above
the other under a single caption ("Upper: ... Lower: ..."), but the journal
wants one file per figure. This script stacks the two rendered plates into that
file.

WHY STACK PIXELS RATHER THAN REDRAW
-----------------------------------
Redrawing both plates on one canvas would mean a third copy of two plotting
routines, and a third copy is how the figure and its sources drift apart.
Stacking the rendered PNGs means Fig. 6 is, pixel for pixel, the two plates the
pipeline writes. Both are rendered at the same DPI and are within 4% of each
other in width; the lower plate is resampled to the upper one's width, so the
two share left and right edges as they do in the manuscript.

USAGE
-----
    python make_fig6_seasonal_cycle.py

Run after the pipeline has written both source plates.
"""

from __future__ import annotations

import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))

OUT = os.path.join(HERE, 'output')
STEM = 'Fig6_seasonal_cycle'
UPPER = os.path.join(REPO, 'Results', 'downscaling', 'Fig_annual_cycle_xgboost.png')
LOWER = os.path.join(REPO, 'Results', 'figures', 'gridded_maps',
                     'Fig_seasonal_mean_0p1deg.png')

# White band between the plates, in inches at the plates' own DPI. Each plate
# already carries matplotlib's 0.1 in tight-bbox pad, so this adds to 0.4 in.
GAP_IN = 0.2


def main() -> int:
    for path in (UPPER, LOWER):
        if not os.path.exists(path):
            print(f'missing source plate: {path}', file=sys.stderr)
            return 1
    upper = Image.open(UPPER)
    lower = Image.open(LOWER)
    dpi = upper.info.get('dpi', (600, 600))
    if round(dpi[0]) != round(lower.info.get('dpi', dpi)[0]):
        print('the two plates are at different DPI; re-render them first',
              file=sys.stderr)
        return 1
    upper, lower = upper.convert('RGB'), lower.convert('RGB')

    width = upper.width
    lower = lower.resize((width, round(lower.height * width / lower.width)),
                         Image.LANCZOS)
    gap = round(GAP_IN * dpi[0])

    fig = Image.new('RGB', (width, upper.height + gap + lower.height), 'white')
    fig.paste(upper, (0, 0))
    fig.paste(lower, (0, upper.height + gap))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{STEM}.png')
    fig.save(path, dpi=dpi)
    print(f'written: {path}  ({fig.width} x {fig.height} px, '
          f'{fig.width / dpi[0]:.2f} x {fig.height / dpi[0]:.2f} in at '
          f'{dpi[0]:.0f} dpi)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
