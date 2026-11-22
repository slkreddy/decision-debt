"""Debug fig4: print overlapping text pairs with geometry."""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl
import matplotlib.pyplot as plt

import orx_figstyle as style

Figs = []


def grab(fig, stem, close=True):
    Figs.append(fig)
    return []


style.save = grab

import fig4_cost_matrix

fig4_cost_matrix.main()
fig = Figs[-1]
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

boxes = []
for text in fig.findobj(mpl.text.Text):
    if not text.get_visible() or not text.get_text().strip():
        continue
    try:
        b = text.get_window_extent(renderer)
    except Exception:
        continue
    boxes.append((text, b))

print("figure size:", fig.get_size_inches())
for text, b in boxes:
    print(f"  {text.get_text()!r} xy=({b.x0:.0f},{b.y0:.0f}) wh=({b.width:.0f},{b.height:.0f})")

for a, b in itertools.combinations(boxes, 2):
    inter = mpl.transforms.Bbox.intersection(a[1], b[1])
    if inter is None:
        continue
    if inter.width > 1.5 and inter.height > 1.5:
        print(f"OVERLAP: {a[0].get_text()!r} over {b[0].get_text()!r}")
