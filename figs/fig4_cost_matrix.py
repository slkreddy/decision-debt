"""Cost-matrix diagram: cost(true, predicted) = delay[true] + handling[predicted].

Sources: out/suite_meta.json (delay/handling parameters from the audit trail).
Regenerate: figs/.venv/bin/python figs/fig4_cost_matrix.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orx_figstyle import annotate_matrix, figure, save, use_style


def main():
    use_style()
    meta = json.load(open("out/suite_meta.json"))
    actions = meta["actions"]
    delay, handling = meta["delay"], meta["handling"]
    m = np.array([[delay[t] + handling[p] for p in actions] for t in actions])
    np.fill_diagonal(m, 0)
    fig, ax = figure(width=3.25, ratio=1.0)
    image = ax.imshow(
        m, cmap="viridis", vmin=0, vmax=m.max(),
        interpolation="nearest", aspect="equal",
    )
    annotate_matrix(ax, m, fmt="{:.0f}", image=image)
    # Category names sit just outside the grid edges, not on the shared
    # axes: the six labels are too wide for the pitch and would overlap.
    ax.set_xticks([])
    # Y ticks carry the true actions; their labels sit outside the image.
    ax.set_yticks(range(len(actions)), actions)
    # Column names sit just below the grid, not on the shared axes.
    for c, name in enumerate(actions):
        ax.text(c, -0.65, name, ha="center", va="top", fontsize=6)
    ax.set_xlabel("Predicted action")
    fig.colorbar(image, ax=ax, label="USD", fraction=0.046, pad=0.03)
    save(fig, "figs/fig4_cost_matrix")


if __name__ == "__main__":
    main()
