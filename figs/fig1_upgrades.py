"""Upgrade churn: share of decisions that change between two model versions.

Sources: out/analysis.json (pair_stats over out/runs/).
Regenerate: figs/.venv/bin/python figs/fig1_upgrades.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orx_figstyle import COLUMN, PALETTE, figure, save, use_style

PAIRS = [
    ("llama3.1:8b v1->v2 (prompt)", "llama3.1:8b [v1] -> llama3.1:8b [v2]"),
    ("llama3.1:8b -> 0.5b", "llama3.1:8b [v1] -> qwen2.5:0.5b [v1]"),
    ("llama3.1:8b -> 1.5b", "llama3.1:8b [v1] -> qwen2.5:1.5b [v1]"),
    ("1.5b -> 3b", "qwen2.5:1.5b [v1] -> qwen2.5:3b [v1]"),
    ("llama3.1:8b -> coder", "llama3.1:8b [v1] -> qwen2.5-coder:7b [v1]"),
]


def main():
    use_style()
    data = json.load(open("out/analysis.json"))
    stats = [data[key] for _, key in PAIRS]
    churn = [s["churn"] * 100 for s in stats]
    err = np.array(
        [
            [s["churn"] * 100 - s["churn_ci"][0] * 100 for s in stats],
            [s["churn_ci"][1] * 100 - s["churn"] * 100 for s in stats],
        ]
    )
    fig, ax = figure(width=COLUMN, ratio=0.5)
    bars = ax.barh(
        range(len(PAIRS)),
        churn,
        height=0.7,
        color=PALETTE["blue"],
        edgecolor="black",
        linewidth=0.4,
        xerr=err,
        capsize=3,
        ecolor="black",
    )
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in churn], fontsize=6, padding=1.5)
    ax.set_yticks(range(len(PAIRS)), [label for label, _ in PAIRS])
    ax.set_xlabel("Churn of decisions (%)")
    ax.set_xlim(0, max(churn) * 1.3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    save(fig, "figs/fig1_upgrades")


if __name__ == "__main__":
    main()
