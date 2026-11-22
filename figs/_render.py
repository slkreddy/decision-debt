"""Render the figures as PNGs at 150 dpi for visual inspection (debug).

Run: figs/.venv/bin/python figs/_render.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

import orx_figstyle as style


def save_png(fig, stem, close=True):
    plt.savefig(f"{stem}.png", format="png", dpi=150)
    if close:
        plt.close(fig)
    return [f"{stem}.png"]


style.save = save_png

import fig1_upgrades
import fig2_fraud_accuracy
import fig3_fraud_churn
import fig4_cost_matrix

for mod in (fig1_upgrades, fig2_fraud_accuracy, fig3_fraud_churn, fig4_cost_matrix):
    mod.main()
