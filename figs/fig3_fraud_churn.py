"""Fraud-answer churn: share of cases whose fraud answer differs from llama3.1:8b.

Sources: out/fraud/<model>__v1.jsonl (predictions).
Regenerate: figs/.venv/bin/python figs/fig3_fraud_churn.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orx_figstyle import COLUMN, PALETTE, figure, save, use_style

MODELS = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5-coder:7b"]


def churn():
    base = {json.loads(l)["id"]: json.loads(l)["pred"]
            for l in open("out/fraud/llama3.1-8b__v1.jsonl")}
    out = []
    for model in MODELS:
        slug = model.replace(":", "-").replace("/", "-")
        preds = {json.loads(l)["id"]: json.loads(l)["pred"]
                 for l in open(f"out/fraud/{slug}__v1.jsonl")}
        out.append(sum(1 for i in preds if preds[i] != base[i]) / len(preds) * 100)
    return out


def main():
    use_style()
    vals = churn()
    fig, ax = figure(width=COLUMN, ratio=0.45)
    bars = ax.bar(range(len(MODELS)), vals, width=0.7,
                  color=PALETTE["blue"], edgecolor="black", linewidth=0.4)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in vals], fontsize=6, padding=1.5)
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_ylabel("Churn of fraud answer (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", length=0)
    save(fig, "figs/fig3_fraud_churn")


if __name__ == "__main__":
    main()
