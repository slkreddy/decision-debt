"""Fraud-agent accuracy per model: overall and fraud-case.

Sources: out/cases.jsonl (gold), out/fraud/<model>__v1.jsonl (predictions).
Regenerate: figs/.venv/bin/python figs/fig2_fraud_accuracy.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orx_figstyle import COLUMN, PALETTE, figure, save, use_style

MODELS = ["llama3.1:8b", "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5-coder:7b"]


def stats():
    cases = {c["id"]: c for c in map(json.loads, open("out/cases.jsonl"))}
    out = {}
    for model in MODELS:
        slug = model.replace(":", "-").replace("/", "-")
        preds = {json.loads(l)["id"]: json.loads(l)["pred"]
                 for l in open(f"out/fraud/{slug}__v1.jsonl")}
        gold_fraud = [i for i, c in cases.items() if c["gold"] == "fraud"]
        overall = sum(1 for i in cases if preds[i] == cases[i]["gold"]) / len(cases)
        fraud_case = sum(1 for i in gold_fraud if preds[i] == "fraud") / len(gold_fraud)
        out[model] = (overall * 100, fraud_case * 100)
    return out


def main():
    use_style()
    data = stats()
    fig, ax = figure(width=COLUMN, ratio=0.9)
    y = list(range(len(MODELS)))
    w = 0.4
    overall = [data[m][0] for m in MODELS]
    fraud = [data[m][1] for m in MODELS]
    b1 = ax.barh([v - w / 2 for v in y], overall, height=w,
                 color=PALETTE["blue"], edgecolor="black", linewidth=0.4)
    b2 = ax.barh([v + w / 2 for v in y], fraud, height=w,
                 color=PALETTE["orange"], edgecolor="black", linewidth=0.4)
    ax.bar_label(b1, labels=[f"{v:.1f}%" for v in overall], fontsize=6, padding=1.5)
    ax.bar_label(b2, labels=[f"{v:.1f}%" for v in fraud], fontsize=6, padding=1.5)
    ax.set_yticks(y, MODELS)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0)
    ax.legend(["overall", "fraud-case"], fontsize=7, frameon=False, loc="lower right")
    save(fig, "figs/fig2_fraud_accuracy")


if __name__ == "__main__":
    main()
