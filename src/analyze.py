"""Analyse decision-debt pilot runs.

Reports, per run: accuracy and expected cost per 1,000 decisions.
Reports, per pair of runs on the same suite: churn, regressions, cost-weighted
churn, decision loss, and a McNemar test on the accuracy discordance.

Usage: python src/analyze.py
"""

import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = {c["id"]: c for c in map(json.loads, (ROOT / "out" / "cases.jsonl").open())}
RUNS = ROOT / "out" / "runs"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load() -> dict[str, dict[str, str | None]]:
    runs: dict[str, dict[str, str | None]] = {}
    for path in sorted(RUNS.glob("*.jsonl")):
        recs = [json.loads(line) for line in path.open()]
        if not recs:
            continue
        key = f"{recs[0]['model']} [{recs[0]['prompt']}]"
        runs[key] = {r["id"]: r["pred"] for r in recs}
    return runs


def run_stats(preds: dict[str, str | None]) -> dict:
    ids = [i for i in preds if i in CASES]
    n = len(ids)
    correct = sum(1 for i in ids if preds[i] == CASES[i]["gold"])
    cost = sum(CASES[i]["cost"].get(preds[i], max(CASES[i]["cost"].values())) for i in ids)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "cost_per_1k": 1000 * cost / n if n else 0.0,
    }


def pair_stats(a: dict[str, str | None], b: dict[str, str | None]) -> dict:
    ids = sorted(set(a) & set(b) & set(CASES))
    n = len(ids)
    changed = [i for i in ids if a[i] != b[i]]
    regressions = [i for i in changed if a[i] == CASES[i]["gold"] and b[i] != CASES[i]["gold"]]
    improvements = [i for i in changed if a[i] != CASES[i]["gold"] and b[i] == CASES[i]["gold"]]

    cost_a = sum(CASES[i]["cost"].get(a[i], max(CASES[i]["cost"].values())) for i in ids)
    cost_b = sum(CASES[i]["cost"].get(b[i], max(CASES[i]["cost"].values())) for i in ids)
    exposure = sum(max(CASES[i]["cost"].values()) for i in ids)
    moved = sum(CASES[i]["cost"].get(b[i], max(CASES[i]["cost"].values())) for i in changed)

    # McNemar on correctness
    b_only = sum(1 for i in ids if a[i] == CASES[i]["gold"] and b[i] != CASES[i]["gold"])
    c_only = sum(1 for i in ids if a[i] != CASES[i]["gold"] and b[i] == CASES[i]["gold"])

    lo, hi = wilson(len(changed), n)
    return {
        "n": n,
        "churn": len(changed) / n,
        "churn_ci": [lo, hi],
        "regressions": len(regressions) / n,
        "improvements": len(improvements) / n,
        "cost_weighted_churn": moved / exposure if exposure else 0.0,
        "cost_per_1k_a": 1000 * cost_a / n,
        "cost_per_1k_b": 1000 * cost_b / n,
        "decision_loss_per_1k": 1000 * (cost_b - cost_a) / n,
        "mcnemar_p": mcnemar_exact(b_only, c_only),
    }


def main() -> None:
    runs = load()
    if not runs:
        raise SystemExit("no runs yet")

    print("\n== Runs ==")
    for name, preds in runs.items():
        s = run_stats(preds)
        print(f"{name:32s} n={s['n']:3d}  acc={s['accuracy']:.3f}  ${s['cost_per_1k']:.0f}/1k")

    results = {}
    print("\n== Pairs ==")
    for (na, a), (nb, b) in itertools.combinations(runs.items(), 2):
        p = pair_stats(a, b)
        results[f"{na} -> {nb}"] = p
        print(
            f"{na} -> {nb}\n"
            f"  churn {p['churn']:.1%} [{p['churn_ci'][0]:.1%}, {p['churn_ci'][1]:.1%}]"
            f"  regressions {p['regressions']:.1%}  improvements {p['improvements']:.1%}\n"
            f"  cost-weighted churn {p['cost_weighted_churn']:.1%}"
            f"  loss ${p['decision_loss_per_1k']:+.0f}/1k  McNemar p={p['mcnemar_p']:.3g}"
        )

    (ROOT / "out" / "analysis.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {ROOT/'out'/'analysis.json'}")


if __name__ == "__main__":
    main()
