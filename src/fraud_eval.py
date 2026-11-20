"""Fraud-detection pilot.

A local agent that answers 'fraud' or 'not_fraud' for each case in the suite.
Scored on the pilot suite against the gold label (gold action == 'fraud').

Also computes decision churn of the fraud answer across model versions:
when a bigger model replaces another, does 'is fraud' stay stable?

Usage:
  python3 src/fraud_eval.py
"""

import json
from pathlib import Path

from run_eval import call

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "out" / "cases.jsonl"
OUT = ROOT / "out" / "fraud"

PROMPT = (
    "You are a bank fraud detector. Decide whether the message reports fraud:\n"
    "unauthorized transactions, a stolen or compromised card, or a transaction\n"
    "you did not make. Answer with one word: fraud or not_fraud.\n\n"
    "Message: {text}\nAnswer:"
)

MODELS = ["llama3.1:8b", "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5-coder:7b"]


def fraud_agent(model: str, text: str) -> str | None:
    raw, _ = call(model, PROMPT.format(text=text))
    low = raw.lower()
    if "not_fraud" in low or "not fraud" in low:
        return "not_fraud"
    if "fraud" in low:
        return "fraud"
    return None


def main() -> None:
    cases = [json.loads(line) for line in CASES.open()]
    gold = {c["id"]: ("fraud" if c["gold"] == "fraud" else "not_fraud") for c in cases}

    OUT.mkdir(parents=True, exist_ok=True)
    preds: dict[str, dict[str, str | None]] = {}

    print("model                  n_fraud_gold  accuracy  parse_fail")
    for model in MODELS:
        path = OUT / f"{model.replace(':', '-').replace('/', '-')}__v1.jsonl"
        done = {json.loads(l)["id"]: json.loads(l)["pred"] for l in path.open()} if path.exists() else {}
        recs = {}
        with path.open("a") as f:
            for c in cases:
                if c["id"] in done:
                    recs[c["id"]] = done[c["id"]]
                    continue
                p = fraud_agent(model, c["text"])
                recs[c["id"]] = p
                f.write(json.dumps({"id": c["id"], "model": model, "pred": p}) + "\n")
                f.flush()
        preds[model] = recs

    summary = {}
    for model in MODELS:
        ids = [i for i in gold if gold[i] == "fraud"]
        acc = sum(1 for i in gold if preds[model][i] == gold[i]) / len(gold)
        fails = sum(1 for i in gold if preds[model][i] is None)
        print(f"{model:22s} {len(ids):4d}           {acc:.3f}    {fails:4d}")
        summary[model] = {"n_fraud_gold": len(ids), "accuracy": acc, "parse_fail": fails}

    # churn of the fraud answer across model pairs, vs the first model
    print("\nchurn of the fraud answer (vs llama3.1:8b):")
    for model in MODELS[1:]:
        churn = sum(1 for i in preds[model] if preds[model][i] != preds["llama3.1:8b"][i]) / len(preds[model])
        print(f"  llama3.1:8b -> {model}: {churn:.1%}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
