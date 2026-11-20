"""Run the decision-debt pilot against local Ollama models.

For every (model, prompt version) pair, route each case and append one JSONL
record. The runner is resumable: re-running skips case ids already present.

Usage:
  python src/run_eval.py --models llama3.1:8b --prompt v1 --limit 20
  python src/run_eval.py --models qwen2.5:0.5b,qwen2.5:1.5b --prompt v1
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "out" / "cases.jsonl"
OUT = ROOT / "out" / "runs"
OLLAMA = "http://localhost:11434/api/generate"

TEAMS = {
    "fraud": "unauthorised transactions, stolen or compromised cards, fraud reports",
    "disputes": "refunds, duplicate charges, wrong amounts, disputed fees",
    "cards": "card ordering, activation, PINs, card not working, delivery",
    "payments": "transfers, top-ups, pending or failed payments, balances",
    "fees": "questions about charges, exchange rates, limits, availability",
    "account": "identity verification, personal details, closing the account",
}

TEAM_BLOCK = "\n".join(f"- {k}: {v}" for k, v in TEAMS.items())

PROMPTS = {
    "v1": (
        "You are a bank support router. Route the customer message to exactly one team.\n\n"
        f"Teams:\n{TEAM_BLOCK}\n\n"
        "Answer with only the team name, lowercase, no punctuation.\n\n"
        'Message: "{text}"\nTeam:'
    ),
    "v2": (
        "You are a bank support router. Route the customer message to exactly one team.\n\n"
        f"Teams:\n{TEAM_BLOCK}\n\n"
        "Rules: if money is missing or a card may be compromised, choose fraud. "
        "If the customer wants money back, choose disputes. "
        "If the message is about a charge or rate, choose fees.\n\n"
        'Respond as JSON only: {{"team": "<team>"}}\n\n'
        'Message: "{text}"\n'
    ),
}


def call(model: str, prompt: str) -> tuple[str, float]:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_predict": 16},
        }
    ).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = json.loads(r.read())
    return payload.get("response", ""), time.time() - t0


def parse(raw: str) -> str | None:
    low = raw.lower()
    hits = [a for a in TEAMS if a in low]
    if not hits:
        return None
    # longest match wins, so "payments" does not shadow anything shorter
    return max(hits, key=len)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--prompt", default="v1", choices=sorted(PROMPTS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = [json.loads(line) for line in CASES.open()]
    if args.limit:
        cases = cases[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)

    for model in args.models.split(","):
        slug = model.replace(":", "-").replace("/", "-")
        path = OUT / f"{slug}__{args.prompt}.jsonl"
        done = {json.loads(line)["id"] for line in path.open()} if path.exists() else set()
        todo = [c for c in cases if c["id"] not in done]
        print(f"{model} [{args.prompt}]: {len(done)} done, {len(todo)} to go -> {path.name}")

        with path.open("a") as f:
            for i, c in enumerate(todo, 1):
                raw, dt = call(model, PROMPTS[args.prompt].format(text=c["text"]))
                rec = {
                    "id": c["id"],
                    "model": model,
                    "prompt": args.prompt,
                    "raw": raw.strip(),
                    "pred": parse(raw),
                    "latency_s": round(dt, 3),
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")


if __name__ == "__main__":
    main()
