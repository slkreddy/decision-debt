# Decision Debt - pilot

Does a model upgrade silently change business decisions that accuracy says are fine?

## What is here

| Path | Purpose |
|---|---|
| `data/banking77_train.csv` | BANKING77 training split (10,003 customer messages, 77 intents) |
| `src/build_cases.py` | Maps 77 intents to 6 routing actions and samples a 300-case suite |
| `out/cases.jsonl` | The suite: text, gold action, per-action cost |
| `out/suite_meta.json` | Cost-function parameters and sampling seed |
| `src/run_eval.py` | Resumable Ollama runner; one JSONL per (model, prompt) |
| `src/analyze.py` | Accuracy, churn, regressions, cost-weighted churn, McNemar |

## The suite

300 cases, 50 per action, stratified sample of BANKING77, seed 7.

Actions: `fraud`, `disputes`, `cards`, `payments`, `fees`, `account`.

Cost of routing a case to the wrong team (USD, by assumption):

```
cost(true, pred) = delay[true] + handling[pred]        (0 if pred == true)
```

| Action | Delay (USD) | Handling (USD) |
|---|---:|---:|
| fraud | 60 | 30 |
| disputes | 25 | 20 |
| payments | 18 | 14 |
| cards | 12 | 12 |
| fees | 7 | 9 |
| account | 5 | 8 |

These are the parameters used by `src/build_cases.py` to generate `out/cases.jsonl` and `out/suite_meta.json`. The delay term makes the cost matrix asymmetric: the cost depends on both the true action and the predicted team. For example, misrouting a fraud case to the account team costs $68 under these assumptions, while routing an account case to fraud costs $35.

## Metrics

- **Churn** — share of cases whose predicted action changes between two runs.
- **Regression rate** — share that changes from correct to wrong.
- **Cost-weighted churn** — cost carried by changed cases / total cost exposure.
- **Decision loss** — change in expected cost per 1,000 decisions.
- **McNemar exact** — whether the accuracy change is distinguishable from noise.

Churn and cost are reported together: churn is the assumption-free behavioural
number, cost is the business translation.

## Reproduce

```sh
python3 src/build_cases.py
python3 src/run_eval.py --models llama3.1:8b --prompt v1
python3 src/analyze.py
```

Runs are resumable; delete a file in `out/runs/` to redo that configuration.

## Scope and limits

- BANKING77 is intent classification, not enterprise policy decisions. The
  cost table is a stated assumption, not measured ground truth.
- Prompt `v2` adds routing rules, so `v1 -> v2` is a policy change as well as a
  prompt change; interpret it as a realistic upgrade, not a clean ablation.
- This is a pilot. It tests whether the effect exists, not how large it is in
  production.
