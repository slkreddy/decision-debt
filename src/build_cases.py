"""Build the decision-debt pilot suite from BANKING77.

Each case is a customer message, a gold routing action, and a cost for every
possible action. Costs are generated from a small, explicit cost function so
the table is auditable and easy to re-parameterise.

Usage: python src/build_cases.py
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "banking77_train.csv"
OUT = ROOT / "out"
SEED = 7
PER_GROUP = 50

ACTIONS = ["fraud", "disputes", "cards", "payments", "fees", "account"]

# 77 BANKING77 intents -> 6 routing actions.
MAPPING = {
    # fraud / urgent loss
    "compromised_card": "fraud",
    "lost_or_stolen_card": "fraud",
    "lost_or_stolen_phone": "fraud",
    "cash_withdrawal_not_recognised": "fraud",
    "card_payment_not_recognised": "fraud",
    "direct_debit_payment_not_recognised": "fraud",
    # disputes / refunds
    "Refund_not_showing_up": "disputes",
    "request_refund": "disputes",
    "transaction_charged_twice": "disputes",
    "reverted_card_payment?": "disputes",
    "extra_charge_on_statement": "disputes",
    "wrong_amount_of_cash_received": "disputes",
    "wrong_exchange_rate_for_cash_withdrawal": "disputes",
    "card_payment_wrong_exchange_rate": "disputes",
    # cards
    "activate_my_card": "cards",
    "card_about_to_expire": "cards",
    "card_acceptance": "cards",
    "card_arrival": "cards",
    "card_delivery_estimate": "cards",
    "card_linking": "cards",
    "card_not_working": "cards",
    "card_swallowed": "cards",
    "change_pin": "cards",
    "contactless_not_working": "cards",
    "get_disposable_virtual_card": "cards",
    "get_physical_card": "cards",
    "getting_spare_card": "cards",
    "getting_virtual_card": "cards",
    "order_physical_card": "cards",
    "passcode_forgotten": "cards",
    "pin_blocked": "cards",
    "virtual_card_not_working": "cards",
    "visa_or_mastercard": "cards",
    "supported_cards_and_currencies": "cards",
    "disposable_card_limits": "cards",
    "apple_pay_or_google_pay": "cards",
    # payments / transfers / top-ups
    "automatic_top_up": "payments",
    "balance_not_updated_after_bank_transfer": "payments",
    "balance_not_updated_after_cheque_or_cash_deposit": "payments",
    "beneficiary_not_allowed": "payments",
    "cancel_transfer": "payments",
    "declined_card_payment": "payments",
    "declined_cash_withdrawal": "payments",
    "declined_transfer": "payments",
    "exchange_via_app": "payments",
    "failed_transfer": "payments",
    "pending_card_payment": "payments",
    "pending_cash_withdrawal": "payments",
    "pending_top_up": "payments",
    "pending_transfer": "payments",
    "receiving_money": "payments",
    "top_up_by_bank_transfer_charge": "payments",
    "top_up_by_card_charge": "payments",
    "top_up_by_cash_or_cheque": "payments",
    "top_up_failed": "payments",
    "top_up_limits": "payments",
    "top_up_reverted": "payments",
    "topping_up_by_card": "payments",
    "transfer_into_account": "payments",
    "transfer_not_received_by_recipient": "payments",
    "transfer_timing": "payments",
    "verify_top_up": "payments",
    # fees
    "card_payment_fee_charged": "fees",
    "cash_withdrawal_charge": "fees",
    "exchange_charge": "fees",
    "exchange_rate": "fees",
    "transfer_fee_charged": "fees",
    "fiat_currency_support": "fees",
    "atm_support": "fees",
    "age_limit": "fees",
    # account
    "country_support": "account",
    "edit_personal_details": "account",
    "terminate_account": "account",
    "unable_to_verify_identity": "account",
    "verify_my_identity": "account",
    "verify_source_of_funds": "account",
    "why_verify_identity": "account",
}

# Business cost parameters (USD).
# delay[true]  = cost of the delay caused when a case sits in the wrong queue
# handling[pred] = cost of the wrong team touching the case
DELAY = {"fraud": 60, "disputes": 25, "payments": 18, "cards": 12, "fees": 7, "account": 5}
HANDLING = {"fraud": 30, "disputes": 20, "payments": 14, "cards": 12, "fees": 9, "account": 8}


def cost_of(true_action: str, predicted: str) -> int:
    if true_action == predicted:
        return 0
    return DELAY[true_action] + HANDLING[predicted]


def main() -> None:
    rows = list(csv.DictReader(DATA.open()))
    missing = {r["category"] for r in rows} - set(MAPPING)
    if missing:
        raise SystemExit(f"unmapped intents: {sorted(missing)}")

    by_group: dict[str, list[dict]] = {a: [] for a in ACTIONS}
    for r in rows:
        by_group[MAPPING[r["category"]]].append(r)

    rng = random.Random(SEED)
    cases = []
    for action in ACTIONS:
        pool = by_group[action]
        for r in rng.sample(pool, min(PER_GROUP, len(pool))):
            cases.append(
                {
                    "id": f"b77_{len(cases):04d}",
                    "source": "banking77",
                    "text": r["text"].strip(),
                    "gold": action,
                    "intent": r["category"],
                    "cost": {a: cost_of(action, a) for a in ACTIONS},
                }
            )

    OUT.mkdir(exist_ok=True)
    with (OUT / "cases.jsonl").open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    meta = {
        "n_cases": len(cases),
        "actions": ACTIONS,
        "per_group": PER_GROUP,
        "seed": SEED,
        "delay": DELAY,
        "handling": HANDLING,
    }
    (OUT / "suite_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {len(cases)} cases -> {OUT/'cases.jsonl'}")


if __name__ == "__main__":
    main()
