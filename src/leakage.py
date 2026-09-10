# Leakage guards: fail loudly on golden contact or taxonomy drift.
# The golden label file doubles as the ID registry for refusal checks (IDs only are read for this purpose; label values never enter training).

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_CSV = ROOT / "artifacts" / "evaluation" / "golden_labels.csv"
TAXONOMY = ROOT / "config" / "taxonomy.json"

INTENTS = ["delay_status", "cancellation_status", "live_departure_info",
           "rebooking_alternative_route", "ticket_acceptance",
           "refund_delay_repay", "booking_ticket_issue", "lost_property",
           "onboard_experience", "general_feedback"]

TAXONOMY_FP = "c43c9831911d3a61"

def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

def load_golden():
    rows = list(csv.DictReader(open(GOLDEN_CSV, encoding="utf-8")))
    if len(rows) != 200:
        raise AssertionError(f"golden is {len(rows)} rows, expected 200")
    return rows

def golden_dyad_ids():
    return {r["dyad_id"] for r in load_golden()}

def assert_no_golden_dyads(dyad_ids, where):
    leaked = set(dyad_ids) & golden_dyad_ids()
    if leaked:
        raise AssertionError(
            f"LEAKAGE: {len(leaked)} golden dyads in {where}: "
            f"{sorted(leaked)[:5]}")

def assert_taxonomy_unchanged(expected=TAXONOMY_FP):
    actual = sha16(TAXONOMY)
    if actual != expected:
        raise AssertionError(
            f"taxonomy changed: {actual} != {expected}")

def assert_taxonomy_ids(doc):
    ids = [t["intent_id"] for t in doc["intents"]]
    if ids != INTENTS:
        raise AssertionError(f"taxonomy intent list changed: {ids}")

def assert_predictions_match_golden(pred_ids):
    golden_ids = [r["golden_id"] for r in load_golden()]
    pred_ids = list(pred_ids)
    if len(pred_ids) != 200 or set(pred_ids) != set(golden_ids):
        missing = set(golden_ids) - set(pred_ids)
        extra = set(pred_ids) - set(golden_ids)
        raise AssertionError(
            f"prediction IDs must match the 200 golden IDs exactly: "
            f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    if len(set(pred_ids)) != 200:
        raise AssertionError("duplicate prediction IDs")

def assert_valid_intent(value, where):
    if value not in INTENTS and value not in ("", None, "abstain"):
        raise AssertionError(f"invalid intent {value!r} in {where}")