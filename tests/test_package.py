# Package integrity + headline + end-to-end tests.
# The end-to-end test downloads MiniLM weights once and runs one live query; everything else is offline and fast.

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def is_hidden(p):
    return any(part.startswith(".") for part in Path(p).parts)


def test_exactly_two_markdown_files():
    mds = sorted(p for p in ROOT.rglob("*.md") if not is_hidden(p))
    assert [p.name for p in mds] == ["README.md", "REPORT.md"], mds


def test_no_history_naming():
    import re
    # patterns built dynamically so this file itself stays clean
    bad_name = re.compile("ST" + "EP_|Pha" + "se |pha" + "se_|st" + "ep_")
    hits = []
    for p in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.json")) + \
            list(ROOT.rglob("*.md")):
        if is_hidden(p):
            continue
        if bad_name.search(p.name):
            hits.append(str(p))
    assert hits == [], hits
    for p in [ROOT / "README.md", ROOT / "REPORT.md"]:
        text = open(p, encoding="utf-8").read()
        assert ("ST" + "EP_") not in text, p
        assert ("Pha" + "se ") not in text, p


def test_taxonomy_locked():
    tax = json.load(open(ROOT / "config" / "taxonomy.json", encoding="utf-8"))
    assert [t["intent_id"] for t in tax["intents"]] == [
        "delay_status", "cancellation_status", "live_departure_info",
        "rebooking_alternative_route", "ticket_acceptance",
        "refund_delay_repay", "booking_ticket_issue", "lost_property",
        "onboard_experience", "general_feedback"]


def test_golden_headline_reproduced():
    from scripts.evaluate_classifier import evaluate, load_predictions
    rep = evaluate(load_predictions(
        ROOT / "artifacts" / "evaluation" / "hybrid_predictions.jsonl"))
    assert rep["primary_accuracy"] == 0.69
    assert rep["macro_f1"] == 0.6777
    assert rep["primary_accuracy_wilson95"] == [0.6228, 0.75]
    assert rep["n_golden"] == 200


def test_golden_set_shape():
    rows = list(csv.DictReader(open(
        ROOT / "artifacts" / "evaluation" / "golden_labels.csv",
        encoding="utf-8")))
    assert len(rows) == 200
    assert len({r["golden_id"] for r in rows}) == 200


def test_end_to_end_agent():
    from src.agent import handle
    rec = handle("Where is my refund for the delayed train?")
    assert rec["intent"] == "refund_delay_repay"
    assert rec["decision"] in ("auto_handle", "escalate")
    assert rec["escalation_reason_code"] in (
        "NONE", "NO_EVIDENCE", "INSUFFICIENT_APPLICABLE_EVIDENCE",
        "CONFLICTING_EVIDENCE", "LOW_INTENT_CONFIDENCE",
        "EVENT_TIME_SENSITIVE", "VALIDATION_FAILED")
    assert all(rec["validation"].values())
    assert "VT-" not in rec["reply"]
