# Grounding gate tests (fast, no models)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grounding import age_bucket, evaluate

BASE_DOC = {"dyad_id": "VT-1-2", "customer_id": "2",
            "brand_texts": ["Hi, please claim via this link https://t.co/x "
                            "from our refunds team."],
            "first_ts": "Tue Oct 31 10:00:00 +0000 2017",
            "intent": "refund_delay_repay", "secondary": ""}
BASE_Q = {"dyad_id": "VT-9-9", "customer_id": "9",
          "first_ts": "Wed Nov 01 10:00:00 +0000 2017",
          "intent": "refund_delay_repay", "secondary": ""}


def test_same_dyad_and_customer_rejected():
    assert "H1_same_dyad" in evaluate(
        {**BASE_Q, "dyad_id": "VT-1-2"}, BASE_DOC)["rejection_reasons"]
    assert "H2_same_customer" in evaluate(
        {**BASE_Q, "customer_id": "2"}, BASE_DOC)["rejection_reasons"]


def test_true_identifier_rejected_audit_flag_separate():
    doc = {**BASE_DOC, "brand_texts": ["Your booking reference is AB123456."]}
    assert "H5_true_identifier" in evaluate(BASE_Q, doc)["rejection_reasons"]
    doc2 = {**BASE_DOC, "brand_texts": ["Hi @123241 please see our reply."]}
    r2 = evaluate(BASE_Q, doc2)
    assert r2["eligibility"] != "rejected"
    assert "W_identifier_audit" in r2["warning_flags"]


def test_missing_provenance_rejected():
    assert "H3_no_brand_turns" in evaluate(
        BASE_Q, {**BASE_DOC, "brand_texts": []})["rejection_reasons"]
    assert "H4_missing_provenance" in evaluate(
        BASE_Q, {**BASE_DOC, "first_ts": ""})["rejection_reasons"]


def test_event_and_url_flags():
    doc = {**BASE_DOC, "first_ts": "Wed Nov 22 10:00:00 +0000 2017",
           "brand_texts": ["Due to flooding today, buses replace trains."]}
    assert "H8_event_nongeneralizable" in evaluate(BASE_Q, doc)[
        "rejection_reasons"]
    assert "W_url" in evaluate(BASE_Q, BASE_DOC)["warning_flags"]
    assert age_bucket("Wed Nov 01 10:00:00 +0000 2017",
                      "Sat Jan 10 10:00:00 +0000 2015") == ">180d"


def test_evidence_preserved_and_deterministic():
    raw = "Hi,  claim   via this link."
    r = evaluate(BASE_Q, {**BASE_DOC, "brand_texts": [raw]})
    assert r["evidence_text"] == raw
    assert evaluate(BASE_Q, BASE_DOC) == evaluate(BASE_Q, BASE_DOC)
