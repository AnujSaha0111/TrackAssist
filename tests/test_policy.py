# Escalation policy + final validator tests (fast, no models)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.policy import check_decision
from src.response import AUTO_CONFIDENCE_THRESHOLD, draft
from src.validation import validate

EV_OK = {"source_dyad_id": "VT-1-2", "source_intent": "refund_delay_repay",
         "source_date": "Tue Oct 31 10:00:00 +0000 2017",
         "warning_flags": [],
         "provenance": {"event_period": False},
         "evidence_text": "Hi, please claim via the refunds page of our site.",
         "signals": {"has_action": True}}


def test_auto_handle_path():
    reply, dec, code, reason, claims, warns, abst, gs = draft(
        "Where is my refund?", "refund_delay_repay", 0.9, [EV_OK], False, False)
    assert dec == "auto_handle" and code == "NONE" and not abst
    assert gs == "grounded" and len(claims) == 1
    assert "VT-1-2" not in reply
    check_decision(dec, code)


def test_no_evidence_abstains():
    reply, dec, code, reason, claims, warns, abst, gs = draft(
        "Where is my refund?", "refund_delay_repay", 0.9, [], False, True)
    assert dec == "escalate" and code == "NO_EVIDENCE" and abst
    assert "won't guess" in reply and claims == []
    check_decision(dec, code)


def test_conflict_escalates():
    _, dec, code, *_ = draft("q", "delay_status", 0.9, [EV_OK],
                             True, False)
    assert dec == "escalate" and code == "CONFLICTING_EVIDENCE"


def test_low_confidence_escalates():
    _, dec, code, *_ = draft("q", "delay_status", 0.1, [EV_OK],
                             False, False)
    assert dec == "escalate" and code == "LOW_INTENT_CONFIDENCE"
    assert AUTO_CONFIDENCE_THRESHOLD == 0.5


def test_warned_evidence_escalates():
    ev = dict(EV_OK, warning_flags=["W_url"])
    _, dec, code, *_ = draft("q", "refund_delay_repay", 0.9, [ev],
                             False, False)
    assert dec == "escalate" and code in ("EVENT_TIME_SENSITIVE",
                                          "INSUFFICIENT_APPLICABLE_EVIDENCE")


def test_validator_catches_leaks():
    rec = {"intent": "x", "intent_confidence": 0.9, "decision": "auto_handle",
           "escalation_reason_code": "NONE", "escalation_reason": "",
           "reply": "See VT-1-2 and call 01234567890 https://evil.example.com",
           "evidence": [{"source_dyad_id": "VT-1-2",
                         "provenance": {"event_period": False}}],
           "warnings": [], "abstained": False}
    v = validate(rec)
    assert not v["no_private_ids"]
    assert not v["no_identifier_leak"]
    assert not v["no_rendered_urls"]


def test_validator_passes_clean_record():
    rec = {"intent": "refund_delay_repay", "intent_confidence": 0.9,
           "decision": "auto_handle", "escalation_reason_code": "NONE",
           "escalation_reason": "",
           "reply": "Based on a similar past case, claims go through the "
                    "refunds page. Note this reflects historical practice.",
           "evidence": [{"source_dyad_id": "VT-1-2",
                         "provenance": {"event_period": False}}],
           "claims": [{"evidence_source_ids": ["VT-1-2"]}],
           "warnings": [], "abstained": False}
    assert all(validate(rec).values())
