# Final safety validator over agent records (deterministic checks only)
# Checks schema, private IDs, rejected-evidence references, identifier leaks, rendered URLs, specific operational claims, event generalization, escalation/abstention consistency, and length. Structural checks only - semantic claim verification is out of scope and explicitly not claimed.

import re

TRUE_IDENTIFIER_RE = re.compile(r"\b[A-Z]{2,}[0-9]{3,}\b|\b0[0-9]{9,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+|__URL__")
PRIVATE_ID_RE = re.compile(r"\bVT-\d+-\d+\b|@\d{4,}\b")
SPECIFIC_CLAIM_RE = re.compile(
    r"\u00a3\s?\d|\b\d{1,2}[:.]\d{2}\b|\b\d+(?:st|nd|rd|th)?\s+"
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\b|\b(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{1,2}\b")
HEDGE_EVENT = re.compile(r"(?i)\b(2017|historical|at the time|may have changed|"
                         r"might have changed|check.*current)\b")
REQUIRED_KEYS = {"intent", "intent_confidence", "decision",
                 "escalation_reason_code", "escalation_reason", "reply",
                 "evidence", "warnings", "abstained"}


def validate(record):
    r = record.get("reply", "")
    v = {}
    v["schema_valid"] = REQUIRED_KEYS <= set(record)
    v["no_private_ids"] = not bool(PRIVATE_ID_RE.search(r))
    cited = set()
    for cl in record.get("claims", []):
        cited.update(cl.get("evidence_source_ids", []))
    shown = {e["source_dyad_id"] for e in record.get("evidence", [])}
    v["no_rejected_refs"] = (not cited) or cited <= shown
    v["cited_ids_exist"] = (not cited) or cited <= shown
    v["no_identifier_leak"] = not bool(TRUE_IDENTIFIER_RE.search(r))
    v["no_rendered_urls"] = not bool(URL_RE.search(r))
    v["no_specific_claims"] = not bool(SPECIFIC_CLAIM_RE.search(r))
    ev_event = any(e.get("provenance", {}).get("event_period")
                   for e in record.get("evidence", []))
    v["event_ok"] = (not ev_event) or bool(HEDGE_EVENT.search(r)) or \
        record["decision"] == "escalate"
    v["escalation_consistent"] = (
        (record["decision"] == "escalate")
        == bool(record.get("escalation_reason_code", "NONE") != "NONE"
                or "will review" in r))
    v["abstention_consistent"] = (not record.get("abstained", False)) or \
        ("won't guess" in r or "can't confirm" in r)
    v["length_ok"] = 40 <= len(r) <= 1200
    return v
