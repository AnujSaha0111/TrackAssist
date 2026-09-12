# Escalation policy: explicit reason codes, no silent failures.
# Codes: NO_EVIDENCE, INSUFFICIENT_APPLICABLE_EVIDENCE, CONFLICTING_EVIDENCE, LOW_INTENT_CONFIDENCE, EVENT_TIME_SENSITIVE, VALIDATION_FAILED, NONE (auto-handle only). Every escalation carries a machine-readable code and a human-readable reason.

CODES = {"NO_EVIDENCE", "INSUFFICIENT_APPLICABLE_EVIDENCE",
         "CONFLICTING_EVIDENCE", "LOW_INTENT_CONFIDENCE",
         "EVENT_TIME_SENSITIVE", "VALIDATION_FAILED", "NONE"}


def check_decision(decision, code):
    if code not in CODES:
        raise AssertionError(f"unknown reason code: {code}")
    if decision == "auto_handle" and code != "NONE":
        raise AssertionError("auto-handle must carry code NONE")
    if decision == "escalate" and code == "NONE":
        raise AssertionError("escalation requires a reason code")
    if decision not in ("auto_handle", "escalate"):
        raise AssertionError(f"unknown decision: {decision}")
    return True
