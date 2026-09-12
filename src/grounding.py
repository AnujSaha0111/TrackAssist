# Deterministic grounding/applicability gate (policy rules, not learned).
# For each retrieved dyad: HARD REJECT (H1-H8), SOFT WARNING (W-*), positive signals recorded but never scored, top-2 conflict audit, abstain when nothing survives. Evidence text is preserved verbatim; nothing redacted.

import re
from datetime import datetime

TRUE_IDENTIFIER_RE = re.compile(r"\b[A-Z]{2,}[0-9]{3,}\b|\b0[0-9]{9,}\b")
BOOKING_SPECIFIC_RE = re.compile(
    r"(?i)\b(booking reference|reference number|case number|"
    r"confirmation number|your reference|booking ref|case ref)\b")
DISRUPTION_NOUN_RE = re.compile(
    r"(?i)\b(flood\w*|strike|striking|engineering work\w*|"
    r"rail replacement|bus replacement|signal failure|overrunning)\b")
RELDAY_RE = re.compile(r"(?i)\b(today|tonight|this morning|this evening)\b")
ACTION_RE = re.compile(
    r"(?i)\b(visit|go to|claim at|call|contact|dm us|direct message|"
    r"check|see|follow|use|fill|complete|send|provide|reply)\b.{0,40}"
    r"(http|www\.|__URL__|link|form|site|page|number|team|line\b)|"
    r"https?://|__URL__")
TIME_SENSITIVE_RE = re.compile(
    r"\b\d{1,2}[:.]\d{2}\b|\bplatform \d+\b|\bon time\b|\brunning\b|"
    r"\bdelayed \d+\b|\b[A-Z\u00a3$]\s?\d|\b\d+ ?mins?\b")
POLICY_RE = re.compile(
    r"(?i)\b(policy|policies|entitled|valid on|accepted on|terms\b|"
    r"conditions\b)\b")
CONSERVATIVE_NUM_RE = re.compile(r"@\d+\b|\b\d{6,}\b")
URL_RE = re.compile(r"https?://\S+|__URL__")
WORD_RE = re.compile(r"[A-Za-z]+")
EVENT_DAYS = {"Nov 22", "Nov 23", "Dec 02", "Dec 03"}


def parse_ts(ts):
    try:
        return datetime.strptime(ts, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None


def age_bucket(q_ts, s_ts):
    q, s = parse_ts(q_ts or ""), parse_ts(s_ts or "")
    if q is None or s is None:
        return "unknown"
    d = abs((q - s).days)
    if d <= 30:
        return "<=30d"
    if d <= 90:
        return "31-90d"
    if d <= 180:
        return "91-180d"
    return ">180d"


def is_event(first_ts):
    return bool("2017" in (first_ts or "")
                and (first_ts or "")[4:10] in EVENT_DAYS)


def content_words(text):
    import html
    return WORD_RE.findall(html.unescape(text or "").lower())


def evaluate(query, doc):
    """query/doc mappings; returns the candidate record."""
    rej, warn = [], []
    brand_joined = " ".join(doc["brand_texts"])
    if doc["dyad_id"] == query["dyad_id"]:
        rej.append("H1_same_dyad")
    if str(doc["customer_id"]) == str(query["customer_id"]):
        rej.append("H2_same_customer")
    if not doc["brand_texts"]:
        rej.append("H3_no_brand_turns")
    if not doc["dyad_id"] or not brand_joined.strip() or not doc["first_ts"]:
        rej.append("H4_missing_provenance")
    true_ident = bool(TRUE_IDENTIFIER_RE.search(brand_joined))
    if true_ident:
        rej.append("H5_true_identifier")
    has_action = bool(ACTION_RE.search(brand_joined))
    if doc["intent"] == "general_feedback" and not has_action:
        rej.append("H6_chatter_no_action")
    if BOOKING_SPECIFIC_RE.search(brand_joined):
        rej.append("H7_booking_specific")
    event = is_event(doc["first_ts"])
    if event and (DISRUPTION_NOUN_RE.search(brand_joined)
                 or RELDAY_RE.search(brand_joined)):
        rej.append("H8_event_nongeneralizable")

    bucket = age_bucket(query.get("first_ts", ""), doc["first_ts"])
    if not rej:
        if bucket == ">180d":
            warn.append("W_old")
        if URL_RE.search(brand_joined):
            warn.append("W_url")
        if event:
            warn.append("W_event")
        if TIME_SENSITIVE_RE.search(brand_joined):
            warn.append("W_time_sensitive")
        if doc["intent"] != query.get("intent", ""):
            warn.append("W_ambiguous")
        if POLICY_RE.search(brand_joined):
            warn.append("W_policy")
        if not true_ident and CONSERVATIVE_NUM_RE.search(brand_joined):
            warn.append("W_identifier_audit")

    resolution_present = any(len(content_words(t)) >= 8
                             for t in doc["brand_texts"])
    return {
        "source_dyad_id": doc["dyad_id"],
        "source_intent": doc["intent"],
        "source_date": doc["first_ts"],
        "age_bucket": bucket,
        "eligibility": "rejected" if rej else ("warning" if warn else "eligible"),
        "rejection_reasons": rej,
        "warning_flags": warn,
        "evidence_text": "\n---\n".join(doc["brand_texts"]),
        "provenance": {"dyad_id": doc["dyad_id"],
                       "customer_id": str(doc["customer_id"]),
                       "first_ts": doc["first_ts"],
                       "event_period": event,
                       "n_brand_turns": len(doc["brand_texts"])},
        "signals": {
            "same_primary": doc["intent"] == query.get("intent", ""),
            "compatible_secondary": bool(
                doc.get("secondary", "") == query.get("secondary", "")
                or not query.get("secondary", "")),
            "has_action": has_action,
            "resolution_present": resolution_present,
        },
    }


def ground(query, retrieved):
    """Attach rank/similarity, compute survivors, conflict, decision."""
    cands = []
    for rank, item in enumerate(retrieved, 1):
        rec = evaluate(query, item["doc"])
        rec["rank"] = rank
        rec["similarity"] = item["similarity"]
        cands.append(rec)
    surv = [c for c in cands if c["eligibility"] != "rejected"]
    for c in surv:
        c["signals"]["support_n"] = sum(
            1 for o in surv
            if o["source_dyad_id"] != c["source_dyad_id"]
            and o["source_intent"] == c["source_intent"])
    conflict = (len(surv) >= 2 and surv[0]["source_intent"]
                != surv[1]["source_intent"]
                and surv[0]["signals"]["has_action"]
                != surv[1]["signals"]["has_action"])
    return {"candidates": cands, "survivors": surv, "conflict": conflict,
            "decision": "evidence_available" if surv else "abstain"}
