# Deterministic evidence-grounded fallback drafting (not an LLM)
# Auto-handle requires ALL of: same-intent surviving evidence with a concrete action, zero warnings on the cited evidence, no conflict, intent confidence >= 0.5 (predeclared policy constant, never tuned). Anything else escalates with an explicit reason. URLs are stripped from replies; identifiers are never rendered. Abstention produces no fabricated answer.

import re

AUTO_CONFIDENCE_THRESHOLD = 0.5

URL_RE = re.compile(r"https?://\S+|www\.\S+|__URL__")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ACTION_VERB_RE = re.compile(
    r"(?i)\b(visit|go to|claim|call|contact|check|see|follow|use|fill|"
    r"complete|send|provide|reply|speak|ask|book|apply)\b")


def strip_urls(text):
    text, n = URL_RE.subn("[link withheld — verify the current page]", text)
    return text, n


def action_sentence(brand_texts):
    for t in brand_texts:
        for s in SENT_SPLIT_RE.split(t or ""):
            if ACTION_VERB_RE.search(s) and len(s.strip()) > 12:
                return s.strip()
    for t in brand_texts:
        for s in SENT_SPLIT_RE.split(t or ""):
            if len(s.strip()) > 12:
                return s.strip()
    return ""


def draft(query_text, intent, confidence, survivors, conflict, abstain):
    """Returns (reply, decision, code, reason, claims, warnings, abstained,
    grounding_status). Deterministic."""
    warnings = []
    same_action = [c for c in survivors
                   if c["source_intent"] == intent and c["signals"]["has_action"]]
    clean = [c for c in same_action if not c["warning_flags"]]

    if abstain or not survivors:
        return (_no_evidence(), "escalate", "NO_EVIDENCE",
                "No eligible historical evidence survived grounding.",
                [], ["no evidence available; human review required"], True,
                "insufficient_evidence")
    if conflict:
        return (_conflict(), "escalate", "CONFLICTING_EVIDENCE",
                "Top surviving evidence sources disagree; must not silently choose.",
                [], ["conflicting historical evidence; human review required"],
                False, "partially_grounded")
    if confidence < AUTO_CONFIDENCE_THRESHOLD:
        return (_uncertain(intent), "escalate", "LOW_INTENT_CONFIDENCE",
                f"Intent {intent} confidence {confidence:.2f} below policy "
                f"threshold {AUTO_CONFIDENCE_THRESHOLD}.",
                [], ["low intent confidence; human review required"],
                False, "partially_grounded")
    if not same_action:
        return (_uncertain(intent), "escalate",
                "INSUFFICIENT_APPLICABLE_EVIDENCE",
                "No same-intent evidence with a concrete action; answering "
                "would require unsupported claims.",
                [], ["no applicable action evidence; human review required"],
                False, "partially_grounded")
    cited = clean[0] if clean else None
    if cited is None:
        c0 = same_action[0]
        for w in c0["warning_flags"]:
            warnings.append(f"historical evidence carries {w}")
        if any("W_event" in c["warning_flags"] or
               c["provenance"]["event_period"] for c in same_action):
            return (_event(), "escalate", "EVENT_TIME_SENSITIVE",
                    "Applicable evidence is event-period; must not generalize "
                    "disruption instructions as current policy.",
                    [], warnings + ["event-specific evidence; human review required"],
                    False, "partially_grounded")
        return (_cautious(c0), "escalate",
                "EVENT_TIME_SENSITIVE"
                if any("W_time_sensitive" in c["warning_flags"] or "W_url" in
                       c["warning_flags"] for c in same_action)
                else "INSUFFICIENT_APPLICABLE_EVIDENCE",
                "Applicable evidence carries unresolved warnings.",
                [], warnings + ["human review required"], False,
                "partially_grounded")
    texts = [t for t in cited["evidence_text"].split("\n---\n")]
    sent = action_sentence(texts)
    sent, n_stripped = strip_urls(sent)
    if n_stripped:
        warnings.append("historical link withheld — verify the current page")
    reply = (f"Based on a similar past case from {cited['source_date'][-4:]} "
             f"({intent} requests like yours were handled as follows): "
             f"\"{sent}\" Note this reflects historical practice and may have "
             f"changed — please confirm current policy if anything differs.")
    return (reply, "auto_handle", "NONE", "",
            [{"claim": sent, "evidence_source_ids": [cited["source_dyad_id"]],
              "support": "supported"}], warnings, False, "grounded")


def _no_evidence():
    return ("I can't confirm this from available historical cases, so I won't "
            "guess. A human support agent will review your message shortly.")


def _conflict():
    return ("Similar past cases were handled in different ways, so I can't "
            "safely pick one for you. A human support agent will review your "
            "message shortly.")


def _uncertain(intent):
    return (f"I'm not confident enough about how to handle your {intent} "
            f"request from historical cases. A human support agent will "
            f"review your message shortly.")


def _event():
    return ("The closest historical cases come from a disruption period, and "
            "those instructions may not apply normally. A human support agent "
            "will review your message shortly.")


def _cautious(c0):
    return ("There is related historical evidence but it carries warnings "
            f"({', '.join(c0['warning_flags'])}), so I won't answer directly. "
            "A human support agent will review your message shortly.")
