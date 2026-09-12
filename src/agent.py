# Support-agent orchestration: classify -> retrieve -> ground -> draft -> validate
# Frozen components only; no learning, no tuning, no golden data. On final validation failure the agent escalates with VALIDATION_FAILED — it never silently repairs a reply into a different claim.

import time

from . import classifier as clf_mod
from . import grounding as ground
from . import policy as pol
from . import response as resp
from . import retrieval as ret
from . import validation as val


def handle(text, prior_texts=None, customer_id="LIVE-QUERY", dyad_id="QUERY"):
    t0 = time.time()
    prior_texts = list(prior_texts or [])
    intent, conf = clf_mod.predict(text, prior_texts)
    retrieved = ret.retrieve(text, prior_texts,
                             query_dyad_id=dyad_id,
                             query_customer_id=str(customer_id))
    query = {"dyad_id": dyad_id, "customer_id": str(customer_id),
             "first_ts": "", "intent": intent, "secondary": ""}
    g = ground.ground(query, retrieved)
    surv = g["survivors"]
    reply, decision, code, reason, claims, warnings, abstained, gstatus = \
        resp.draft(text, intent, conf, surv, g["conflict"], not surv)
    record = {"intent": intent, "intent_confidence": conf,
              "decision": decision, "escalation_reason_code": code,
              "escalation_reason": reason, "reply": reply,
              "evidence": [{"source_dyad_id": c["source_dyad_id"],
                            "rank": c["rank"], "similarity": c["similarity"],
                            "source_intent": c["source_intent"],
                            "source_date": c["source_date"],
                            "age_bucket": c["age_bucket"],
                            "eligibility": c["eligibility"],
                            "warning_flags": c["warning_flags"],
                            "provenance": c["provenance"],
                            "signals": c["signals"]} for c in surv],
              "warnings": warnings, "abstained": abstained,
              "grounding_status": gstatus, "conflict": g["conflict"],
              "claims": claims, "query_dyad_id": dyad_id,
              "latency_s": round(time.time() - t0, 1)}
    pol.check_decision(decision, code)
    flags = val.validate(record)
    record["validation"] = flags
    if not all(flags.values()):
        record.update(
            decision="escalate", escalation_reason_code="VALIDATION_FAILED",
            escalation_reason="Generated reply failed final safety "
                              "validation: " + ", ".join(
                                  k for k, v in flags.items() if not v),
            reply=("I can't safely answer from the available evidence. A "
                   "human support agent will review your message shortly."),
            abstained=False, grounding_status="insufficient_evidence",
            claims=[])
    return record
