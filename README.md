# TrackAssist

### An Evidence-Grounded AI Support Agent for Rail Customer Service

TrackAssist is an AI customer-support system that classifies customer requests, retrieves historically similar support resolutions, applies grounding and safety checks, drafts a conservative response and decides whether the request can be handled automatically or should be escalated.

TrackAssist is an evidence-grounded customer-support agent developed and evaluated using the Customer Support on Twitter dataset, with VirginTrains selected as the target brand. TrackAssist is an independent engineering project and is not an official VirginTrains product.

- **Target brand:** VirginTrains (selected by transparent scoring over 108 brands)
- **Headline:** intent classification accuracy **69.0%** (Wilson 95% CI 62.3–75.0, macro F1 0.6777) on the frozen 200-row golden set. This is INTENT CLASSIFICATION accuracy, not customer-resolution rate.
- **Reproduce:** `py scripts/evaluate_classifier.py --predictions artifacts/evaluation/hybrid_predictions.jsonl` (~1 s)
- **Architecture:** customer text → hybrid classifier → historical retrieval → grounding gate → fallback draft + escalation → safety validator
- **Key limitations:** n=200 with overlapping intervals across variants; 2017-stale data; noisy weak training labels; 0/180 auto-handle by conservative policy; no human quality judgments exist.

## Quick start (<15 min, CPU-only, no API key)

```bash
py -m pip install -r requirements.txt
py scripts/evaluate_classifier.py --predictions artifacts/evaluation/hybrid_predictions.jsonl
# -> accuracy 0.690, macro F1 0.6777 (frozen predictions, ~1 s)
py scripts/run_agent.py --text "Where is my refund for the delayed train?"
# -> intent + evidence + escalation decision (~3 min first start: MiniLM
#    weights download once ~90MB, then corpus encoding; cached afterwards)
py -m pytest tests/ -q
```

## Architecture

```text
Customer message
        |
Intent classifier (word+char TF-IDF + MiniLM -> logistic regression)
        |
Historical resolution retrieval (MiniLM cosine over 3,985 dyads)
        |
Grounding + safety gate (hard rejects, warnings, conflict audit, abstention)
        |
Response drafting (deterministic evidence-grounded fallback)
        |
Auto-handle / escalate (reason-coded)
        |
Safety validation (failures escalate, never silently repaired)
```

## Evaluation

- **Golden set** (`artifacts/evaluation/golden_labels.csv`): 200 hand-adjudicated customer messages with primary/secondary intent labels, sampled across months, thread sizes and intents with leakage guards (unique dyads/customers/targets). Reserved strictly for evaluation.
- **Response set** (`artifacts/examples/response_eval_cases.jsonl`): 180 separate non-golden cases with grounded evidence for answer-handling study. No human annotations exist yet, so only automated validator/escalation diagnostics are reported - never presented as human quality judgments.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority | 0.195 | 0.033 |
| TF-IDF (word 1–2g) | 0.640 | 0.634 |
| Word + character TF-IDF | 0.650 | 0.6565 |
| Semantic (MiniLM) | 0.670 | 0.6605 |
| Hybrid (word+char + MiniLM) | **0.690** | **0.6777** |

All on the frozen n=200 golden set (CIs overlap - ranking is suggestive, not proven). Strict customer-overlap reruns change nothing (Δ≈0). Retrieval Hit@3 ≈ 0.77–0.78 (proxy agreement with weak labels, not human relevance). Full numbers: `artifacts/evaluation/classification_results.json`.

## Intent Taxonomy

| Intent | Customer need |
|---|---|
| delay_status | Train already delayed; delay/arrival questions |
| cancellation_status | Service cancelled / is it cancelled |
| live_departure_info | Will it run / on time / platform / works / strikes |
| rebooking_alternative_route | Change journey: train/route/mode, replacement transport |
| ticket_acceptance | Use EXISTING ticket on another operator/service/day |
| refund_delay_repay | Money back: refunds, Delay Repay, compensation |
| booking_ticket_issue | Buy, reference, collect, reserve, upgrade, policy |
| lost_property | Left/lost/found item |
| onboard_experience | Concrete onboard/station problem |
| general_feedback | No actionable request: praise, venting, banter |

## Historical Grounding

TrackAssist uses a derived VirginTrains support corpus from the Customer Support on Twitter dataset: 3,985 VirginTrains dyads (`data/corpus.jsonl`, derived - raw 500MB CSV not shipped) with customer turns + brand replies. Retrieval returns dyads; the gate rejects same-dyad/customer matches, identifier-bearing replies, booking-specific and non-generalizable event content and action-less chatter, while flagging URLs, event periods, stale content and numeric artifacts. Historical evidence is never proof of current policy - all corpus content is from 2017.

## Escalation

Reason-coded, deterministic: NO_EVIDENCE, INSUFFICIENT_APPLICABLE_EVIDENCE, CONFLICTING_EVIDENCE, LOW_INTENT_CONFIDENCE, EVENT_TIME_SENSITIVE, VALIDATION_FAILED (NONE for auto-handle). The current conservative policy escalates all 180 evaluated response cases - that is safety behavior under a strict gate, not a success metric. Every escalation carries a code and a human-readable reason.

## Failure Analysis

Top modes (real evaluation rows):
1. Feedback/onboard tone boundary (sarcasm, venting, praise).
2. Question-shaped chatter read as status requests (and reverse).
3. Seat/reservation vocabulary shared by onboard and booking.
4. Validity-vs-change boundary (earlier/missed-train/barrier wording).
5. Established-vs-open delay/cancel/live wording.
Plus: stale/event evidence, identifier quarantine, fabrication-driven abstention, conflict deadlock.

## Limitations

n=200 with overlapping variant intervals; rare intents unstable (delay 8, acceptance 9); weak noisy training labels; 2017-stale corpus; retrieval applicability is not current correctness; validator pass is not response quality; 0% auto-handle (safe, not useful); no human annotations or agreement anywhere; 6 GB RAM blocks larger local models; CPU-only.

## What Is Misleading About the Headline Number?

69.0% is intent-label agreement on 200 curated messages — not the share of customers receiving good answers. Intervals overlap all variants; rare classes swing on 1–2 examples; training labels are noisy keyword rules; dyad-overlap sensitivity is ~zero but author-style leakage can't be excluded; response quality is unmeasured.

## Sources

- Customer Support on Twitter, Kaggle (`thoughtvector/customer-support-on-twitter`) - primary data; subsampled per the assignment. Download separately if rebuilding the corpus; not required to run this package.
- Banking77 (Hugging Face `PolyAI/banking77`) - intent-exploration reference only; never merged; not shipped.
- `sentence-transformers` + `all-MiniLM-L6-v2` - frozen retrieval representation and one classifier variant (no fine-tuning).
- Qwen2.5-0.5B-Instruct (Qwen) — evaluated once as a response generator, found inadequate, excluded from the agent; documented in REPORT.
- scikit-learn (TF-IDF, logistic regression, metrics), SciPy (sparse matrices), PyTorch (CPU inference), transformers (model loading). All application code in `src/` and `scripts/` is original.
