# TrackAssist

## Evidence-Grounded AI Support for Rail Customer Service

- **Project:** TrackAssist
- **Target brand:** VirginTrains
- **Dataset:** Customer Support on Twitter

TrackAssist is an evidence-grounded customer-support agent developed and evaluated using the Customer Support on Twitter dataset, with VirginTrains selected as the target brand. TrackAssist is an independent engineering project and is not an official VirginTrains product.

## 1. Problem Framing

Twitter customer support for VirginTrains: noisy, multi-turn, time-pressured messages. Good means: correct intent on support requests, no action on chatter, answers grounded in how similar issues were actually resolved, and honest escalation whenever evidence is missing, conflicting, stale, or customer-specific. Deliberately not built: live-policy RAG, LLM classification or judging, auto-send, escalation ML, golden top-ups.

## 2. Data and Intent Taxonomy

Customer Support on Twitter (Kaggle): 2.8M tweets, 108 brands. VirginTrains was selected as the target brand because its support conversations contain practical, resolution-oriented interactions, as measured by transparent scoring for on-Twitter grounding quality. Threads were reconstructed into 18,188 per-customer dyads (7,991 eligible); a locked 10-intent taxonomy (delay, cancellation, live departures, rebooking, ticket acceptance, refund, booking, lost property, onboard, general feedback) was adjudicated over a 200-row golden set sampled across months, thread sizes, and intents with strict leakage guards. Training used 7,791 non-golden dyads with conservative keyword weak labels (63.9% coverage; ambiguity preserved, never forced).

## 3. System Design

Hybrid intent classifier (word+char TF-IDF 58,104 dims + MiniLM 384d → logistic regression) → MiniLM-first top-10 retrieval over 3,985 dyads → deterministic grounding gate (hard rejects H1–H8, warnings W-*, conflict audit, abstention) → deterministic evidence-quoting fallback drafting → reason-coded escalation → final safety validator (failures escalate, never silently repaired). No LLM in the agent: a 0.5B local model failed the frozen response contract and larger models exceed available hardware.

## 4. Evaluation Methodology

Golden 200 reserved strictly for evaluation (never trained/selected/tuned on); customer-disjoint development split; strict golden-customer reruns; single frozen evaluation pass per configuration; Wilson intervals throughout. Response study uses a separate 180-case set with automated validator/escalation diagnostics only - no human annotations exist, so no human quality numbers are reported anywhere.

## 5. Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority | 0.195 | 0.033 |
| TF-IDF | 0.640 | 0.634 |
| Word + character | 0.650 | 0.6565 |
| Semantic (MiniLM) | 0.670 | 0.6605 |
| Hybrid | **0.690** (CI 0.623–0.750) | **0.6777** |

Hybrid leads on accuracy, macro, and weighted F1 with the largest macro gain observed; strict rerun identical. Retrieval Hit@3 ≈ 0.78 (proxy). Response set: 180 escalate / 0 auto-handle under the conservative gate; all 180 pass final validation.

## 6. Historical Resolution Grounding

Retrieval returns full dyads with provenance; the gate rejects identifier-bearing, booking-specific, non-generalizable event, and action-less chatter evidence while flagging URLs, event periods, stale content, and numeric artifacts. Historical evidence is applicability-filtered material, never proof of current policy - the corpus is from 2017.

## 7. Response and Escalation Behavior

Fallback quotes the top same-intent actionable sentence verbatim with staleness framing; URLs stripped; identifiers never rendered. Escalation codes (NO_EVIDENCE, INSUFFICIENT_APPLICABLE_EVIDENCE, CONFLICTING_EVIDENCE, LOW_INTENT_CONFIDENCE, EVENT_TIME_SENSITIVE, VALIDATION_FAILED) fire on predeclared rules. The 0%-auto-handle outcome is conservative safety behavior, not automation success.

## 8. Failure Analysis

1. **Tone boundary** (feedback↔onboard, ~27 errors): sarcastic wifi complaint ("Paid £5 for wi-fi that doesn't work. Fantastic, thank you") and coach-lettering jokes defeat lexical/tone cues. Hypothesis: no sarcasm or ask-vs-report syntax. Fix: tone-aware features/sequence model.
2. **Question-shaped chatter → live info** (polls, praise with times, jokes phrased as questions): interrogative shape outweighs no-ask content. Fix: ask-shape vs report-shape modeling.
3. **Seat/reservation overlap** (standing reports → booking): shared vocabulary across onboard/booking. Fix: seating-specific semantics.
4. **Validity-vs-change** ("earlier train with later ticket", barrier disputes): needs syntactic understanding of validity. Fix: validity syntax modeling + targeted examples.
5. **Established-vs-open status** (delay/cancel/live): bag-level features cannot do state inference. Fix: temporal/state features. Stale-event, identifier-quarantine, fabrication-driven abstention, and conflict deadlock complete the top list with real case IDs in artifacts.

## 9. What Is Misleading About My Headline Number?

69.0% is intent-label agreement on 200 curated messages - not customers receiving good answers. n=200: the CI (62.3–75.0) overlaps every variant, so the ranking is suggestive, not proven. Class imbalance (feedback 44 vs delay 8) skews macro F1; rare classes swing on 1–2 examples. Training labels are noisy keyword rules (agreement ≠ accuracy). Dyad-overlap sensitivity is ~zero but author-style leakage can't be excluded. 2017 data cannot establish current policy. Retrieval applicability is not correctness today. Validator pass rates are guardrail mechanics, not response quality. 100% escalation is conservative safety behavior, not successful automation. No satisfaction, production, or human-agreement claim is made anywhere.

## 11. Decisions

| Decision | Why | Consequence |
|---|---|---|
| VirginTrains over larger brands | On-Twitter grounding quality beats volume | Smaller, usable corpus |
| Dyad (not raw thread) as unit | Broadcasts fuse unrelated issues | Clean eval units |
| 10-intent Twitter-derived taxonomy | Banking77 would distort rail needs | Support-shaped labels |
| AI labels re-adjudicated, 4 flips | First pass confirmed too easily | Honest 98%, not 100% |
| No general-information intent | Curiosity ≈0.06%, no support action | Two awkward rows, correctly scoped |
| Chatter stays in feedback | Deployed agents receive everything | Hardest class (F1 .59) |
| Weak labels, 63.9% coverage | Hand-labeling 7,791 infeasible | Noisy ceiling, zero leakage |
| Dyad-disjoint pool + strict reruns | Customer-disjoint impossible at n=7,791 | Flagged overlap, Δ≈0 everywhere |
| balanced class weights, predeclared | Rare intents need recall | Selected 3/3 pre-evaluation |
| Structured features dropped at 0.600 | Coarse counts drowned lexical precision | Reported loss, not tuned |
| Semantic as co-incumbent on +0.004 | Ahead on all metrics but CIs overlap | Tied-with-lean honesty |
| Hybrid wins (+0.021 macro), delay regresses | Complementarity real, uneven | Stated plainly incl. worst-of-three |
| 0.5B generator rejected, not patched | Schema echo, no hedging, fabrications | Single frozen run stands as evidence |
| Deterministic fallback, 0/180 auto-handle | RAM/API limits rule out larger LLMs | Safe, limited, disclosed |
| Escalation gate kept strict | Loosening on outcomes would be tuning | 0% auto-handle preserved as limitation |

## 12. Limitations and Evidence Boundaries

Demonstrated: intent classification to 0.69 ± 0.06, leakage-audited retrieval with provenance, deterministic safety gating, honest abstention. Not demonstrated: response quality, human agreement, current-policy correctness, satisfaction, production performance. Every number above traces to a frozen artifact; nothing is quoted without provenance.
