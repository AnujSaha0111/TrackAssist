# Aggregate automated diagnostics over agent batch output.
# Reads the JSONL produced by run_agent.py --batch and reports validator-based metrics plus the escalation distribution. Human-judged helpfulness/correctness are NOT computed here (no human annotations exist).

import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/agent_results.jsonl")
    ap.add_argument("--out", default="artifacts/agent_aggregates.json")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    agg = {
        "n": len(rows),
        "decisions": dict(Counter(r["decision"] for r in rows)),
        "reason_codes": dict(Counter(r["escalation_reason_code"]
                                     for r in rows)),
        "validator_all_pass": sum(1 for r in rows
                                  if all(r["validation"].values())),
        "abstained": sum(1 for r in rows if r["abstained"]),
        "conflicts": sum(1 for r in rows if r["conflict"]),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows)
                               / max(len(rows), 1), 2),
        "median_confidence": sorted(r["intent_confidence"]
                                    for r in rows)[len(rows) // 2],
        "note": "automated validator/escalation diagnostics ONLY; no "
                "human-judged helpfulness/correctness (no annotations exist)",
    }
    json.dump(agg, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
