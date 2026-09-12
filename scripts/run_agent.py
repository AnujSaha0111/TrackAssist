# TrackAssist support agent entry point (target brand: VirginTrains)
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import handle


def main():
    ap = argparse.ArgumentParser(
        description="TrackAssist — Evidence-Grounded AI Support Agent for "
                    "Rail Customer Service (target brand: VirginTrains).")
    ap.add_argument("--text", default=None)
    ap.add_argument("--context", action="append", default=[])
    ap.add_argument("--batch", default=None)
    ap.add_argument("--out", default="artifacts/agent_results.jsonl")
    a = ap.parse_args()
    if a.batch:
        n = 0
        with open(a.out, "w", encoding="utf-8") as f:
            for line in open(a.batch, encoding="utf-8"):
                r = json.loads(line)
                ctx = r.get("prior_customer_context") or r.get("context") or []
                ctx = [m if isinstance(m, str) else m.get("text", "")
                       for m in ctx]
                rec = handle(r.get("customer_query") or r.get("target_text", ""),
                             ctx,
                             customer_id="EVAL-" + r.get("case_id",
                                                         r.get("query_dyad_id",
                                                               "?")),
                             dyad_id=r.get("query_dyad_id", "QUERY"))
                rec["case_id"] = r.get("case_id", r.get("query_dyad_id"))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"batch done: {n} -> {a.out}")
    else:
        if not a.text:
            ap.error("--text required without --batch")
        print("TrackAssist", file=sys.stderr)
        print("Target brand: VirginTrains", file=sys.stderr)
        print(json.dumps(handle(a.text, a.context), ensure_ascii=False,
                         indent=1))


if __name__ == "__main__":
    main()
