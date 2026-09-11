# Intent-classifier harness: predictions vs the frozen 200-row golden set
# Reports accuracy, Wilson 95% CI, macro/weighted F1, per-intent P/R/F1, confusion matrix, secondary exact-match recall, coverage, abstention rate. Guards assert 200 exact golden IDs, valid intents, unchanged taxonomy
# Predictions format (CSV or JSONL): golden_id, predicted_primary, predicted_secondary (optional), abstained (optional)

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import leakage as lg

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_CSV = ROOT / "artifacts" / "evaluation" / "golden_labels.csv"


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def load_predictions(path):
    path = Path(path)
    rows = []
    if path.suffix == ".jsonl":
        for line in open(path, encoding="utf-8"):
            rows.append(json.loads(line))
    else:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for r in rows:
        gid = r["golden_id"]
        if gid in out:
            raise AssertionError(f"duplicate prediction for {gid}")
        lg.assert_valid_intent(r.get("predicted_primary"), f"predictions:{gid}")
        lg.assert_valid_intent(r.get("predicted_secondary") or "",
                               f"predictions:{gid}")
        out[gid] = {"primary": r.get("predicted_primary"),
                    "secondary": r.get("predicted_secondary") or "",
                    "abstained": str(r.get("abstained", "0")).strip()
                    in ("1", "true")}
    return out


def evaluate(preds):
    golden = lg.load_golden()
    lg.assert_predictions_match_golden(preds.keys())
    targets = [r["target_text"] for r in golden]
    if len(set(targets)) != 200:
        raise AssertionError("duplicate golden targets")
    lg.assert_taxonomy_ids(json.load(
        open(ROOT / "config" / "taxonomy.json", encoding="utf-8")))

    intents = lg.INTENTS
    y_true = [r["human_intent"] for r in golden]
    y_pred = [preds[r["golden_id"]]["primary"] for r in golden]
    abst = [preds[r["golden_id"]]["abstained"] for r in golden]
    scored = [(t, p) for t, p, a in zip(y_true, y_pred, abst) if not a]
    n_abst = sum(abst)

    correct = sum(1 for t, p in scored if t == p)
    acc = correct / len(scored) if scored else 0.0
    ci_lo, ci_hi = wilson(acc, len(scored))

    per, macro_f = {}, []
    for iid in intents:
        tp = sum(1 for t, p in scored if t == iid and p == iid)
        fp = sum(1 for t, p in scored if t != iid and p == iid)
        fn = sum(1 for t, p in scored if t == iid and p != iid)
        p, r, f = prf(tp, fp, fn)
        per[iid] = {"precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(f, 4),
                    "support": sum(1 for t in y_true if t == iid)}
        macro_f.append(f)
    macro_f1 = sum(macro_f) / len(macro_f)
    tot = sum(per[i]["support"] for i in intents)
    weighted_f1 = sum(per[i]["f1"] * per[i]["support"] for i in intents) / tot

    matrix = {t: {p: 0 for p in intents} for t in intents}
    for t, p in scored:
        if p in intents:
            matrix[t][p] += 1

    gold_sec = [(r["human_secondary_intent"] or "") for r in golden]
    pred_sec = [preds[r["golden_id"]]["secondary"] for r in golden]
    sec_both = sum(1 for g, p in zip(gold_sec, pred_sec) if g and g == p)
    sec_gold_n = sum(1 for g in gold_sec if g)

    return {
        "n_golden": 200,
        "n_scored": len(scored),
        "n_abstained": n_abst,
        "abstention_rate": round(n_abst / 200, 4),
        "coverage": round(len(scored) / 200, 4),
        "primary_accuracy": round(acc, 4),
        "primary_accuracy_wilson95": [round(ci_lo, 4), round(ci_hi, 4)],
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_intent": per,
        "confusion_true_rows_x_pred_cols": {"intents": intents, "matrix":
            [[matrix[t][p] for p in intents] for t in intents]},
        "secondary_exact_matches": sec_both,
        "secondary_gold_nonempty": sec_gold_n,
        "secondary_recall_on_nonempty": round(sec_both / sec_gold_n, 4)
        if sec_gold_n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = evaluate(load_predictions(a.predictions))
    if a.out:
        json.dump(report, open(a.out, "w", encoding="utf-8"), indent=1)
        print(f"wrote {a.out}")
    else:
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ("per_intent",
                                       "confusion_true_rows_x_pred_cols")},
                         indent=1))


if __name__ == "__main__":
    main()
