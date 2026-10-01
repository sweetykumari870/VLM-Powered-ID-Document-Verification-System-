from typing import Dict, List, Optional

from .ocr import FIELDS
from .validate import normalize


def field_counts(pred: Dict[str, Optional[str]], truth: Dict[str, Optional[str]]):
    """Per-sample (tp, fp, fn) and per-field correctness. A wrong value is both an fp and an fn."""
    tp = fp = fn = 0
    correct = {}
    for f in FIELDS:
        p, t = normalize(f, pred.get(f)), normalize(f, truth.get(f))
        correct[f] = p == t and t is not None
        if t is None:
            fp += p is not None
        elif p is None:
            fn += 1
        elif p == t:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn, correct


def evaluate(preds: List[dict], truths: List[dict]) -> dict:
    TP = FP = FN = 0
    per_field = {f: 0 for f in FIELDS}
    exact_docs = 0
    errors = []
    for i, (p, t) in enumerate(zip(preds, truths)):
        tp, fp, fn, correct = field_counts(p, t)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        for f in FIELDS:
            per_field[f] += correct[f]
        exact_docs += all(correct.values())
        for f in FIELDS:
            if not correct[f]:
                errors.append({"sample": i, "field": f, "pred": p.get(f), "truth": t.get(f)})
    n = max(len(truths), 1)
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n_docs": len(truths),
        "exact_match_doc": round(exact_docs / n, 3),
        "field_accuracy": {f: round(per_field[f] / n, 3) for f in FIELDS},
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "errors": errors,
    }
