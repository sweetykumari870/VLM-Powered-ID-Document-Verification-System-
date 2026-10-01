"""Benchmark OCR baseline vs. VLM prompt versions on synthetic (or your own labelled) samples.

python scripts/generate_synthetic_ids.py --n 20
python scripts/benchmark.py --n 20            # needs ANTHROPIC_API_KEY (+ tesseract for OCR baseline)
python scripts/benchmark.py --skip-vlm        # OCR baseline only
Outputs results/benchmark.csv and results/errors.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifier.metrics import evaluate  # noqa: E402
from verifier.ocr import FIELDS  # noqa: E402
from verifier.pipeline import ocr_extract  # noqa: E402
from verifier.vlm import PROMPTS, extract_with_vlm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--skip-vlm", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    a = ap.parse_args()

    root = Path(a.data)
    labels = json.loads((root / "labels.json").read_text())[: a.n]
    truths = [{f: s[f] for f in FIELDS} for s in labels]
    images = [(root / s["file"]).read_bytes() for s in labels]

    systems = {}
    if not a.skip_ocr:
        systems["ocr_tesseract_baseline"] = lambda b: ocr_extract(b)
    if not a.skip_vlm:
        for v in PROMPTS:
            systems[f"vlm_{v}"] = (lambda ver: (lambda b: extract_with_vlm(b, ver)))(v)

    rows, all_errors = [], {}
    for name, fn in systems.items():
        preds = []
        for b in images:
            try:
                preds.append(fn(b))
            except Exception as e:  # count failures as empty predictions
                print(f"[{name}] error: {type(e).__name__}: {e}")
                preds.append({f: None for f in FIELDS})
        res = evaluate(preds, truths)
        all_errors[name] = res.pop("errors")
        rows.append({"system": name, "n_docs": res["n_docs"], "exact_match_doc": res["exact_match_doc"],
                     "precision": res["precision"], "recall": res["recall"], "f1": res["f1"],
                     **{f"acc_{k}": v for k, v in res["field_accuracy"].items()}})

    Path("results").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/benchmark.csv", index=False)
    Path("results/errors.json").write_text(json.dumps(all_errors, indent=2))
    print(df.to_string(index=False))
    print("Error analysis saved to results/errors.json (field-level mistakes per system)")


if __name__ == "__main__":
    main()
