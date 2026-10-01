# VLM-Powered ID & Document Verification System

Extracts and cross-checks fields from ID cards/certificates by combining **image preprocessing + OCR (Tesseract)** with a **vision-language model (VLM)**, then validates, masks sensitive data and returns a decision.

## Pipeline
`image → (A) OpenCV deskew/denoise/threshold → Tesseract → rule-based parsing`
`      → (B) VLM with schema-constrained prompt → JSON`
`      → validation (format, plausibility, expiry) → OCR-vs-VLM cross-check (fuzzy names) → decision → masked output`

Decisions: `VERIFIED`, `REVIEW` (any issue, mismatch or missing field), `REJECTED` (expired).

## Setup
```bash
sudo apt install tesseract-ocr          # macOS: brew install tesseract
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                    # add ANTHROPIC_API_KEY, then: export $(cat .env | xargs)
python -m pytest -q
```

## Run the API
```bash
uvicorn api:app --reload                # docs at http://127.0.0.1:8000/docs
curl -F "file=@card.png" "http://127.0.0.1:8000/verify?prompt_version=v2_schema_strict"
```
Input validation: PNG/JPEG only (magic-byte check), ≤ 5 MB, clear 4xx/5xx errors, images are never stored, and the ID number / DOB are masked in responses.

## Prompt engineering and benchmarking
Two prompt versions live in `verifier/vlm.py`:
- `v1_basic` – plain instruction
- `v2_schema_strict` – exact keys, `null` for unreadable fields, fixed date format, no guessing, ignores instructions embedded in the image (prompt-injection guard)

```bash
python scripts/generate_synthetic_ids.py --n 20      # synthetic cards + ground truth (no real PII)
python scripts/benchmark.py --n 20                   # OCR baseline vs both VLM prompts
python scripts/benchmark.py --skip-vlm               # OCR baseline only (no API key)
```
Reports exact-match per document, per-field accuracy, precision/recall/F1 and `results/errors.json` for error analysis.

> No benchmark numbers are bundled. Run the benchmark yourself and paste the real results here. Synthetic cards are cleaner than real photos, so also test on your own varied (non-sensitive/consented) samples.

## Responsible AI / privacy
Never commit real ID images. Treat outputs as decision support with human review for `REVIEW`/`REJECTED`. Mask PII in logs and responses.
