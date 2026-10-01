"""Generate SYNTHETIC ID-card images + ground truth for benchmarking (no real PII).
python scripts/generate_synthetic_ids.py --n 20 --out data/synthetic
"""
import argparse
import json
import random
import string
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vikram", "Isha", "Arjun", "Sneha"]
LAST = ["Sharma", "Verma", "Singh", "Patel", "Gupta", "Mehta", "Rao", "Iyer", "Nair", "Das"]


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_record(rng: random.Random) -> dict:
    dob = date(1970, 1, 1) + timedelta(days=rng.randint(0, 12000))
    exp = date(2024, 1, 1) + timedelta(days=rng.randint(0, 3000))
    pan = "".join(rng.choices(string.ascii_uppercase, k=5)) + "".join(rng.choices(string.digits, k=4)) \
        + rng.choice(string.ascii_uppercase)
    return {"name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "dob": dob.strftime("%d/%m/%Y"),
            "id_number": pan, "expiry_date": exp.strftime("%d/%m/%Y")}


def render(rec: dict, path: Path):
    img = Image.new("RGB", (1000, 560), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 989, 549], outline="black", width=4)
    d.text((40, 30), "SAMPLE IDENTITY CARD (SYNTHETIC)", fill="black", font=font(34))
    y = 140
    for label, key in (("Name", "name"), ("DOB", "dob"), ("ID Number", "id_number"), ("Expiry", "expiry_date")):
        d.text((60, y), f"{label}: {rec[key]}", fill="black", font=font(42))
        y += 85
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    labels = []
    for i in range(a.n):
        rec = make_record(rng)
        fname = f"id_{i:03d}.png"
        render(rec, out / fname)
        labels.append({"file": fname, **rec})
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"Wrote {a.n} synthetic samples to {out}")


if __name__ == "__main__":
    main()
