#to augment any newly added (then labelled images in glm_new)
#skip any copy that already exists, and skip any image that has no text label

import csv, random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "glm_new"
IMAGES, LABELS = D / "images", D / "labels.csv"
COPIES = 3
rng = random.Random(7)

def augment(img):
    out = img.convert("RGB")
    if rng.random() < 0.8:
        out = out.rotate(rng.uniform(-4, 4), expand=True, fillcolor=(255, 255, 255))
    if rng.random() < 0.6:
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.2))
    if rng.random() < 0.6:
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
    return out

rows = list(csv.DictReader(LABELS.open(encoding="utf-8-sig")))
existing = {r["image"] for r in rows}
new_rows = list(rows)
for r in rows:
    name = r["image"]
    if "_aug" in name or not r["text"].strip():
        continue
    src = IMAGES / name
    img = Image.open(src)
    for i in range(COPIES):
        new_name = f"{src.stem}_aug{i}{src.suffix}"
        if new_name in existing:
            continue
        augment(img).save(IMAGES / new_name, quality=90)
        new_rows.append({"image": new_name, "text": r["text"], "task": "text"})
        existing.add(new_name)

with LABELS.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "text", "task"])
    w.writeheader(); w.writerows(new_rows)
print(f"{len(rows)} -> {len(new_rows)} rows")