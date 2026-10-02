"""
Generates a DRAFT ShareGPT-format dataset from raw prescription images,
using GLM-OCR's own (unfine-tuned) reading as a starting point.

IMPORTANT: this is a draft generator, not a labeler. Every "gpt" value
must be checked against the actual image and corrected by hand before
training -- an uncorrected draft teaches the model its own mistakes,
including plausible-looking hallucinations that are easy to miss on a
skim. Review every line against the image, not just for fluency.

Usage:
    python prepare_glmocr_finetune.py

Requires: ocr_engine.py's LLaMA-Factory API running (see its own
error message for the two-terminal setup), OR adjust _draft_text()
below to call the model directly instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_engine import ocr_image  # reuses your existing GLM-OCR pipeline

RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "glmocr_finetune"
IMAGES_DIR = OUT_DIR / "images"
DRAFT_PATH = OUT_DIR / "prescriptions_draft.json"

INSTRUCTION = "<image>Transcribe this handwritten prescription exactly as written."


def _draft_text(image_path: Path) -> str:
    """Get GLM-OCR's own (unfine-tuned) reading as a starting draft."""
    lines = ocr_image(image_path)
    return "\n".join(lines) if lines else "(no text detected)"


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    photos = sorted(
        p for p in RAW.glob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not photos:
        raise SystemExit(f"No images found in {RAW}")

    entries = []
    for i, photo in enumerate(photos):
        print(f"[{i+1}/{len(photos)}] Drafting: {photo.name}")
        dest_name = f"img_{i:04d}{photo.suffix.lower()}"
        dest_path = IMAGES_DIR / dest_name
        dest_path.write_bytes(photo.read_bytes())

        try:
            draft = _draft_text(photo)
        except Exception as e:
            print(f"  WARNING: OCR failed on {photo.name} ({e}) -- leaving blank for manual entry")
            draft = "(OCR FAILED -- fill in manually)"

        entries.append({
            "conversations": [
                {"from": "human", "value": INSTRUCTION},
                {"from": "gpt", "value": draft},
            ],
            "images": [f"images/{dest_name}"],
            "_source_file": photo.name,  # for your own reference while reviewing; remove before training if LLaMA-Factory complains about extra keys
        })

    DRAFT_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(entries)} draft entries -> {DRAFT_PATH}")
    print("\nNEXT STEP (do not skip): open this file and correct every")
    print("'gpt' value against the actual image before using it for training.")
    print("Draft text is a starting point, not verified ground truth.")


if __name__ == "__main__":
    main()
