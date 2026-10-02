"""GLM-OCR via direct transformers load (same approach as the working
Colab script) -- no LLaMA-Factory API server required.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_MODEL = None
_PROCESSOR = None

MODEL_ID = "zai-org/GLM-OCR"
PROMPT = "Text Recognition:"
MAX_NEW_TOKENS = 512


def get_backend() -> str:
    return "transformers_direct"


def last_backend() -> str:
    return "transformers_direct"


def _load_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration
        import torch

        print(f"Loading {MODEL_ID} ...")
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
        _MODEL = GlmOcrForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.float16, device_map="auto"
        )
    return _MODEL, _PROCESSOR


def _dedupe_ocr_text(text: str) -> str:
    """Collapse exact-repeat runs of 3+ consecutive lines only -- never
    truncate on a single matching line, which can silently drop real
    content (e.g. a page with several 'Tab' lines)."""
    lines = [p.strip() for p in text.splitlines() if p.strip()]
    if not lines:
        return ""
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        # look for a run where lines[i:i+k] repeats immediately after itself
        run_found = False
        for k in range(1, min(6, (len(lines) - i) // 2 + 1)):
            if lines[i:i + k] == lines[i + k:i + 2 * k]:
                cleaned.extend(lines[i:i + k])
                i += 2 * k
                run_found = True
                break
        if not run_found:
            cleaned.append(lines[i])
            i += 1
    return "\n".join(cleaned)


def _ocr_direct(image_path: Path) -> str:
    from PIL import Image

    model, processor = _load_model()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": PROMPT},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    text = processor.decode(output[0], skip_special_tokens=True)
    return text.strip()


def ocr_image(image_path: Path, output_dir: Path | None = None) -> list[str]:
    """Run GLM-OCR directly (loaded once, reused across calls) and
    return text lines."""
    text = _dedupe_ocr_text(_ocr_direct(Path(image_path)))
    lines = [part for part in text.splitlines() if part.strip()]
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{Path(image_path).stem}_ocr.json").write_text(
            json.dumps(
                {"backend": "transformers_direct", "text": text, "lines": lines},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return lines
