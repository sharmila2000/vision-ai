"""
verifier.py — Stage 3: Florence-2 crop verification (second opinion).

Why a verifier:
  OWLv2 matches patches by text-image cosine similarity. On a moving camera,
  textured walls, phone screens, and book spines can score 0.25–0.29 for almost
  any query. Raising the threshold to suppress these also cuts real objects.

  The correct solution is a two-stage approach:
    1. OWLv2 proposes candidate boxes (generous threshold, high recall).
    2. Florence-2 receives the *cropped region* and generates a caption.
       Florence-2 was trained on 5.4B image-text pairs — its captions are
       grounded in what is visually present in the crop.
    3. We check whether the query term (or a synonym) appears in the caption.
       If it does → verified. If not → rejected as a false positive.

  Why caption-based, not VQA-based:
    Florence-2's <VQA> task token is unstable on MPS — it echoes back the
    prompt instead of answering. <CAPTION> is stable, fast, and produces
    accurate natural-language descriptions of crop content.

  A detection is accepted only if BOTH models agree:
    - OWLv2: finds it even when tiny or partially occluded  (high recall)
    - Florence-2: rejects crops that don't contain the queried object (high precision)

Memory: Florence-2-large ~1.5 GB on MPS (float32). OWLv2-large ~1.5 GB.
Total ~3 GB — safe on 25.8 GB RAM. Verifier only runs on detected crop
regions, not full frames, so overhead is minimal.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from models.types import Detection

# ── constants ──────────────────────────────────────────────────────────────── #

FLORENCE_MODEL_ID = "microsoft/Florence-2-large"
CROP_PAD = 0.15   # expand OWLv2 box by 15% each side for context

# Synonym map: if any of these words appears in the Florence-2 caption,
# treat the crop as a verified match for the given query term.
_SYNONYMS: Dict[str, List[str]] = {
    "calendar":   ["whiteboard", "white board", "calendar", "board", "notice board", "display board"],
    "whiteboard": ["whiteboard", "white board", "board", "notice board", "display board"],
    "board":      ["board", "whiteboard", "white board"],
    "phone":      ["phone", "smartphone", "mobile", "cellphone", "cell phone", "device"],
    "mobile":     ["phone", "smartphone", "mobile", "cellphone", "device"],
    "tv":         ["television", "tv", "screen", "monitor", "display"],
    "television": ["television", "tv", "screen", "monitor"],
    "laptop":     ["laptop", "computer", "notebook", "macbook"],
    "sofa":       ["sofa", "couch", "settee", "seat"],
    "chair":      ["chair", "armchair", "seat"],
    "bag":        ["bag", "handbag", "backpack", "tote"],
    "bottle":     ["bottle", "water bottle"],
    "cup":        ["cup", "mug", "glass"],
    "book":       ["book", "notebook", "textbook"],
    "person":     ["person", "man", "woman", "people", "human", "individual",
                   "sitting", "standing", "walking", "holding", "wearing",
                   "boy", "girl", "child", "he ", "she "],
    "key":        ["key", "keys"],
    "remote":     ["remote", "controller"],
    "glasses":    ["glasses", "spectacles", "eyeglasses"],
}


# ── verifier class ─────────────────────────────────────────────────────────── #

class Florence2Verifier:
    """
    Uses Florence-2-large CAPTION task to verify whether a cropped region
    actually contains the queried object. Stable on MPS (float32).
    """

    def __init__(self):
        print(f"[Verifier] Loading {FLORENCE_MODEL_ID} …")
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_ID,
            dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",   # required for MPS (avoids SDPA crash)
        ).to(self.device)
        self.model.eval()
        print(f"[Verifier] Ready on {device_str}")

    def verify(
        self,
        pil_image: Image.Image,
        detections: List[Detection],
    ) -> List[Detection]:
        """
        For each detection, crop the region (with padding), generate a
        Florence-2 caption, and set detection.verified if the caption
        mentions the queried object (or a synonym).
        """
        w, h = pil_image.size
        for det in detections:
            crop    = self._padded_crop(pil_image, det.box, w, h)
            caption = self._caption(crop)
            det.verified = self._matches(caption, det.label)
            # Debug: uncomment to see captions during development
            # print(f"  [Verifier] '{det.label}' crop caption: {caption!r} → {det.verified}")
        return detections

    # ── internals ──────────────────────────────────────────────────────────── #

    def _padded_crop(self, img: Image.Image, box: List[int], w: int, h: int) -> Image.Image:
        x1, y1, x2, y2 = box
        px = max(int((x2 - x1) * CROP_PAD), 4)
        py = max(int((y2 - y1) * CROP_PAD), 4)
        return img.crop((max(0, x1-px), max(0, y1-py), min(w, x2+px), min(h, y2+py)))

    def _caption(self, crop: Image.Image) -> str:
        """Generate a short caption for the crop using Florence-2 <CAPTION> task."""
        inputs = self.processor(text="<CAPTION>", images=crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=40,
                do_sample=False,
                use_cache=False,   # required for MPS stability
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()

    def _matches(self, caption: str, query: str) -> bool:
        """Return True if the caption mentions the query or any of its synonyms."""
        synonyms = _SYNONYMS.get(query.lower(), [query.lower()])
        return any(syn in caption for syn in synonyms)
