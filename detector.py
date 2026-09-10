"""
detector.py — Stage 2: OWLv2 open-vocabulary frame-level detector.

Why OWLv2 (proven in Phase 1):
- Full-resolution patch grid — no spatial downsampling, catches tiny objects.
- Direct box regression, no language generation — zero coordinate hallucination.
- Detection-trained (not instruction-tuned) — deterministic, threshold-stable.

Query expansion is applied here: OWLv2 scores vary significantly with exact
phrasing ("white board" scores 0.54 vs "whiteboard" scores 0.29 on the same
object). We probe all variants and collapse results back to the user's term.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from models.types import Detection

# ── constants ──────────────────────────────────────────────────────────────── #

MODEL_ID        = "google/owlv2-large-patch14-ensemble"
SCORE_THRESHOLD = 0.25   # 0.20 too noisy for video; 0.30 cuts real objects
NMS_IOU_THRESH  = 0.50

# Query expansion table — probe multiple phrasings, keep highest-scoring box.
# Keys are the user's term (lowercased); values are OWLv2 prompt variants.
QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "calendar":   ["white board", "display board", "notice board", "calendar", "whiteboard"],
    "whiteboard": ["white board", "whiteboard", "display board", "notice board"],
    "board":      ["white board", "display board", "notice board", "board"],
    "tv":         ["television", "tv screen", "flat screen tv", "monitor"],
    "television": ["television", "tv screen", "flat screen tv", "monitor"],
    "person":     ["someone sitting", "man", "woman", "person", "someone standing"],
    "phone":      ["mobile phone", "smartphone", "cell phone", "phone"],
    "mobile":     ["mobile phone", "smartphone", "cell phone"],
    "laptop":     ["laptop computer", "laptop", "notebook computer"],
    "sofa":       ["sofa", "couch", "settee"],
    "chair":      ["chair", "armchair", "seat"],
    "bag":        ["bag", "handbag", "backpack", "tote bag"],
    "bottle":     ["bottle", "water bottle", "plastic bottle"],
    "cup":        ["cup", "mug", "coffee cup"],
    "key":        ["key", "house key", "door key", "metal key"],
    "remote":     ["remote control", "tv remote", "remote"],
    "glasses":    ["glasses", "spectacles", "eyeglasses"],
    "book":       ["book", "notebook", "textbook"],
    "watch":      ["watch", "wristwatch", "wrist watch"],
    "wallet":     ["wallet", "purse", "billfold"],
}


# ── NMS helpers ────────────────────────────────────────────────────────────── #

def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _is_contained(inner: List[int], outer: List[int], thresh: float = 0.85) -> bool:
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_inner = max(1, (ax2 - ax1) * (ay2 - ay1))
    return (inter / area_inner) >= thresh


def nms(detections: List[Detection]) -> List[Detection]:
    """IoU + containment NMS (identical to Phase 1 logic)."""
    dets = sorted(detections, key=lambda d: d.score, reverse=True)
    keep: List[Detection] = []
    for d in dets:
        if not any(
            _iou(d.box, k.box) > NMS_IOU_THRESH or _is_contained(d.box, k.box)
            for k in keep
        ):
            keep.append(d)
    return keep


# ── detector class ─────────────────────────────────────────────────────────── #

class OWLv2Detector:
    """
    Loads OWLv2-large-patch14-ensemble once and runs open-vocabulary detection
    on individual PIL frames. Applies query expansion + NMS per frame.
    """

    def __init__(self):
        print(f"[Detector] Loading {MODEL_ID} …")
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.processor = Owlv2Processor.from_pretrained(MODEL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        print(f"[Detector] Ready on {device_str}")

    def detect(self, pil_image: Image.Image, queries: List[str]) -> List[Detection]:
        """
        Detect all queried objects in one PIL frame.
        Expands each query to its variant phrasings, runs OWLv2 once (all
        variants in a single forward pass), then collapses results back to the
        user's original term. Returns NMS-filtered detections above threshold.
        """
        # Build expanded prompt list + reverse map: variant → user term
        expanded: List[str] = []
        variant_to_original: Dict[str, str] = {}
        for q in queries:
            variants = QUERY_EXPANSIONS.get(q.lower(), [q])
            for v in variants:
                if v not in expanded:
                    expanded.append(v)
                variant_to_original[v] = q

        prompts = [f"a photo of a {v}" for v in expanded]
        inputs = self.processor(text=[prompts], images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=SCORE_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        detections: List[Detection] = []
        for score, label_idx, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            variant = expanded[int(label_idx)]
            detections.append(Detection(
                score=float(score),
                box=[int(v) for v in box.tolist()],
                label=variant_to_original[variant],  # always user's term
            ))

        return nms(detections)
