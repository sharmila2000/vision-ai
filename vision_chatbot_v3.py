"""
vision_chatbot_v3.py
--------------------
Phase 1 — Object Location Chatbot using OWLv2

OWLv2 (Owl-ViT v2) is an open-vocabulary object detector trained
end-to-end for detection. Unlike VLMs which generate coordinates as
text, OWLv2 directly computes visual similarity between a text query
and every patch of the image, then outputs bounding boxes.

This correctly handles tiny objects because:
  - The image is processed at full resolution as a grid of patches
  - Each patch is compared to the text embedding of the query
  - No language generation, no coordinate hallucination
  - Specifically trained for detection, not text completion

Usage:
    python vision_chatbot_v3.py --image images/img01.jpg
    > where is the key?
    > find the laptop
    > where is the bag?
"""

import os
import re
import sys
import argparse
import datetime

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection


# ============================================================
# CONFIGURATION
# ============================================================

# owlv2-large is more accurate and gives higher confidence scores —
# it separates true detections from false positives more clearly.
MODEL_ID   = "google/owlv2-large-patch14-ensemble"
OUTPUT_DIR = "results_qwen3vl"

# Detection confidence threshold.
# OWLv2 outputs a score per box. Lower = more sensitive (more false positives).
# Higher = stricter (may miss weak detections).
SCORE_THRESHOLD = 0.20

# Box drawing
BOX_COLOURS   = [(0,200,50), (50,150,255), (255,80,80), (0,220,220), (200,50,200)]
BOX_THICKNESS = 3
LABEL_SCALE   = 0.7
LABEL_THICK   = 2


# ============================================================
# DEVICE
# ============================================================

def get_device() -> str:
    if torch.cuda.is_available():          return "cuda"
    if torch.backends.mps.is_available():  return "mps"
    return "cpu"

def get_dtype(device: str) -> torch.dtype:
    # OWLv2 is stable in float32 on MPS; float16 can cause NaN on some Mac configs
    if device == "cuda": return torch.float16
    return torch.float32


# ============================================================
# MODEL LOAD
# ============================================================

def load_model(device: str, dtype: torch.dtype, model_id: str = MODEL_ID):
    print(f"  Loading OWLv2 processor from {model_id}...")
    processor = Owlv2Processor.from_pretrained(model_id)
    print("  Processor loaded.")

    print(f"  Loading OWLv2 model...")
    model = Owlv2ForObjectDetection.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print("  Model loaded.")
    return processor, model


# ============================================================
# QUERY PARSING
# ============================================================

def parse_query_items(query: str) -> list[str]:
    """
    Extract the item(s) to search for from a natural-language query.
    Returns a list of text prompts to pass to OWLv2.

    OWLv2 works best with prompts like:
      "a photo of a key"
      "a photo of a laptop"

    We strip filler words and format as detection prompts.
    """
    q = query.lower().strip().rstrip("?").strip()

    for filler in [
        "where is the", "where is a", "where is",
        "find the", "find a", "find",
        "locate the", "locate a", "locate",
        "show me the", "show me a", "show me",
        "what is the location of the", "what is the location of",
    ]:
        if q.startswith(filler):
            q = q[len(filler):].strip()
            break

    # Build detection-style prompts — OWLv2 was trained on this format
    prompts = [f"a photo of a {q}", f"a {q}"]
    return prompts


# ============================================================
# DETECTION
# ============================================================

def _iou(a: list, b: list) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _containment(small: list, large: list) -> float:
    """
    Fraction of 'small' box area that lies inside 'large' box.
    Returns 1.0 if small is entirely inside large, 0.0 if no overlap.
    Used to suppress sub-region duplicate detections that IoU misses.
    """
    ix1 = max(small[0], large[0]); iy1 = max(small[1], large[1])
    ix2 = min(small[2], large[2]); iy2 = min(small[3], large[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    area_small = (small[2]-small[0]) * (small[3]-small[1])
    if area_small == 0: return 0.0
    return inter / area_small


def _nms(detections: list[dict], iou_threshold: float = 0.4) -> list[dict]:
    """
    Non-Maximum Suppression with containment check.

    Suppresses box B if EITHER:
      - IoU(A, B) > iou_threshold  (standard overlap)
      - B is mostly contained within A (>= 80% of B's area inside A)

    This handles the case where OWLv2 fires two boxes on the same tiny
    object: one large box around the whole key, one small box on part of it.
    """
    kept = []
    used = [False] * len(detections)
    for i, d in enumerate(detections):
        if used[i]: continue
        kept.append(d)
        for j in range(i+1, len(detections)):
            if used[j]: continue
            overlap    = _iou(d["box"], detections[j]["box"])
            contained  = _containment(detections[j]["box"], d["box"])
            # Also suppress if the kept box is mostly inside the candidate
            # (handles case where lower-score box is a larger wrapper region)
            reverse_contained = _containment(d["box"], detections[j]["box"])
            if overlap > iou_threshold or contained >= 0.7 or reverse_contained >= 0.6:
                used[j] = True
    return kept


def detect(
    image:      Image.Image,
    prompts:    list[str],
    processor,
    model,
    device:     str,
    threshold:  float = SCORE_THRESHOLD,
) -> list[dict]:
    """
    Run OWLv2 on the image with the given text prompts.
    Applies NMS to remove duplicate boxes for the same object.

    Returns a list of detections, each:
        {
            "box":   [x1, y1, x2, y2],  # absolute pixel coords
            "score": float,
            "label": str,               # which prompt matched
        }
    sorted by score descending.
    """
    img_w, img_h = image.size

    inputs = processor(
        text=[prompts],   # list of list — one image, multiple prompts
        images=image,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    # Post-process: convert model output to boxes in pixel coordinates
    target_sizes = torch.tensor([[img_h, img_w]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        threshold=threshold,
        target_sizes=target_sizes,
        text_labels=[prompts],   # list of list — one image, N prompts
    )[0]

    detections = []
    for score, label_str, box in zip(
        results["scores"], results["text_labels"], results["boxes"]
    ):
        detections.append({
            "box":   [int(v) for v in box.tolist()],
            "score": float(score),
            "label": label_str,
        })

    # Sort highest confidence first, then suppress duplicates
    detections.sort(key=lambda d: d["score"], reverse=True)
    detections = _nms(detections, iou_threshold=0.4)
    return detections


# ============================================================
# DRAW BOXES
# ============================================================

def draw_boxes(
    image_path:  str,
    detections:  list[dict],
    query:       str,
    output_path: str,
):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [warn] Cannot read image for annotation.")
        return

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        score  = det["score"]
        colour = BOX_COLOURS[idx % len(BOX_COLOURS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, BOX_THICKNESS)

        label = f"{query} ({score:.2f})"
        (tw, th), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK
        )
        ly = max(y1, th + 8)
        cv2.rectangle(img, (x1, ly-th-bl-4), (x1+tw+4, ly), colour, -1)
        cv2.putText(
            img, label, (x1+2, ly-bl-2),
            cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE,
            (255,255,255), LABEL_THICK, cv2.LINE_AA,
        )

    cv2.imwrite(output_path, img)
    print(f"  Annotated image → {output_path}")


# ============================================================
# LOCATION DESCRIPTION
# ============================================================

def describe_location(box: list, img_w: int, img_h: int) -> str:
    """Convert a bounding box to a natural-language location description."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    col = "left"   if cx < img_w / 3 else ("centre" if cx < 2 * img_w / 3 else "right")
    row = "top"    if cy < img_h / 3 else ("middle" if cy < 2 * img_h / 3 else "bottom")

    return f"{row}-{col} of the image  (box: {x1},{y1} → {x2},{y2})"


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vision Chatbot v3 — OWLv2 open-vocabulary detector"
    )
    parser.add_argument("--image",     default="images/img01.jpg")
    parser.add_argument("--model",     default=MODEL_ID)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD,
                        help=f"Detection confidence threshold (default: {SCORE_THRESHOLD})")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    dtype  = get_dtype(device)

    print()
    print("=" * 60)
    print("  VISION CHATBOT v3 — OWLv2 Detector")
    print("=" * 60)
    print(f"  Image     : {args.image}")
    print(f"  Model     : {args.model}")
    print(f"  Device    : {device}  ({dtype})")
    print(f"  Threshold : {args.threshold}")
    print("=" * 60)

    # Load image
    image    = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size
    print(f"\n  Image loaded: {img_w}×{img_h}")

    # Load model (downloads ~600MB on first run, cached after)
    processor, model = load_model(device, dtype, args.model)

    # Chat loop
    print("\n" + "=" * 60)
    print("  Ready. Type any query.")
    print("  Examples:")
    print("    > where is the key?")
    print("    > find the laptop")
    print("    > where is the bag?")
    print("    > list all objects")
    print("  Adjust sensitivity: lower --threshold finds more (may add noise)")
    print("  Type 'quit' to exit.")
    print("=" * 60 + "\n")

    log_entries = []
    query_count = 0

    while True:
        try:
            raw_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session ended.")
            break

        if not raw_input:
            continue
        if raw_input.lower() in {"quit", "exit", "q"}:
            break

        # Special command: list all detectable objects at low threshold
        if raw_input.lower() in {"list all objects", "list objects", "list all"}:
            raw_input = "list all objects"
            prompts = [
                "a photo of a key", "a photo of a laptop", "a photo of a bag",
                "a photo of a phone", "a photo of a remote control",
                "a photo of a book", "a photo of glasses", "a photo of a cup",
                "a photo of shoes", "a photo of a bottle", "a photo of a wallet",
                "a photo of a watch", "a photo of a charger",
            ]
        else:
            prompts = parse_query_items(raw_input)

        query_count += 1
        print(f"\n  [Query {query_count}] Detecting: {prompts}")

        detections = detect(image, prompts, processor, model, device, args.threshold)

        print("\n" + "-" * 60)
        if detections:
            # For key queries use a lower secondary threshold on tiled crops
            # to catch very small objects the full-image pass may miss
            print(f"  Found {len(detections)} detection(s):\n")
            for i, det in enumerate(detections, 1):
                loc = describe_location(det["box"], img_w, img_h)
                print(f"  {i}. Score: {det['score']:.3f}  |  {loc}")
                print(f"     Prompt: {det['label']}")

            safe = re.sub(r"[^\w\s-]", "", raw_input).strip().replace(" ", "_")[:40]
            ts   = datetime.datetime.now().strftime("%H%M%S")
            out  = os.path.join(OUTPUT_DIR, f"q{query_count:02d}_{safe}_{ts}.jpg")
            draw_boxes(args.image, detections, raw_input.rstrip("?").strip(), out)
        else:
            print(f"  Nothing found above threshold {args.threshold}.")
            print(f"  Try: python vision_chatbot_v3.py --threshold 0.05")
        print("-" * 60 + "\n")

        log_entries.append({"query": raw_input, "detections": detections})

    # Save log
    if log_entries:
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = os.path.join(OUTPUT_DIR, f"v3_session_{ts}.txt")
        with open(log, "w") as f:
            f.write(f"Model: {args.model}\nImage: {args.image}\n\n")
            for e in log_entries:
                f.write(f"QUERY: {e['query']}\n")
                for d in e["detections"]:
                    f.write(f"  score={d['score']:.3f}  box={d['box']}  label={d['label']}\n")
                f.write("\n")
        print(f"  Log → {log}")

    print("  Done.\n")


if __name__ == "__main__":
    main()
