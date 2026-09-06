"""
vision_chatbot_raw.py
--------------------
OWLv2-large  —  raw, unrestricted output

This version removes every restriction from v3 and exposes the model
exactly as it runs:

  1. NO prompt formatting.
     Your input is sent to OWLv2 exactly as you typed it.
     You can type  "key",  "a photo of a key",  "shiny metal object",
     "silver house key on leather sofa" — the model gets it verbatim.

  2. NO threshold filtering by default.
     All boxes the model returns are shown, ranked by score.
     Use  --threshold  to cut off at a custom level (default: 0.0 = show all).

  3. NO NMS / duplicate suppression by default.
     Every single box the model outputs is shown, including overlapping ones.
     Use  --nms  flag to enable NMS if you want deduplication.

  4. RAW scores printed in full.
     Each box shows its exact floating-point confidence score.

  5. RAW box coordinates printed in full.
     Exact pixel [x1, y1, x2, y2] printed alongside every detection.

  6. RAW label printed as-is.
     The label is whatever you typed — no reformatting, no stripping.

  7. Multiple prompts at once.
     Separate with  |  to query several things in one shot:
       > key | laptop | bag

  8. "raw scan" command — sends a broad unfiltered probe using single-word
     prompts so you can see what scores the model assigns to common objects.

Usage:
    python vision_chatbot_v6.py --image images/img01.jpg
    python vision_chatbot_v6.py --image images/img01.jpg --threshold 0.05
    python vision_chatbot_v6.py --image images/img01.jpg --nms
    python vision_chatbot_v6.py --image images/img01.jpg --threshold 0.0 --nms

    > key
    > a photo of a key
    > silver house key on leather sofa armrest
    > key | laptop | bag
    > raw scan
"""

import os
import re
import sys
import argparse
import datetime

import cv2
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

MODEL_ID   = "google/owlv2-large-patch14-ensemble"
OUTPUT_DIR = "results_qwen3vl"

# Default threshold — 0.0 means show every box the model outputs.
# Raise with --threshold to filter noise.
DEFAULT_THRESHOLD = 0.10

# Box colours (cycling)
BOX_COLOURS   = [
    (0, 200, 50), (50, 150, 255), (255, 80, 80),
    (0, 220, 220), (200, 50, 200), (255, 165, 0),
    (128, 255, 0), (255, 0, 128),
]
BOX_THICKNESS = 2
LABEL_SCALE   = 0.55
LABEL_THICK   = 1

# "raw scan" sends these verbatim — single words, no formatting
RAW_SCAN_PROMPTS = [
    "person", "man", "woman", "face", "hand",
    "key", "keys", "laptop", "computer", "phone", "mobile",
    "bag", "backpack", "suitcase", "luggage",
    "bottle", "cup", "mug", "glass",
    "book", "notebook", "paper",
    "remote", "charger", "cable",
    "watch", "glasses", "wallet",
    "chair", "sofa", "couch", "table", "desk",
    "door", "window", "shelf", "shelf",
    "shoes", "shoe", "slipper",
    "clock", "frame", "picture",
]


# ──────────────────────────────────────────────────────────────
# DEVICE / DTYPE
# ──────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():          return "cuda"
    if torch.backends.mps.is_available():  return "mps"
    return "cpu"

def get_dtype(device: str) -> torch.dtype:
    if device == "cuda": return torch.float16
    return torch.float32


# ──────────────────────────────────────────────────────────────
# MODEL LOAD
# ──────────────────────────────────────────────────────────────

def load_model(device: str, dtype: torch.dtype, model_id: str = MODEL_ID):
    print(f"  Loading OWLv2 processor  [{model_id}]...")
    processor = Owlv2Processor.from_pretrained(model_id)
    print("  Processor loaded.")

    print(f"  Loading OWLv2 model...")
    model = Owlv2ForObjectDetection.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print("  Model loaded.\n")
    return processor, model


# ──────────────────────────────────────────────────────────────
# RAW DETECTION
# No prompt formatting. No threshold pre-filtering inside here.
# Returns every single box the model produces.
# ──────────────────────────────────────────────────────────────

def detect_raw(
    image:     Image.Image,
    prompts:   list[str],      # sent verbatim to OWLv2
    processor,
    model,
    device:    str,
    threshold: float = 0.0,
) -> list[dict]:
    """
    Run OWLv2 with the given prompts, exactly as supplied.
    Returns ALL detections above threshold (default 0.0 = everything).

    Each detection:
        {
            "box":   [x1, y1, x2, y2],   # absolute pixel coords
            "score": float,               # raw confidence 0..1
            "label": str,                 # the prompt that matched
        }
    Sorted by score descending. NMS is NOT applied here.
    """
    img_w, img_h = image.size

    inputs = processor(
        text=[prompts],          # [[prompt1, prompt2, ...]] — one image
        images=image,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    # post_process_grounded_object_detection converts raw logits → boxes + scores
    target_sizes = torch.tensor([[img_h, img_w]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        threshold=threshold,
        target_sizes=target_sizes,
        text_labels=[prompts],
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["text_labels"], results["boxes"]
    ):
        detections.append({
            "box":   [int(v) for v in box.tolist()],
            "score": float(score),
            "label": label,
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# ──────────────────────────────────────────────────────────────
# OPTIONAL NMS  (only when --nms flag is passed)
# ──────────────────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0: return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def apply_nms(detections: list[dict], iou_thr: float = 0.4) -> list[dict]:
    """Standard greedy NMS — keeps highest-score box when two overlap."""
    kept = []
    used = [False] * len(detections)
    for i, d in enumerate(detections):
        if used[i]: continue
        kept.append(d)
        for j in range(i + 1, len(detections)):
            if used[j]: continue
            if _iou(d["box"], detections[j]["box"]) > iou_thr:
                used[j] = True
    return kept


# ──────────────────────────────────────────────────────────────
# DRAW BOXES
# Draws all detections. Label = exact matched prompt + raw score.
# ──────────────────────────────────────────────────────────────

def draw_boxes(image_path: str, detections: list[dict], output_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print("  [warn] Cannot read image for annotation.")
        return

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        colour = BOX_COLOURS[idx % len(BOX_COLOURS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, BOX_THICKNESS)

        # Label = raw prompt + score
        label = f"{det['label']}  {det['score']:.4f}"
        (tw, th), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK
        )
        ly = max(y1, th + 6)
        cv2.rectangle(img, (x1, ly - th - bl - 3), (x1 + tw + 4, ly), colour, -1)
        cv2.putText(
            img, label, (x1 + 2, ly - bl - 1),
            cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE,
            (255, 255, 255), LABEL_THICK, cv2.LINE_AA,
        )

    cv2.imwrite(output_path, img)
    print(f"  Annotated image → {output_path}")


# ──────────────────────────────────────────────────────────────
# PRINT RAW DETECTIONS TABLE
# ──────────────────────────────────────────────────────────────

def print_detections(detections: list[dict], img_w: int, img_h: int):
    if not detections:
        print("  (no detections)")
        return

    # Header
    print(f"  {'#':>3}  {'score':>7}  {'x1':>5} {'y1':>5} {'x2':>5} {'y2':>5}  "
          f"{'w':>5} {'h':>5}  label")
    print("  " + "-" * 90)

    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["box"]
        w  = x2 - x1
        h  = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        col  = "L" if cx < img_w / 3 else ("C" if cx < 2 * img_w / 3 else "R")
        row  = "T" if cy < img_h / 3 else ("M" if cy < 2 * img_h / 3 else "B")
        zone = f"[{row}{col}]"

        print(f"  {i:>3}  {det['score']:>7.4f}  "
              f"{x1:>5} {y1:>5} {x2:>5} {y2:>5}  "
              f"{w:>5} {h:>5}  {zone}  {det['label']}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vision Chatbot v6 — OWLv2 raw unrestricted output"
    )
    parser.add_argument("--image",     default="images/img01.jpg",
                        help="Path to input image")
    parser.add_argument("--model",     default=MODEL_ID,
                        help="HuggingFace model ID")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Score threshold (default 0.0 = show ALL boxes)")
    parser.add_argument("--nms",       action="store_true",
                        help="Apply NMS to suppress overlapping boxes")
    parser.add_argument("--nms-iou",   type=float, default=0.4,
                        help="IoU threshold for NMS (default 0.4)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    dtype  = get_dtype(device)

    print()
    print("=" * 65)
    print("  VISION CHATBOT v6 — OWLv2  (raw unrestricted output)")
    print("=" * 65)
    print(f"  Image     : {args.image}")
    print(f"  Model     : {args.model}")
    print(f"  Device    : {device}  ({dtype})")
    print(f"  Threshold : {args.threshold}  {'(show ALL)' if args.threshold == 0.0 else ''}")
    print(f"  NMS       : {'ON  (iou≥' + str(args.nms_iou) + ')' if args.nms else 'OFF (raw boxes)'}")
    print("=" * 65)

    image        = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size
    print(f"\n  Image loaded: {img_w}×{img_h} px")

    processor, model = load_model(device, dtype, args.model)

    print("=" * 65)
    print("  Ready. Your input is sent to OWLv2 verbatim — no formatting.")
    print()
    print("  How to type prompts:")
    print("    > key                              ← single word, raw")
    print("    > a photo of a key                 ← full OWLv2 training format")
    print("    > silver house key on leather sofa ← descriptive phrase")
    print("    > key | laptop | bag               ← multiple prompts at once")
    print("    > raw scan                         ← broad unfiltered probe")
    print("    > quit                             ← exit")
    print("=" * 65 + "\n")

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

        query_count += 1

        # ── "raw scan" ──────────────────────────────────────────
        if raw_input.lower() == "raw scan":
            prompts = RAW_SCAN_PROMPTS
            print(f"\n  [Query {query_count}] RAW SCAN — {len(prompts)} prompts, no formatting")
        else:
            # Split on | for multi-prompt, strip whitespace — NO other processing
            prompts = [p.strip() for p in raw_input.split("|") if p.strip()]
            print(f"\n  [Query {query_count}] Prompts sent verbatim: {prompts}")

        # ── run model ───────────────────────────────────────────
        detections = detect_raw(
            image, prompts, processor, model, device, threshold=args.threshold
        )
        raw_count = len(detections)

        # ── optional NMS ────────────────────────────────────────
        if args.nms and detections:
            before = len(detections)
            detections = apply_nms(detections, iou_thr=args.nms_iou)
            print(f"  NMS: {before} → {len(detections)} boxes")

        # ── print raw table ─────────────────────────────────────
        print(f"\n  Raw detections (threshold={args.threshold}, total={raw_count}"
              f"{', after NMS=' + str(len(detections)) if args.nms else ''}):\n")
        print_detections(detections, img_w, img_h)
        print()

        # ── save annotated image ─────────────────────────────────
        if detections:
            safe = re.sub(r"[^\w\s-]", "", raw_input[:30]).strip().replace(" ", "_")
            ts   = datetime.datetime.now().strftime("%H%M%S")
            out  = os.path.join(OUTPUT_DIR, f"v6_q{query_count:02d}_{safe}_{ts}.jpg")
            draw_boxes(args.image, detections, out)
        else:
            print(f"  Nothing returned above threshold {args.threshold}.")

        print()
        log_entries.append({"query": raw_input, "prompts": prompts, "detections": detections})

    # ── save session log ─────────────────────────────────────────
    if log_entries:
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = os.path.join(OUTPUT_DIR, f"v6_session_{ts}.txt")
        with open(log, "w") as f:
            f.write(f"Model     : {args.model}\n")
            f.write(f"Image     : {args.image}\n")
            f.write(f"Threshold : {args.threshold}\n")
            f.write(f"NMS       : {args.nms}\n\n")
            for e in log_entries:
                f.write(f"QUERY: {e['query']}\n")
                f.write(f"PROMPTS: {e['prompts']}\n")
                for d in e["detections"]:
                    f.write(f"  score={d['score']:.4f}  box={d['box']}  label={d['label']}\n")
                f.write("\n")
        print(f"  Log → {log}")

    print("  Done.\n")


if __name__ == "__main__":
    main()
