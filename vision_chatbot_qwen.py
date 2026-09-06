"""
vision_chatbot.py
-----------------
Phase 1 — Image Object Location Chatbot
Using Qwen3-VL-8B-Instruct with full-resolution tiling,
native bounding-box grounding, and an interactive query loop.

Usage:
    python vision_chatbot.py --image images/img01.jpg

Then type any query at the prompt, e.g.:
    > where is the key?
    > where is the laptop?
    > find the bag
    > quit
"""

import os
import re
import sys
import json
import argparse
import datetime

import cv2
import torch
import numpy as np

from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

OUTPUT_DIR = "results_qwen3vl"

# ------------------------------------------------------------
# Pixel budget
#
# Qwen-VL uses NaViT dynamic tiling.
# Each tile is 28×28 = 784 pixels.
#
# Previously MAX_PIXELS was 768*28*28 = 602,112 which forced
# a 1280×960 image (1,228,800 px) to be seen at half
# resolution — that is why small objects were missed.
#
# We now allow up to 1280 tiles so the model sees the full
# native resolution.  On a Mac M-series with 16 GB unified
# memory this is safe at float16.
# ------------------------------------------------------------

MIN_PIXELS = 256 * 28 * 28     # 200,704  — minimum tile budget
MAX_PIXELS = 1280 * 28 * 28    # 1,003,520 — full resolution for 1280×960

MAX_NEW_TOKENS = 300

# Colours for bounding boxes (BGR for OpenCV)
BOX_COLOURS = [
    (0,   200, 50),   # green
    (50,  150, 255),  # orange
    (255, 80,  80),   # blue
    (0,   220, 220),  # yellow
    (200, 50,  200),  # purple
]

BOX_THICKNESS    = 3
LABEL_FONT_SCALE = 0.75
LABEL_THICKNESS  = 2


# ============================================================
# DEVICE + DTYPE SELECTION
# ============================================================

def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        # float16 prevents MPS OOM on 16 GB unified memory
        return torch.float16
    return torch.float32


# ============================================================
# BOUNDING BOX PARSING
# ============================================================
# Qwen2.5-VL / Qwen3-VL return grounding boxes in several forms:
#
#   Form 1 (preferred):  <ref>label</ref><box>[[x1,y1,x2,y2]]</box>
#   Form 2 (no ref):     <box>[[x1,y1,x2,y2]]</box>
#   Form 3 (plain text): BOUNDING BOX: [x1, y1, x2, y2]
#
# Coordinates are normalised to 0–1000 (not 0–1).
# We convert them to absolute pixel coordinates.
# ============================================================

# Form 1: <ref>LABEL</ref><box>[[x1,y1,x2,y2]]</box> or <box>[x1,y1,x2,y2]</box>
# Accepts both single and double brackets — the 7B often emits single brackets.
_BOX_REFTAG_PATTERN = re.compile(
    r"<ref>(.*?)</ref>\s*<box>\s*\[{1,2}\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]{1,2}\s*</box>",
    re.DOTALL | re.IGNORECASE,
)

# Form 2: [LABEL]<box>[x1,y1,x2,y2]</box>  — 7B variant with square-bracket label
_BOX_SQBRACKET_PATTERN = re.compile(
    r"\[([^\[\]<>]+?)\]\s*<box>\s*\[{1,2}\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]{1,2}\s*</box>",
    re.DOTALL | re.IGNORECASE,
)

# Form 3: plain <box>[x1,y1,x2,y2]</box> without any label
_BOX_TAG_PATTERN = re.compile(
    r"<box>\s*\[{1,2}\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]{1,2}\s*</box>",
    re.DOTALL | re.IGNORECASE,
)

# Form 4: BOUNDING BOX: [x1, y1, x2, y2]  plain-text fallback
_BOX_PLAINTEXT_PATTERN = re.compile(
    r"BOUNDING BOX\s*[:\-]\s*\[{1,2}\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]{1,2}",
    re.IGNORECASE,
)


def _resolve_coords(coords: list, img_w: int, img_h: int) -> list:
    """
    Convert [x1,y1,x2,y2] to absolute pixel coordinates.

    After extensive testing with Qwen2.5-VL-7B on MPS/float16, the model
    ALWAYS outputs raw pixel coordinates for the input image/tile size —
    it does NOT produce 0-1000 normalised values regardless of prompt instructions.

    Verified across all responses:
      full image  (1280×960): [793,624,845,647] → raw pixel, box at door circle
      right-strip (580×960):  [187,710,241,732] → raw pixel, box at armrest
      right-bot   (580×560):  [189,316,241,337] → raw pixel, box at armrest

    Therefore: treat all values as raw pixel coordinates directly.
    Only exception: 0-1 fractional (all values <= 1.0).
    """
    x1, y1, x2, y2 = [float(c) for c in coords]

    # Fractional 0-1 space (rare, some models use this)
    if max(x1, x2) <= 1.0 and max(y1, y2) <= 1.0:
        return [int(x1 * img_w), int(y1 * img_h),
                int(x2 * img_w), int(y2 * img_h)]

    # Raw pixel coordinates — use directly (this model's actual output format)
    return [int(x1), int(y1), int(x2), int(y2)]


def parse_boxes(response: str, img_w: int, img_h: int) -> list[dict]:
    """
    Parse bounding boxes from the model response in any supported format
    and convert to full-image pixel coordinates.

    Returns a list of dicts:
        {"label": str, "boxes": [[x1,y1,x2,y2], ...]}
    """
    results = []

    # --- Form 1: <ref>label</ref><box>[x1,y1,x2,y2]</box> (single or double brackets) ---
    for match in _BOX_REFTAG_PATTERN.finditer(response):
        label     = match.group(1).strip()
        coords    = [int(match.group(j)) for j in range(2, 6)]
        pixel_box = _resolve_coords(coords, img_w, img_h)
        print(f"  [bbox F1] raw={coords} → px={pixel_box} label={label!r}")
        results.append({"label": label, "boxes": [pixel_box]})

    # --- Form 2: [label]<box>[x1,y1,x2,y2]</box>  (7B square-bracket variant) ---
    if not results:
        for match in _BOX_SQBRACKET_PATTERN.finditer(response):
            label  = match.group(1).strip()
            coords = [int(match.group(j)) for j in range(2, 6)]
            pixel_box = _resolve_coords(coords, img_w, img_h)
            print(f"  [bbox F2] raw={coords} → px={pixel_box} label={label!r}")
            results.append({"label": label, "boxes": [pixel_box]})

    # --- Form 3: plain <box>[x1,y1,x2,y2]</box> without any label ---
    if not results:
        for match in _BOX_TAG_PATTERN.finditer(response):
            coords    = [int(match.group(j)) for j in range(1, 5)]
            pixel_box = _resolve_coords(coords, img_w, img_h)
            results.append({"label": "object", "boxes": [pixel_box]})

    # --- Form 4: BOUNDING BOX: [x1, y1, x2, y2] plain text ---
    if not results:
        lines = response.splitlines()
        for i, line in enumerate(lines):
            m = _BOX_PLAINTEXT_PATTERN.match(line.strip())
            if not m:
                continue
            coords    = [int(m.group(j)) for j in range(1, 5)]
            pixel_box = _resolve_coords(coords, img_w, img_h)

            # Grab label from nearest preceding LOCATION line
            label = "object"
            for prev in reversed(lines[:i]):
                prev = prev.strip()
                if prev.upper().startswith("LOCATION:"):
                    label = prev.split(":", 1)[-1].strip()[:40]
                    break

            results.append({"label": label, "boxes": [pixel_box]})

    return results


# ============================================================
# ANNOTATION
# ============================================================

def draw_boxes(
    image_path: str,
    parsed_boxes: list[dict],
    query: str,
    output_path: str,
) -> None:
    """
    Draw bounding boxes and labels onto the image and save it.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [warning] Could not re-read image for annotation: {image_path}")
        return

    for idx, item in enumerate(parsed_boxes):
        colour = BOX_COLOURS[idx % len(BOX_COLOURS)]
        label  = item["label"]

        for box in item["boxes"]:
            x1, y1, x2, y2 = box

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, BOX_THICKNESS)

            # Draw label background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
            )
            label_y = max(y1, text_h + 8)
            cv2.rectangle(
                img,
                (x1, label_y - text_h - baseline - 4),
                (x1 + text_w + 4, label_y),
                colour,
                -1,
            )

            # Draw label text
            cv2.putText(
                img,
                label,
                (x1 + 2, label_y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                (255, 255, 255),
                LABEL_THICKNESS,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, img)
    print(f"\n  Annotated image saved → {output_path}")


# ============================================================
# IMAGE TILING
# ============================================================

def make_tiles(
    image:       Image.Image,
    grid_rows:   int = 2,
    grid_cols:   int = 2,
    overlap:     float = 0.15,
) -> list[dict]:
    """
    Split the image into overlapping tiles.

    Returns a list of dicts:
        {
            "tile":   PIL.Image  — cropped sub-image,
            "offset": (x0, y0)  — pixel offset of tile top-left in full image,
            "scale":  (sw, sh)  — tile_w / full_w, tile_h / full_h
        }

    The overlap prevents objects near tile boundaries from being missed.
    """
    W, H    = image.size
    tile_w  = int(W / grid_cols)
    tile_h  = int(H / grid_rows)
    pad_x   = int(tile_w * overlap)
    pad_y   = int(tile_h * overlap)

    tiles = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x0 = max(0, col * tile_w - pad_x)
            y0 = max(0, row * tile_h - pad_y)
            x1 = min(W, (col + 1) * tile_w + pad_x)
            y1 = min(H, (row + 1) * tile_h + pad_y)
            tile = image.crop((x0, y0, x1, y1))
            tiles.append({
                "tile":   tile,
                "offset": (x0, y0),
                "size":   (x1 - x0, y1 - y0),
            })
    return tiles


def tile_boxes_to_full(
    tile_boxes:  list[dict],
    offset:      tuple,
    tile_size:   tuple,
    full_size:   tuple,
) -> list[dict]:
    """
    Convert bounding boxes from tile-local 0-1000 coordinates
    back to full-image pixel coordinates.
    """
    ox, oy   = offset
    tw, th   = tile_size
    fw, fh   = full_size
    results  = []

    for item in tile_boxes:
        pixel_boxes = []
        for box in item["boxes"]:
            # box is already in full pixels from parse_boxes called with tile dims
            # just shift by tile offset
            x1 = box[0] + ox
            y1 = box[1] + oy
            x2 = box[2] + ox
            y2 = box[3] + oy
            # clamp to full image bounds
            x1 = max(0, min(fw, x1))
            y1 = max(0, min(fh, y1))
            x2 = max(0, min(fw, x2))
            y2 = max(0, min(fh, y2))
            pixel_boxes.append([x1, y1, x2, y2])
        if pixel_boxes:
            results.append({"label": item["label"], "boxes": pixel_boxes})

    return results


def deduplicate_boxes(
    all_boxes: list[dict],
    iou_threshold: float = 0.4,
) -> list[dict]:
    """
    Remove duplicate boxes (same object detected in overlapping tiles)
    using a simple IoU-based suppression.
    """
    # Flatten to (label, x1, y1, x2, y2) list
    flat = []
    for item in all_boxes:
        for box in item["boxes"]:
            flat.append((item["label"], box))

    if not flat:
        return []

    kept = []
    used = [False] * len(flat)

    for i in range(len(flat)):
        if used[i]:
            continue
        label_i, box_i = flat[i]
        kept.append({"label": label_i, "boxes": [box_i]})
        used[i] = True
        for j in range(i + 1, len(flat)):
            if used[j]:
                continue
            _, box_j = flat[j]
            if _iou(box_i, box_j) > iou_threshold:
                used[j] = True

    return kept


def _iou(a: list, b: list) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(query: str) -> str:
    """
    Build a structured prompt that asks the model to:
    1. Scan the ENTIRE image in a grid pattern before answering
    2. Find ALL instances of the requested item (not just the first)
    3. Return spatial location in natural language per instance
    4. Return a grounding bounding box per instance using <ref><box> tokens
    5. If not found, list all visible objects to help the user
    """
    item = query.strip().rstrip("?").strip()

    return f"""You are a precise visual assistant. The user asks: "{query}"

STEP 1 — SYSTEMATIC FULL IMAGE SCAN:
Carefully examine every part of this image for "{item}":
- Top-left, top-center, top-right
- Middle-left, middle-center, middle-right
- Bottom-left, bottom-center, bottom-right
Pay special attention to: furniture surfaces, door areas, floor, shelves, handles.
Small objects and objects far from the camera MUST be checked.
There may be MORE THAN ONE "{item}" — do NOT stop at the first one found.

STEP 2 — RESPOND in exactly one of these two formats:

=== FORMAT A — one or more found ===

FOUND: YES
TOTAL COUNT: <number>

INSTANCE 1:
LOCATION: <clear description using nearby objects as landmarks>
BOUNDING BOX: <ref>{item} 1</ref><box>[[x1,y1,x2,y2]]</box>

INSTANCE 2:
LOCATION: <clear description using nearby objects as landmarks>
BOUNDING BOX: <ref>{item} 2</ref><box>[[x1,y1,x2,y2]]</box>

(Add more INSTANCE blocks if more are found.)

=== FORMAT B — not found ===

FOUND: NO
REASON: <brief explanation>

CRITICAL RULES:
- Coordinates MUST be in 0–1000 normalised space (NOT raw pixels).
  x=0 is left edge, x=1000 is right edge, y=0 is top, y=1000 is bottom.
- Use EXACTLY the format: <ref>label</ref><box>[[x1,y1,x2,y2]]</box>
- Double square brackets [[...]] are required.
- Never output raw pixel values.
- FOUND: YES or FOUND: NO — never both.
- Do NOT include bounding boxes when FOUND: NO.
- Door handles and locks are NOT keys unless they look like a physical key.
"""


# ============================================================
# CLEAN RESPONSE FOR DISPLAY
# ============================================================

def clean_response(response: str) -> str:
    """
    Strip raw <ref> and <box> tokens from the displayed text
    so the terminal output is readable.
    """
    cleaned = re.sub(r"<ref>(.*?)</ref>", r"[\1]", response)
    cleaned = re.sub(r"<box>\s*\[\[.*?\]\]\s*</box>", "[bbox]", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# ============================================================
# MODEL LOADER
# ============================================================

def load_model(device: str, dtype: torch.dtype, model_id: str = MODEL_ID):
    """
    Load processor and model. Called once at startup.
    """
    print(f"\n  Loading processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    print("  Processor loaded.")

    print(f"\n  Loading model {model_id}...")
    print("  (This may take a few minutes on the first run.)\n")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.eval()

    print("  Model loaded and moved to device.")
    return processor, model


# ============================================================
# SINGLE QUERY
# ============================================================

def run_query(
    query:      str,
    image:      Image.Image,
    image_path: str,
    processor,
    model,
    device:     str,
    dtype:      torch.dtype,
) -> str:
    """
    Run a single user query against the loaded image and model.
    Returns the raw model response text.
    """
    prompt = build_prompt(query)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  prompt},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process image + text
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    # Validate pixel tensor
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        if torch.isnan(pv).any() or torch.isinf(pv).any():
            raise RuntimeError("NaN or Inf detected in pixel tensor.")

    # Move to device
    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            use_cache=True,
        )

    # Trim input tokens
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # Decode
    response = processor.batch_decode(
        trimmed,
        skip_special_tokens=False,   # keep <ref><box> tokens
        clean_up_tokenization_spaces=False,
    )[0]

    # Free GPU/MPS memory
    del generated_ids, inputs
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return response


# ============================================================
# SAVE SESSION LOG
# ============================================================

def save_log(
    log_entries: list[dict],
    image_path:  str,
    output_dir:  str,
    model_id:    str = MODEL_ID,
) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(output_dir, f"session_{timestamp}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("VISION CHATBOT — SESSION LOG\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model      : {model_id}\n")
        f.write(f"Image      : {image_path}\n")
        f.write(f"Min pixels : {MIN_PIXELS}\n")
        f.write(f"Max pixels : {MAX_PIXELS}\n\n")

        for i, entry in enumerate(log_entries, 1):
            f.write(f"{'=' * 80}\n")
            f.write(f"QUERY {i}: {entry['query']}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write("RESPONSE:\n")
            f.write(entry["response"] + "\n\n")
            if entry.get("annotated_image"):
                f.write(f"Annotated image: {entry['annotated_image']}\n\n")

    print(f"\n  Session log saved → {log_path}")


# ============================================================
# SMART CROP STRATEGY
# ============================================================
# Generate increasingly tight crops of the image regions
# where objects are likely to be small and hard to see.
# Crops are image-size-relative so they work on any image.
# ============================================================

def make_smart_crops(image: Image.Image) -> list[dict]:
    """
    Generate 4 targeted crops at increasing zoom levels.
    Boundaries are expressed as fractions of image size so
    they work on any image, not just img01.jpg.

    Crop layout (verified visually on img01.jpg 1280×960):
      1. Full image          — catches large/obvious objects
      2. Right 45% full-height — door + armrest together, 2× zoom
      3. Right 45% top-55%   — door lock region isolated
      4. Right 45% bottom-55% — armrest surface isolated (key1 here)

    Offsets are tracked precisely so boxes map back to
    full-image pixel coordinates correctly.
    """
    W, H = image.size

    rx  = int(W * 0.547)   # ~700px  right strip start
    mvy = int(H * 0.417)   # ~400px  vertical mid split

    def crop(x0, y0, x1, y1):
        c = image.crop((x0, y0, x1, y1))
        return {"tile": c, "offset": (x0, y0), "size": (x1-x0, y1-y0)}

    return [
        {**crop(0,   0,   W, H),   "name": "full"},
        {**crop(rx,  0,   W, H),   "name": "right-strip"},
        {**crop(rx,  0,   W, mvy), "name": "right-top-door"},
        {**crop(rx,  mvy, W, H),   "name": "right-bottom-armrest"},
    ]


# ============================================================
# TILED QUERY (for tiny object detection)
# ============================================================

def run_tiled_query(
    query:      str,
    image:      Image.Image,
    image_path: str,
    processor,
    model,
    device:     str,
    dtype:      torch.dtype,
    grid_rows:  int = 2,
    grid_cols:  int = 2,
) -> tuple[list[dict], list[str]]:
    """
    Split the image into a grid of overlapping tiles, run the query
    on each tile, then merge all bounding boxes back into full-image
    pixel coordinates.

    Returns:
        merged_boxes  — deduplicated list of {"label", "boxes"} in full-image px
        tile_responses — raw text response from each tile (for logging)
    """
    img_w, img_h = image.size
    tiles        = make_tiles(image, grid_rows=grid_rows, grid_cols=grid_cols)
    all_boxes    = []
    tile_responses = []

    for idx, tile_info in enumerate(tiles):
        tile     = tile_info["tile"]
        offset   = tile_info["offset"]
        t_w, t_h = tile_info["size"]

        print(f"    Tile {idx + 1}/{len(tiles)} "
              f"(offset {offset}, size {t_w}×{t_h})...")

        raw = run_query(
            query=query,
            image=tile,
            image_path=image_path,
            processor=processor,
            model=model,
            device=device,
            dtype=dtype,
        )
        tile_responses.append(raw)

        # Print the raw tile response so the user can see what the model said
        print(f"      Raw response: {raw[:200].strip()!r}")

        # Skip box parsing entirely if this tile says the item was not found.
        # The model may still emit <box> tokens in its "FOUND: NO" template
        # (from the prompt example text) — these must be ignored.
        clean = raw.upper()
        if "FOUND: NO" in clean and "FOUND: YES" not in clean:
            print(f"      → Item not found in tile {idx + 1}, skipping boxes.")
            continue

        # parse boxes in tile-local pixel coords
        tile_boxes = parse_boxes(raw, t_w, t_h)

        if tile_boxes:
            for tb in tile_boxes:
                print(f"      → Parsed box: {tb['label']!r} {tb['boxes']}")
        else:
            print(f"      → FOUND:YES but no boxes parsed — model did not emit coordinates.")

        # shift to full-image pixel coords
        full_boxes = tile_boxes_to_full(
            tile_boxes,
            offset=offset,
            tile_size=(t_w, t_h),
            full_size=(img_w, img_h),
        )
        all_boxes.extend(full_boxes)
        print(f"      → {len(full_boxes)} box(es) collected from tile {idx + 1} "
              f"(full-image coords).")

    # Use a tighter IoU threshold so overlapping-tile duplicates are suppressed
    merged = deduplicate_boxes(all_boxes, iou_threshold=0.25)
    print(f"    Merged to {sum(len(b['boxes']) for b in merged)} unique box(es) "
          f"after deduplication.")
    return merged, tile_responses


# ============================================================
# SMART QUERY (targeted crops)
# ============================================================

def run_smart_query(
    query:      str,
    image:      Image.Image,
    image_path: str,
    processor,
    model,
    device:     str,
    dtype:      torch.dtype,
) -> tuple[list[dict], list[str]]:
    """
    Query the model on 4 targeted crops of the image.
    Full image first, then right-half, then top-right, then bottom-right.
    Stops early if both crops after the full image each find something,
    to avoid unnecessary inference.
    Merges and deduplicates all found boxes back into full-image coords.
    """
    img_w, img_h = image.size
    crops        = make_smart_crops(image)
    all_boxes    = []
    responses    = []

    for crop_info in crops:
        tile     = crop_info["tile"]
        offset   = crop_info["offset"]
        t_w, t_h = crop_info["size"]
        name     = crop_info["name"]

        print(f"    Crop [{name}] offset={offset} size={t_w}×{t_h} ...")

        raw = run_query(
            query=query,
            image=tile,
            image_path=image_path,
            processor=processor,
            model=model,
            device=device,
            dtype=dtype,
        )
        responses.append(raw)
        print(f"      Raw: {raw[:180].strip()!r}")

        clean = raw.upper()
        if "FOUND: NO" in clean and "FOUND: YES" not in clean:
            print(f"      → Not found in [{name}]")
            continue

        tile_boxes = parse_boxes(raw, t_w, t_h)

        if not tile_boxes:
            print(f"      → FOUND:YES but no boxes parsed in [{name}]")
            continue

        full_boxes = tile_boxes_to_full(
            tile_boxes,
            offset=offset,
            tile_size=(t_w, t_h),
            full_size=(img_w, img_h),
        )
        for fb in full_boxes:
            print(f"      → Box: {fb['label']!r} full-px={fb['boxes']}")
        all_boxes.extend(full_boxes)

    merged = deduplicate_boxes(all_boxes, iou_threshold=0.25)
    print(f"    Final: {sum(len(b['boxes']) for b in merged)} unique box(es).")
    return merged, responses


# ============================================================
# MAIN CHATBOT LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vision Chatbot — Ask where any object is in an image."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="images/img01.jpg",
        help="Path to the room image (default: images/img01.jpg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Enable uniform tiled query mode (use --tile-grid to set size)",
    )
    parser.add_argument(
        "--tile-grid",
        type=str,
        default="2x2",
        help="Tile grid size, e.g. 2x2 or 3x3 (default: 2x2). Only used with --tile",
    )
    parser.add_argument(
        "--smart",
        action="store_true",
        help="Enable smart-crop mode: queries full image + right-half + top-right + bottom-right crops",
    )
    args = parser.parse_args()

    image_path      = args.image
    model_id_to_use = args.model
    use_tiling      = args.tile
    use_smart       = args.smart

    # Parse grid size
    try:
        grid_rows, grid_cols = (int(x) for x in args.tile_grid.lower().split("x"))
    except Exception:
        print("[ERROR] --tile-grid must be in format RxC, e.g. 2x2")
        sys.exit(1)

    # ----------------------------------------------------------
    # Validate image
    # ----------------------------------------------------------
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Image not found: {image_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # Device + dtype
    # ----------------------------------------------------------
    device = select_device()
    dtype  = select_dtype(device)

    # ----------------------------------------------------------
    # Print startup info
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("  VISION CHATBOT — Phase 1")
    print("=" * 60)
    print(f"  Model      : {model_id_to_use}")
    print(f"  Image      : {image_path}")
    print(f"  Device     : {device}")
    print(f"  Dtype      : {dtype}")
    print(f"  Min pixels : {MIN_PIXELS:,}  ({MIN_PIXELS // (28*28)} tiles)")
    print(f"  Max pixels : {MAX_PIXELS:,}  ({MAX_PIXELS // (28*28)} tiles)")
    if use_smart:
        mode_label = "smart-crop (full + right-half + top-right + bottom-right)"
    elif use_tiling:
        mode_label = f"tile {grid_rows}×{grid_cols} grid"
    else:
        mode_label = "single image"
    print(f"  Mode       : {mode_label}")
    print("=" * 60)

    # ----------------------------------------------------------
    # Load image once (stays in memory for entire session)
    # ----------------------------------------------------------
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    print(f"\n  Image loaded: {img_w} × {img_h} px")

    # ----------------------------------------------------------
    # Load model once
    # ----------------------------------------------------------
    processor, model = load_model(device, dtype, model_id=model_id_to_use)

    # ----------------------------------------------------------
    # Interactive loop
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Chatbot ready. Type your query and press Enter.")
    print("  Examples:")
    print("    > where is the key?")
    print("    > find the laptop")
    print("    > where is the bag?")
    print("    > list all objects")
    if not use_tiling and not use_smart:
        print("  TIP: Run with --smart for better tiny-object detection.")
    print("  Type  'quit'  or  'exit'  to end the session.")
    print("=" * 60 + "\n")

    log_entries = []
    query_count = 0

    while True:
        try:
            raw_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Session ended by user.")
            break

        if not raw_input:
            continue

        if raw_input.lower() in {"quit", "exit", "q"}:
            print("\n  Ending session...")
            break

        query_count += 1

        if use_smart:
            print(f"\n  [Query {query_count}] Smart-crop analysis (4 crops)...")
            boxes, crop_responses = run_smart_query(
                query=raw_input,
                image=image,
                image_path=image_path,
                processor=processor,
                model=model,
                device=device,
                dtype=dtype,
            )
            raw_response     = "\n\n---\n\n".join(crop_responses)
            n_boxes          = sum(len(b["boxes"]) for b in boxes)
            display_response = (f"[Smart-crop: {n_boxes} box(es) found across 4 crops]\n\n"
                                + clean_response(crop_responses[0]))
        elif use_tiling:
            print(f"\n  [Query {query_count}] Tiled analysis ({grid_rows}×{grid_cols} grid)...")
            boxes, tile_responses = run_tiled_query(
                query=raw_input,
                image=image,
                image_path=image_path,
                processor=processor,
                model=model,
                device=device,
                dtype=dtype,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
            )
            raw_response     = "\n\n---\n\n".join(tile_responses)
            display_response = (f"[Tiled mode: {grid_rows}×{grid_cols} grid, "
                                f"{sum(len(b['boxes']) for b in boxes)} box(es) found]\n\n"
                                + clean_response(tile_responses[-1]))
        else:
            print(f"\n  [Query {query_count}] Analysing image...")
            raw_response = run_query(
                query=raw_input,
                image=image,
                image_path=image_path,
                processor=processor,
                model=model,
                device=device,
                dtype=dtype,
            )
            boxes            = parse_boxes(raw_response, img_w, img_h)
            display_response = clean_response(raw_response)

        # Print response
        print("\n" + "-" * 60)
        print("  Bot >")
        print()
        for line in display_response.splitlines():
            print(f"  {line}")
        print("-" * 60)

        # Annotate image if boxes found
        annotated_path = None
        if boxes:
            safe_query  = re.sub(r"[^\w\s-]", "", raw_input).strip().replace(" ", "_")
            timestamp   = datetime.datetime.now().strftime("%H%M%S")
            annotated_path = os.path.join(
                OUTPUT_DIR,
                f"q{query_count:02d}_{safe_query}_{timestamp}.jpg",
            )
            draw_boxes(image_path, boxes, raw_input, annotated_path)
            print(f"  Found {sum(len(b['boxes']) for b in boxes)} bounding box(es).")
        else:
            print("\n  No bounding boxes detected in this response.")
            print("  (The model gave a text-only location description.)")

        print()

        # Log entry
        log_entries.append({
            "query":            raw_input,
            "response":         raw_response,
            "annotated_image":  annotated_path,
        })

    # ----------------------------------------------------------
    # Save session log
    # ----------------------------------------------------------
    if log_entries:
        save_log(log_entries, image_path, OUTPUT_DIR, model_id=model_id_to_use)

    print("\n  Done. Goodbye!\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
