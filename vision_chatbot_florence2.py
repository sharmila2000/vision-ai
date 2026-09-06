"""
vision_chatbot_florence2.py
---------------------------
Phase 1 — Object Location Chatbot using Florence-2-large

Florence-2 (Microsoft) is a unified vision foundation model that handles
detection, grounding, segmentation and captioning in a single architecture.

For object detection it uses the task prompt:
  <OPEN_VOCABULARY_DETECTION>  — find any object by text description
  <CAPTION_TO_PHRASE_GROUNDING> — ground every noun in a caption

Why Florence-2 is stronger than OWLv2 for small objects:
  - Uses a DaViT (Dual Attention Vision Transformer) image encoder
    trained on 5.4 billion image-text-annotation triples (FLD-5B dataset)
  - The grounding head directly predicts box coordinates from visual features
  - Trained with multi-scale supervision — explicitly handles objects at
    different scales including very small ones
  - Language backbone understands complex queries and synonyms better
  - ~900M parameters with much richer training signal than OWLv2

MPS fixes applied:
  - attn_implementation="eager"  (avoids SDPA dispatch crash on MPS)
  - use_cache=False              (past_key_values returns None on MPS)
  - dtype= not torch_dtype=      (deprecated param)

Usage:
    python vision_chatbot_florence2.py --image images/img01.jpg
    > where is the key?
    > find the laptop
    > where is the bag?
    > list all objects
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
from transformers import AutoProcessor, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID   = "microsoft/Florence-2-large"
OUTPUT_DIR = "results_florence2"

# Florence-2 detection task tokens
TASK_OVD    = "<OPEN_VOCABULARY_DETECTION>"
TASK_GROUND = "<CAPTION_TO_PHRASE_GROUNDING>"

# Query expansion map
QUERY_EXPANSIONS: dict[str, list[str]] = {
    "key":       ["house key", "metal key", "door key", "key on sofa",
                  "key on armrest", "key ring", "car key"],
    "keys":      ["house key", "metal key", "key ring"],
    "phone":     ["mobile phone", "smartphone", "cell phone"],
    "remote":    ["remote control", "tv remote"],
    "glasses":   ["spectacles", "eyeglasses"],
    "bag":       ["handbag", "backpack", "shoulder bag"],
    "wallet":    ["purse", "wallet"],
    "charger":   ["phone charger", "cable charger", "charging cable"],
    "bottle":    ["water bottle", "plastic bottle"],
    "book":      ["book", "notebook"],
    "shoes":     ["shoe", "sneaker", "footwear"],
    "watch":     ["wristwatch", "watch"],
}

# False-positive suppression
FP_FILTER: dict[str, list[str]] = {
    "key":    ["necklace", "pendant", "jewelry", "jewellery",
               "chain", "person", "man", "woman", "neck", "collar"],
    "keys":   ["necklace", "pendant", "jewelry", "jewellery", "chain"],
    "phone":  ["remote", "laptop"],
    "remote": ["phone"],
}

# Box drawing
BOX_COLOURS   = [(0,200,50), (50,150,255), (255,80,80), (0,220,220), (200,50,200)]
BOX_THICKNESS = 3
LABEL_SCALE   = 0.7
LABEL_THICK   = 2


# ============================================================
# DEVICE
# ============================================================

def get_device() -> str:
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_dtype(device: str) -> torch.dtype:
    if device == "cuda": return torch.float16
    return torch.float32


# ============================================================
# MODEL LOAD
# ============================================================

def load_model(device: str, dtype: torch.dtype, model_id: str = MODEL_ID):
    print(f"  Loading Florence-2 processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("  Processor loaded.")

    print(f"  Loading Florence-2 model (~900 MB, one-time download)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    print("  Model loaded.")
    return processor, model


# ============================================================
# QUERY PARSING
# ============================================================

def parse_query(query: str) -> str:
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
    return q


# ============================================================
# DETECTION
# ============================================================

def run_florence(image, task, text, processor, model, device) -> dict:
    prompt = task + text
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            use_cache=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height)
    )
    if device == "mps":    torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()
    return parsed


def _is_fp(label: str, query_item: str) -> bool:
    banned = FP_FILTER.get(query_item.lower(), [])
    return any(b in label.lower() for b in banned)


def _run_single(image, item, processor, model, device, query_item) -> list[dict]:
    detections = []
    try:
        r1 = run_florence(image, TASK_OVD, item, processor, model, device)
        if TASK_OVD in r1:
            data = r1[TASK_OVD]
            bboxes = data.get("bboxes", [])
            labels = data.get("bboxes_labels", [""] * len(bboxes))
            for box, lbl in zip(bboxes, labels):
                eff = lbl or item
                if not _is_fp(eff, query_item):
                    detections.append({"box": [int(v) for v in box], "label": eff,
                                       "pass": 1, "query": item})
    except Exception as e:
        print(f"    [OVD error for '{item}'] {e}")

    caption = f"A room with a {item} visible."
    try:
        r2 = run_florence(image, TASK_GROUND, caption, processor, model, device)
        if TASK_GROUND in r2:
            data = r2[TASK_GROUND]
            bboxes = data.get("bboxes", [])
            labels = data.get("labels", [""] * len(bboxes))
            for box, lbl in zip(bboxes, labels):
                first_word = item.lower().split()[0]
                if first_word in lbl.lower() and not _is_fp(lbl, query_item):
                    detections.append({"box": [int(v) for v in box], "label": lbl or item,
                                       "pass": 2, "query": item})
    except Exception as e:
        print(f"    [Grounding error for '{item}'] {e}")
    return detections


def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def _containment(small, large):
    ix1=max(small[0],large[0]); iy1=max(small[1],large[1])
    ix2=min(small[2],large[2]); iy2=min(small[3],large[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    area=(small[2]-small[0])*(small[3]-small[1])
    return inter/area if area else 0.0

def _nms(dets, iou_thr=0.4):
    used=[False]*len(dets); kept=[]
    for i,d in enumerate(dets):
        if used[i]: continue
        kept.append(d)
        for j in range(i+1,len(dets)):
            if used[j]: continue
            a,b=d["box"],dets[j]["box"]
            if _iou(a,b)>iou_thr or _containment(b,a)>=0.7 or _containment(a,b)>=0.6:
                used[j]=True
    return kept


def detect(image, item, processor, model, device) -> list[dict]:
    expansions = QUERY_EXPANSIONS.get(item.lower(), [item])
    if item.lower() not in [e.lower() for e in expansions]:
        expansions = [item] + expansions
    print(f"    Expansions for '{item}': {expansions}")
    all_dets = []
    for exp in expansions:
        all_dets.extend(_run_single(image, exp, processor, model, device, item))
    return _nms(all_dets)


# ============================================================
# DRAW BOXES
# ============================================================

def draw_boxes(image_path, detections, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print("  [warn] Cannot read image.")
        return
    for idx, det in enumerate(detections):
        x1,y1,x2,y2 = det["box"]
        colour = BOX_COLOURS[idx % len(BOX_COLOURS)]
        cv2.rectangle(img, (x1,y1), (x2,y2), colour, BOX_THICKNESS)
        label = det["label"]
        (tw,th),bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK)
        ly = max(y1, th+8)
        cv2.rectangle(img, (x1,ly-th-bl-4), (x1+tw+4,ly), colour, -1)
        cv2.putText(img, label, (x1+2,ly-bl-2),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (255,255,255), LABEL_THICK, cv2.LINE_AA)
    cv2.imwrite(output_path, img)
    print(f"  Annotated image → {output_path}")


# ============================================================
# LOCATION DESCRIPTION
# ============================================================

def describe_location(box, img_w, img_h) -> str:
    x1,y1,x2,y2 = box
    cx=(x1+x2)/2; cy=(y1+y2)/2
    col="left" if cx<img_w/3 else ("centre" if cx<2*img_w/3 else "right")
    row="top"  if cy<img_h/3 else ("middle"  if cy<2*img_h/3 else "bottom")
    return f"{row}-{col} of the image  (box: {x1},{y1} → {x2},{y2})"


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vision Chatbot Florence-2 — open-vocabulary detector"
    )
    parser.add_argument("--image",  default="images/img01.jpg")
    parser.add_argument("--model",  default=MODEL_ID)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}"); sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    dtype  = get_dtype(device)

    print()
    print("=" * 60)
    print("  VISION CHATBOT — Florence-2-large Detector")
    print("=" * 60)
    print(f"  Image  : {args.image}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}  ({dtype})")
    print("=" * 60)

    image = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size
    print(f"\n  Image loaded: {img_w}×{img_h}")

    processor, model = load_model(device, dtype, args.model)

    print("\n" + "=" * 60)
    print("  Ready. Type any query.")
    print("  Examples: where is the key? / find the laptop / list all objects")
    print("  Type 'quit' to exit.")
    print("=" * 60 + "\n")

    log_entries = []; query_count = 0

    while True:
        try:
            raw_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session ended."); break
        if not raw_input: continue
        if raw_input.lower() in {"quit", "exit", "q"}: break

        if raw_input.lower() in {"list all objects", "list objects", "list all"}:
            print(f"\n  Generating scene description...")
            result = run_florence(image, "<MORE_DETAILED_CAPTION>", "", processor, model, device)
            caption = result.get("<MORE_DETAILED_CAPTION>", "No caption generated.")
            print("\n" + "-"*60)
            print(f"  Scene: {caption}")
            print("-"*60 + "\n")
            log_entries.append({"query": raw_input, "caption": caption})
            continue

        query_count += 1
        item = parse_query(raw_input)
        print(f"\n  [Query {query_count}] Detecting '{item}'...")
        detections = detect(image, item, processor, model, device)

        print("\n" + "-"*60)
        if detections:
            print(f"  Found {len(detections)} instance(s) of '{item}':\n")
            for i, det in enumerate(detections, 1):
                loc = describe_location(det["box"], img_w, img_h)
                print(f"  {i}. {loc}")
                print(f"     Label : {det['label']}   (Pass {det.get('pass','?')})")
            safe = re.sub(r"[^\w\s-]","",raw_input).strip().replace(" ","_")[:40]
            ts   = datetime.datetime.now().strftime("%H%M%S")
            out  = os.path.join(OUTPUT_DIR, f"flo_q{query_count:02d}_{safe}_{ts}.jpg")
            draw_boxes(args.image, detections, out)
        else:
            print(f"  '{item}' not found. Try rephrasing: 'metal key', 'house key'.")
        print("-"*60 + "\n")
        log_entries.append({"query": raw_input, "detections": detections})

    if log_entries:
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = os.path.join(OUTPUT_DIR, f"flo_session_{ts}.txt")
        with open(log, "w") as f:
            f.write(f"Model: {args.model}\nImage: {args.image}\n\n")
            for e in log_entries:
                f.write(f"QUERY: {e['query']}\n")
                for d in e.get("detections", []):
                    f.write(f"  box={d['box']}  label={d['label']}\n")
                if "caption" in e:
                    f.write(f"  caption: {e['caption']}\n")
                f.write("\n")
        print(f"  Log → {log}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
