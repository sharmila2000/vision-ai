"""
vision_chatbot_hybrid.py
------------------------
Phase 1 — Object Location Chatbot  (Hybrid OWLv2 + Florence-2)

FUSION STRATEGY
───────────────
  1. OWLv2-large runs first  — strong at tiny objects, direct box output
  2. Florence-2-large runs second — scene-aware, synonym-rich
  3. Boxes that appear in BOTH models → source="both"  (highest confidence)
     OWLv2-only boxes → kept (strong score anchor)
     Florence-2-only boxes → kept after false-positive filter
  4. Final NMS across all boxes

This eliminates the necklace false positive:
  OWLv2 doesn't fire on the necklace (score too low).
  Florence-2 necklace box has no OWL consensus → dropped by FP filter.

Usage:
    python vision_chatbot_hybrid.py --image images/img01.jpg
    python vision_chatbot_hybrid.py --image images/img01.jpg --owl-only
    python vision_chatbot_hybrid.py --image images/img01.jpg --flo-only
    > where is the key?
    > find the laptop
    > list all objects
"""

import os
import re
import sys
import argparse
import datetime

import cv2
import torch
from PIL import Image
from transformers import (
    Owlv2Processor, Owlv2ForObjectDetection,
    AutoProcessor, AutoModelForCausalLM,
)


# ============================================================
# CONFIGURATION
# ============================================================

OWL_MODEL_ID  = "google/owlv2-large-patch14-ensemble"
FLOR_MODEL_ID = "microsoft/Florence-2-large"
OUTPUT_DIR    = "results_hybrid"
OWL_MIN_SCORE = 0.20

TASK_OVD    = "<OPEN_VOCABULARY_DETECTION>"
TASK_GROUND = "<CAPTION_TO_PHRASE_GROUNDING>"
TASK_CAP    = "<MORE_DETAILED_CAPTION>"

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
    "cup":       ["coffee cup", "mug", "tea cup"],
    "laptop":    ["laptop", "laptop computer", "notebook computer"],
    "suitcase":  ["suitcase", "luggage", "trolley bag"],
}

FP_FILTER: dict[str, list[str]] = {
    "key":    ["necklace", "pendant", "jewelry", "jewellery",
               "chain", "person", "man", "woman", "neck", "collar"],
    "keys":   ["necklace", "pendant", "jewelry", "jewellery", "chain"],
    "phone":  ["remote", "laptop"],
    "remote": ["phone"],
}

BOX_COLOURS   = [(0,200,50),(50,150,255),(255,80,80),(0,220,220),(200,50,200),(255,165,0)]
BOX_THICKNESS = 3
LABEL_SCALE   = 0.65
LABEL_THICK   = 2
SOURCE_BADGE  = {"owl": "OWL", "flor": "FLO", "both": "✓✓"}


# ============================================================
# DEVICE / DTYPE
# ============================================================

def get_device():
    if torch.cuda.is_available():          return "cuda"
    if torch.backends.mps.is_available():  return "mps"
    return "cpu"

def get_dtype(device):
    return torch.float16 if device == "cuda" else torch.float32


# ============================================================
# MODEL LOADING
# ============================================================

def load_owlv2(device, dtype):
    print(f"  [OWLv2 ] Loading from {OWL_MODEL_ID}...")
    proc  = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(
        OWL_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print("  [OWLv2 ] Ready.")
    return proc, model

def load_florence(device, dtype):
    print(f"  [Flor-2] Loading from {FLOR_MODEL_ID}...")
    proc  = AutoProcessor.from_pretrained(FLOR_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLOR_MODEL_ID, dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to(device)
    model.eval()
    print("  [Flor-2] Ready.")
    return proc, model


# ============================================================
# QUERY HELPERS
# ============================================================

def parse_item(query):
    q = query.lower().strip().rstrip("?").strip()
    for filler in [
        "where is the","where is a","where is","find the","find a","find",
        "locate the","locate a","locate","show me the","show me a","show me",
        "what is the location of the","what is the location of",
        "can you find the","can you find a","can you find",
    ]:
        if q.startswith(filler):
            q = q[len(filler):].strip(); break
    return q

def expand_query(item):
    expansions = QUERY_EXPANSIONS.get(item.lower(), [])
    return [item] + [e for e in expansions if e.lower() != item.lower()]


# ============================================================
# NMS
# ============================================================

def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def _cont(small, large):
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
            if _iou(a,b)>iou_thr or _cont(b,a)>=0.7 or _cont(a,b)>=0.6:
                used[j]=True
    return kept


# ============================================================
# OWLv2 DETECTION
# ============================================================

def owl_detect_single(image, item, proc, model, device, thr):
    img_w, img_h = image.size
    prompts = [f"a photo of a {item}", f"a {item}"]
    inputs = proc(text=[prompts], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    tgt = torch.tensor([[img_h,img_w]], device=device)
    res = proc.post_process_grounded_object_detection(
        outputs=outputs, threshold=thr, target_sizes=tgt, text_labels=[prompts]
    )[0]
    return [{"box":[int(v) for v in b.tolist()],"score":float(s),"label":l,"query":item,"source":"owl"}
            for s,l,b in zip(res["scores"],res["text_labels"],res["boxes"])]

def owl_run(image, item, proc, model, device, thr=OWL_MIN_SCORE):
    all_dets = []
    for it in expand_query(item):
        all_dets.extend(owl_detect_single(image, it, proc, model, device, thr))
    all_dets.sort(key=lambda d: d["score"], reverse=True)
    return _nms(all_dets)


# ============================================================
# FLORENCE-2 DETECTION
# ============================================================

def _flor_task(image, task, text, proc, model, device):
    inputs = proc(text=task+text, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        gids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, do_sample=False, num_beams=1, use_cache=False,
        )
    gen = proc.batch_decode(gids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(gen, task=task, image_size=(image.width,image.height))
    if device=="mps": torch.mps.empty_cache()
    elif device=="cuda": torch.cuda.empty_cache()
    return parsed

def _is_fp(label, qi):
    return any(b in label.lower() for b in FP_FILTER.get(qi.lower(), []))

def flor_detect_single(image, item, proc, model, device, qi):
    dets = []
    try:
        r1 = _flor_task(image, TASK_OVD, item, proc, model, device)
        if TASK_OVD in r1:
            for box, lbl in zip(r1[TASK_OVD].get("bboxes",[]), r1[TASK_OVD].get("bboxes_labels",[])):
                eff = lbl or item
                if not _is_fp(eff, qi):
                    dets.append({"box":[int(v) for v in box],"label":eff,"query":item,"pass":1,"source":"flor"})
    except Exception as e:
        print(f"    [Flor OVD '{item}'] {e}")
    try:
        r2 = _flor_task(image, TASK_GROUND, f"A room with a {item} visible.", proc, model, device)
        if TASK_GROUND in r2:
            for box, lbl in zip(r2[TASK_GROUND].get("bboxes",[]), r2[TASK_GROUND].get("labels",[])):
                if item.lower().split()[0] in lbl.lower() and not _is_fp(lbl, qi):
                    dets.append({"box":[int(v) for v in box],"label":lbl or item,"query":item,"pass":2,"source":"flor"})
    except Exception as e:
        print(f"    [Flor Grnd '{item}'] {e}")
    return dets

def flor_run(image, item, proc, model, device):
    all_dets = []
    for it in expand_query(item):
        all_dets.extend(flor_detect_single(image, it, proc, model, device, qi=item))
    return _nms(all_dets)


# ============================================================
# FUSION
# ============================================================

def fuse(owl_dets, flor_dets, iou_thr=0.35):
    merged = []; flor_matched = [False]*len(flor_dets)
    for od in owl_dets:
        best_iou, best_fi = 0.0, -1
        for fi, fd in enumerate(flor_dets):
            iou = _iou(od["box"], fd["box"])
            if iou > best_iou: best_iou, best_fi = iou, fi
        if best_iou >= iou_thr and best_fi >= 0:
            c = dict(od); c["source"] = "both"
            fl = flor_dets[best_fi]["label"]
            if fl.lower() != od["label"].lower(): c["label"] = f"{od['label']} / {fl}"
            merged.append(c); flor_matched[best_fi] = True
        else:
            merged.append(od)
    for fi, fd in enumerate(flor_dets):
        if not flor_matched[fi]: merged.append(fd)
    priority = {"both":0,"owl":1,"flor":2}
    merged.sort(key=lambda d: (priority.get(d["source"],3), -d.get("score",0.0)))
    return _nms(merged, iou_thr=0.4)


# ============================================================
# DRAW BOXES
# ============================================================

def draw_boxes(image_path, detections, output_path):
    img = cv2.imread(image_path)
    if img is None: print("  [warn] Cannot read image."); return
    for idx, det in enumerate(detections):
        x1,y1,x2,y2 = det["box"]
        colour = BOX_COLOURS[idx % len(BOX_COLOURS)]
        cv2.rectangle(img, (x1,y1), (x2,y2), colour, BOX_THICKNESS)
        badge = SOURCE_BADGE.get(det["source"], det["source"].upper())
        sc    = f" {det['score']:.2f}" if det.get("score") else ""
        label = f"[{badge}]{sc} {det['label']}"
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

def describe_location(box, img_w, img_h):
    x1,y1,x2,y2=box; cx=(x1+x2)/2; cy=(y1+y2)/2
    col="left" if cx<img_w/3 else("centre" if cx<2*img_w/3 else "right")
    row="top"  if cy<img_h/3 else("middle"  if cy<2*img_h/3 else "bottom")
    w=x2-x1; h=y2-y1
    size="tiny" if max(w,h)<60 else("small" if max(w,h)<150 else "large")
    return f"{row}-{col}  [{size}, box: {x1},{y1}→{x2},{y2}]"


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Vision Chatbot Hybrid — OWLv2 + Florence-2")
    parser.add_argument("--image",     default="images/img01.jpg")
    parser.add_argument("--owl-model", default=OWL_MODEL_ID)
    parser.add_argument("--flo-model", default=FLOR_MODEL_ID)
    parser.add_argument("--threshold", type=float, default=OWL_MIN_SCORE)
    parser.add_argument("--owl-only",  action="store_true")
    parser.add_argument("--flo-only",  action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}"); sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device(); dtype = get_dtype(device)

    print(); print("=" * 65)
    print("  VISION CHATBOT — OWLv2 + Florence-2 Hybrid")
    print("=" * 65)
    print(f"  Image     : {args.image}")
    print(f"  Device    : {device}  ({dtype})")
    print(f"  Threshold : {args.threshold}")
    use_owl = not args.flo_only; use_flor = not args.owl_only
    print(f"  Mode      : {'OWLv2' if use_owl else ''}{'+'if use_owl and use_flor else ''}{'Florence-2' if use_flor else ''}")
    print("=" * 65)

    image = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size
    print(f"\n  Image loaded: {img_w}×{img_h}")

    owl_proc=owl_model=flo_proc=flo_model=None
    if use_owl:  owl_proc, owl_model = load_owlv2(device, dtype)
    if use_flor: flo_proc, flo_model = load_florence(device, dtype)

    print("\n" + "="*65)
    print("  Ready. Examples: where is the key? / find the laptop / list all objects")
    print("  Type 'quit' to exit.")
    print("="*65 + "\n")

    log_entries=[]; query_count=0

    while True:
        try:
            raw_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session ended."); break
        if not raw_input: continue
        if raw_input.lower() in {"quit","exit","q"}: break

        if raw_input.lower() in {"list all objects","list objects","list all"}:
            if use_flor:
                result = _flor_task(image, TASK_CAP, "", flo_proc, flo_model, device)
                caption = result.get(TASK_CAP, "No caption.")
                print("\n" + "-"*65); print(f"  Scene: {caption}"); print("-"*65 + "\n")
                log_entries.append({"query": raw_input, "caption": caption})
            continue

        query_count += 1
        item = parse_item(raw_input)
        print(f"\n  [Query {query_count}] '{item}'  expansions: {expand_query(item)}")

        owl_dets=[]; flor_dets=[]
        if use_owl:
            print("    Running OWLv2...")
            owl_dets = owl_run(image, item, owl_proc, owl_model, device, args.threshold)
        if use_flor:
            print("    Running Florence-2...")
            flor_dets = flor_run(image, item, flo_proc, flo_model, device)

        if use_owl and use_flor: detections = fuse(owl_dets, flor_dets)
        elif use_owl:            detections = owl_dets
        else:                    detections = flor_dets

        print("\n" + "-"*65)
        if detections:
            print(f"  Found {len(detections)} instance(s) of '{item}':\n")
            for i, det in enumerate(detections, 1):
                badge = SOURCE_BADGE.get(det["source"], det["source"])
                sc    = f"  score={det['score']:.3f}" if det.get("score") else ""
                print(f"  {i}. [{badge}]{sc}")
                print(f"     Location : {describe_location(det['box'], img_w, img_h)}")
                print(f"     Label    : {det['label']}")
            safe = re.sub(r"[^\w\s-]","",raw_input).strip().replace(" ","_")[:40]
            ts   = datetime.datetime.now().strftime("%H%M%S")
            out  = os.path.join(OUTPUT_DIR, f"hyb_q{query_count:02d}_{safe}_{ts}.jpg")
            draw_boxes(args.image, detections, out)
        else:
            print(f"  '{item}' not found. Try --threshold 0.10 or rephrase.")
        print("-"*65 + "\n")
        log_entries.append({"query": raw_input, "item": item, "detections": detections})

    if log_entries:
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = os.path.join(OUTPUT_DIR, f"hyb_session_{ts}.txt")
        with open(log, "w") as f:
            f.write(f"OWL: {args.owl_model}\nFlo: {args.flo_model}\nImage: {args.image}\n\n")
            for e in log_entries:
                f.write(f"QUERY: {e['query']}\n")
                for d in e.get("detections", []):
                    src=SOURCE_BADGE.get(d["source"],d["source"])
                    sc=f"  score={d['score']:.3f}" if d.get("score") else ""
                    f.write(f"  [{src}]{sc}  box={d['box']}  label={d['label']}\n")
                if "caption" in e: f.write(f"  caption: {e['caption']}\n")
                f.write("\n")
        print(f"  Log → {log}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
