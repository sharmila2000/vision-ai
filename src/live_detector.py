#!/usr/bin/env python3
"""
src/live_detector.py — Live webcam object detection using OWLv2 and Florence-2.

This script opens your MacBook's FaceTime HD camera, runs the open-vocabulary OWLv2 detector
on a background thread (to prevent the camera display from lagging), and renders live,
fully annotated bounding boxes at a buttery-smooth 30 FPS on the main GUI thread.

Usage:
    source .venv/bin/activate
    python run.py --queries "laptop, cup, phone, person"

Optional flags:
    --verify           Enable Florence-2 verification (high precision, slightly slower box updates)
    --threshold 0.25   Set the OWLv2 confidence threshold (default 0.25)
    --camera 0         Index of the camera to use (default 0 for built-in FaceTime camera)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from typing import List, Optional, Tuple

import cv2
import torch
import numpy as np
from PIL import Image

# Ensure the root of the repo is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import OWLv2Detector
from src.verifier import Florence2Verifier
from src.models.types import Detection

# Color palette (BGR) for drawing
_PALETTE = [
    (0, 200, 80),    # green
    (255, 100, 0),   # orange
    (0, 100, 255),   # blue
    (200, 0, 200),   # purple
    (0, 200, 200),   # teal
]

def get_color(label: str, queries: List[str], auto_detect: bool = False) -> Tuple[int, int, int]:
    """Assigns a consistent color from the palette to a specific query class."""
    if auto_detect:
        # Dynamic color based on class name hashing so each auto-detected class gets a unique color
        return _PALETTE[hash(label) % len(_PALETTE)]
    try:
        idx = queries.index(label)
        return _PALETTE[idx % len(_PALETTE)]
    except ValueError:
        return (128, 128, 128)  # gray fallback


class LiveDetectionWorker:
    """
    Handles heavy model inference on a separate thread to maintain high preview FPS.
    """
    def __init__(self, queries: List[str], use_verifier: bool, threshold: float, auto_detect: bool = False, florence_model_id: str = "microsoft/Florence-2-base"):
        self.queries = queries
        self.use_verifier = use_verifier
        self.threshold = threshold
        self.auto_detect = auto_detect
        self.florence_model_id = florence_model_id
        
        # Load models depending on mode
        if self.auto_detect:
            print(f"[Worker] Initializing Florence-2 ({florence_model_id}) for Auto-Detection (Open World)...")
            self.verifier = Florence2Verifier(model_id=florence_model_id)
            self.detector = None
        else:
            self.detector = OWLv2Detector()
            # Override the default threshold in OWLv2Detector
            import src.detector as detector
            detector.SCORE_THRESHOLD = threshold
            self.verifier = Florence2Verifier(model_id=florence_model_id) if use_verifier else None
        
        # Threading state variables
        self.frame_to_process: Optional[Image.Image] = None
        self.latest_detections: List[Detection] = []
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def update_frame(self, pil_img: Image.Image):
        """Offers a new frame to process if the background thread is currently idle."""
        with self.lock:
            if self.frame_to_process is None:
                self.frame_to_process = pil_img

    def get_detections(self) -> List[Detection]:
        """Gets the latest verified detections list."""
        with self.lock:
            return list(self.latest_detections)

    def _worker_loop(self):
        print("[Worker] Detection thread started.")
        while self.running:
            pil_img = None
            with self.lock:
                if self.frame_to_process is not None:
                    # Snatch the image and reset the container to accept the next one
                    pil_img = self.frame_to_process
                    self.frame_to_process = None

            if pil_img is None:
                # Idle: Sleep briefly to avoid pegging the CPU
                time.sleep(0.01)
                continue

            try:
                if self.auto_detect:
                    # Run Florence-2 <OD> (Object Detection) to automatically detect everything in the frame
                    inputs = self.verifier.processor(text="<OD>", images=pil_img, return_tensors="pt")
                    inputs = {k: v.to(self.verifier.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = self.verifier.model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=256,  # Reduced from 1024 to massively lower latency!
                            do_sample=False,
                            use_cache=False,  # required for MPS stability (prevents NoneType.shape crash)
                        )
                    
                    # Do NOT skip special tokens here! The bounding box coordinates <loc_xxx> are special tokens.
                    generated_text = self.verifier.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed = self.verifier.processor.post_process_generation(
                        generated_text,
                        task="<OD>",
                        image_size=(pil_img.width, pil_img.height)
                    )
                    
                    od_data = parsed.get("<OD>", {})
                    if not od_data and "OD" in parsed:
                        od_data = parsed.get("OD", {})
                    elif not od_data and "<od>" in parsed:
                        od_data = parsed.get("<od>", {})
                        
                    bboxes = od_data.get("bboxes", [])
                    labels = od_data.get("labels", [])
                    
                    dets = []
                    for box, label in zip(bboxes, labels):
                        # Convert parsed to standard Detections [x1, y1, x2, y2]
                        dets.append(Detection(
                            score=1.0,  # absolute confidence for auto detection
                            box=[int(b) for b in box],
                            label=label,
                            verified=True
                        ))
                else:
                    # 1. Run OWLv2 Open-Vocabulary Detection
                    dets = self.detector.detect(pil_img, self.queries)

                    # 2. Run Optional Florence-2 Crop Verification
                    if self.verifier and dets:
                        # Verify each class detected
                        for q in self.queries:
                            class_dets = [d for d in dets if d.label == q]
                            if class_dets:
                                self.verifier.verify(pil_img, class_dets)
                        # Filter verified detections, or let verifier state stand
                        # If verification is enabled, we mark verified ones
                        pass
                    else:
                        # Treat all detections as verified if verifier is turned off
                        for d in dets:
                            d.verified = True

                # 3. Store result
                with self.lock:
                    self.latest_detections = dets

            except Exception as e:
                import traceback
                print(f"[Worker Error]: {e}")
                traceback.print_exc()
                time.sleep(0.5)


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    color: Tuple[int, int, int],
    queries: List[str]
) -> None:
    """Draws custom bounded box and label on the BGR frame in-place."""
    x1, y1, x2, y2 = detection.box
    thickness = 2
    verified = detection.verified

    # Draw bounding box (solid if verified, dashed/thin if unverified)
    if verified:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    else:
        # Dashed box fallback
        for i in range(x1, x2, 10):
            cv2.line(frame, (i, y1), (min(i+5, x2), y1), color, 1)
            cv2.line(frame, (i, y2), (min(i+5, x2), y2), color, 1)
        for i in range(y1, y2, 10):
            cv2.line(frame, (x1, i), (x1, min(i+5, y2)), color, 1)
            cv2.line(frame, (x2, i), (x2, min(i+5, y2)), color, 1)

    # Label text construction (OpenCV putText only supports ASCII; Unicode checkmarks render as '???')
    v_symbol = " *" if verified else ""
    text = f"{detection.label}{v_symbol} {detection.score:.2f}"
    
    font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Text background bubble
    pad = 4
    bubble_y1 = max(0, y1 - text_height - baseline - pad * 2)
    cv2.rectangle(
        frame,
        (x1, bubble_y1),
        (x1 + text_width + pad * 2, y1),
        color,
        cv2.FILLED,
    )
    # White text overlay
    cv2.putText(
        frame,
        text,
        (x1 + pad, max(text_height, y1 - baseline)),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time MacBook Camera Object Detection using OWLv2 & Florence-2.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--queries", "-q",
        default="laptop, cup, phone, person",
        help="Comma-separated list of target classes to detect in real-time (ignored if --auto is used)."
    )
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Enable Auto-Detection mode using Florence-2 (detect all available objects automatically)."
    )
    parser.add_argument(
        "--florence-model", "-f",
        default="microsoft/Florence-2-base",
        choices=["microsoft/Florence-2-base", "microsoft/Florence-2-large"],
        help="Florence-2 model size to use for auto-detect or verification (default: Base for high-speed live feed)."
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Enable Florence-2 second-opinion verification (higher precision)."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.25,
        help="OWLv2 confidence score threshold [0.0 - 1.0] (default: 0.25)."
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="OpenCV index of the MacBook camera (default: 0)."
    )

    args = parser.parse_args()

    # Load environmental API keys for spacebar analysis
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    auto_detect = args.auto
    queries = []
    if not auto_detect:
        # Parse and trim query list
        queries = [item.strip() for item in args.queries.split(",") if item.strip()]
        if not queries:
            print("[Error] No query terms provided.")
            sys.exit(1)

    print("=" * 70)
    print(" Live Object Detection Setup")
    if auto_detect:
        print(" Mode:               AUTO-DETECT (Open World - detects everything)")
        print(f" Model:              {args.florence_model} (<OD> task)")
    else:
        print(f" Target Queries:     {queries}")
        print(" Model:              google/owlv2-large-patch14-ensemble")
        print(f" Florence-2 Verifier: {'ENABLED (High Precision)' if args.verify else 'DISABLED (High Speed)'}")
        print(f" Florence-2 Model:   {args.florence_model}")
        print(f" OWLv2 Threshold:    {args.threshold}")
    print("=" * 70)

    # 1. Open webcam
    print(f"[Camera] Initializing camera index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[Error] Cannot open MacBook camera (index {args.camera}).")
        print("Please check that your terminal has 'Camera' access in System Settings -> Privacy & Security.")
        sys.exit(1)

    # Attempt to request 1280x720 resolution for high visual clarity
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] Active resolution: {width}x{height}")

    # 2. Spawn and start our heavy lifting worker thread
    worker = LiveDetectionWorker(
        queries=queries, 
        use_verifier=args.verify, 
        threshold=args.threshold,
        auto_detect=auto_detect,
        florence_model_id=args.florence_model
    )
    worker.start()

    print("\n[Preview] Live preview window opening. Press 'q' or 'ESC' in the window to quit.")
    
    fps_start_time = time.time()
    fps_frames = 0
    fps_display = 0.0

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                print("[Camera] Failed to capture a frame from the webcam.")
                break

            # macOS camera inputs can sometimes be mirrored.
            # Flipping makes it natural to look at (like a mirror).
            bgr_frame = cv2.flip(bgr_frame, 1)
            
            # Keep a completely clean frame COPY (no bounding box lines) for Gemini analysis
            clean_frame = bgr_frame.copy()

            # Convert to PIL Image for the detector
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Submit to worker (this is non-blocking — if worker is busy, it just skips processing this frame)
            worker.update_frame(pil_image)

            # Get latest detections (from whichever frame completed processing last)
            current_dets = worker.get_detections()

            # Render detections on the active frame
            for d in current_dets:
                color = get_color(d.label, queries, auto_detect=auto_detect)
                draw_detection(bgr_frame, d, color, queries)

            # Calculate actual preview FPS
            fps_frames += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps_display = fps_frames / elapsed
                fps_frames = 0
                fps_start_time = time.time()

            # Overlay Status & FPS in the top-left corner
            if auto_detect:
                status_text = f"FPS: {fps_display:.1f} | Mode: AUTO-DETECT (all objects)"
            else:
                status_text = f"FPS: {fps_display:.1f} | Active Queries: {', '.join(queries)}"
            cv2.putText(
                bgr_frame,
                status_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr_frame,
                "Press 'q' to Quit",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            
            # Show Spacebar instructions for full-screen analysis
            space_instr = "Press SPACEBAR for Gemini full-screen analysis"
            cv2.putText(
                bgr_frame,
                space_instr,
                (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )

            # Display frame in OpenCV window
            cv2.imshow("MacBook Camera Live Object Detector (OWLv2)", bgr_frame)

            # Read keyboard inputs (wait 10ms to keep GUI running)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACEBAR
                print("\n" + "=" * 70)
                print(" 📸 Spacebar Pressed: Capturing clean snapshot for Gemini Analysis...")
                print("=" * 70)
                
                if not gemini_api_key:
                    print("\n[Notice] Gemini API Key is not set.")
                    print("To enable full screen analysis, set the GEMINI_API_KEY environment variable")
                    print("or place it in your '.env' file: GEMINI_API_KEY=\"your_key\"\n")
                else:
                    temp_path = "temp_live_snapshot.jpg"
                    try:
                        # Save the clean frame
                        cv2.imwrite(temp_path, clean_frame)
                        
                        # Import and initialize Gemini
                        from google import genai
                        client = genai.Client(api_key=gemini_api_key)
                        pil_snap = Image.open(temp_path)
                        
                        print("Sending full-screen snapshot to Gemini 1.5 Flash...")
                        response = client.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=[
                                pil_snap,
                                "Analyze this webcam snapshot of my room/environment in detail. "
                                "Describe every single visible object across the entire screen, their placement, "
                                "the person, their actions/pose, and summarize the overall context in a highly "
                                "conversational and friendly manner."
                            ]
                        )
                        print("\n--- 📝 Gemini Full Screen Analysis ---")
                        print(response.text)
                        print("=" * 70 + "\n")
                    except Exception as ex:
                        print(f"[Gemini Error]: {ex}\n")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Teardown camera and worker thread cleanly
        print("\nCleaning up resources...")
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Live detector closed cleanly. Goodbye!\n")


if __name__ == "__main__":
    main()
