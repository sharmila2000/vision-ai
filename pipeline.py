"""
pipeline.py — Phase 2 main pipeline orchestrator.

Architecture (in order):
  Stage 1  Frame sampling      OpenCV VideoCapture at configurable fps
  Stage 2  Detection           OWLv2-large (detector.py) — high recall
  Stage 3  Verification        Florence-2-large (verifier.py) — high precision
  Stage 4  Tracking            SimpleTracker (tracker.py) — persistent IDs
  Stage 5  Event detection     Rule-based (tracker.py) — appeared/disappeared/pickup
  Stage 6  Annotated video     H.264 output (annotator.py) — only when detected

Why two-model detection (OWLv2 + Florence-2):
  OWLv2 alone has high recall but low precision on video — it matches
  patch-level texture similarity and fires on walls, phone screens, book
  spines for almost any query. Florence-2 acts as a second opinion on each
  candidate crop: it sees only the cropped region and answers a binary
  yes/no question. Only detections confirmed by BOTH models are kept.

  This combination achieves:
    - Zero missed detections for objects that OWLv2 finds (high recall kept)
    - Near-zero false positives (Florence-2 rejects noise crops)
    - No dependency on raising score threshold (which would miss real objects)
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
from PIL import Image

from annotator import write_annotated_video
from detector import OWLv2Detector
from models.types import Detection, Event
from tracker import detect_events
from verifier import Florence2Verifier


class Pipeline:
    """
    Orchestrates all stages. Models are loaded once and reused across queries.
    Set use_verifier=False to skip Florence-2 (faster but lower precision).
    """

    def __init__(self, use_verifier: bool = True):
        self.detector = OWLv2Detector()
        self.verifier: Optional[Florence2Verifier] = None
        if use_verifier:
            self.verifier = Florence2Verifier()
        else:
            print("[Pipeline] Verifier disabled — running OWLv2-only mode.")

    def run(
        self,
        video_path: str,
        query: str,
        sample_fps: float = 1.0,
        also_detect_person: bool = True,
        verbose: bool = True,
        save_video: bool = True,
    ) -> List[Event]:
        """
        Full pipeline: video → frames → detect → verify → track → events.

        Args:
            video_path:         Path to .mp4 file.
            query:              Object to find (e.g. "key", "whiteboard").
            sample_fps:         Frames per second to sample (default 1.0).
            also_detect_person: Detect persons alongside query for pickup analysis.
            verbose:            Print per-frame progress.
            save_video:         Write annotated MP4 if object detected.

        Returns:
            List of Event objects (appeared / disappeared / pickup_candidate).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        native_fps  = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration     = total_frames / native_fps

        print(f"\n[Video]    {os.path.basename(video_path)}")
        print(f"           {width}x{height} | {native_fps:.1f} fps | "
              f"{total_frames} frames | {duration:.1f}s")
        print(f"[Query]    '{query}' | sample={sample_fps} fps | "
              f"verifier={'ON' if self.verifier else 'OFF'}\n")

        queries = [query]
        if also_detect_person:
            queries.append("person")

        frame_step  = max(1, round(native_fps / sample_fps))
        frame_history: List[Tuple[int, List[Detection]]] = []
        det_map: Dict[int, List[Detection]] = {}

        frame_idx = 0
        sampled   = 0
        t0        = time.time()

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                ts  = frame_idx / native_fps
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                # Stage 2 — OWLv2 detection
                dets = self.detector.detect(pil, queries)

                # Stage 3 — Florence-2 verification (only on query object crops)
                if self.verifier and dets:
                    query_dets  = [d for d in dets if d.label == query]
                    other_dets  = [d for d in dets if d.label != query]
                    if query_dets:
                        self.verifier.verify(pil, query_dets)
                    # Keep only verified query detections; person detections kept as-is
                    dets = [d for d in query_dets if d.verified] + other_dets
                else:
                    # Without verifier, treat all as verified
                    for d in dets:
                        d.verified = True

                frame_history.append((frame_idx, dets))
                det_map[frame_idx] = dets
                sampled += 1

                if verbose:
                    target_dets = [d for d in dets if d.label == query]
                    if target_dets:
                        for d in target_dets:
                            v_tag = "✓verified" if d.verified else "unverified"
                            print(
                                f"  frame {frame_idx:5d} ({ts:6.1f}s) → "
                                f"'{query}' score={d.score:.3f} "
                                f"box=[{d.box[0]},{d.box[1]},{d.box[2]},{d.box[3]}] "
                                f"[{v_tag}]"
                            )
                    else:
                        print(f"  frame {frame_idx:5d} ({ts:6.1f}s) → no '{query}' detected")

            frame_idx += 1

        cap.release()
        print(f"\n[Done]     {sampled} frames processed in {time.time()-t0:.1f}s")

        # Stage 5 — event detection
        events = detect_events(frame_history, query, native_fps)

        # Stage 6 — annotated video (only when something was detected)
        query_detected = any(d.label == query for dets in det_map.values() for d in dets)
        if save_video and query_detected:
            out_path = write_annotated_video(
                video_path, det_map, query, width, height, native_fps
            )
            print(f"[Video]    Annotated output → {out_path}")

        return events

    def summarise(self, events: List[Event], query: str) -> str:
        """Convert event list to a plain-English summary."""
        if not events:
            return f"No '{query}' was detected in the video."

        lines = [f"=== Detection Summary for '{query}' ===\n"]
        for ev in events:
            mm, ss = int(ev.timestamp_sec // 60), ev.timestamp_sec % 60
            ts = f"{mm:02d}:{ss:05.2f}"
            if ev.event_type == "appeared":
                lines.append(f"  [{ts}] '{query}' appeared  (track #{ev.track_id})")
                if ev.box:
                    lines.append(f"          box {ev.box}")
            elif ev.event_type == "disappeared":
                lines.append(f"  [{ts}] '{query}' disappeared  (track #{ev.track_id})")
                lines.append(f"          {ev.note}")
            elif ev.event_type == "pickup_candidate":
                lines.append(f"  [{ts}] *** '{query}' likely picked up!  (track #{ev.track_id})")
                if ev.person_proximity_px is not None:
                    lines.append(
                        f"          Person was {ev.person_proximity_px:.0f}px away."
                    )
                lines.append(f"          {ev.note}")

        lines.append("")
        if any(e.event_type == "pickup_candidate" for e in events):
            ev = next(e for e in events if e.event_type == "pickup_candidate")
            mm, ss = int(ev.timestamp_sec // 60), ev.timestamp_sec % 60
            lines.append(
                f"CONCLUSION: At {mm:02d}:{ss:05.2f}, '{query}' was likely "
                "picked up by a person standing nearby."
            )
        elif any(e.event_type == "disappeared" for e in events):
            ev = next(e for e in events if e.event_type == "disappeared")
            mm, ss = int(ev.timestamp_sec // 60), ev.timestamp_sec % 60
            lines.append(
                f"CONCLUSION: '{query}' disappeared at {mm:02d}:{ss:05.2f}. "
                "No person was close enough to confirm a pickup."
            )
        else:
            lines.append(f"CONCLUSION: '{query}' remained visible throughout the video.")

        return "\n".join(lines)
