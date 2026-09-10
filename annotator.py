"""
annotator.py — Stage 6: Draw bounding boxes on every frame and write output MP4.

Uses H.264 (avc1) codec — universally playable on macOS, Windows, Linux.
Bounding boxes are held between sampled frames so they appear on all
intermediate frames, not just the 1-per-second detection frames.
Writes output only when the query object was actually detected.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from models.types import Detection

# ── colour palette (BGR) ──────────────────────────────────────────────────── #
# Index 0 → queried object (green), Index 1 → person (orange), rest → others
_PALETTE: List[Tuple[int, int, int]] = [
    (0, 200, 80),    # green  — queried object
    (255, 100, 0),   # orange — person
    (0, 100, 255),   # blue
    (200, 0, 200),   # purple
    (0, 200, 200),   # teal
]

OUTPUT_DIR = "results_video"


def _colour(label: str, query: str) -> Tuple[int, int, int]:
    if label == query:
        return _PALETTE[0]
    if label == "person":
        return _PALETTE[1]
    return _PALETTE[(hash(label) % (len(_PALETTE) - 2)) + 2]


def _draw_box(
    frame: np.ndarray,
    box: List[int],
    label: str,
    score: float,
    colour: Tuple[int, int, int],
    verified: bool,
) -> None:
    """Draw box + label on frame in-place. Dashed outline if unverified."""
    x1, y1, x2, y2 = box
    thickness = 2

    if verified:
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    else:
        # Draw dashed rectangle for unverified (fallback — verifier disabled)
        for i in range(x1, x2, 10):
            cv2.line(frame, (i, y1), (min(i+5, x2), y1), colour, thickness)
            cv2.line(frame, (i, y2), (min(i+5, x2), y2), colour, thickness)
        for i in range(y1, y2, 10):
            cv2.line(frame, (x1, i), (x1, min(i+5, y2)), colour, thickness)
            cv2.line(frame, (x2, i), (x2, min(i+5, y2)), colour, thickness)

    # Label background + text
    text = f"{label} {score:.2f}{'*' if verified else ''}"
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), bl = cv2.getTextSize(text, font, fs, ft)
    pad = 3
    cv2.rectangle(
        frame,
        (x1, max(0, y1 - th - bl - pad * 2)),
        (x1 + tw + pad * 2, y1),
        colour, cv2.FILLED,
    )
    cv2.putText(
        frame, text,
        (x1 + pad, max(th, y1 - bl - pad)),
        font, fs, (255, 255, 255), ft, cv2.LINE_AA,
    )


def write_annotated_video(
    video_path: str,
    det_map: Dict[int, List[Detection]],
    query: str,
    width: int,
    height: int,
    native_fps: float,
) -> str:
    """
    Re-read the source video, overlay bounding boxes on every frame, write
    H.264 MP4 to results_video/. Raises RuntimeError if writing fails.
    Only called when the query object was detected (caller's responsibility).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_{query.replace(' ', '_')}_annotated.mp4")

    # Always remove stale file first — prevents broken videos surviving re-runs
    if os.path.exists(out_path):
        os.remove(out_path)

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"avc1"), native_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"VideoWriter failed to open with avc1 codec for '{out_path}'."
        )

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frames_written = 0
    current_dets: List[Detection] = []   # held between sampled frames

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx in det_map:
                current_dets = det_map[frame_idx]
            for d in current_dets:
                _draw_box(bgr, d.box, d.label, d.score, _colour(d.label, query), d.verified)

            # Timestamp overlay (top-left)
            ts = frame_idx / native_fps
            ts_str = f"{int(ts//60):02d}:{ts%60:05.2f}"
            cv2.putText(bgr, ts_str, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr, ts_str, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)

            writer.write(bgr)
            frames_written += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    file_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    if file_size < 1024:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(
            f"Output video is suspiciously small ({file_size} bytes, "
            f"{frames_written} frames written). Codec may have failed silently."
        )

    return out_path
