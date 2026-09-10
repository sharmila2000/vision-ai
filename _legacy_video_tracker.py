"""
video_tracker.py — Phase 2 core pipeline
=========================================
Stage 1: Extract frames from video at 1 fps (configurable).
Stage 2: Run OWLv2-large on each frame — same model/settings as Phase 1.
Stage 3: ByteTrack-style multi-object tracker (pure NumPy/SciPy — no extra install).
Stage 4: Event detector — disappearance + person proximity → timestamped event log.
Stage 5: Annotated output video — bounding boxes drawn on every frame for the query object.

Usage (module):
    from video_tracker import VideoTracker
    tracker = VideoTracker()
    events = tracker.run("videos/video_01.mp4", query="key", sample_fps=1.0)
    # → also writes results_video/video_01_key_annotated.mp4
"""

import os
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection


# ─────────────────────────────── constants ────────────────────────────────── #

MODEL_ID         = "google/owlv2-large-patch14-ensemble"
SCORE_THRESHOLD  = 0.25         # balanced: 0.20 too noisy for video, 0.30 misses real objects
NMS_IOU_THRESH   = 0.50
PROXIMITY_PX     = 150          # person bbox centre within this distance → "close"
MAX_MISS_FRAMES  = 3            # drop track sooner to avoid carrying stale boxes
IOU_MATCH_THRESH = 0.25         # min IoU to link a detection to an existing track
MIN_TRACK_FRAMES = 2            # detection must appear in ≥ N sampled frames to be "real"

# Query expansion: OWLv2 is sensitive to exact phrasing.
# For ambiguous terms, probe multiple variants and keep the highest-scoring box.
# All variants map back to the user's original term in output.
_QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "calendar":    ["calendar", "white board", "display board", "notice board", "whiteboard"],
    "whiteboard":  ["white board", "whiteboard", "display board", "notice board"],
    "board":       ["white board", "display board", "notice board", "board"],
    "tv":          ["television", "tv screen", "flat screen tv", "monitor"],
    "television":  ["television", "tv screen", "flat screen tv"],
    "phone":       ["mobile phone", "smartphone", "cell phone", "phone"],
    "laptop":      ["laptop computer", "laptop", "notebook computer"],
    "sofa":        ["sofa", "couch", "settee"],
    "chair":       ["chair", "armchair", "seat"],
}


# ──────────────────────────────── data types ──────────────────────────────── #

@dataclass
class Detection:
    score: float
    box: List[int]   # [x1, y1, x2, y2] in pixels
    label: str

@dataclass
class Track:
    track_id: int
    label: str
    box: List[int]
    last_seen_frame: int
    first_seen_frame: int
    missed_frames: int = 0
    history: List[Tuple[int, List[int]]] = field(default_factory=list)  # [(frame_idx, box), ...]

@dataclass
class Event:
    frame_idx: int
    timestamp_sec: float
    event_type: str          # "appeared" | "disappeared" | "pickup_candidate"
    label: str
    track_id: int
    box: Optional[List[int]]
    person_proximity_px: Optional[float] = None
    note: str = ""


# ──────────────────────────────── NMS helpers ─────────────────────────────── #

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
    """True if most of inner is inside outer (handles sub-region duplicates)."""
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_inner = max(1, (ax2 - ax1) * (ay2 - ay1))
    return (inter / area_inner) >= thresh


def nms(detections: List[Detection]) -> List[Detection]:
    """IoU + containment NMS — identical logic to Phase 1."""
    dets = sorted(detections, key=lambda d: d.score, reverse=True)
    keep = []
    for i, d in enumerate(dets):
        suppress = False
        for k in keep:
            if _iou(d.box, k.box) > NMS_IOU_THRESH or _is_contained(d.box, k.box):
                suppress = True
                break
        if not suppress:
            keep.append(d)
    return keep


# ──────────────────────────────── tracker ─────────────────────────────────── #

class SimpleTracker:
    """
    Lightweight detection-agnostic tracker (Hungarian matching by IoU).
    Inspired by ByteTrack / SORT but self-contained — no extra dependencies.
    """

    def __init__(self):
        self._next_id = 1
        self._active: Dict[int, Track] = {}

    def update(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        """
        Match detections to existing tracks by IoU, create new tracks for unmatched
        detections, increment missed_frames for unmatched tracks.
        Returns the list of currently active tracks after this frame.
        """
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Greedy IoU matching (high-score detections first)
        for det_idx in list(unmatched_dets):
            det = detections[det_idx]
            best_iou = IOU_MATCH_THRESH
            best_tid = None
            for tid, track in self._active.items():
                if tid in matched_track_ids:
                    continue
                if track.label != det.label:
                    continue
                iou_val = _iou(det.box, track.box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_tid = tid
            if best_tid is not None:
                track = self._active[best_tid]
                track.box = det.box
                track.last_seen_frame = frame_idx
                track.missed_frames = 0
                track.history.append((frame_idx, det.box))
                matched_track_ids.add(best_tid)
                unmatched_dets.remove(det_idx)

        # Increment missed counter for unmatched tracks
        for tid in list(self._active.keys()):
            if tid not in matched_track_ids:
                self._active[tid].missed_frames += 1

        # Remove stale tracks
        stale = [tid for tid, t in self._active.items() if t.missed_frames > MAX_MISS_FRAMES]
        for tid in stale:
            del self._active[tid]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            t = Track(
                track_id=self._next_id,
                label=det.label,
                box=det.box,
                last_seen_frame=frame_idx,
                first_seen_frame=frame_idx,
                history=[(frame_idx, det.box)],
            )
            self._active[self._next_id] = t
            self._next_id += 1

        return list(self._active.values())

    def all_tracks(self) -> Dict[int, Track]:
        return dict(self._active)


# ──────────────────────────────── detector ────────────────────────────────── #

class OWLv2Detector:
    """Phase 1 OWLv2 detection logic, adapted for single-frame batch use."""

    def __init__(self):
        print(f"[OWLv2] Loading {MODEL_ID} …")
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.processor = Owlv2Processor.from_pretrained(MODEL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        print(f"[OWLv2] Loaded on {device_str}")

    def detect(self, pil_image: Image.Image, queries: List[str]) -> List[Detection]:
        """
        Run OWLv2 on a single PIL image with given text queries.
        Automatically expands queries using _QUERY_EXPANSIONS and collapses
        variant labels back to the original user term. Returns detections
        above SCORE_THRESHOLD after NMS.
        """
        # Build expanded prompt list and a reverse map: variant → original term
        expanded: List[str] = []
        variant_to_original: Dict[str, str] = {}
        for q in queries:
            variants = _QUERY_EXPANSIONS.get(q.lower(), [q])
            for v in variants:
                if v not in expanded:
                    expanded.append(v)
                variant_to_original[v] = q  # collapse back to user's term

        prompts = [f"a photo of a {v}" for v in expanded]
        inputs = self.processor(
            text=[prompts],
            images=pil_image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to absolute pixel boxes
        target_sizes = torch.tensor([pil_image.size[::-1]])  # (H, W)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=SCORE_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        detections = []
        for score, label_idx, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            variant = expanded[int(label_idx)]
            original = variant_to_original[variant]
            detections.append(Detection(
                score=float(score),
                box=[int(v) for v in box.tolist()],
                label=original,           # always the user's term, never the variant
            ))

        return nms(detections)


# ──────────────────────────── event detector ──────────────────────────────── #

def _box_centre(box: List[int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def detect_events(
    frame_history: List[Tuple[int, List[Detection]]],
    target_label: str,
    video_fps: float,
) -> List[Event]:
    """
    Walk through per-frame detection history and emit events for the target object.

    Stability filter: a track must be seen in >= MIN_TRACK_FRAMES consecutive
    sampled frames before it is emitted as a real "appeared" event. This removes
    single-frame false positives caused by camera motion or visually similar objects.
    """
    tracker = SimpleTracker()
    events: List[Event] = []

    # confirmed_ids: tracks that have passed the stability filter
    confirmed_ids: set = set()
    # candidate_ids: seen once, waiting for confirmation
    candidate_first_seen: Dict[int, Tuple[int, float, List[int]]] = {}  # tid → (frame, ts, box)
    active_target_ids: set = set()

    for frame_idx, dets in frame_history:
        ts = frame_idx / video_fps
        active_tracks = tracker.update(dets, frame_idx)

        current_target_ids = {t.track_id for t in active_tracks if t.label == target_label}

        for t in active_tracks:
            if t.label != target_label:
                continue
            tid = t.track_id

            if tid in confirmed_ids:
                continue  # already confirmed, nothing to do on appear

            if tid not in candidate_first_seen:
                # First time we see this track — record as candidate
                candidate_first_seen[tid] = (frame_idx, ts, t.box)
            else:
                # Seen again — number of sampled frames this track has appeared in
                frames_seen = len(t.history)
                if frames_seen >= MIN_TRACK_FRAMES:
                    # Confirmed: emit appeared using the *first* observed frame/box
                    first_fi, first_ts, first_box = candidate_first_seen[tid]
                    events.append(Event(
                        frame_idx=first_fi,
                        timestamp_sec=first_ts,
                        event_type="appeared",
                        label=target_label,
                        track_id=tid,
                        box=first_box,
                        note=f"Confirmed after {frames_seen} frames",
                    ))
                    confirmed_ids.add(tid)

        # Disappearances: only for confirmed tracks that just left
        for gone_id in (active_target_ids - current_target_ids):
            if gone_id in confirmed_ids:
                events.append(Event(
                    frame_idx=frame_idx,
                    timestamp_sec=ts,
                    event_type="disappeared",
                    label=target_label,
                    track_id=gone_id,
                    box=None,
                    note=f"Track {gone_id} lost at frame {frame_idx} (~{ts:.1f}s)",
                ))

        active_target_ids = current_target_ids

    # Second pass: annotate disappearance events with person proximity
    # Build a frame → person boxes map from the raw detection history
    frame_person_map: Dict[int, List[List[int]]] = {}
    for frame_idx, dets in frame_history:
        frame_person_map[frame_idx] = [d.box for d in dets if d.label == "person"]

    # Build a frame → target box map
    frame_target_map: Dict[int, List[int]] = {}
    for frame_idx, dets in frame_history:
        for d in dets:
            if d.label == target_label:
                frame_target_map[frame_idx] = d.box

    for ev in events:
        if ev.event_type == "disappeared":
            # Look back up to 3 frames for the last known target position
            last_box = None
            for look_back in range(1, 4):
                fi = ev.frame_idx - look_back
                if fi in frame_target_map:
                    last_box = frame_target_map[fi]
                    break

            if last_box is not None:
                person_boxes = frame_person_map.get(ev.frame_idx, [])
                if not person_boxes and ev.frame_idx > 0:
                    person_boxes = frame_person_map.get(ev.frame_idx - 1, [])
                if person_boxes:
                    target_c = _box_centre(last_box)
                    min_d = min(_dist(target_c, _box_centre(pb)) for pb in person_boxes)
                    ev.person_proximity_px = min_d
                    if min_d < PROXIMITY_PX:
                        ev.event_type = "pickup_candidate"
                        ev.note += f" — person {min_d:.0f}px away (< {PROXIMITY_PX}px threshold)"

    return events



# ──────────────────────── annotated video writer ──────────────────────────── #

# Colour palette per label (BGR for OpenCV)
_LABEL_COLOURS: Dict[str, Tuple[int, int, int]] = {}
_PALETTE = [
    (0, 200, 80),    # green  — primary query object
    (255, 100, 0),   # orange — person
    (0, 100, 255),   # blue
    (200, 0, 200),   # purple
    (0, 200, 200),   # teal
]


def _label_colour(label: str, query: str) -> Tuple[int, int, int]:
    """Return a consistent BGR colour for a given label."""
    if label == query:
        return _PALETTE[0]          # always green for the queried object
    if label == "person":
        return _PALETTE[1]          # always orange for persons
    idx = (hash(label) % (len(_PALETTE) - 2)) + 2
    return _PALETTE[idx]


def _draw_box(
    frame: "np.ndarray",
    box: List[int],
    label: str,
    score: float,
    colour: Tuple[int, int, int],
) -> None:
    """Draw a bounding box + label on a BGR numpy frame (in-place)."""
    x1, y1, x2, y2 = box
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    text = f"{label} {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

    # Background pill behind text so it's readable on any frame colour
    pad = 3
    cv2.rectangle(
        frame,
        (x1, max(0, y1 - th - baseline - pad * 2)),
        (x1 + tw + pad * 2, y1),
        colour,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (x1 + pad, max(th, y1 - baseline - pad)),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def _write_annotated_video(
    video_path: str,
    det_map: Dict[int, List["Detection"]],
    query: str,
    width: int,
    height: int,
    native_fps: float,
    frame_step: int,
) -> str:
    """
    Re-read the video, draw bounding boxes on every frame (carrying the last
    known detection forward between sampled frames), and write an output MP4.

    The last detection is held until the next sampled frame so boxes appear
    on all intermediate frames, not just the 1-per-second sampled ones.
    Always removes any pre-existing file first so a stale/broken video is
    never left behind if writing fails.
    """
    os.makedirs("results_video", exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join("results_video", f"{base}_{query.replace(' ', '_')}_annotated.mp4")

    # Remove any pre-existing file — prevents stale broken videos from surviving
    if os.path.exists(out_path):
        os.remove(out_path)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, native_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"VideoWriter failed to open '{out_path}'. "
            "avc1 (H.264) codec may not be available on this system."
        )

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frames_written = 0
    current_dets: List["Detection"] = []   # held between sampled frames

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            # Update held detections at every sampled frame
            if frame_idx in det_map:
                current_dets = det_map[frame_idx]

            # Draw all current detections
            for d in current_dets:
                colour = _label_colour(d.label, query)
                _draw_box(bgr_frame, d.box, d.label, d.score, colour)

            # Overlay timestamp (top-left corner)
            ts = frame_idx / native_fps
            mm = int(ts // 60)
            ss = ts % 60
            ts_str = f"{mm:02d}:{ss:05.2f}"
            cv2.putText(
                bgr_frame, ts_str, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                bgr_frame, ts_str, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
            )

            writer.write(bgr_frame)
            frames_written += 1
            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    # Verify the output is a valid, non-empty file
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(
            f"Output video '{out_path}' is missing or suspiciously small "
            f"({os.path.getsize(out_path) if os.path.exists(out_path) else 0} bytes). "
            f"Wrote {frames_written} frames — codec may have silently failed."
        )

    return out_path



# ──────────────────────────── main pipeline ───────────────────────────────── #

class VideoTracker:
    """
    Full Phase 2 pipeline: video → frames → OWLv2 detections → tracking → events.
    """

    def __init__(self):
        self.detector = OWLv2Detector()

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
        Process a video file and return a list of timestamped events for the queried object.
        Also writes an annotated output video with bounding boxes drawn on every frame.

        Args:
            video_path:         Path to the .mp4 file.
            query:              Object to search for (e.g. "key").
            sample_fps:         Frames per second to sample for detection (default 1).
            also_detect_person: Also detect "person" for proximity/pickup analysis.
            verbose:            Print per-frame progress including bbox coordinates.
            save_video:         Write annotated output video to results_video/ folder.

        Returns:
            List of Event objects (appeared / disappeared / pickup_candidate).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / native_fps

        print(f"\n[Video] {os.path.basename(video_path)}")
        print(f"        {width}x{height} | {native_fps:.1f} fps | {total_frames} frames | {duration:.1f}s")
        print(f"[Track] Query='{query}' | Sampling at {sample_fps} fps\n")

        queries = [query]
        if also_detect_person:
            queries.append("person")

        frame_step = max(1, round(native_fps / sample_fps))
        frame_history: List[Tuple[int, List[Detection]]] = []

        # Build a frame_idx → detections map for the annotation pass
        det_map: Dict[int, List[Detection]] = {}

        frame_idx = 0
        sampled = 0
        start_time = time.time()

        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                ts = frame_idx / native_fps
                rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                dets = self.detector.detect(pil_img, queries)
                frame_history.append((frame_idx, dets))
                det_map[frame_idx] = dets
                sampled += 1

                if verbose:
                    target_dets = [d for d in dets if d.label == query]
                    if target_dets:
                        for d in target_dets:
                            print(
                                f"  frame {frame_idx:5d} ({ts:6.1f}s) → "
                                f"'{query}' score={d.score:.3f} "
                                f"box=[{d.box[0]},{d.box[1]},{d.box[2]},{d.box[3]}]"
                            )
                    else:
                        print(f"  frame {frame_idx:5d} ({ts:6.1f}s) → no '{query}' detected")

            frame_idx += 1

        cap.release()
        elapsed = time.time() - start_time
        print(f"\n[Done] Detection pass: {sampled} frames in {elapsed:.1f}s")

        # ── Annotated output video ────────────────────────────────────────────
        # Only write if the query object was actually detected in at least one frame
        query_detected = any(
            d.label == query for dets in det_map.values() for d in dets
        )
        if save_video and query_detected:
            out_path = _write_annotated_video(
                video_path, det_map, query, width, height, native_fps, frame_step
            )
            print(f"[bbox] Annotated video → {out_path}")

        # Run event detection
        events = detect_events(frame_history, query, native_fps)
        return events

    def summarise(self, events: List[Event], query: str) -> str:
        """Convert the event list to a plain-English summary."""
        if not events:
            return f"No '{query}' was detected in the video."

        lines = [f"=== Detection Summary for '{query}' ===\n"]
        for ev in events:
            mm = int(ev.timestamp_sec // 60)
            ss = ev.timestamp_sec % 60
            ts_str = f"{mm:02d}:{ss:05.2f}"

            if ev.event_type == "appeared":
                lines.append(f"  [{ts_str}] '{query}' appeared (track #{ev.track_id})")
                if ev.box:
                    lines.append(f"            location: box {ev.box}")

            elif ev.event_type == "disappeared":
                lines.append(f"  [{ts_str}] '{query}' disappeared from view (track #{ev.track_id})")
                lines.append(f"            {ev.note}")

            elif ev.event_type == "pickup_candidate":
                lines.append(f"  [{ts_str}] *** '{query}' likely picked up! (track #{ev.track_id})")
                if ev.person_proximity_px is not None:
                    lines.append(
                        f"            A person was {ev.person_proximity_px:.0f}px away — "
                        "within pickup proximity."
                    )
                lines.append(f"            {ev.note}")

        lines.append("")
        if any(e.event_type == "pickup_candidate" for e in events):
            pickup = next(e for e in events if e.event_type == "pickup_candidate")
            mm = int(pickup.timestamp_sec // 60)
            ss = pickup.timestamp_sec % 60
            lines.append(
                f"CONCLUSION: At {mm:02d}:{ss:05.2f}, the '{query}' was most likely "
                "picked up by a person standing nearby."
            )
        elif any(e.event_type == "disappeared" for e in events):
            dis = next(e for e in events if e.event_type == "disappeared")
            mm = int(dis.timestamp_sec // 60)
            ss = dis.timestamp_sec % 60
            lines.append(
                f"CONCLUSION: The '{query}' disappeared from view at {mm:02d}:{ss:05.2f}. "
                "No person was detected nearby at that moment."
            )
        else:
            lines.append(f"CONCLUSION: The '{query}' remained visible throughout the video.")

        return "\n".join(lines)
