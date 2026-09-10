"""
tracker.py — Stage 4: Lightweight IoU-based multi-object tracker.

Inspired by SORT/ByteTrack but self-contained (no extra dependencies).
Assigns persistent track IDs across frames using greedy IoU matching.

Stability filter (MIN_TRACK_FRAMES): a track must survive in at least N
consecutive sampled frames before it is promoted to a confirmed detection.
This eliminates single-frame false positives from camera motion.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from models.types import Detection, Event, Track

# ── constants ──────────────────────────────────────────────────────────────── #

IOU_MATCH_THRESH = 0.25   # min IoU to link a detection to an existing track
MAX_MISS_FRAMES  = 3      # drop track after N consecutive missed frames
MIN_TRACK_FRAMES = 2      # track must appear in ≥ N frames to be confirmed
PROXIMITY_PX     = 150    # person centre within this px → pickup candidate


# ── geometry helpers ───────────────────────────────────────────────────────── #

def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6)


def _centre(box: List[int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# ── tracker ────────────────────────────────────────────────────────────────── #

class SimpleTracker:
    """Greedy IoU tracker with stability filter and event emission."""

    def __init__(self):
        self._next_id = 1
        self._active: Dict[int, Track] = {}

    def update(
        self, detections: List[Detection], frame_idx: int
    ) -> List[Track]:
        """
        Match detections → existing tracks by IoU. Create new tracks for
        unmatched detections. Increment missed_frames for unmatched tracks
        and drop those exceeding MAX_MISS_FRAMES.
        """
        unmatched = list(range(len(detections)))
        matched_tids: Set[int] = set()

        for det_idx in list(unmatched):
            det = detections[det_idx]
            best_iou, best_tid = IOU_MATCH_THRESH, None
            for tid, track in self._active.items():
                if tid in matched_tids or track.label != det.label:
                    continue
                v = _iou(det.box, track.box)
                if v > best_iou:
                    best_iou, best_tid = v, tid
            if best_tid is not None:
                t = self._active[best_tid]
                t.box = det.box
                t.last_seen_frame = frame_idx
                t.missed_frames = 0
                t.history.append((frame_idx, det.box))
                matched_tids.add(best_tid)
                unmatched.remove(det_idx)

        for tid in list(self._active):
            if tid not in matched_tids:
                self._active[tid].missed_frames += 1

        # Drop stale tracks
        for tid in [t for t, tr in self._active.items()
                    if tr.missed_frames > MAX_MISS_FRAMES]:
            del self._active[tid]

        # New tracks for unmatched detections
        for det_idx in unmatched:
            det = detections[det_idx]
            t = Track(
                track_id=self._next_id,
                label=det.label,
                box=det.box,
                first_seen_frame=frame_idx,
                last_seen_frame=frame_idx,
                history=[(frame_idx, det.box)],
            )
            self._active[self._next_id] = t
            self._next_id += 1

        return list(self._active.values())


def detect_events(
    frame_history: List[Tuple[int, List[Detection]]],
    target_label: str,
    video_fps: float,
) -> List[Event]:
    """
    Walk frame_history, run the tracker, apply the stability filter, and
    emit timestamped events (appeared / disappeared / pickup_candidate).
    """
    tracker = SimpleTracker()
    events: List[Event] = []

    confirmed_ids: Set[int] = set()
    candidate_first_seen: Dict[int, Tuple[int, float, List[int]]] = {}
    active_target_ids: Set[int] = set()

    # Build lookup maps for second pass
    frame_person_map: Dict[int, List[List[int]]] = {}
    frame_target_map: Dict[int, List[int]] = {}

    for frame_idx, dets in frame_history:
        ts = frame_idx / video_fps
        active_tracks = tracker.update(dets, frame_idx)

        # Build maps for proximity analysis
        frame_person_map[frame_idx] = [d.box for d in dets if d.label == "person"]
        for d in dets:
            if d.label == target_label:
                frame_target_map[frame_idx] = d.box

        current_target_ids = {
            t.track_id for t in active_tracks if t.label == target_label
        }

        # Stability filter: confirm tracks seen in ≥ MIN_TRACK_FRAMES
        for t in active_tracks:
            if t.label != target_label or t.track_id in confirmed_ids:
                continue
            tid = t.track_id
            if tid not in candidate_first_seen:
                candidate_first_seen[tid] = (frame_idx, ts, list(t.box))
            elif len(t.history) >= MIN_TRACK_FRAMES:
                fi0, ts0, box0 = candidate_first_seen[tid]
                events.append(Event(
                    frame_idx=fi0, timestamp_sec=ts0,
                    event_type="appeared",
                    label=target_label, track_id=tid, box=box0,
                    note=f"Confirmed after {len(t.history)} frames",
                ))
                confirmed_ids.add(tid)

        # Disappearances: only for confirmed tracks
        for gone_id in (active_target_ids - current_target_ids):
            if gone_id in confirmed_ids:
                events.append(Event(
                    frame_idx=frame_idx, timestamp_sec=ts,
                    event_type="disappeared",
                    label=target_label, track_id=gone_id, box=None,
                    note=f"Track {gone_id} lost at frame {frame_idx} (~{ts:.1f}s)",
                ))

        active_target_ids = current_target_ids

    # After loop: promote any single-frame candidate at the last sampled frame.
    # The object just came into view — it couldn't accumulate 2 frames because
    # the video ended. Emit as "appeared" using the current (last) box.
    last_frame_idx = frame_history[-1][0] if frame_history else -1
    last_frame_ts  = last_frame_idx / video_fps
    for tid, (fi0, ts0, box0) in candidate_first_seen.items():
        if tid not in confirmed_ids:
            # Only promote if the track is still active at the last frame
            if any(
                t.track_id == tid
                for t in tracker._active.values()
            ):
                events.append(Event(
                    frame_idx=fi0, timestamp_sec=ts0,
                    event_type="appeared",
                    label=target_label, track_id=tid, box=box0,
                    note="Appeared at end of video (single-frame — unconfirmed)",
                ))
                confirmed_ids.add(tid)

    # Second pass: upgrade disappearances to pickup_candidate if person nearby
    for ev in events:
        if ev.event_type != "disappeared":
            continue
        last_box: Optional[List[int]] = None
        for look_back in range(1, 4):
            fb = ev.frame_idx - look_back
            if fb in frame_target_map:
                last_box = frame_target_map[fb]
                break
        if last_box is None:
            continue
        person_boxes = frame_person_map.get(ev.frame_idx) or \
                       frame_person_map.get(ev.frame_idx - 1, [])
        if not person_boxes:
            continue
        tc = _centre(last_box)
        min_d = min(_dist(tc, _centre(pb)) for pb in person_boxes)
        ev.person_proximity_px = min_d
        if min_d < PROXIMITY_PX:
            ev.event_type = "pickup_candidate"
            ev.note += f" — person {min_d:.0f}px away"

    return events
