"""
models/types.py — shared data classes used across all pipeline modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Detection:
    score: float
    box: List[int]        # [x1, y1, x2, y2] absolute pixels
    label: str            # always the user's original query term
    verified: bool = False  # set True by Verifier after second-opinion check


@dataclass
class Track:
    track_id: int
    label: str
    box: List[int]
    first_seen_frame: int
    last_seen_frame: int
    missed_frames: int = 0
    history: List[Tuple[int, List[int]]] = field(default_factory=list)


@dataclass
class Event:
    frame_idx: int
    timestamp_sec: float
    event_type: str       # "appeared" | "disappeared" | "pickup_candidate"
    label: str
    track_id: int
    box: Optional[List[int]]
    person_proximity_px: Optional[float] = None
    note: str = ""
