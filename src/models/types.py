"""
src/models/types.py — shared data classes used across live detection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    score: float
    box: List[int]          # [x1, y1, x2, y2] absolute pixels
    label: str              # the detected query class label
    verified: bool = False  # set True by Verifier after second-opinion check
