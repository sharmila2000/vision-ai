"""
video_chatbot.py — Phase 2 interactive chatbot.

Usage:
    source .venv/bin/activate
    python video_chatbot.py --video videos/video_01.mp4

Optional flags:
    --fps 2.0          Sample rate (default 1.0 fps)
    --no-verify        Disable Florence-2 verifier (faster, lower precision)
"""
from __future__ import annotations

import argparse
import os
import re
import sys

try:
    from pipeline import Pipeline
except ImportError as exc:
    print(f"Import error: {exc}\nRun: source .venv/bin/activate first.")
    sys.exit(1)


# ── query parsing ──────────────────────────────────────────────────────────── #

_FILLER = re.compile(
    r"\b(where|is|the|a|an|are|was|were|find|show|me|in|this|video|"
    r"what|happened|to|with|did|pick|picked|up|put|down|locate|"
    r"can|you|please|it)\b",
    re.IGNORECASE,
)

_CORRECTIONS = {
    "calender": "calendar",
    "colour":   "color",
    "grey":     "gray",
    "scren":    "screen",
    "labtop":   "laptop",
    "moblie":   "mobile",
}

def parse_query(user_text: str) -> str:
    clean = _FILLER.sub(" ", user_text.strip().rstrip("?.,!"))
    tokens = clean.split()
    corrected = [_CORRECTIONS.get(t.lower(), t) for t in tokens if len(t) > 1]
    return " ".join(corrected).strip() or user_text.strip()


# ── result formatter ───────────────────────────────────────────────────────── #

def _ts(sec: float) -> str:
    return f"{int(sec//60):02d}:{sec%60:05.2f}"


def format_response(events: list, query: str, video_name: str) -> str:
    if not events:
        return (
            f"I couldn't detect a '{query}' in '{video_name}'. "
            "Try a more specific term, e.g. 'house key', 'mobile phone'."
        )

    appeared  = [e for e in events if e.event_type == "appeared"]
    pickups   = [e for e in events if e.event_type == "pickup_candidate"]
    disappear = [e for e in events if e.event_type == "disappeared"]
    parts = []

    if appeared:
        ev = appeared[0]
        loc = _describe_position(ev.box) if ev.box else ""
        loc_str = f" ({loc})" if loc else ""
        parts.append(f"The '{query}' was first seen at {_ts(ev.timestamp_sec)}{loc_str}.")

    if pickups:
        ev = pickups[0]
        parts.append(
            f"At {_ts(ev.timestamp_sec)}, the '{query}' disappeared and a person "
            f"was {ev.person_proximity_px:.0f}px away — likely picked up."
        )
    elif disappear:
        ev = disappear[0]
        parts.append(
            f"At {_ts(ev.timestamp_sec)}, the '{query}' disappeared. "
            "No person was close enough to confirm a pickup."
        )
    elif appeared:
        parts.append(f"The '{query}' remained visible throughout the video.")

    return "\n".join(parts)


def _describe_position(box: list) -> str:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    h = "left" if cx < 283 else ("right" if cx > 565 else "centre")
    v = "top"  if cy < 159 else ("bottom" if cy > 319 else "middle")
    if h == "centre" and v == "middle":
        return "centre of frame"
    return f"{v}-{h} of frame"


# ── result cache ───────────────────────────────────────────────────────────── #

class _Cache:
    def __init__(self):
        self._d: dict = {}

    def _key(self, video: str, query: str) -> str:
        return f"{os.path.abspath(video)}|{query.lower()}"

    def get(self, video: str, query: str):
        return self._d.get(self._key(video, query))

    def put(self, video: str, query: str, events: list):
        self._d[self._key(video, query)] = events


# ── chatbot loop ───────────────────────────────────────────────────────────── #

def run_chatbot(video_path: str, sample_fps: float, use_verifier: bool):
    if not os.path.exists(video_path):
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    pipe  = Pipeline(use_verifier=use_verifier)
    cache = _Cache()

    print("=" * 60)
    print("  Vision AI — Phase 2  |  Video Object Chatbot")
    print(f"  Video  : {os.path.basename(video_path)}")
    print(f"  Mode   : {'OWLv2 + Florence-2 (verified)' if use_verifier else 'OWLv2 only'}")
    print(f"  Sample : {sample_fps} fps")
    print("=" * 60)
    print("  Examples: 'where is the whiteboard?'  'find the laptop'")
    print("            'list all objects'           'quit'\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if re.search(r"\b(list|scan|all|everything)\b", user_input, re.I):
            _broad_scan(pipe, cache, video_path, sample_fps)
            continue

        query = parse_query(user_input)
        if not query:
            print("Bot > Please enter an object to search for.\n")
            continue

        # Show spelling correction to user if it changed
        original_word = user_input.strip().lower().rstrip("?.,!")
        if query != original_word:
            print(f"Bot > (Searching for '{query}'…)\n")
        else:
            print(f"Bot > Searching for '{query}' …")

        cached = cache.get(video_path, query)
        if cached is not None:
            print("Bot > (cached)\n")
            events = cached
        else:
            events = pipe.run(
                video_path, query=query,
                sample_fps=sample_fps,
                also_detect_person=True,
                verbose=True,
                save_video=True,
            )
            cache.put(video_path, query, events)

        print("\n" + pipe.summarise(events, query))
        print(f"\nBot > {format_response(events, query, os.path.basename(video_path))}\n")


def _broad_scan(pipe, cache, video_path, sample_fps):
    common = [
        "person", "laptop", "phone", "whiteboard", "tv",
        "sofa", "chair", "bag", "bottle", "book",
    ]
    print("Bot > Scanning for common objects …\n")
    found = []
    for obj in common:
        cached = cache.get(video_path, obj)
        if cached is not None:
            events = cached
        else:
            events = pipe.run(
                video_path, query=obj,
                sample_fps=sample_fps,
                also_detect_person=False,
                verbose=False,
                save_video=False,
            )
            cache.put(video_path, obj, events)
        appeared = [e for e in events if e.event_type in
                    ("appeared", "pickup_candidate", "disappeared")]
        if appeared:
            found.append((obj, appeared[0]))

    if found:
        print("Bot > Objects detected:")
        for obj, ev in found:
            print(f"       • {obj:<12} first seen at {_ts(ev.timestamp_sec)}")
    else:
        print("Bot > No common objects detected above confidence threshold.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 2 Video Chatbot")
    ap.add_argument("--video",     default="videos/video_01.mp4")
    ap.add_argument("--fps",       type=float, default=1.0)
    ap.add_argument("--no-verify", action="store_true",
                    help="Disable Florence-2 verifier (faster, lower precision)")
    args = ap.parse_args()
    run_chatbot(args.video, args.fps, use_verifier=not args.no_verify)
