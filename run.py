#!/usr/bin/env python3
"""
run.py — Project main entrypoint.
Organizes imports and kicks off the live camera object detection session.
"""
import sys
import os

# Put the root of the repo on the search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.live_detector import main

if __name__ == "__main__":
    main()
