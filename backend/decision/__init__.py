"""
============================================================
SmartVisionAI
============================================================

Package:
    backend.decision

Purpose:
    Decision and navigation engine.

This package converts:
    YOLO detections
    +
    MiDaS depth information
    +
    scene information

into:

    - risk scores
    - safety levels
    - emergency-stop decisions
    - navigation direction
    - voice instructions
    - primary obstacle
    - free-path information

============================================================
"""

from .engine import DecisionEngine


__all__ = [
    "DecisionEngine",
]
