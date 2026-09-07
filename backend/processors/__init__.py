"""
============================================================
SmartVisionAI
============================================================

Package:
    backend.processors

Purpose:
    Image/frame processing and integration pipeline.

The FrameProcessor connects:

    YOLOv8
       ↓
    MiDaS
       ↓
    Decision Engine
       ↓
    Navigation Result

============================================================
"""

from .frame_processor import FrameProcessor


__all__ = [
    "FrameProcessor",
]