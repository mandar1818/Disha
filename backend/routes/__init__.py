"""
============================================================
SmartVisionAI
============================================================

Package:
    backend.routes

Purpose:
    FastAPI route definitions for SmartVisionAI.

Available routes:
    - Detection API
    - Detection health
    - Detection status

============================================================
"""

from .detection import router as detection_router


__all__ = [
    "detection_router",
]