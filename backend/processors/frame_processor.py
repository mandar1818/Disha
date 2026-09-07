"""
SmartVisionAI - Frame Processor
===============================
Pipeline:

Camera Frame
    ↓
YOLO Object Detection
    ↓
MiDaS Relative Depth
    ↓
Object + Depth Fusion
    ↓
Unknown Obstacle Detection
    ↓
Decision Engine
    ↓
Final Navigation Result

Phase 14 Real-Time Optimization:
- Eliminates redundant full-array nanmin/nanmax scans in _sample_bbox_depth and _depth_map_to_scene
- Pre-allocated morphological kernels and direct boolean mask construction for unknown obstacle detection
- Monotonic microsecond-accurate stage profiling (optional processing_breakdown_ms)
- Preserves full Phase 1-13 navigation, safety, and step guidance contracts
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.yolo.detector import YOLODetector
from backend.depth.midas import MiDaSDepthEstimator
from backend.decision.engine import DecisionEngine
from backend.config import (
    INFERENCE_BACKEND,
    FALLBACK_TO_PYTORCH_ON_ERROR,
    YOLO_TFLITE_PATH,
    YOLO_TFLITE_MODEL,
    COCO_CLASSES_PATH,
    COCO_CLASSES,
    MIDAS_ONNX_PATH,
    MIDAS_ONNX_MODEL,
    YOLO_PYTORCH_PATH,
    YOLO_PYTORCH_MODEL,
    MIDAS_PYTORCH_MODEL,
    DEFAULT_YOLO_CONFIDENCE,
)

# Pre-allocated morphological kernels for unknown obstacle segmentation
_MORPH_KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_MORPH_KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


class FrameProcessor:
    """
    Main backend coordinator.

    Combines:
        YOLO (PyTorch or TFLite)
        MiDaS (PyTorch or ONNX)
        DecisionEngine

    into one processing pipeline.
    """

    def __init__(
        self,
        yolo_model: Optional[str] = None,
        yolo_confidence: float = DEFAULT_YOLO_CONFIDENCE,
        midas_model: Optional[str] = None,
        device: Optional[str] = None,
        inference_backend: Optional[str] = None,
    ) -> None:

        target_backend = (
            inference_backend or INFERENCE_BACKEND or "pytorch"
        ).strip().lower()

        self.requested_backend = target_backend
        self.active_backend = "pytorch"
        self.fallback_active = False
        self.loaded = False

        print(f"[FrameProcessor] Initializing (requested backend: {self.requested_backend})...")

        if self.requested_backend == "tflite_onnx":
            try:
                print("[FrameProcessor] Initializing TFLite+ONNX Edge-AI models...")
                self.yolo = YOLODetector(
                    model_path=yolo_model or YOLO_TFLITE_MODEL,
                    confidence=yolo_confidence,
                    backend="tflite",
                    labels_path=COCO_CLASSES,
                )
                self.midas = MiDaSDepthEstimator(
                    onnx_model_path=midas_model or MIDAS_ONNX_MODEL,
                    backend="onnx",
                )
                self.active_backend = "tflite_onnx"
                self.fallback_active = False
                print("[FrameProcessor] TFLite+ONNX Edge-AI backend initialized successfully.")
            except Exception as exc:
                if FALLBACK_TO_PYTORCH_ON_ERROR:
                    print(
                        f"[FrameProcessor] WARNING: TFLite/ONNX initialization failed: {exc}. "
                        "Safely falling back to PyTorch backend."
                    )
                    self.yolo = YOLODetector(
                        model_path=YOLO_PYTORCH_MODEL,
                        confidence=yolo_confidence,
                        backend="pytorch",
                        device=device,
                    )
                    self.midas = MiDaSDepthEstimator(
                        model_name=MIDAS_PYTORCH_MODEL,
                        backend="pytorch",
                        device=device,
                    )
                    self.active_backend = "pytorch"
                    self.fallback_active = True
                else:
                    raise
        else:
            print("[FrameProcessor] Initializing PyTorch backend...")
            self.yolo = YOLODetector(
                model_path=yolo_model or YOLO_PYTORCH_MODEL,
                confidence=yolo_confidence,
                backend="pytorch",
                device=device,
            )
            self.midas = MiDaSDepthEstimator(
                model_name=midas_model or MIDAS_PYTORCH_MODEL,
                backend="pytorch",
                device=device,
            )
            self.active_backend = "pytorch"
            self.fallback_active = False

        self.decision_engine = DecisionEngine()
        self.loaded = True

        print(
            f"[FrameProcessor] Initialization completed. "
            f"Active backend: {self.active_backend} "
            f"(fallback: {self.fallback_active})"
        )

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> Dict[str, Any]:
        """
        Return complete processor status including active backend.
        """
        return {
            "loaded": self.loaded,
            "processor": "FrameProcessor",
            "active_backend": self.active_backend,
            "requested_backend": self.requested_backend,
            "fallback_active": self.fallback_active,
            "yolo": self._component_status(self.yolo),
            "midas": self._component_status(self.midas),
            "decision_engine": self._component_status(self.decision_engine),
        }

    # ========================================================
    # COMPONENT STATUS
    # ========================================================

    @staticmethod
    def _component_status(
        component: Any,
    ) -> Dict[str, Any]:
        """
        Safely obtain component status.
        """

        try:

            if hasattr(
                component,
                "status",
            ):

                result = component.status()

                if isinstance(
                    result,
                    dict,
                ):
                    return result

        except Exception:
            pass

        return {
            "loaded": bool(
                getattr(
                    component,
                    "loaded",
                    False,
                )
            )
        }

    # ========================================================
    # IMAGE VALIDATION
    # ========================================================

    @staticmethod
    def _validate_frame(
        frame: Any,
    ) -> np.ndarray:
        """
        Validate and normalize an OpenCV frame.
        """

        if frame is None:
            raise ValueError(
                "Frame cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "Frame must be a numpy.ndarray."
            )

        if frame.size == 0:
            raise ValueError(
                "Frame is empty."
            )

        if frame.ndim not in (
            2,
            3,
        ):
            raise ValueError(
                "Frame must be a 2D or 3D image."
            )

        return frame

    # ========================================================
    # YOLO DETECTION
    # ========================================================

    def detect_objects(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Run YOLO detection.

        The detector implementation may expose different
        method names, so this function supports the standard
        methods used by our backend.
        """

        frame = self._validate_frame(
            frame
        )

        # Preferred API
        if hasattr(
            self.yolo,
            "detect",
        ):

            result = self.yolo.detect(
                frame
            )

            if isinstance(
                result,
                list,
            ):
                return result

            if isinstance(
                result,
                dict,
            ):

                objects = result.get(
                    "objects",
                    result.get(
                        "detections",
                        [],
                    ),
                )

                if isinstance(
                    objects,
                    list,
                ):
                    return objects

        # Compatibility API
        if hasattr(
            self.yolo,
            "detect_objects",
        ):

            result = (
                self.yolo.detect_objects(
                    frame
                )
            )

            if isinstance(
                result,
                list,
            ):
                return result

            if isinstance(
                result,
                dict,
            ):

                objects = result.get(
                    "objects",
                    result.get(
                        "detections",
                        [],
                    ),
                )

                if isinstance(
                    objects,
                    list,
                ):
                    return objects

        raise RuntimeError(
            "YOLODetector does not provide "
            "a compatible detection method."
        )

    # ========================================================
    # DEPTH ESTIMATION
    # ========================================================

    def estimate_depth_map(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Return the full normalized MiDaS depth map."""

        frame = self._validate_frame(frame)

        if not hasattr(self.midas, "estimate_depth_map"):
            raise RuntimeError(
                "MiDaSDepthEstimator does not provide estimate_depth_map()."
            )

        depth_map = self.midas.estimate_depth_map(frame)

        if not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
            raise RuntimeError("MiDaS returned an invalid depth map.")

        return depth_map.astype(np.float32, copy=False)

    def estimate_depth(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """Estimate relative scene depth and return scene statistics."""

        depth_map = self.estimate_depth_map(frame)
        return self._depth_map_to_scene(depth_map)

    # ========================================================
    # DEPTH MAP -> SCENE INFORMATION
    # ========================================================

    @staticmethod
    def _depth_map_to_scene(
        depth_map: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Convert a depth map into simple scene regions.

        Important:
            Values are relative depth values in [0.0, 1.0].
            They are NOT guaranteed meters.
        """

        if not isinstance(
            depth_map,
            np.ndarray,
        ) or depth_map.size == 0:
            return {
                "average_depth": 1.0,
                "center_depth": 1.0,
                "left_depth": 1.0,
                "center_region_depth": 1.0,
                "right_depth": 1.0,
            }

        height, width = depth_map.shape[:2]

        # Whole scene
        average_depth = float(
            np.mean(depth_map)
        )

        # Center region
        cx1 = int(
            width * 0.35
        )
        cx2 = int(
            width * 0.65
        )

        center_region = depth_map[
            :,
            cx1:cx2,
        ]

        center_depth = float(
            np.mean(center_region)
        ) if center_region.size else 1.0

        # Left
        left_region = depth_map[
            :,
            :int(width * 0.33),
        ]

        left_depth = float(
            np.mean(left_region)
        ) if left_region.size else 1.0

        # Right
        right_region = depth_map[
            :,
            int(width * 0.67):,
        ]

        right_depth = float(
            np.mean(right_region)
        ) if right_region.size else 1.0

        return {
            "average_depth":
                round(
                    average_depth,
                    4,
                ),

            "center_depth":
                round(
                    center_depth,
                    4,
                ),

            "left_depth":
                round(
                    left_depth,
                    4,
                ),

            "center_region_depth":
                round(
                    center_depth,
                    4,
                ),

            "right_depth":
                round(
                    right_depth,
                    4,
                ),
        }

    # ========================================================
    # OBJECT DEPTH
    # ========================================================

    def estimate_object_depth(
        self,
        depth_map: np.ndarray,
        bbox: Dict[str, Any],
    ) -> float:
        """Estimate an object's relative depth from an existing depth map."""

        if not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
            return 1.0

        return self._sample_bbox_depth(depth_map, bbox)

    # ========================================================
    # SAMPLE BOUNDING BOX DEPTH
    # ========================================================

    @staticmethod
    def _sample_bbox_depth(
        depth_map: np.ndarray,
        bbox: Dict[str, Any],
    ) -> float:
        """
        Sample depth from the center of a bounding box.

        Phase 14 Optimization:
        Depth map is already normalized to [0.0, 1.0] by estimate_depth_map.
        Avoids redundant full-array nanmin/nanmax scans over the entire frame.
        """

        if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
            return 1.0

        height, width = depth_map.shape[:2]

        try:
            x1 = int(float(bbox.get("x1", 0)))
            y1 = int(float(bbox.get("y1", 0)))
            x2 = int(float(bbox.get("x2", width)))
            y2 = int(float(bbox.get("y2", height)))
        except Exception:
            return 1.0

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            return 1.0

        # Central 50% portion of the box
        cx1 = int(x1 + (x2 - x1) * 0.25)
        cx2 = int(x1 + (x2 - x1) * 0.75)
        cy1 = int(y1 + (y2 - y1) * 0.25)
        cy2 = int(y1 + (y2 - y1) * 0.75)

        region = depth_map[cy1:cy2, cx1:cx2]

        if region.size == 0:
            region = depth_map[y1:y2, x1:x2]

        if region.size == 0:
            return 1.0

        # Median is robust to outlier pixels
        value = float(np.median(region))

        return max(0.0, min(1.0, value))

    # ========================================================
    # FUSE OBJECTS + DEPTH
    # ========================================================

    def fuse_objects_and_depth(
        self,
        frame: np.ndarray,
        objects: List[Dict[str, Any]],
        scene_depth: Dict[str, Any],
        depth_map: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Add depth information to each YOLO object.

        If native object depth is unavailable, scene center
        depth is used as a safe fallback.
        """

        fused: List[Dict[str, Any]] = []

        map_for_depth = (
            depth_map
            if depth_map is not None
            and isinstance(depth_map, np.ndarray)
            and depth_map.size > 0
            else None
        )

        for obj in objects:
            if not isinstance(obj, dict):
                continue

            new_object = dict(obj)
            bbox = new_object.get("bbox", {})
            if not isinstance(bbox, dict):
                bbox = {}

            existing_depth = new_object.get("depth", None)

            if existing_depth is not None:
                try:
                    depth = float(existing_depth)
                except Exception:
                    depth = self._fallback_depth(scene_depth)
            else:
                depth = None

            if depth is None:
                try:
                    if map_for_depth is not None:
                        depth = self.estimate_object_depth(map_for_depth, bbox)
                    else:
                        depth = self._fallback_depth(scene_depth)
                except Exception:
                    depth = self._fallback_depth(scene_depth)

            if not np.isfinite(depth):
                depth = self._fallback_depth(scene_depth)

            depth = max(0.0, min(1.0, float(depth)))
            new_object["depth"] = round(depth, 4)
            new_object["bbox"] = bbox

            fused.append(new_object)

        return fused

    # ========================================================
    # FALLBACK DEPTH
    # ========================================================

    @staticmethod
    def _fallback_depth(
        scene_depth: Dict[str, Any],
    ) -> float:
        """
        Safe fallback if object-level depth cannot be obtained.
        """

        if not isinstance(
            scene_depth,
            dict,
        ):
            return 1.0

        for key in (
            "center_region_depth",
            "center_depth",
            "average_depth",
        ):
            value = scene_depth.get(key, None)
            if value is not None:
                try:
                    value = float(value)
                    if np.isfinite(value):
                        return max(0.0, min(1.0, value))
                except Exception:
                    continue

        return 1.0

    # ========================================================
    # UNKNOWN OBSTACLE CANDIDATE DETECTION
    # ========================================================

    def detect_unknown_candidates(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        known_objects: List[Dict[str, Any]],
        frame_width: float,
        frame_height: float,
    ) -> List[Dict[str, Any]]:
        """
        Detect genuine unknown physical obstacle candidates using depth analysis.

        OpenCV-based foreground depth segmentation identifies candidate
        physical obstacles that are not recognized by the 80 YOLO classes.
        Candidates overlapping known YOLO detections are suppressed to prevent
        duplicate representations of the same physical entity.
        """

        if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
            return []

        h, w = depth_map.shape[:2]
        frame_area = float(w * h)
        if frame_area <= 0:
            return []

        # 1. Direct foreground mask: relative foreground objects (< 0.55)
        # Depth map is already clean float32 [0.0, 1.0] from estimate_depth_map
        fg_mask = (depth_map < 0.55).astype(np.uint8) * 255

        # 2. Morphological cleanup using pre-allocated kernels
        clean_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, _MORPH_KERNEL_OPEN)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL_CLOSE)

        # 3. Contours
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Dict[str, Any]] = []

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            box_area = float(bw * bh)
            cnt_area = float(cv2.contourArea(cnt))

            # Size filters: calibrated to 0.8% - 45% of frame area
            if box_area < 0.008 * frame_area or box_area > 0.45 * frame_area:
                continue
            if cnt_area < 0.005 * frame_area:
                continue

            # Aspect ratio: avoid extreme horizontal/vertical slivers
            ar = bw / float(bh) if bh > 0 else 0.0
            if ar > 4.5 or ar < 0.15:
                continue

            # Ground proximity: candidate must extend into bottom 70% of frame
            if (by + bh) / float(h) < 0.30:
                continue

            # Sample median depth
            roi = depth_map[by : by + bh, bx : bx + bw]
            if roi.size == 0:
                continue
            cand_depth = float(np.median(roi))
            if cand_depth > 0.55:
                continue

            cand_box = {
                "x1": float(bx),
                "y1": float(by),
                "x2": float(bx + bw),
                "y2": float(by + bh),
            }

            # 4. Known YOLO box suppression
            suppressed = False
            for k_obj in known_objects:
                k_box = k_obj.get("bbox", {})
                kx1 = float(k_box.get("x1", 0))
                ky1 = float(k_box.get("y1", 0))
                kx2 = float(k_box.get("x2", 0))
                ky2 = float(k_box.get("y2", 0))

                ix1 = max(cand_box["x1"], kx1)
                iy1 = max(cand_box["y1"], ky1)
                ix2 = min(cand_box["x2"], kx2)
                iy2 = min(cand_box["y2"], ky2)

                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter_area = iw * ih

                if inter_area <= 0:
                    continue

                k_area = max(1.0, (kx2 - kx1) * (ky2 - ky1))
                union_area = box_area + k_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                cand_overlap = inter_area / box_area if box_area > 0 else 0.0
                k_overlap = inter_area / k_area if k_area > 0 else 0.0

                if iou >= 0.25 or cand_overlap >= 0.35 or (k_overlap >= 0.50 and box_area <= 2.5 * k_area):
                    suppressed = True
                    break

            if suppressed:
                continue

            # Confidence based on solidity and depth contrast
            solidity = cnt_area / max(1.0, box_area)
            depth_contrast = max(0.0, 0.60 - cand_depth) / 0.60
            conf = round(float(np.clip(0.40 + 0.30 * solidity + 0.30 * depth_contrast, 0.40, 0.85)), 4)

            candidates.append({
                "bbox": cand_box,
                "cand_depth": cand_depth,
                "confidence": conf,
                "box_area": box_area,
                "class_name": "unknown_obstacle",
            })

        # Non-Maximum Suppression (Deduplication among candidates)
        candidates.sort(key=lambda c: c["box_area"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for c in candidates:
            dup = False
            for k in kept:
                b1, b2 = c["bbox"], k["bbox"]
                ix1 = max(b1["x1"], b2["x1"])
                iy1 = max(b1["y1"], b2["y1"])
                ix2 = min(b1["x2"], b2["x2"])
                iy2 = min(b1["y2"], b2["y2"])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                ia = iw * ih
                if ia > 0:
                    a1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
                    a2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
                    u = a1 + a2 - ia
                    if u > 0 and (ia / u) >= 0.35:
                        dup = True
                        break
            if not dup:
                kept.append(c)

        return kept[:3]

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Process one camera frame.

        Complete pipeline:

            frame
              ↓
            YOLO
              ↓
            MiDaS
              ↓
            fusion
              ↓
            Unknown Obstacle Detection
              ↓
            Decision Engine
              ↓
            final result
        """

        start_time = time.perf_counter()

        frame = self._validate_frame(
            frame
        )

        height, width = frame.shape[:2]

        # ----------------------------------------------------
        # 1. YOLO Detection
        # ----------------------------------------------------
        t_yolo_start = time.perf_counter()
        objects = self.detect_objects(
            frame
        )

        if not isinstance(
            objects,
            list,
        ):
            objects = []
        t_yolo_end = time.perf_counter()

        # ----------------------------------------------------
        # 2. MiDaS Depth Estimation
        # ----------------------------------------------------
        t_midas_start = time.perf_counter()
        depth_map = None
        try:
            depth_map = self.estimate_depth_map(
                frame
            )
        except Exception:
            depth_map = None

        if depth_map is not None:
            scene_depth = self._depth_map_to_scene(
                depth_map
            )
        else:
            scene_depth = {
                "average_depth": 1.0,
                "center_depth": 1.0,
                "left_depth": 1.0,
                "center_region_depth": 1.0,
                "right_depth": 1.0,
            }
        t_midas_end = time.perf_counter()

        # ----------------------------------------------------
        # 3. Object-Depth Fusion
        # ----------------------------------------------------
        t_fusion_start = time.perf_counter()
        fused_objects = (
            self.fuse_objects_and_depth(
                frame,
                objects,
                scene_depth,
                depth_map=depth_map,
            )
        )
        t_fusion_end = time.perf_counter()

        # ----------------------------------------------------
        # 4. Unknown Obstacle Candidate Detection
        # ----------------------------------------------------
        t_unknown_start = time.perf_counter()
        unknown_candidates: List[Dict[str, Any]] = []
        if depth_map is not None:
            try:
                unknown_candidates = self.detect_unknown_candidates(
                    frame=frame,
                    depth_map=depth_map,
                    known_objects=fused_objects,
                    frame_width=width,
                    frame_height=height,
                )
            except Exception:
                unknown_candidates = []
        t_unknown_end = time.perf_counter()

        # ----------------------------------------------------
        # 5. Decision Engine
        # ----------------------------------------------------
        t_decision_start = time.perf_counter()
        decision = (
            self.decision_engine.analyze(
                objects=fused_objects,
                scene_depth=scene_depth,
                frame_width=width,
                frame_height=height,
                unknown_objects=unknown_candidates,
            )
        )
        t_decision_end = time.perf_counter()

        # ----------------------------------------------------
        # Processing time & Telemetry Breakdown
        # ----------------------------------------------------
        total_elapsed = time.perf_counter() - start_time
        processing_time_ms = total_elapsed * 1000.0

        breakdown_ms = {
            "yolo": round((t_yolo_end - t_yolo_start) * 1000.0, 2),
            "midas": round((t_midas_end - t_midas_start) * 1000.0, 2),
            "fusion": round((t_fusion_end - t_fusion_start) * 1000.0, 2),
            "unknown": round((t_unknown_end - t_unknown_start) * 1000.0, 2),
            "decision": round((t_decision_end - t_decision_start) * 1000.0, 2),
            "total": round(processing_time_ms, 2),
        }

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {
            "success": True,

            "frame": {
                "width": width,
                "height": height,
            },

            "objects": decision.get(
                "objects",
                [],
            ),

            "unknown_objects": decision.get(
                "unknown_objects",
                [],
            ),

            "scene": decision.get(
                "scene",
                scene_depth,
            ),

            "free_path": decision.get(
                "free_path",
                {},
            ),

            "multiple_objects": decision.get(
                "multiple_objects",
                {},
            ),

            "safety_level": decision.get(
                "safety_level",
                "SAFE",
            ),

            "emergency_stop": bool(
                decision.get(
                    "emergency_stop",
                    False,
                )
            ),

            "primary_obstacle": decision.get(
                "primary_obstacle",
                None,
            ),

            "navigation": decision.get(
                "navigation",
                {},
            ),

            "step_guidance": decision.get(
                "step_guidance",
                {},
            ),

            "predictive_threat": decision.get(
                "predictive_threat",
                {
                    "obstacle_id": None,
                    "class_name": "none",
                    "position": "NONE",
                    "distance_m": 0.0,
                    "motion_state": "STATIONARY",
                    "approach_rate_mps": 0.0,
                    "time_to_collision_s": None,
                    "approach_score": 0.0,
                    "threat_level": "NONE",
                    "is_approaching": False,
                    "warning_issued": False,
                },
            ),

            "approaching": bool(
                decision.get(
                    "approaching",
                    False,
                )
            ),

            "approach_rate": float(
                decision.get(
                    "approach_rate",
                    0.0,
                )
            ),

            "obstacle_stability": int(
                decision.get(
                    "obstacle_stability",
                    0,
                )
            ),

            "voice_instruction": decision.get(
                "voice_instruction",
                "Continue forward.",
            ),

            "emergency_alert": bool(
                decision.get(
                    "emergency_alert",
                    False,
                )
            ),

            "processing_time_ms": round(
                processing_time_ms,
                2,
            ),

            # Optional telemetry breakdown (internal/diagnostic)
            "processing_breakdown_ms": breakdown_ms,

            "backend": self.active_backend,
            "active_backend": self.active_backend,
            "fallback_active": self.fallback_active,

            "api": {
                "endpoint": "/detect",
                "service": "SmartVisionAI",
            },

            # Backward compatibility aliases
            "detections": decision.get(
                "objects",
                [],
            ),

            "multi_object": decision.get(
                "multiple_objects",
                {},
            ),

            "navigation_confidence": float(
                0.92 if decision.get("navigation", {}).get("confidence") == "HIGH" else 0.70
            ),
        }

    # ========================================================
    # ALIASES
    # ========================================================

    def process(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Short alias for process_frame().
        """

        return self.process_frame(
            frame
        )

    def analyze(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Alias used by API routes.
        """

        return self.process_frame(
            frame
        )


# ============================================================
# EXPORT
# ============================================================

__all__ = [
    "FrameProcessor",
]