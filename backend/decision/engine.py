"""
============================================================
SmartVisionAI
============================================================

File:
    backend/decision/engine.py

Purpose:
    Decision Engine for SmartVisionAI.

Responsibilities:
    - Filter invalid detections
    - Remove duplicate detections using IoU
    - Assign object priorities
    - Determine object position
    - Combine confidence + depth + priority
    - Calculate risk score
    - Identify primary obstacle
    - Analyze free paths
    - Determine safety level
    - Emergency-stop logic
    - Navigation decision
    - Multi-object reasoning
    - Generate voice instruction

Important:
    MiDaS provides RELATIVE depth, not guaranteed metric distance.

============================================================
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# DECISION ENGINE
# ============================================================

class DecisionEngine:
    """
    Rule-based decision engine for SmartVisionAI.

    The engine converts YOLO + MiDaS information into:

        risk
        safety level
        emergency stop
        navigation
        voice instruction
    """

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    EMERGENCY_THRESHOLD = 0.18
    AWARE_THRESHOLD = 0.32
    CAUTION_THRESHOLD = 0.50
    DANGER_THRESHOLD = 0.68

    # --------------------------------------------------------
    # General configuration
    # --------------------------------------------------------

    DEFAULT_CONFIDENCE = 0.50
    IOU_THRESHOLD = 0.60

    # Smaller MiDaS value = closer object.
    # These values are relative-depth thresholds.

    VERY_CLOSE_DEPTH = 0.20
    CLOSE_DEPTH = 0.35
    MEDIUM_DEPTH = 0.55

    # --------------------------------------------------------
    # COCO object priorities
    # --------------------------------------------------------

    OBJECT_PRIORITIES: Dict[str, float] = {
        # Humans
        "person": 1.00,

        # Vehicles
        "car": 0.95,
        "bus": 0.95,
        "truck": 0.95,
        "motorcycle": 0.90,
        "bicycle": 0.85,
        "train": 0.90,

        # Large / important obstacles
        "dining table": 0.75,
        "table": 0.75,
        "chair": 0.55,
        "bench": 0.65,
        "couch": 0.65,
        "bed": 0.60,

        # Common objects
        "backpack": 0.45,
        "suitcase": 0.55,
        "handbag": 0.40,

        "bottle": 0.15,
        "cup": 0.15,
        "book": 0.10,
        "cell phone": 0.15,
        "laptop": 0.15,

        # Animals
        "dog": 0.65,
        "cat": 0.55,
        "horse": 0.70,
        "cow": 0.70,
        "sheep": 0.60,

        # Outdoor obstacles
        "traffic light": 0.45,
        "stop sign": 0.60,
        "fire hydrant": 0.60,

        # Sports / miscellaneous
        "sports ball": 0.20,
        "skateboard": 0.40,
        "surfboard": 0.30,

        # Kitchen / household
        "refrigerator": 0.70,
        "microwave": 0.30,
        "oven": 0.50,
        "sink": 0.45,
        "toilet": 0.35,

        # Default environmental objects
        "potted plant": 0.25,
        "tv": 0.20,
        "keyboard": 0.10,
        "mouse": 0.10,
    }

    # --------------------------------------------------------
    # Human / vehicle overrides
    # --------------------------------------------------------

    HUMAN_CLASSES = {
        "person",
    }

    VEHICLE_CLASSES = {
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "train",
    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        emergency_threshold: float = EMERGENCY_THRESHOLD,
        danger_threshold: float = DANGER_THRESHOLD,
        caution_threshold: float = CAUTION_THRESHOLD,
        aware_threshold: float = AWARE_THRESHOLD,
    ) -> None:

        self.emergency_threshold = float(
            emergency_threshold
        )

        self.danger_threshold = float(
            danger_threshold
        )

        self.caution_threshold = float(
            caution_threshold
        )

        self.aware_threshold = float(
            aware_threshold
        )

        self.loaded = True
        self.track_history: List[Dict[str, Any]] = []
        self.object_tracks: Dict[str, Dict[str, Any]] = {}
        self._next_track_num: int = 1
        self.frame_index: int = 0
        self.last_action: str = "FORWARD"
        self.candidate_action: str = "FORWARD"
        self.action_vote_count: int = 0
        self.last_direction: str = "CENTER"
        self.direction_vote_count: int = 0
        self.turn_state: str = "NONE"
        self.step_length_m: float = 0.75
        self.safety_margin_m: float = 0.75
        self.max_guidance_distance_m: float = 6.0
        self.last_steps: int = 0

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> Dict[str, Any]:
        """
        Return engine status.
        """

        return {
            "loaded": True,
            "engine": "DecisionEngine",
            "emergency_threshold":
                self.emergency_threshold,
            "danger_threshold":
                self.danger_threshold,
            "caution_threshold":
                self.caution_threshold,
            "aware_threshold":
                self.aware_threshold,
        }

    # ========================================================
    # NUMERIC HELPERS
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to float.
        """

        try:
            result = float(value)

            if result != result:
                return default

            return result

        except Exception:
            return default

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """
        Clamp a number between minimum and maximum.
        """

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # ========================================================
    # WALKING STEPS & MOVEMENT DISTANCE (PHASE 10 & 11)
    # ========================================================

    @classmethod
    def estimate_walking_steps(
        cls,
        distance_m: Any,
        step_length_m: Any = 0.75,
    ) -> int:
        """
        Convert distance in meters to approximate walking steps.
        Bounded: [1, 8] for positive finite distance, 8 for +inf, 0 for <= 0 or NaN/invalid.
        """
        try:
            d = float(distance_m)
            if math.isnan(d) or d <= 0.0:
                return 0
            if math.isinf(d):
                return 8 if d > 0 else 0

            sl = 0.75
            if step_length_m is not None:
                try:
                    cand_sl = float(step_length_m)
                    if math.isfinite(cand_sl) and 0.35 <= cand_sl <= 2.5:
                        sl = cand_sl
                except Exception:
                    sl = 0.75

            steps = int(round(d / sl))
            return max(1, min(8, steps))
        except Exception:
            return 0

    def estimate_movement_distance(
        self,
        distance_m: Any,
        distance_category: str = "NEAR",
        action: str = "FORWARD",
        free_path: Optional[Dict[str, Any]] = None,
        safety_margin_m: Optional[float] = None,
        max_guidance_distance_m: Optional[float] = None,
    ) -> Tuple[float, str, str]:
        """
        Estimate safe guidance movement distance with safety margin.
        Returns: (movement_distance_m, step_confidence, step_reason)
        """
        try:
            d = float(distance_m)
        except Exception:
            d = 0.0

        cat = str(distance_category).upper() if distance_category else ""
        act = str(action).upper() if action else ""

        if not math.isfinite(d) or d <= 0.0 or cat == "UNKNOWN_DISTANCE":
            return 0.0, "LOW", "Unknown obstacle distance; exercise caution"

        if act in ("STOP", "TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT"):
            return 0.0, "HIGH", "No forward movement during stop/turn"

        sm = float(safety_margin_m) if safety_margin_m is not None and safety_margin_m >= 0 else self.safety_margin_m
        max_d = float(max_guidance_distance_m) if max_guidance_distance_m is not None and max_guidance_distance_m > 0 else self.max_guidance_distance_m

        if d <= sm:
            return 0.0, "HIGH", "Obstacle within safety margin; stop forward movement"

        eff = max(0.0, d - sm)
        if act == "SLOW_DOWN":
            eff = max(0.5, eff * 0.5)
            return round(min(max_d, eff), 2), "MEDIUM", "Cautionary forward movement at reduced speed"

        return round(min(max_d, eff), 2), "HIGH", "Forward movement safe"

    # ========================================================
    # BOUNDING BOX
    # ========================================================

    @classmethod
    def _bbox(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Normalize bounding box.
        """

        bbox = obj.get(
            "bbox",
            {},
        )

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return {
                "x1": cls._safe_float(bbox[0]),
                "y1": cls._safe_float(bbox[1]),
                "x2": cls._safe_float(bbox[2]),
                "y2": cls._safe_float(bbox[3]),
            }

        if not isinstance(
            bbox,
            dict,
        ):
            bbox = {}

        return {
            "x1": cls._safe_float(
                bbox.get("x1", 0)
            ),
            "y1": cls._safe_float(
                bbox.get("y1", 0)
            ),
            "x2": cls._safe_float(
                bbox.get("x2", 0)
            ),
            "y2": cls._safe_float(
                bbox.get("y2", 0)
            ),
        }

    # ========================================================
    # IOU
    # ========================================================

    @classmethod
    def calculate_iou(
        cls,
        box_a: Dict[str, Any],
        box_b: Dict[str, Any],
    ) -> float:
        """
        Calculate Intersection over Union.

        IoU is used to identify duplicate detections.
        """

        try:

            ax1 = cls._safe_float(
                box_a.get("x1", 0)
            )
            ay1 = cls._safe_float(
                box_a.get("y1", 0)
            )
            ax2 = cls._safe_float(
                box_a.get("x2", 0)
            )
            ay2 = cls._safe_float(
                box_a.get("y2", 0)
            )

            bx1 = cls._safe_float(
                box_b.get("x1", 0)
            )
            by1 = cls._safe_float(
                box_b.get("y1", 0)
            )
            bx2 = cls._safe_float(
                box_b.get("x2", 0)
            )
            by2 = cls._safe_float(
                box_b.get("y2", 0)
            )

            intersection_x1 = max(
                ax1,
                bx1,
            )

            intersection_y1 = max(
                ay1,
                by1,
            )

            intersection_x2 = min(
                ax2,
                bx2,
            )

            intersection_y2 = min(
                ay2,
                by2,
            )

            intersection_width = max(
                0.0,
                intersection_x2
                - intersection_x1,
            )

            intersection_height = max(
                0.0,
                intersection_y2
                - intersection_y1,
            )

            intersection_area = (
                intersection_width
                * intersection_height
            )

            area_a = max(
                0.0,
                ax2 - ax1,
            ) * max(
                0.0,
                ay2 - ay1,
            )

            area_b = max(
                0.0,
                bx2 - bx1,
            ) * max(
                0.0,
                by2 - by1,
            )

            union_area = (
                area_a
                + area_b
                - intersection_area
            )

            if union_area <= 0:
                return 0.0

            return cls._clamp(
                intersection_area
                / union_area
            )

        except Exception:
            return 0.0

    # ========================================================
    # DUPLICATE FILTER
    # ========================================================

    def remove_duplicates(
        self,
        objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate detections.

        Two detections of the same class with high IoU
        are considered duplicates.

        The higher-confidence detection is retained.
        """

        if not objects:
            return []

        sorted_objects = sorted(
            objects,
            key=lambda item:
                self._safe_float(
                    item.get(
                        "confidence",
                        0.0,
                    )
                ),
            reverse=True,
        )

        kept: List[Dict[str, Any]] = []

        for current in sorted_objects:

            current_class = str(
                current.get(
                    "class_name",
                    current.get(
                        "label",
                        "unknown",
                    ),
                )
            ).lower().strip()

            current_box = self._bbox(
                current
            )

            duplicate = False

            for existing in kept:

                existing_class = str(
                    existing.get(
                        "class_name",
                        existing.get(
                            "label",
                            "unknown",
                        ),
                    )
                ).lower().strip()

                if (
                    current_class
                    != existing_class
                ):
                    continue

                existing_box = self._bbox(
                    existing
                )

                iou = self.calculate_iou(
                    current_box,
                    existing_box,
                )

                if (
                    iou
                    >= self.IOU_THRESHOLD
                ):
                    duplicate = True
                    break

            if not duplicate:
                kept.append(
                    current
                )

        return kept

    # ========================================================
    # OBJECT PRIORITY
    # ========================================================

    def get_priority(
        self,
        class_name: str,
    ) -> float:
        """
        Return object priority.

        Humans and vehicles receive explicit priority
        overrides because they are especially important
        for an assistive navigation system.
        """

        name = str(
            class_name
            or "unknown"
        ).lower().strip()

        if name in self.HUMAN_CLASSES:
            return 1.0

        if name in self.VEHICLE_CLASSES:
            return 0.95

        return self._clamp(
            self.OBJECT_PRIORITIES.get(
                name,
                0.30,
            )
        )

    # ========================================================
    # POSITION
    # ========================================================

    def get_position(
        self,
        bbox: Dict[str, Any],
        frame_width: float,
    ) -> str:
        """
        Determine whether an object is:

            LEFT
            CENTER
            RIGHT
        """

        width = self._safe_float(
            frame_width,
            1.0,
        )

        if width <= 0:
            width = 1.0

        x1 = self._safe_float(
            bbox.get("x1", 0)
        )

        x2 = self._safe_float(
            bbox.get("x2", 0)
        )

        center_x = (
            x1 + x2
        ) / 2.0

        normalized_x = (
            center_x
            / width
        )

        if normalized_x < 0.35:
            return "LEFT"

        if normalized_x > 0.65:
            return "RIGHT"

        return "CENTER"

    # ========================================================
    # DEPTH NORMALIZATION
    # ========================================================

    def normalize_depth(
        self,
        depth: Any,
    ) -> float:
        """
        Normalize relative depth.

        MiDaS output may differ depending on preprocessing.

        The returned value is interpreted as:

            0.0 -> very close
            1.0 -> relatively far
        """

        value = self._safe_float(
            depth,
            1.0,
        )

        return self._clamp(
            value
        )

    # ========================================================
    # DEPTH RISK
    # ========================================================

    def depth_risk(
        self,
        depth: float,
    ) -> float:
        """
        Convert relative depth into risk.

        Smaller depth means greater risk.
        """

        depth = self.normalize_depth(
            depth
        )

        risk = 1.0 - depth

        return self._clamp(
            risk
        )

    # ========================================================
    # OBJECT RISK
    # ========================================================

    def calculate_risk(
        self,
        confidence: float,
        depth: float,
        priority: float,
        position: str,
        bbox: Optional[Dict[str, Any]] = None,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
        distance_m: Optional[float] = None,
    ) -> float:
        """
        Calculate object risk.

        Risk considers:
            distance (metric proximity)
            relative depth (MiDaS)
            confidence
            object priority
            position (CENTER elevated)
            object size / bbox
        """

        confidence = self._clamp(
            self._safe_float(
                confidence
            )
        )

        depth_risk = self.depth_risk(
            depth
        )

        priority = self._clamp(
            self._safe_float(
                priority
            )
        )

        # ----------------------------------------------------
        # Proximity risk: blend relative depth with distance
        # ----------------------------------------------------
        if distance_m is not None:
            dist_val = self._safe_float(distance_m)
            if dist_val > 0.0 and math.isfinite(dist_val):
                # 0.4m or less = 1.0 (imminent collision), 5.0m or more = 0.0
                dist_risk = self._clamp(1.0 - (dist_val - 0.4) / 4.6)
                prox_risk = 0.50 * depth_risk + 0.50 * dist_risk
            else:
                prox_risk = depth_risk
        else:
            prox_risk = depth_risk

        # ----------------------------------------------------
        # Base risk
        # ----------------------------------------------------
        risk = (
            0.40 * prox_risk
            + 0.30 * priority
            + 0.15 * confidence
        )

        # ----------------------------------------------------
        # Position danger
        # ----------------------------------------------------
        pos_upper = str(position).upper().strip()
        if pos_upper == "CENTER":
            risk += 0.12
        elif pos_upper in {
            "LEFT",
            "RIGHT",
        }:
            risk += 0.03

        # ----------------------------------------------------
        # Close distance bonus (< 1.2m)
        # ----------------------------------------------------
        if distance_m is not None:
            dist_val = self._safe_float(distance_m)
            if 0.0 < dist_val <= 0.8:
                risk += 0.15
            elif 0.8 < dist_val <= 1.2:
                risk += 0.08

        # ----------------------------------------------------
        # Bounding-box size
        # ----------------------------------------------------
        if (
            bbox is not None
            and frame_width
            and frame_height
        ):
            width = max(
                0.0,
                self._safe_float(
                    bbox.get("x2", 0)
                )
                - self._safe_float(
                    bbox.get("x1", 0)
                ),
            )

            height = max(
                0.0,
                self._safe_float(
                    bbox.get("y2", 0)
                )
                - self._safe_float(
                    bbox.get("y1", 0)
                ),
            )

            frame_area = (
                self._safe_float(
                    frame_width
                )
                * self._safe_float(
                    frame_height
                )
            )

            object_area = (
                width * height
            )

            if frame_area > 0:
                area_ratio = (
                    object_area
                    / frame_area
                )

                # Large objects are more likely
                # to represent immediate obstacles.
                size_bonus = self._clamp(
                    area_ratio * 0.30
                )

                risk += size_bonus

        return round(
            self._clamp(risk),
            4,
        )

    # ========================================================
    # OBJECT NORMALIZATION
    # ========================================================

    def normalize_objects(
        self,
        objects: Any,
        frame_width: float,
        frame_height: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Normalize detector output into the common format.
        """

        if not isinstance(
            objects,
            list,
        ):
            return []

        normalized: List[
            Dict[str, Any]
        ] = []

        for idx, item in enumerate(objects):

            if not isinstance(
                item,
                dict,
            ):
                continue

            class_name = str(
                item.get(
                    "class_name",
                    item.get(
                        "label",
                        "unknown",
                    ),
                )
            ).lower().strip()

            raw_conf = item.get("confidence", None)
            if raw_conf is None:
                confidence = 0.85
            else:
                confidence = self._clamp(
                    self._safe_float(
                        raw_conf,
                        0.0,
                    )
                )

            if confidence < self.DEFAULT_CONFIDENCE:
                continue

            bbox = self._bbox(
                item
            )

            depth = self.normalize_depth(
                item.get(
                    "depth",
                    1.0,
                )
            )

            priority = self.get_priority(
                class_name
            )

            if "position" in item and item["position"]:
                position = str(item["position"]).upper().strip()
            else:
                position = self.get_position(
                    bbox,
                    frame_width,
                )

            dist_info = self.estimate_distance_and_steps(
                depth=depth,
                bbox=bbox,
                class_name=class_name,
                confidence=confidence,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            if "distance_m" in item and item["distance_m"] is not None:
                cand_d = self._safe_float(item["distance_m"])
                if math.isfinite(cand_d) and cand_d > 0:
                    dist_m = round(cand_d, 2)
                    dist_cat = self.get_distance_category(dist_m)
                    dist_conf = str(item.get("distance_confidence", dist_info.get("distance_confidence", "HIGH")))
                else:
                    dist_m = float(dist_info.get("estimated_distance_m", 2.5))
                    dist_cat = str(dist_info.get("distance_category", "NEAR"))
                    dist_conf = str(dist_info.get("distance_confidence", "MEDIUM"))
            else:
                dist_m = float(dist_info.get("estimated_distance_m", 2.5))
                dist_cat = str(dist_info.get("distance_category", "NEAR"))
                dist_conf = str(dist_info.get("distance_confidence", "MEDIUM"))

            if "risk_score" in item and item["risk_score"] is not None:
                risk = self._clamp(self._safe_float(item["risk_score"]))
            else:
                risk = self.calculate_risk(
                    confidence=confidence,
                    depth=depth,
                    priority=priority,
                    position=position,
                    bbox=bbox,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    distance_m=dist_m,
                )

            if risk >= 0.70 or (position == "CENTER" and dist_m < 2.5):
                nav_relevance = "HIGH"
            elif risk >= 0.40 or dist_m <= 3.5:
                nav_relevance = "MEDIUM"
            elif risk >= 0.15:
                nav_relevance = "LOW"
            else:
                nav_relevance = "NONE"

            raw_id = item.get("id", None)
            if raw_id is not None:
                try:
                    obj_id = int(raw_id)
                except Exception:
                    obj_id = idx + 1
            else:
                obj_id = idx + 1

            bx1 = self._safe_float(bbox.get("x1", 0.0))
            by1 = self._safe_float(bbox.get("y1", 0.0))
            bx2 = self._safe_float(bbox.get("x2", 0.0))
            by2 = self._safe_float(bbox.get("y2", 0.0))
            obj_cx = round((bx1 + bx2) / 2.0, 2)
            obj_cy = round((by1 + by2) / 2.0, 2)
            obj_w = round(max(0.0, bx2 - bx1), 2)
            obj_h = round(max(0.0, by2 - by1), 2)

            normalized.append(
                {
                    "id":
                        obj_id,

                    "class_name":
                        class_name,

                    "confidence":
                        round(
                            confidence,
                            4,
                        ),

                    "bbox":
                        bbox,

                    "center_x":
                        obj_cx,

                    "center_y":
                        obj_cy,

                    "width":
                        obj_w,

                    "height":
                        obj_h,

                    "position":
                        position,

                    "distance_m":
                        dist_m,

                    "distance_category":
                        dist_cat,

                    "distance_confidence":
                        dist_conf,

                    "priority":
                        round(
                            priority,
                            4,
                        ),

                    "risk_score":
                        risk,

                    "approaching":
                        False,

                    "is_approaching":
                        False,

                    "rapidly_approaching":
                        False,

                    "is_rapidly_approaching":
                        False,

                    "approach_rate":
                        0.0,

                    "approach_rate_mps":
                        0.0,

                    "approach_score":
                        0.0,

                    "motion_state":
                        "UNKNOWN_MOTION",

                    "time_to_collision_s":
                        None,

                    "navigation_relevance":
                        nav_relevance,

                    # Backward compatibility fields
                    "depth":
                        round(
                            depth,
                            4,
                        ),

                    "estimated_distance_m":
                        dist_m,

                    "estimated_steps":
                        dist_info.get("estimated_steps", 0),
                }
            )

        return normalized

    # ========================================================
    # UNKNOWN OBJECT NORMALIZATION
    # ========================================================

    def normalize_unknown_objects(
        self,
        unknown_candidates: Any,
        frame_width: float,
        frame_height: Optional[float] = None,
        known_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Normalize unknown obstacle candidates into the canonical Phase 2/4 contract.

        Each entry satisfies:
            id: int (1001+)
            position: "LEFT" | "CENTER" | "RIGHT"
            bbox: {x1, y1, x2, y2}
            distance_m: float
            distance_category: "VERY_CLOSE" | "NEAR" | "FAR" | "UNKNOWN_DISTANCE"
            distance_confidence: "HIGH" | "MEDIUM" | "LOW"
            risk_score: float
            navigation_relevance: "HIGH" | "MEDIUM" | "LOW" | "NONE"
            confidence: float (candidate physical obstacle confidence)
        """
        if not isinstance(unknown_candidates, list):
            return []

        normalized: List[Dict[str, Any]] = []

        for idx, item in enumerate(unknown_candidates):
            if not isinstance(item, dict):
                continue

            bbox = self._bbox(item)

            # Suppress candidate if it overlaps significantly with an existing known object
            if known_objects:
                suppressed = False
                for k_obj in known_objects:
                    k_box = self._bbox(k_obj)
                    if self.calculate_iou(bbox, k_box) >= 0.30:
                        suppressed = True
                        break
                if suppressed:
                    continue

            depth = self.normalize_depth(
                item.get("cand_depth", item.get("depth", 0.5))
            )
            confidence = self._clamp(
                self._safe_float(item.get("confidence", 0.50)),
                0.40,
                0.85,
            )

            cand_id = int(item.get("id", 1001 + idx))
            position = self.get_position(bbox, frame_width)

            # Conservative priority for unknown physical obstacle
            priority = 0.60

            dist_info = self.estimate_distance_and_steps(
                depth=depth,
                bbox=bbox,
                class_name="unknown",
                confidence=confidence,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            dist_m = float(dist_info.get("estimated_distance_m", 2.5))
            dist_cat = str(dist_info.get("distance_category", "NEAR"))
            dist_conf = str(dist_info.get("distance_confidence", "MEDIUM"))
            if not math.isfinite(dist_m) or dist_m <= 0:
                dist_cat = "UNKNOWN_DISTANCE"
                dist_conf = "LOW"

            risk = self.calculate_risk(
                confidence=confidence,
                depth=depth,
                priority=priority,
                position=position,
                bbox=bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                distance_m=dist_m,
            )

            if risk >= 0.70 or (position == "CENTER" and dist_m < 2.5):
                nav_relevance = "HIGH"
            elif risk >= 0.40 or dist_m <= 3.5:
                nav_relevance = "MEDIUM"
            elif risk >= 0.15:
                nav_relevance = "LOW"
            else:
                nav_relevance = "NONE"

            bx1 = self._safe_float(bbox.get("x1", 0.0))
            by1 = self._safe_float(bbox.get("y1", 0.0))
            bx2 = self._safe_float(bbox.get("x2", 0.0))
            by2 = self._safe_float(bbox.get("y2", 0.0))
            cand_cx = round((bx1 + bx2) / 2.0, 2)
            cand_cy = round((by1 + by2) / 2.0, 2)
            cand_w = round(max(0.0, bx2 - bx1), 2)
            cand_h = round(max(0.0, by2 - by1), 2)

            normalized.append(
                {
                    "id": cand_id,
                    "class_name": "unknown_obstacle",
                    "position": position,
                    "bbox": bbox,
                    "center_x": cand_cx,
                    "center_y": cand_cy,
                    "width": cand_w,
                    "height": cand_h,
                    "distance_m": round(dist_m, 2),
                    "distance_category": dist_cat,
                    "distance_confidence": dist_conf,
                    "risk_score": risk,
                    "navigation_relevance": nav_relevance,
                    "confidence": round(confidence, 4),
                    # Internal helper fields for DecisionEngine integration
                    "depth": round(depth, 4),
                    "priority": priority,
                    "estimated_distance_m": round(dist_m, 2),
                    "estimated_steps": int(dist_info.get("estimated_steps", 0)),
                }
            )

        return normalized

    # ========================================================
    # DISTANCE & STEP ESTIMATION
    # ========================================================

    @classmethod
    def estimate_distance_and_steps(
        cls,
        depth: float,
        bbox: Any,
        class_name: str,
        confidence: float,
        frame_width: Optional[float] = 640.0,
        frame_height: Optional[float] = 480.0,
        step_length_m: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Multi-cue approximate distance estimator for SmartVisionAI.
        
        Combines:
        1. MiDaS normalized relative depth (0.0=close, 1.0=far)
        2. Bounding box relative height & area geometry
        3. Object class typical physical scale
        4. Vertical ground plane proximity
        """
        # Validate depth input
        is_valid_depth = True
        try:
            depth_val = float(depth)
            if math.isnan(depth_val) or math.isinf(depth_val) or depth_val < 0.0:
                is_valid_depth = False
        except (TypeError, ValueError):
            is_valid_depth = False

        if not is_valid_depth:
            return {
                "estimated_distance_m": 2.5,
                "distance_m": 2.5,
                "distance_category": "UNKNOWN_DISTANCE",
                "distance_confidence": "LOW",
                "estimated_steps": 3,
            }

        depth_val = cls._clamp(depth_val)
        # Base relative distance curve (approx 0.5m to 6.0m)
        dist_base = 0.5 + depth_val * 5.5

        # Bounding box geometry
        if isinstance(bbox, dict):
            x1 = cls._safe_float(bbox.get("x1", 0.0))
            y1 = cls._safe_float(bbox.get("y1", 0.0))
            x2 = cls._safe_float(bbox.get("x2", 0.0))
            y2 = cls._safe_float(bbox.get("y2", 0.0))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1 = cls._safe_float(bbox[0])
            y1 = cls._safe_float(bbox[1])
            x2 = cls._safe_float(bbox[2])
            y2 = cls._safe_float(bbox[3])
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        frame_w = max(1.0, float(frame_width or 640.0))
        frame_h = max(1.0, float(frame_height or 480.0))

        box_h_ratio = cls._clamp(box_h / frame_h)
        box_area_ratio = cls._clamp((box_w * box_h) / (frame_w * frame_h))
        bottom_ratio = cls._clamp(y2 / frame_h)

        # Scale by class
        cname = str(class_name or "").lower().strip()
        if cname in cls.HUMAN_CLASSES:
            geometry_dist = max(0.5, 3.5 * (1.0 - box_h_ratio * 0.85))
            dist = 0.65 * dist_base + 0.35 * geometry_dist
        elif cname in cls.VEHICLE_CLASSES:
            geometry_dist = max(0.8, 5.0 * (1.0 - box_area_ratio * 0.90))
            dist = 0.60 * dist_base + 0.40 * geometry_dist
        elif cname in {"dining table", "table", "chair", "couch", "bed", "bench"}:
            geometry_dist = max(0.5, 4.0 * (1.0 - box_h_ratio * 0.75))
            dist = 0.70 * dist_base + 0.30 * geometry_dist
        else:
            dist = dist_base

        if bottom_ratio > 0.85 and box_h_ratio > 0.25:
            dist = min(dist, dist * 0.85)

        dist = max(0.5, min(7.0, dist))
        estimated_distance_m = round(dist, 2)

        conf = cls._safe_float(confidence, 0.5)
        if conf >= 0.65 and box_area_ratio >= 0.04 and 0.15 <= depth_val <= 0.85:
            confidence_level = "HIGH"
        elif conf >= 0.45:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        if estimated_distance_m < 1.4:
            distance_category = "VERY_CLOSE"
        elif estimated_distance_m <= 3.5:
            distance_category = "NEAR"
        else:
            distance_category = "FAR"

        raw_steps = round(estimated_distance_m / max(0.1, step_length_m))
        estimated_steps = max(1, min(8, raw_steps))

        return {
            "estimated_distance_m": estimated_distance_m,
            "distance_m": estimated_distance_m,
            "distance_category": distance_category,
            "distance_confidence": confidence_level,
            "estimated_steps": estimated_steps,
        }

    @classmethod
    def get_distance_category(cls, dist_m: float) -> str:
        """
        Map a metric distance to canonical category: VERY_CLOSE, NEAR, FAR, UNKNOWN_DISTANCE.
        """
        try:
            d = float(dist_m)
            if not math.isfinite(d) or d <= 0:
                return "UNKNOWN_DISTANCE"
            if d < 1.4:
                return "VERY_CLOSE"
            elif d <= 3.5:
                return "NEAR"
            else:
                return "FAR"
        except Exception:
            return "UNKNOWN_DISTANCE"

    # ========================================================
    # MULTI-OBJECT TRACKING & TEMPORAL DISTANCE SMOOTHING
    # ========================================================

    def track_and_smooth_distances(
        self,
        objects: List[Dict[str, Any]],
        unknown_objects: List[Dict[str, Any]],
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> None:
        """
        Multi-object temporal tracking and adaptive distance smoothing.

        Performs:
        1. Multi-object track association per detection via IoU + centroid proximity.
        2. Adaptive EMA distance smoothing per track to stabilize noise without lag.
        3. SAFETY-CRITICAL BYPASS: alpha = 1.0 (immediate update with 0 lag) whenever
           obstacle is dangerously close (<= 1.2m) or makes a sudden rapid approach.
        4. Independent per-object tracks so distance histories never cross-pollinate.
        5. Track retention across intermittent frame drops (up to 3 frames) without crashing.
        """
        self.frame_index += 1
        frame_w = max(1.0, float(frame_width or 640.0))
        frame_h = max(1.0, float(frame_height or 480.0))

        # Flatten list of item references with origin tag
        all_items: List[tuple[Dict[str, Any], bool]] = []
        for obj in objects:
            all_items.append((obj, False))
        for unk in unknown_objects:
            all_items.append((unk, True))

        # Extract features for association
        detections_info = []
        for idx, (item, is_unknown) in enumerate(all_items):
            box = item.get("bbox", {})
            if isinstance(box, dict):
                bx1 = self._safe_float(box.get("x1", 0.0))
                by1 = self._safe_float(box.get("y1", 0.0))
                bx2 = self._safe_float(box.get("x2", 0.0))
                by2 = self._safe_float(box.get("y2", 0.0))
            elif isinstance(box, (list, tuple)) and len(box) >= 4:
                bx1 = self._safe_float(box[0])
                by1 = self._safe_float(box[1])
                bx2 = self._safe_float(box[2])
                by2 = self._safe_float(box[3])
            else:
                bx1, by1, bx2, by2 = 0.0, 0.0, 0.0, 0.0

            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            cname = str(item.get("class_name", "unknown" if is_unknown else "")).lower().strip()
            raw_dist = self._safe_float(
                item.get("distance_m", item.get("estimated_distance_m", 2.5)),
                2.5,
            )

            detections_info.append({
                "index": idx,
                "item": item,
                "is_unknown": is_unknown,
                "class_name": cname,
                "box": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
                "cx": cx,
                "cy": cy,
                "raw_dist": raw_dist,
                "item_id": item.get("id"),
            })

        # Pairwise association scoring
        match_candidates = []
        for det in detections_info:
            for track_id, trk in self.object_tracks.items():
                if det["class_name"] != trk["class_name"]:
                    continue

                det_item_id = det.get("item_id")
                trk_det_id = trk.get("detection_id")
                if det_item_id is not None and trk_det_id is not None:
                    if det_item_id == trk_det_id:
                        match_candidates.append((2.0, det["index"], track_id))
                        continue
                    else:
                        continue

                b1, b2 = det["box"], trk["box"]
                ix1 = max(b1["x1"], b2["x1"])
                iy1 = max(b1["y1"], b2["y1"])
                ix2 = min(b1["x2"], b2["x2"])
                iy2 = min(b1["y2"], b2["y2"])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                a1 = max(1.0, (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"]))
                a2 = max(1.0, (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"]))
                union = a1 + a2 - inter
                iou = inter / union if union > 0 else 0.0

                norm_cdist = math.hypot((det["cx"] - trk["cx"]) / frame_w, (det["cy"] - trk["cy"]) / frame_h)
                px_cdist = math.hypot(det["cx"] - trk["cx"], det["cy"] - trk["cy"])

                if iou >= 0.18 or norm_cdist <= 0.22 or px_cdist <= 140.0:
                    score = iou * 0.70 + (1.0 - min(1.0, norm_cdist)) * 0.30
                    match_candidates.append((score, det["index"], track_id))

        # Greedy match assignment
        match_candidates.sort(key=lambda x: x[0], reverse=True)
        assigned_dets = set()
        assigned_tracks = set()
        matched_pairs = []

        for score, d_idx, trk_id in match_candidates:
            if d_idx in assigned_dets or trk_id in assigned_tracks:
                continue
            assigned_dets.add(d_idx)
            assigned_tracks.add(trk_id)
            matched_pairs.append((d_idx, trk_id))

        # Process matched tracks
        for d_idx, trk_id in matched_pairs:
            det = detections_info[d_idx]
            item = det["item"]
            trk = self.object_tracks[trk_id]
            raw_dist = det["raw_dist"]
            prev_smoothed = trk["smoothed_distance_m"]
            prev_raw = trk["raw_distance_m"]

            # Adaptive smoothing factor
            # Safety bypass: immediate update for close dangerous obstacles or rapid approaches
            if raw_dist <= 1.2:
                alpha = 1.0
            elif raw_dist <= 1.4 and prev_smoothed > 1.8:
                alpha = 1.0
            elif raw_dist < prev_smoothed - 0.50:
                alpha = 0.90
            elif raw_dist < prev_raw - 0.12:
                alpha = 0.65
            else:
                alpha = 0.35

            smoothed_dist = round(alpha * raw_dist + (1.0 - alpha) * prev_smoothed, 2)
            smoothed_dist = max(0.5, min(7.0, smoothed_dist))

            stability = min(15, trk["stability"] + 1)

            trk["raw_history"].append(raw_dist)
            if len(trk["raw_history"]) > 6:
                trk["raw_history"].pop(0)

            trk["smoothed_history"].append(smoothed_dist)
            if len(trk["smoothed_history"]) > 6:
                trk["smoothed_history"].pop(0)

            # Timestamp & delta_time calculation
            curr_ts = time.time()
            prev_ts = trk.get("current_timestamp", None)
            trk["previous_timestamp"] = prev_ts
            trk["current_timestamp"] = curr_ts
            if prev_ts is not None:
                dt = curr_ts - prev_ts
            else:
                dt = 0.70
            if dt <= 0.05 or dt > 2.0 or not math.isfinite(dt):
                dt = 0.70

            # Distance history tracking
            trk["previous_distance_m"] = prev_smoothed
            trk["current_distance_m"] = smoothed_dist
            trk["distance_history"] = list(trk["smoothed_history"])

            # Compute approach rate: delta distance / delta time
            rate = (prev_smoothed - smoothed_dist) / dt
            sm_hist = trk["smoothed_history"]
            raw_h = trk["raw_history"]

            if len(sm_hist) >= 3:
                sm_total = sm_hist[-3] - sm_hist[-1]
                sm_window_rate = sm_total / (2.0 * dt)
                if sm_total > 0.15:
                    rate = max(rate, sm_window_rate)
            elif len(raw_h) >= 3:
                raw_total = raw_h[-3] - raw_h[-1]
                raw_window_rate = raw_total / (2.0 * dt)
                if raw_total > 0.20:
                    rate = max(rate, raw_window_rate)

            if not math.isfinite(rate):
                rate = 0.0
            approach_rate_mps = round(rate, 2)

            # Trend consistency & noise filtering
            if len(sm_hist) >= 2:
                pairwise_diffs = [sm_hist[i - 1] - sm_hist[i] for i in range(1, len(sm_hist))]
                descent_count = sum(1 for d in pairwise_diffs if d > 0.02)
                trend_consistency = descent_count / float(len(pairwise_diffs))
                total_descent = sm_hist[0] - sm_hist[-1]
            else:
                trend_consistency = 0.0
                total_descent = 0.0

            consecutive_app = trk.get("consecutive_approaching_frames", 0)
            consecutive_stat = trk.get("consecutive_stationary_frames", 0)
            consecutive_rec = trk.get("consecutive_receding_frames", 0)

            approaching = False
            rapidly_approaching = False

            # Motion state classification with consecutive-frame confirmation
            # 1. Safety bypass: sudden close drop into danger zone
            if raw_dist <= 1.4 and (prev_smoothed - raw_dist) >= 0.50:
                approaching = True
                rapidly_approaching = True
                motion_state = "RAPIDLY_APPROACHING"
                consecutive_app += 1
                consecutive_stat = 0
                consecutive_rec = 0
                approach_rate_mps = max(0.50, round((prev_smoothed - raw_dist) / dt, 2))
            elif raw_dist <= 1.2 and (prev_smoothed - raw_dist) >= 0.25:
                approaching = True
                consecutive_app += 1
                consecutive_stat = 0
                consecutive_rec = 0
                if approach_rate_mps >= 0.50:
                    rapidly_approaching = True
                    motion_state = "RAPIDLY_APPROACHING"
                else:
                    motion_state = "APPROACHING"
            # 2. Rapid approach: rate >= 0.50 m/s
            elif approach_rate_mps >= 0.50:
                consecutive_app += 1
                consecutive_stat = 0
                consecutive_rec = 0
                if consecutive_app >= 2 or total_descent >= 0.45 or (prev_smoothed - smoothed_dist) >= 0.35:
                    rapidly_approaching = True
                    approaching = True
                    motion_state = "RAPIDLY_APPROACHING"
                else:
                    approaching = False
                    rapidly_approaching = False
                    motion_state = "UNKNOWN_MOTION"
            # 3. Moderate approach: 0.15 <= rate < 0.50 m/s
            elif approach_rate_mps >= 0.15:
                consecutive_stat = 0
                consecutive_rec = 0
                if (prev_smoothed - smoothed_dist) >= 0.04 or (prev_raw - raw_dist) >= 0.04:
                    consecutive_app += 1
                else:
                    consecutive_app = 1

                if consecutive_app >= 2 and (trend_consistency >= 0.60 or total_descent >= 0.25):
                    approaching = True
                    rapidly_approaching = False
                    motion_state = "APPROACHING"
                else:
                    approaching = False
                    rapidly_approaching = False
                    motion_state = "UNKNOWN_MOTION"
            # 4. Receding: rate <= -0.15 m/s
            elif approach_rate_mps <= -0.15:
                consecutive_rec += 1
                consecutive_app = 0
                consecutive_stat = 0
                approaching = False
                rapidly_approaching = False
                motion_state = "RECEDING"
            # 5. Stationary: -0.15 < rate < 0.15 m/s
            else:
                consecutive_stat += 1
                consecutive_app = 0
                consecutive_rec = 0
                approaching = False
                rapidly_approaching = False
                motion_state = "STATIONARY"

            # Bounded approach score: [0.0, 1.0]
            if approaching or rapidly_approaching:
                rate_norm = min(1.0, max(0.0, approach_rate_mps / 1.0))
                consec_norm = min(1.0, consecutive_app / 3.0)
                trend_norm = max(0.0, min(1.0, trend_consistency))
                stab_norm = min(1.0, stability / 4.0)
                det_conf = float(det.get("confidence", item.get("confidence", 0.70)))
                conf_norm = max(0.0, min(1.0, det_conf))
                raw_score = (
                    0.35 * rate_norm +
                    0.25 * consec_norm +
                    0.20 * trend_norm +
                    0.10 * stab_norm +
                    0.10 * conf_norm
                )
                if rapidly_approaching:
                    raw_score = max(0.65, raw_score * 1.25)
                approach_score = round(max(0.0, min(1.0, raw_score)), 2)
            else:
                approach_score = 0.0

            # Time to collision (TTC) in seconds
            if smoothed_dist > 0.0 and approach_rate_mps > 0.05 and math.isfinite(smoothed_dist) and math.isfinite(approach_rate_mps):
                ttc = smoothed_dist / approach_rate_mps
                time_to_collision_s = round(min(60.0, max(0.1, ttc)), 1)
            else:
                time_to_collision_s = None

            trk["box"] = det["box"]
            trk["cx"] = det["cx"]
            trk["cy"] = det["cy"]
            trk["raw_distance_m"] = raw_dist
            trk["smoothed_distance_m"] = smoothed_dist
            trk["previous_distance_m"] = prev_smoothed
            trk["current_distance_m"] = smoothed_dist
            trk["distance_history"] = list(trk["smoothed_history"])
            trk["stability"] = stability
            trk["missed_frames"] = 0
            trk["last_seen_frame"] = self.frame_index
            trk["approaching"] = approaching
            trk["rapidly_approaching"] = rapidly_approaching
            trk["approach_rate"] = approach_rate_mps
            trk["approach_rate_mps"] = approach_rate_mps
            trk["approach_score"] = approach_score
            trk["motion_state"] = motion_state
            trk["consecutive_approaching_frames"] = consecutive_app
            trk["consecutive_stationary_frames"] = consecutive_stat
            trk["consecutive_receding_frames"] = consecutive_rec
            trk["time_to_collision_s"] = time_to_collision_s

            # Canonical categorization and confidence from smoothed distance
            if item.get("distance_category") == "UNKNOWN_DISTANCE":
                cat = "UNKNOWN_DISTANCE"
                conf = "LOW"
            elif smoothed_dist < 1.4:
                cat = "VERY_CLOSE"
                conf = "HIGH" if stability >= 2 and item.get("distance_confidence") != "LOW" else item.get("distance_confidence", "MEDIUM")
            elif smoothed_dist <= 3.5:
                cat = "NEAR"
                conf = "HIGH" if stability >= 2 and item.get("distance_confidence") != "LOW" else item.get("distance_confidence", "MEDIUM")
            else:
                cat = "FAR"
                conf = item.get("distance_confidence", "LOW")

            # Update item in-place
            item["distance_m"] = smoothed_dist
            item["estimated_distance_m"] = smoothed_dist
            item["distance_category"] = cat
            item["distance_confidence"] = conf
            item["track_id"] = trk_id
            item["approaching"] = approaching
            item["is_approaching"] = approaching
            item["rapidly_approaching"] = rapidly_approaching
            item["is_rapidly_approaching"] = rapidly_approaching
            item["approach_rate"] = approach_rate_mps
            item["approach_rate_mps"] = approach_rate_mps
            item["approach_score"] = approach_score
            item["motion_state"] = motion_state
            item["time_to_collision_s"] = time_to_collision_s
            item["obstacle_stability"] = stability
            item["consecutive_approaching_frames"] = consecutive_app
            item["consecutive_stationary_frames"] = consecutive_stat
            item["consecutive_receding_frames"] = consecutive_rec
            item["estimated_steps"] = max(1, min(8, round(smoothed_dist / 0.75)))

        # Process unmatched detections (initialize new tracks)
        for det in detections_info:
            if det["index"] in assigned_dets:
                continue
            item = det["item"]
            raw_dist = det["raw_dist"]
            trk_id = f"trk_{self._next_track_num}"
            self._next_track_num += 1
            curr_ts = time.time()

            self.object_tracks[trk_id] = {
                "track_id": trk_id,
                "detection_id": det.get("item_id"),
                "class_name": det["class_name"],
                "box": det["box"],
                "cx": det["cx"],
                "cy": det["cy"],
                "raw_distance_m": raw_dist,
                "smoothed_distance_m": raw_dist,
                "previous_distance_m": raw_dist,
                "current_distance_m": raw_dist,
                "raw_history": [raw_dist],
                "smoothed_history": [raw_dist],
                "distance_history": [raw_dist],
                "previous_timestamp": None,
                "current_timestamp": curr_ts,
                "stability": 1,
                "missed_frames": 0,
                "last_seen_frame": self.frame_index,
                "approaching": False,
                "rapidly_approaching": False,
                "approach_rate": 0.0,
                "approach_rate_mps": 0.0,
                "approach_score": 0.0,
                "motion_state": "UNKNOWN_MOTION",
                "consecutive_approaching_frames": 0,
                "consecutive_stationary_frames": 0,
                "consecutive_receding_frames": 0,
                "time_to_collision_s": None,
            }

            item["track_id"] = trk_id
            item["obstacle_stability"] = 1
            item["approaching"] = False
            item["is_approaching"] = False
            item["rapidly_approaching"] = False
            item["is_rapidly_approaching"] = False
            item["approach_rate"] = 0.0
            item["approach_rate_mps"] = 0.0
            item["approach_score"] = 0.0
            item["motion_state"] = "UNKNOWN_MOTION"
            item["time_to_collision_s"] = None
            item["consecutive_approaching_frames"] = 0
            item["consecutive_stationary_frames"] = 0
            item["consecutive_receding_frames"] = 0
            item["estimated_distance_m"] = raw_dist

        # Age out unmatched tracks (retain across up to 3 intermittent misses)
        stale_track_ids = []
        for trk_id, trk in self.object_tracks.items():
            if trk_id not in assigned_tracks:
                trk["missed_frames"] += 1
                if trk["missed_frames"] > 3:
                    stale_track_ids.append(trk_id)

        for trk_id in stale_track_ids:
            del self.object_tracks[trk_id]

    # ========================================================
    # PREDICTIVE THREAT SELECTION (PHASE 11)
    # ========================================================

    def compute_predictive_threat(
        self,
        objects: List[Dict[str, Any]],
        unknown_objects: Optional[List[Dict[str, Any]]] = None,
        primary_obstacle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Identify the highest predictive threat across known and unknown obstacles.
        Combines corridor priority (CENTER > sides), approach rate, approach score,
        proximity distance, and risk score.
        """
        all_candidates: List[Dict[str, Any]] = []
        if isinstance(objects, list):
            all_candidates.extend(objects)
        if isinstance(unknown_objects, list):
            all_candidates.extend(unknown_objects)

        approaching_candidates: List[Dict[str, Any]] = []
        for cand in all_candidates:
            if not isinstance(cand, dict):
                continue
            is_app = bool(cand.get("approaching", False) or cand.get("is_approaching", False) or cand.get("rapidly_approaching", False))
            motion = str(cand.get("motion_state", "")).upper()
            if is_app or motion in ("APPROACHING", "RAPIDLY_APPROACHING"):
                approaching_candidates.append(cand)

        if not approaching_candidates:
            p_id = primary_obstacle.get("track_id", primary_obstacle.get("id")) if primary_obstacle else None
            p_cls = str(primary_obstacle.get("class_name", "none")) if primary_obstacle else "none"
            p_pos = str(primary_obstacle.get("position", "NONE")).upper() if primary_obstacle else "NONE"
            p_dist = round(self._safe_float(primary_obstacle.get("distance_m", 0.0)), 2) if primary_obstacle else 0.0
            p_mot = str(primary_obstacle.get("motion_state", "STATIONARY")).upper() if primary_obstacle else "STATIONARY"
            return {
                "obstacle_id": p_id,
                "track_id": str(primary_obstacle.get("track_id", "")) if primary_obstacle else "",
                "class_name": p_cls,
                "position": p_pos,
                "distance_m": p_dist,
                "approach_rate_mps": 0.0,
                "approach_score": 0.0,
                "time_to_collision_s": None,
                "motion_state": p_mot,
                "threat_level": "NONE",
                "is_approaching": False,
                "warning_issued": False,
            }

        def threat_scorer(c: Dict[str, Any]) -> float:
            pos = str(c.get("position", "CENTER")).upper()
            corridor_weight = 1.35 if pos == "CENTER" else 0.80
            rate = self._safe_float(c.get("approach_rate", c.get("approach_rate_mps", 0.0)))
            dist = self._safe_float(c.get("distance_m", c.get("estimated_distance_m", 3.0)), 3.0)
            score = self._safe_float(c.get("approach_score", 0.0))
            risk = self._safe_float(c.get("risk_score", 0.0))
            is_rapid = bool(c.get("rapidly_approaching", False) or str(c.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING")

            rapid_bonus = 0.35 if is_rapid else 0.0
            dist_factor = max(0.0, 1.0 - min(1.0, dist / 6.0))
            rate_factor = min(1.0, max(0.0, rate / 1.5))

            return corridor_weight * (0.35 * score + 0.25 * rate_factor + 0.25 * dist_factor + 0.15 * risk + rapid_bonus)

        approaching_candidates.sort(key=threat_scorer, reverse=True)
        best = approaching_candidates[0]

        best_dist = round(self._safe_float(best.get("distance_m", best.get("estimated_distance_m", 0.0))), 2)
        raw_dist = round(self._safe_float(best.get("raw_distance_m", best_dist)), 2)
        eval_dist = min(best_dist, raw_dist) if raw_dist > 0 else best_dist
        best_rate = round(self._safe_float(best.get("approach_rate", best.get("approach_rate_mps", 0.0))), 2)
        best_score = round(self._safe_float(best.get("approach_score", 0.0)), 2)
        best_motion = str(best.get("motion_state", "APPROACHING")).upper()
        if best_motion not in ("STATIONARY", "APPROACHING", "RAPIDLY_APPROACHING", "RECEDING", "UNKNOWN_MOTION"):
            best_motion = "APPROACHING"

        best_pos = str(best.get("position", "CENTER")).upper()
        best_ttc = best.get("time_to_collision_s", None)
        ttc_val = float(best_ttc) if best_ttc is not None and math.isfinite(float(best_ttc)) else None

        # Determine threat level
        if best_pos == "CENTER" and (
            (best_motion == "RAPIDLY_APPROACHING" and (eval_dist <= 1.4 or (ttc_val is not None and ttc_val <= 2.0)))
            or (ttc_val is not None and ttc_val <= 2.0 and eval_dist <= 2.0)
        ):
            threat_lvl = "EMERGENCY"
        elif best_pos == "CENTER" and (
            (best_motion == "RAPIDLY_APPROACHING" and eval_dist <= 2.5)
            or (best_motion in ("APPROACHING", "RAPIDLY_APPROACHING") and (ttc_val is not None and ttc_val <= 3.0))
        ):
            threat_lvl = "DANGER"
        elif (best_pos == "CENTER" and eval_dist <= 2.2) or (ttc_val is not None and ttc_val <= 6.0) or (best_motion == "RAPIDLY_APPROACHING"):
            threat_lvl = "CAUTION"
        elif best_motion in ("APPROACHING", "RAPIDLY_APPROACHING"):
            threat_lvl = "AWARE"
        else:
            threat_lvl = "NONE"

        warning_issued = bool(threat_lvl in ("CAUTION", "DANGER", "EMERGENCY"))

        return {
            "obstacle_id": best.get("track_id", best.get("id")),
            "track_id": str(best.get("track_id", "trk_1")),
            "class_name": str(best.get("class_name", "obstacle")),
            "position": best_pos,
            "distance_m": best_dist,
            "approach_rate_mps": best_rate,
            "approach_score": best_score,
            "time_to_collision_s": best_ttc,
            "motion_state": best_motion,
            "threat_level": threat_lvl,
            "is_approaching": True,
            "warning_issued": warning_issued,
        }

    # ========================================================
    # TEMPORAL TRACKING & APPROACH DETECTION
    # ========================================================

    def update_tracking_and_approach(
        self,
        primary_obstacle: Optional[Dict[str, Any]],
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract primary obstacle tracking and approach status.
        Maintains backward compatibility for tracking_info and self.track_history.
        """
        if not primary_obstacle and not predictive_threat:
            self.track_history = []
            return {
                "approaching": False,
                "approach_rate": 0.0,
                "approach_score": 0.0,
                "rapidly_approaching": False,
                "motion_state": "UNKNOWN_MOTION",
                "time_to_collision_s": None,
                "obstacle_stability": 0,
            }

        ref = predictive_threat if predictive_threat else primary_obstacle
        approaching = bool(ref.get("approaching", False) or ref.get("motion_state") in ("APPROACHING", "RAPIDLY_APPROACHING"))
        rapidly_approaching = bool(ref.get("rapidly_approaching", False) or ref.get("motion_state") == "RAPIDLY_APPROACHING")
        approach_rate = float(ref.get("approach_rate_mps", ref.get("approach_rate", 0.0)))
        approach_score = float(ref.get("approach_score", 0.0))
        motion_state = str(ref.get("motion_state", "UNKNOWN_MOTION"))
        ttc = ref.get("time_to_collision_s", None)
        stability = int(primary_obstacle.get("obstacle_stability", 1) if primary_obstacle else 1)

        # Backward compatibility track history
        if primary_obstacle:
            record = {
                "class_name": str(primary_obstacle.get("class_name", "")),
                "estimated_distance_m": float(primary_obstacle.get("distance_m", primary_obstacle.get("estimated_distance_m", 2.0))),
                "stability": stability,
                "approaching": approaching,
                "approach_rate": approach_rate,
            }
            self.track_history.append(record)
            if len(self.track_history) > 5:
                self.track_history.pop(0)

        return {
            "approaching": approaching,
            "approach_rate": approach_rate,
            "approach_score": approach_score,
            "rapidly_approaching": rapidly_approaching,
            "motion_state": motion_state,
            "time_to_collision_s": ttc,
            "obstacle_stability": stability,
        }

    # ========================================================
    # FREE PATH & SAFE WALKING CORRIDOR (PHASE 7)
    # ========================================================

    @staticmethod
    def _compute_interval_coverage(
        intervals: List[Tuple[float, float]],
        span_start: float,
        span_end: float,
    ) -> float:
        """
        Compute normalized union coverage of 1D intervals within [span_start, span_end].
        """
        if not intervals or span_end <= span_start:
            return 0.0

        clamped: List[Tuple[float, float]] = []
        for s, e in intervals:
            cs = max(span_start, s)
            ce = min(span_end, e)
            if ce > cs:
                clamped.append((cs, ce))

        if not clamped:
            return 0.0

        clamped.sort(key=lambda x: x[0])
        merged = [clamped[0]]
        for cur in clamped[1:]:
            prev_s, prev_e = merged[-1]
            if cur[0] <= prev_e:
                merged[-1] = (prev_s, max(prev_e, cur[1]))
            else:
                merged.append(cur)

        total_len = sum(e - s for s, e in merged)
        return min(1.0, max(0.0, total_len / (span_end - span_start)))

    def analyze_free_path(
        self,
        scene_depth: Optional[Dict[str, Any]] = None,
        objects: Optional[List[Dict[str, Any]]] = None,
        unknown_objects: Optional[List[Dict[str, Any]]] = None,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> Dict[str, Any]:
        """
        Phase 7 Free-Path Analysis and Safe Walking Corridor Detection.

        Combines:
        1. Monocular relative depth map regional baselines (LEFT, CENTER, RIGHT)
        2. Detected object bounding boxes (known YOLO + Phase 4 genuine unknowns)
        3. Phase 5 smoothed approximate distances and distance categories
        4. Spatial corridor coverage and horizontal overlap
        5. Walking-relevant vertical ground proximity (suppresses top-of-frame objects)
        6. Independent side corridor clearances and least-dangerous best-direction selection
        """
        scene = scene_depth if isinstance(scene_depth, dict) else {}

        left_depth = self.normalize_depth(
            scene.get("left_depth", scene.get("left", 1.0))
        )
        center_depth = self.normalize_depth(
            scene.get("center_region_depth", scene.get("center_depth", 1.0))
        )
        right_depth = self.normalize_depth(
            scene.get("right_depth", scene.get("right", 1.0))
        )

        scene_left_risk = max(0.0, 1.0 - left_depth)
        scene_center_risk = max(0.0, 1.0 - center_depth)
        scene_right_risk = max(0.0, 1.0 - right_depth)

        width = float(frame_width) if frame_width and frame_width > 0 else 640.0
        height = float(frame_height) if frame_height and frame_height > 0 else 480.0

        all_obstacles: List[Dict[str, Any]] = []
        if isinstance(objects, list):
            all_obstacles.extend(objects)
        if isinstance(unknown_objects, list):
            all_obstacles.extend(unknown_objects)

        center_intervals: List[Tuple[float, float]] = []
        left_intervals: List[Tuple[float, float]] = []
        right_intervals: List[Tuple[float, float]] = []

        center_contribs: List[float] = []
        left_contribs: List[float] = []
        right_contribs: List[float] = []

        center_blocking_flags: List[bool] = []
        left_blocking_flags: List[bool] = []
        right_blocking_flags: List[bool] = []

        max_center_dist_weight = 0.0
        max_left_dist_weight = 0.0
        max_right_dist_weight = 0.0

        has_low_conf = False
        has_med_conf = False

        for obs in all_obstacles:
            bbox = obs.get("bbox") if isinstance(obs.get("bbox"), dict) else {}
            x1 = max(0.0, min(1.0, self._safe_float(bbox.get("x1", 0)) / width))
            y1 = max(0.0, min(1.0, self._safe_float(bbox.get("y1", 0)) / height))
            x2 = max(0.0, min(1.0, self._safe_float(bbox.get("x2", 0)) / width))
            y2 = max(0.0, min(1.0, self._safe_float(bbox.get("y2", 0)) / height))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            obj_width = x2 - x1

            # Vertical ground-proximity filter:
            # Physical walking hazards are located on or near the ground plane.
            # Bottom edge y2 < 0.35 is in top background/sky/ceiling -> not a walking hazard.
            if y2 < 0.35:
                vertical_factor = 0.0
            else:
                vertical_factor = min(1.0, max(0.0, (y2 - 0.30) / 0.30))

            # Horizontal overlaps with normalized corridors:
            # Left: [0.0, 0.35], span = 0.35
            # Center: [0.35, 0.65], span = 0.30
            # Right: [0.65, 1.0], span = 0.35
            cov_c = max(0.0, min(x2, 0.65) - max(x1, 0.35)) / 0.30
            cov_l = max(0.0, min(x2, 0.35) - max(x1, 0.0)) / 0.35
            cov_r = max(0.0, min(x2, 1.0) - max(x1, 0.65)) / 0.35

            dist_m = self._safe_float(obs.get("distance_m", obs.get("estimated_distance_m", 2.0)))
            dist_cat = str(obs.get("distance_category", "")).upper()
            if not dist_cat or dist_cat == "NONE":
                if dist_m < 1.4:
                    dist_cat = "VERY_CLOSE"
                elif dist_m < 3.5:
                    dist_cat = "NEAR"
                else:
                    dist_cat = "FAR"

            risk = self._safe_float(obs.get("risk_score", 0.0))

            if dist_cat == "VERY_CLOSE":
                dist_weight = 1.0
            elif dist_cat == "NEAR":
                dist_weight = 0.85
            elif dist_cat == "FAR":
                dist_weight = 0.50 if obj_width > 0.60 else 0.25
            else:
                dist_weight = 0.60

            dist_conf = str(obs.get("distance_confidence", "HIGH")).upper()
            det_conf = self._safe_float(obs.get("confidence", 0.8))
            is_unknown = bool(obs.get("class_name") in ("unknown_obstacle", "obstacle", None, ""))
            if dist_conf == "LOW" or det_conf < 0.40:
                has_low_conf = True
            elif dist_conf == "MEDIUM" or det_conf < 0.65 or is_unknown:
                has_med_conf = True

            if vertical_factor > 0.0:
                # Center corridor
                if cov_c > 0.02:
                    center_intervals.append((x1, x2))
                    c_contrib = risk * dist_weight * vertical_factor * min(1.0, cov_c * 1.5)
                    center_contribs.append(c_contrib)
                    max_center_dist_weight = max(max_center_dist_weight, dist_weight)
                    if vertical_factor >= 0.35:
                        if (dist_cat == "VERY_CLOSE" and cov_c >= 0.15 and risk >= 0.40) or \
                           (dist_cat == "NEAR" and cov_c >= 0.20 and risk >= 0.45) or \
                           (risk >= 0.65 and cov_c >= 0.25) or \
                           (cov_c >= 0.50 and dist_cat in ("VERY_CLOSE", "NEAR")):
                            center_blocking_flags.append(True)

                # Left corridor
                if cov_l > 0.02:
                    left_intervals.append((x1, x2))
                    l_contrib = risk * dist_weight * vertical_factor * min(1.0, cov_l * 1.5)
                    left_contribs.append(l_contrib)
                    max_left_dist_weight = max(max_left_dist_weight, dist_weight)
                    if vertical_factor >= 0.35:
                        if (dist_cat == "VERY_CLOSE" and cov_l >= 0.15 and risk >= 0.40) or \
                           (dist_cat == "NEAR" and cov_l >= 0.20 and risk >= 0.45) or \
                           (risk >= 0.65 and cov_l >= 0.25) or \
                           (cov_l >= 0.50 and dist_cat in ("VERY_CLOSE", "NEAR")):
                            left_blocking_flags.append(True)

                # Right corridor
                if cov_r > 0.02:
                    right_intervals.append((x1, x2))
                    r_contrib = risk * dist_weight * vertical_factor * min(1.0, cov_r * 1.5)
                    right_contribs.append(r_contrib)
                    max_right_dist_weight = max(max_right_dist_weight, dist_weight)
                    if vertical_factor >= 0.35:
                        if (dist_cat == "VERY_CLOSE" and cov_r >= 0.15 and risk >= 0.40) or \
                           (dist_cat == "NEAR" and cov_r >= 0.20 and risk >= 0.45) or \
                           (risk >= 0.65 and cov_r >= 0.25) or \
                           (cov_r >= 0.50 and dist_cat in ("VERY_CLOSE", "NEAR")):
                            right_blocking_flags.append(True)

        center_coverage = self._compute_interval_coverage(center_intervals, 0.35, 0.65)
        left_coverage = self._compute_interval_coverage(left_intervals, 0.0, 0.35)
        right_coverage = self._compute_interval_coverage(right_intervals, 0.65, 1.0)

        max_c_risk = max(center_contribs) if center_contribs else 0.0
        max_l_risk = max(left_contribs) if left_contribs else 0.0
        max_r_risk = max(right_contribs) if right_contribs else 0.0

        center_risk = min(1.0, max(scene_center_risk, max_c_risk, center_coverage * max_center_dist_weight))
        left_risk = min(1.0, max(scene_left_risk, max_l_risk, left_coverage * max_left_dist_weight))
        right_risk = min(1.0, max(scene_right_risk, max_r_risk, right_coverage * max_right_dist_weight))

        center_blocked = bool(
            center_depth <= self.CLOSE_DEPTH
            or any(center_blocking_flags)
            or center_risk >= 0.50
            or (center_coverage >= 0.45 and max_center_dist_weight >= 0.70)
        )
        center_clear = bool(not center_blocked)

        left_blocked = bool(
            left_depth <= self.CLOSE_DEPTH
            or any(left_blocking_flags)
            or left_risk >= 0.50
            or (left_coverage >= 0.45 and max_left_dist_weight >= 0.70)
        )
        left_clear = bool(not left_blocked)

        right_blocked = bool(
            right_depth <= self.CLOSE_DEPTH
            or any(right_blocking_flags)
            or right_risk >= 0.50
            or (right_coverage >= 0.45 and max_right_dist_weight >= 0.70)
        )
        right_clear = bool(not right_blocked)

        safe_directions: List[str] = []
        if left_clear:
            safe_directions.append("LEFT")
        if center_clear:
            safe_directions.append("CENTER")
        if right_clear:
            safe_directions.append("RIGHT")

        no_safe_path = bool(len(safe_directions) == 0)

        # Best direction selection:
        # Priority:
        # 1. CENTER forward corridor if clear and risk is balanced
        # 2. Safest clear direction among remaining safe directions
        # 3. Least dangerous direction when all corridors are blocked
        if center_clear and (center_risk <= min(left_risk, right_risk) + 0.15):
            best_direction = "CENTER"
        elif safe_directions:
            risk_map = {"LEFT": left_risk, "CENTER": center_risk, "RIGHT": right_risk}
            best_direction = min(safe_directions, key=lambda d: risk_map[d])
        else:
            risk_map = {"LEFT": left_risk, "CENTER": center_risk, "RIGHT": right_risk}
            best_direction = min(risk_map, key=risk_map.get)

        if best_direction == "CENTER":
            corridor_width = round(0.30 * max(0.0, 1.0 - center_coverage), 4)
        elif best_direction == "LEFT":
            corridor_width = round(0.35 * max(0.0, 1.0 - left_coverage), 4)
        else:
            corridor_width = round(0.35 * max(0.0, 1.0 - right_coverage), 4)

        if has_low_conf:
            path_confidence = "LOW"
        elif has_med_conf:
            path_confidence = "MEDIUM"
        else:
            path_confidence = "HIGH"

        return {
            "left_clear": left_clear,
            "center_clear": center_clear,
            "right_clear": right_clear,
            "left_depth": round(left_depth, 4),
            "center_depth": round(center_depth, 4),
            "right_depth": round(right_depth, 4),
            "left_risk": round(left_risk, 4),
            "center_risk": round(center_risk, 4),
            "right_risk": round(right_risk, 4),
            "best_direction": best_direction,

            # Phase 7 Safe Walking Corridor fields
            "center_blocked": center_blocked,
            "left_blocked": left_blocked,
            "right_blocked": right_blocked,
            "center_coverage": round(center_coverage, 4),
            "left_coverage": round(left_coverage, 4),
            "right_coverage": round(right_coverage, 4),
            "corridor_width": corridor_width,
            "safe_directions": safe_directions,
            "no_safe_path": no_safe_path,
            "path_confidence": path_confidence,

            # Backward compatibility
            "left": round(left_depth, 4),
            "center": round(center_depth, 4),
            "right": round(right_depth, 4),
            "is_center_blocked": center_blocked,
        }

    # ========================================================
    # SAFETY LEVEL
    # ========================================================

    def get_safety_level(
        self,
        primary_risk: float,
        emergency_stop: bool,
    ) -> str:
        """
        Convert risk into safety level.
        """

        risk = self._clamp(
            self._safe_float(
                primary_risk
            )
        )

        if emergency_stop:
            return "EMERGENCY"

        # Risk increases from SAFE -> AWARE -> CAUTION -> DANGER.
        # The thresholds are evaluated from highest to lowest so
        # the safety levels remain mutually reachable.
        if risk >= self.danger_threshold:
            return "DANGER"

        if risk >= self.caution_threshold:
            return "CAUTION"

        if risk >= self.aware_threshold:
            return "AWARE"

        return "SAFE"

    # ========================================================
    # EMERGENCY LOGIC
    # ========================================================

    def should_emergency_stop(
        self,
        primary_obstacle: Optional[
            Dict[str, Any]
        ],
        scene: Dict[str, Any],
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine whether the system should command an
        emergency stop.

        Emergency stop requires a genuinely dangerous
        situation rather than merely detecting an object.

        Strong emergency conditions:

            - predictive threat EMERGENCY
            - rapidly approaching center obstacle <= 1.4m or TTC <= 2.0s
            - very close center obstacle
            - very high risk center obstacle
            - very close human/vehicle in center
            - center path severely blocked
        """

        # ----------------------------------------------------
        # Predictive threat evaluation (Phase 11)
        # ----------------------------------------------------
        if predictive_threat and isinstance(predictive_threat, dict):
            pt_pos = str(predictive_threat.get("position", "")).upper()
            pt_mot = str(predictive_threat.get("motion_state", "")).upper()
            pt_dist = self._safe_float(predictive_threat.get("distance_m", 99.0), 99.0)
            pt_ttc = predictive_threat.get("time_to_collision_s")
            pt_ttc_val = float(pt_ttc) if pt_ttc is not None and math.isfinite(float(pt_ttc)) else None
            pt_threat = str(predictive_threat.get("threat_level", "NONE")).upper()

            if pt_threat == "EMERGENCY":
                return True
            if pt_pos == "CENTER":
                if pt_mot == "RAPIDLY_APPROACHING" and (pt_dist <= 1.4 or (pt_ttc_val is not None and pt_ttc_val <= 2.0)):
                    return True
                if pt_ttc_val is not None and pt_ttc_val <= 2.0 and pt_dist <= 2.0 and pt_mot in ("APPROACHING", "RAPIDLY_APPROACHING"):
                    return True

        if not primary_obstacle:
            return False

        risk = self._safe_float(
            primary_obstacle.get(
                "risk_score",
                0.0,
            )
        )

        depth = self.normalize_depth(
            primary_obstacle.get(
                "depth",
                1.0,
            )
        )

        position = str(
            primary_obstacle.get(
                "position",
                "",
            )
        ).upper()

        class_name = str(
            primary_obstacle.get(
                "class_name",
                "",
            )
        ).lower()

        center_depth = self.normalize_depth(
            scene.get(
                "center_region_depth",
                scene.get(
                    "center_depth",
                    1.0,
                ),
            )
        )

        curr_dist_m = self._safe_float(
            primary_obstacle.get(
                "distance_m",
                primary_obstacle.get(
                    "estimated_distance_m",
                    99.0,
                ),
            ),
            99.0,
        )

        # ----------------------------------------------------
        # Rapidly approaching center obstacle (Phase 11 / Requirement 8)
        # ----------------------------------------------------
        obs_mot = str(primary_obstacle.get("motion_state", "")).upper()
        obs_ttc = primary_obstacle.get("time_to_collision_s")
        obs_ttc_val = float(obs_ttc) if obs_ttc is not None and math.isfinite(float(obs_ttc)) else None

        if obs_mot == "RAPIDLY_APPROACHING":
            if curr_dist_m <= 2.0 or (obs_ttc_val is not None and obs_ttc_val <= 2.5) or position == "CENTER":
                return True

        if position == "CENTER":
            if obs_ttc_val is not None and obs_ttc_val <= 2.0 and curr_dist_m <= 2.0 and obs_mot in ("APPROACHING", "RAPIDLY_APPROACHING"):
                return True

        # ----------------------------------------------------
        # Very close obstacle <= 0.8m -> immediate STOP (Requirement 7)
        # ----------------------------------------------------
        if curr_dist_m <= 0.8:
            return True

        # ----------------------------------------------------
        # Extremely close relative depth <= 0.20 (Requirement 7)
        # ----------------------------------------------------
        if depth <= self.VERY_CLOSE_DEPTH:
            if position == "CENTER" or curr_dist_m <= 1.2 or risk >= 0.50:
                return True

        # ----------------------------------------------------
        # Very dangerous / high-risk obstacle (Requirement 7)
        # ----------------------------------------------------
        if risk >= 0.85 or (risk >= 0.75 and curr_dist_m <= 1.0 and position == "CENTER"):
            return True

        if (
            position == "CENTER"
            and (depth <= self.VERY_CLOSE_DEPTH or curr_dist_m <= 1.2)
            and risk >= 0.55
        ):
            return True

        # ----------------------------------------------------
        # Person / vehicle very close
        # ----------------------------------------------------
        if (
            position == "CENTER"
            and class_name
            in (
                self.HUMAN_CLASSES
                | self.VEHICLE_CLASSES
            )
            and (depth <= 0.20 or curr_dist_m <= 1.2)
        ):
            return True

        # ----------------------------------------------------
        # Center path severely blocked
        # ----------------------------------------------------
        if (
            position == "CENTER"
            and center_depth
            <= 0.12
            and risk >= 0.55
        ):
            return True

        return False

    # ========================================================
    # NAVIGATION DECISION ENGINE (PHASE 8)
    # ========================================================

    def _compute_navigation_candidate(
        self,
        safety_level: str,
        emergency_stop: bool,
        primary_obstacle: Optional[Dict[str, Any]],
        free_path: Optional[Dict[str, Any]],
        backward_path_clear: bool = False,
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str]:
        """
        Compute deterministic navigation candidate (action, direction, reason) without hysteresis.
        """
        fp = free_path if isinstance(free_path, dict) else {}
        left_clear = bool(fp.get("left_clear", False))
        center_clear = bool(fp.get("center_clear", False))
        right_clear = bool(fp.get("right_clear", False))
        no_safe_path = bool(fp.get("no_safe_path", False))
        left_risk = self._safe_float(fp.get("left_risk", 1.0))
        right_risk = self._safe_float(fp.get("right_risk", 1.0))
        center_risk = self._safe_float(fp.get("center_risk", 1.0))
        left_cov = self._safe_float(fp.get("left_coverage", 0.0))
        right_cov = self._safe_float(fp.get("right_coverage", 0.0))

        # 1. Emergency stop override
        if emergency_stop or safety_level == "EMERGENCY":
            return (
                "STOP",
                "CENTER",
                "Emergency stop: immediate obstacle in path",
            )

        # Extract primary obstacle info
        obs_pos = str(primary_obstacle.get("position", "")).upper() if primary_obstacle else ""
        if not obs_pos and primary_obstacle and "bbox" in primary_obstacle:
            obs_pos = self.get_position(primary_obstacle["bbox"], 640.0)
        if not obs_pos and primary_obstacle:
            obs_pos = "CENTER"
        obs_risk = self._safe_float(primary_obstacle.get("risk_score", 0.0)) if primary_obstacle else 0.0
        obs_dist = self._safe_float(
            primary_obstacle.get("distance_m", primary_obstacle.get("estimated_distance_m", 99.0)),
            99.0,
        ) if primary_obstacle else 99.0
        obs_depth = self.normalize_depth(primary_obstacle.get("depth", 1.0)) if primary_obstacle else 1.0
        obs_cat = str(primary_obstacle.get("distance_category", "")).upper() if primary_obstacle else ""

        # Extract predictive threat / tracking info
        threat_mot = str(predictive_threat.get("motion_state", "")).upper() if predictive_threat else ""
        threat_rapid = bool(threat_mot == "RAPIDLY_APPROACHING")
        threat_app = bool(predictive_threat.get("is_approaching", False)) if predictive_threat else False
        obs_mot = str(primary_obstacle.get("motion_state", "")).upper() if primary_obstacle else ""
        obs_rapid = bool(obs_mot == "RAPIDLY_APPROACHING" or (primary_obstacle and primary_obstacle.get("rapidly_approaching")))
        obs_app = bool(obs_rapid or (primary_obstacle and (primary_obstacle.get("approaching") or primary_obstacle.get("is_approaching"))))

        is_rapid = threat_rapid or obs_rapid
        is_app = threat_app or obs_app

        # 2. Very close or high-risk obstacle -> immediate STOP (Requirement 7)
        if primary_obstacle:
            if obs_dist <= 0.8 or obs_depth <= self.VERY_CLOSE_DEPTH:
                return (
                    "STOP",
                    "CENTER",
                    "Obstacle very close: immediate stop",
                )
            if (obs_risk >= 0.85 or (obs_risk >= 0.75 and obs_dist <= 1.0)) and obs_pos == "CENTER":
                return (
                    "STOP",
                    "CENTER",
                    "High-risk obstacle ahead: immediate stop",
                )

        # 3. Rapidly approaching obstacle -> STOP or SLOW_DOWN (Requirement 8)
        if is_rapid:
            if obs_pos == "CENTER" or obs_dist <= 2.0:
                return (
                    "STOP",
                    "CENTER",
                    "Rapidly approaching obstacle in path",
                )
            elif right_clear and not left_clear:
                return (
                    "TURN_RIGHT",
                    "RIGHT",
                    "Rapidly approaching obstacle; detour right",
                )
            elif left_clear and not right_clear:
                return (
                    "TURN_LEFT",
                    "LEFT",
                    "Rapidly approaching obstacle; detour left",
                )
            else:
                return (
                    "SLOW_DOWN",
                    "CENTER",
                    "Rapidly approaching obstacle detected",
                )

        # 4. Moderately approaching center obstacle -> detour or SLOW_DOWN (Requirement 8)
        if is_app and obs_pos == "CENTER" and obs_dist <= 2.5:
            if left_clear and not right_clear:
                return (
                    "TURN_LEFT",
                    "LEFT",
                    "Approaching center obstacle; left corridor is clear",
                )
            elif right_clear and not left_clear:
                return (
                    "TURN_RIGHT",
                    "RIGHT",
                    "Approaching center obstacle; right corridor is clear",
                )
            elif left_clear and right_clear:
                chosen_side = "RIGHT" if right_risk <= left_risk else "LEFT"
                # Anti-oscillation hysteresis: maintain active turn direction unless other side is substantially safer
                if "LEFT" in self.last_action and left_risk <= right_risk + 0.08:
                    chosen_side = "LEFT"
                elif "RIGHT" in self.last_action and right_risk <= left_risk + 0.08:
                    chosen_side = "RIGHT"
                else:
                    chosen_side = "RIGHT" if right_risk <= left_risk else "LEFT"
                return (
                    f"TURN_{chosen_side}",
                    chosen_side,
                    f"Approaching center obstacle; {chosen_side.lower()} corridor is safer",
                )
            else:
                if obs_dist <= 1.8:
                    return (
                        "STOP",
                        "CENTER",
                        "Approaching center obstacle with no safe detour",
                    )
                return (
                    "SLOW_DOWN",
                    "CENTER",
                    "Approaching center obstacle; slow down",
                )

        # 5. All forward corridors blocked or no safe path
        if no_safe_path or (not left_clear and not center_clear and not right_clear):
            if backward_path_clear:
                return (
                    "BACK",
                    "BACK",
                    "All forward corridors are blocked; safe backward path clear",
                )
            return (
                "STOP",
                "NONE",
                "All forward corridors are blocked",
            )

        # 6. Center corridor blocked or unsafe (Requirement 5)
        center_unsafe = bool(
            not center_clear
            or (obs_pos == "CENTER" and (obs_risk >= 0.45 or obs_dist <= 2.5 or safety_level in ("CAUTION", "DANGER", "EMERGENCY")))
        )

        if center_unsafe:
            # CENTER obstacle: choose a genuinely safe left/right path when available. If no safe path -> STOP.
            if left_clear and not right_clear:
                return (
                    "TURN_LEFT",
                    "LEFT",
                    "Center obstacle blocks forward path; left corridor is clear",
                )
            elif right_clear and not left_clear:
                return (
                    "TURN_RIGHT",
                    "RIGHT",
                    "Center obstacle blocks forward path; right corridor is clear",
                )
            elif left_clear and right_clear:
                # Anti-oscillation hysteresis: favor existing active turn direction
                if self.last_action in ("TURN_LEFT", "MOVE_LEFT") and left_risk <= right_risk + 0.08:
                    return (
                        "TURN_LEFT",
                        "LEFT",
                        "Center obstacle blocks forward path; continuing safe left turn",
                    )
                elif self.last_action in ("TURN_RIGHT", "MOVE_RIGHT") and right_risk <= left_risk + 0.08:
                    return (
                        "TURN_RIGHT",
                        "RIGHT",
                        "Center obstacle blocks forward path; continuing safe right turn",
                    )
                elif right_risk < left_risk - 0.04:
                    return (
                        "TURN_RIGHT",
                        "RIGHT",
                        "Center obstacle blocks forward path; right corridor is safer",
                    )
                elif left_risk < right_risk - 0.04:
                    return (
                        "TURN_LEFT",
                        "LEFT",
                        "Center obstacle blocks forward path; left corridor is safer",
                    )
                elif right_cov < left_cov - 0.05:
                    return (
                        "TURN_RIGHT",
                        "RIGHT",
                        "Center obstacle blocks forward path; right corridor has more clearance",
                    )
                elif left_cov < right_cov - 0.05:
                    return (
                        "TURN_LEFT",
                        "LEFT",
                        "Center obstacle blocks forward path; left corridor has more clearance",
                    )
                else:
                    return (
                        "TURN_RIGHT",
                        "RIGHT",
                        "Center obstacle blocks forward path; right corridor is clear",
                    )
            else:
                if backward_path_clear:
                    return (
                        "BACK",
                        "BACK",
                        "Center blocked and side corridors unsafe; safe backward path clear",
                    )
                return (
                    "STOP",
                    "NONE",
                    "Center blocked and no safe side corridor available",
                )

        # 7. LEFT obstacle -> prefer RIGHT when safe (Requirement 6)
        if obs_pos == "LEFT":
            if right_clear:
                if obs_risk >= 0.40 or obs_dist <= 2.5:
                    return (
                        "MOVE_RIGHT",
                        "RIGHT",
                        "Obstacle on left; shift right into clear corridor",
                    )
                elif center_clear:
                    return (
                        "FORWARD",
                        "CENTER",
                        "Obstacle on left; path is clear to proceed",
                    )
                else:
                    return (
                        "TURN_RIGHT",
                        "RIGHT",
                        "Obstacle on left; right corridor is clear",
                    )
            elif center_clear:
                return (
                    "FORWARD",
                    "CENTER",
                    "Obstacle on left; center path is clear",
                )
            elif left_clear:
                return (
                    "SLOW_DOWN",
                    "CENTER",
                    "Obstacle on left and right blocked; proceed with caution",
                )
            else:
                return (
                    "STOP",
                    "NONE",
                    "Obstacle on left and other paths blocked",
                )

        # 8. RIGHT obstacle -> prefer LEFT when safe (Requirement 6)
        if obs_pos == "RIGHT":
            if left_clear:
                if obs_risk >= 0.40 or obs_dist <= 2.5:
                    return (
                        "MOVE_LEFT",
                        "LEFT",
                        "Obstacle on right; shift left into clear corridor",
                    )
                elif center_clear:
                    return (
                        "FORWARD",
                        "CENTER",
                        "Obstacle on right; path is clear to proceed",
                    )
                else:
                    return (
                        "TURN_LEFT",
                        "LEFT",
                        "Obstacle on right; left corridor is clear",
                    )
            elif center_clear:
                return (
                    "FORWARD",
                    "CENTER",
                    "Obstacle on right; center path is clear",
                )
            elif right_clear:
                return (
                    "SLOW_DOWN",
                    "CENTER",
                    "Obstacle on right and left blocked; proceed with caution",
                )
            else:
                return (
                    "STOP",
                    "NONE",
                    "Obstacle on right and other paths blocked",
                )

        # 9. Center corridor is clear and safe
        if center_clear:
            if primary_obstacle and obs_cat == "FAR":
                reason = "Distant obstacle ahead; center corridor is safe to proceed"
            else:
                reason = "Center path is clear"
            return ("FORWARD", "CENTER", reason)

        # 10. Fallback if neither center nor sides are clear
        return ("STOP", "NONE", "All forward corridors are blocked")

    def estimate_turn_angle(
        self,
        action: str,
        direction: str,
        free_path: Optional[Dict[str, Any]] = None,
        primary_obstacle: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str, str]:
        """
        Estimate approximate navigation turn angle target and direction.
        Returns: (turn_angle_deg, turn_direction, turn_confidence)

        IMPORTANT: A single RGB forward-facing camera cannot measure physical user rotation.
        turn_angle_deg is a recommended navigation target, NOT a measured physical angle.
        """
        act = str(action).upper() if action else ""
        d = str(direction).upper() if direction else ""

        if act in ("STOP", "BACK", "GO_BACK", "FORWARD", "GO_FORWARD", "CONTINUE_FORWARD", "SLOW_DOWN") or d in ("NONE", "BACK", "CENTER"):
            return 0.0, "NONE", "HIGH"

        if act in ("TURN_LEFT", "MOVE_LEFT") or d == "LEFT":
            turn_dir = "LEFT"
        elif act in ("TURN_RIGHT", "MOVE_RIGHT") or d == "RIGHT":
            turn_dir = "RIGHT"
        else:
            return 0.0, "NONE", "HIGH"

        fp = free_path if isinstance(free_path, dict) else {}
        center_cov = self._safe_float(fp.get("center_coverage", 0.0))
        center_risk = self._safe_float(fp.get("center_risk", 0.0))
        path_conf = str(fp.get("path_confidence", "")).upper()

        side_clear = bool(fp.get("left_clear", False) if turn_dir == "LEFT" else fp.get("right_clear", False))
        side_cov = self._safe_float(fp.get("left_coverage", 0.0) if turn_dir == "LEFT" else fp.get("right_coverage", 0.0))
        side_risk = self._safe_float(fp.get("left_risk", 1.0) if turn_dir == "LEFT" else fp.get("right_risk", 1.0))

        has_geometry = bool(fp and ("center_coverage" in fp or "center_risk" in fp or "left_clear" in fp or "right_clear" in fp))

        if not has_geometry:
            return 30.0, turn_dir, "LOW"

        # Lateral shift vs full turn angle
        if act in ("MOVE_LEFT", "MOVE_RIGHT"):
            angle = 15.0
        elif center_cov >= 0.70 or center_risk >= 0.80 or (center_cov >= 0.50 and side_cov <= 0.15):
            angle = 45.0
        elif center_cov >= 0.35 or center_risk >= 0.50:
            angle = 30.0
        else:
            angle = 15.0

        angle = max(15.0, min(60.0, float(angle)))

        if path_conf == "HIGH" and side_clear and side_risk < 0.40:
            conf = "HIGH"
        elif path_conf in ("MEDIUM", "HIGH") or side_clear:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        return angle, turn_dir, conf

    def determine_navigation(
        self,
        safety_level: str,
        emergency_stop: bool,
        primary_obstacle: Optional[Dict[str, Any]],
        free_path: Optional[Dict[str, Any]],
        backward_path_clear: bool = False,
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Determine navigation command with temporal hysteresis to prevent oscillation.
        Returns canonical navigation dictionary.
        """
        cand_action, cand_direction, cand_reason = self._compute_navigation_candidate(
            safety_level=safety_level,
            emergency_stop=emergency_stop,
            primary_obstacle=primary_obstacle,
            free_path=free_path,
            backward_path_clear=backward_path_clear,
            predictive_threat=predictive_threat,
        )

        threat_is_rapid_center = bool(
            predictive_threat
            and str(predictive_threat.get("position", "")).upper() == "CENTER"
            and str(predictive_threat.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING"
        )
        if not threat_is_rapid_center and primary_obstacle:
            if (
                str(primary_obstacle.get("position", "")).upper() == "CENTER"
                and str(primary_obstacle.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING"
            ):
                threat_is_rapid_center = True

        # Extract obstacle position and risk for hysteresis decisions
        obs_pos = str(primary_obstacle.get("position", "")).upper() if primary_obstacle else ""
        obs_risk = self._safe_float(primary_obstacle.get("risk_score", 0.0)) if primary_obstacle else 0.0
        fp = free_path if isinstance(free_path, dict) else {}
        center_is_clear = bool(fp.get("center_clear", True))

        # Hysteresis filtering
        if cand_action in ("STOP", "EMERGENCY_STOP") or emergency_stop:
            # Immediate bypass on danger/stop
            self.last_action = "STOP"
            self.candidate_action = "STOP"
            self.action_vote_count = 2
            action = "STOP"
            direction = cand_direction
            reason = cand_reason
        elif cand_action == "SLOW_DOWN":
            # Immediate safety response for slowing down
            self.last_action = "SLOW_DOWN"
            self.candidate_action = "SLOW_DOWN"
            self.action_vote_count = 2
            action = "SLOW_DOWN"
            direction = cand_direction
            reason = cand_reason
        elif (threat_is_rapid_center or not center_is_clear or (primary_obstacle and obs_risk >= 0.40)) and cand_action in ("TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT"):
            # Immediate avoidance when path is obstructed (safety first: never walk into obstacles)
            self.last_action = cand_action
            self.candidate_action = cand_action
            self.action_vote_count = 2
            action = cand_action
            direction = cand_direction
            reason = cand_reason
        elif cand_action in ("BACK", "GO_BACK") and backward_path_clear:
            # Immediate escape when all forward corridors are blocked
            self.last_action = cand_action
            self.candidate_action = cand_action
            self.action_vote_count = 2
            action = cand_action
            direction = cand_direction
            reason = cand_reason
        else:
            is_same = (cand_action == self.last_action) or (
                cand_action in ("FORWARD", "GO_FORWARD") and self.last_action in ("FORWARD", "GO_FORWARD")
            ) or (
                cand_action in ("BACK", "GO_BACK") and self.last_action in ("BACK", "GO_BACK")
            )
            if is_same:
                self.candidate_action = cand_action
                self.action_vote_count = 2
                action = cand_action
                direction = cand_direction
                reason = cand_reason
            else:
                # Action differs from last confirmed action
                if cand_action == self.candidate_action:
                    self.action_vote_count += 1
                    if self.action_vote_count >= 2:
                        self.last_action = cand_action
                        action = cand_action
                        direction = cand_direction
                        reason = cand_reason
                    else:
                        # 1st vote: hold previous action, BUT never hold FORWARD if center is obstructed
                        if not center_is_clear and self.last_action in ("FORWARD", "GO_FORWARD"):
                            action = "SLOW_DOWN"
                            direction = "CENTER"
                            reason = "Center path obstructed: slowing down pending confirmation"
                        else:
                            action = self.last_action
                            direction = (
                                "CENTER"
                                if action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD", "SLOW_DOWN")
                                else ("LEFT" if action in ("TURN_LEFT", "MOVE_LEFT") else ("RIGHT" if action in ("TURN_RIGHT", "MOVE_RIGHT") else "NONE"))
                            )
                            reason = f"Holding {action.lower()} pending confirmation"
                else:
                    self.candidate_action = cand_action
                    self.action_vote_count = 1
                    if not center_is_clear and self.last_action in ("FORWARD", "GO_FORWARD"):
                        action = "SLOW_DOWN"
                        direction = "CENTER"
                        reason = "Center path obstructed: slowing down pending confirmation"
                    else:
                        action = self.last_action
                        direction = (
                            "CENTER"
                            if action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD", "SLOW_DOWN")
                            else ("LEFT" if action in ("TURN_LEFT", "MOVE_LEFT") else ("RIGHT" if action in ("TURN_RIGHT", "MOVE_RIGHT") else "NONE"))
                        )
                        reason = f"Holding {action.lower()} pending confirmation"

        self.last_direction = direction
        self.direction_vote_count = self.action_vote_count

        # Determine turn angle and direction (Phase 9)
        turn_angle_deg, turn_direction, turn_confidence = self.estimate_turn_angle(
            action=action,
            direction=direction,
            free_path=free_path,
            primary_obstacle=primary_obstacle,
        )

        # Stage and movement action mapping (Phase 9 & Requirement 4)
        if emergency_stop or action in ("STOP", "EMERGENCY_STOP"):
            navigation_stage = "STOP"
            turn_required = False
            movement_required = False
            turn_direction = "NONE"
            movement_action = "NONE"
            turn_angle_deg = 0.0
            self.turn_state = "NONE"
        elif action == "TURN_LEFT":
            navigation_stage = "TURN"
            turn_required = True
            movement_required = False
            turn_direction = "LEFT"
            movement_action = "NONE"
            self.turn_state = "TURNING_LEFT"
        elif action == "TURN_RIGHT":
            navigation_stage = "TURN"
            turn_required = True
            movement_required = False
            turn_direction = "RIGHT"
            movement_action = "NONE"
            self.turn_state = "TURNING_RIGHT"
        elif action == "MOVE_LEFT":
            navigation_stage = "MOVE"
            turn_required = False
            movement_required = True
            turn_direction = "LEFT"
            movement_action = "WALK_LEFT"
            self.turn_state = "NONE"
        elif action == "MOVE_RIGHT":
            navigation_stage = "MOVE"
            turn_required = False
            movement_required = True
            turn_direction = "RIGHT"
            movement_action = "WALK_RIGHT"
            self.turn_state = "NONE"
        elif action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
            navigation_stage = "MOVE"
            turn_required = False
            movement_required = True
            turn_direction = "NONE"
            movement_action = "WALK_FORWARD"
            if self.turn_state in ("TURNING_LEFT", "TURNING_RIGHT"):
                self.turn_state = "READY_TO_MOVE"
            else:
                self.turn_state = "NONE"
        elif action == "SLOW_DOWN":
            navigation_stage = "MOVE"
            turn_required = False
            movement_required = True
            turn_direction = "NONE"
            movement_action = "WALK_FORWARD"
            self.turn_state = "NONE"
        elif action in ("BACK", "GO_BACK"):
            if backward_path_clear:
                navigation_stage = "MOVE"
                turn_required = False
                movement_required = True
                turn_direction = "NONE"
                movement_action = "WALK_BACK"
                self.turn_state = "NONE"
            else:
                navigation_stage = "STOP"
                turn_required = False
                movement_required = False
                turn_direction = "NONE"
                movement_action = "NONE"
                turn_angle_deg = 0.0
                self.turn_state = "NONE"
        else:
            navigation_stage = "IDLE"
            turn_required = False
            movement_required = False
            turn_direction = "NONE"
            movement_action = "NONE"
            turn_angle_deg = 0.0
            self.turn_state = "NONE"

        primary_dist = 0.0
        conf = "HIGH"

        if primary_obstacle:
            primary_dist = round(self._safe_float(primary_obstacle.get("distance_m", 0.0)), 2)
            conf = str(primary_obstacle.get("distance_confidence", "HIGH")).upper()
            if conf not in ("HIGH", "MEDIUM", "LOW"):
                conf = "HIGH"

        # Estimate movement distance and steps (Phase 10 & 11)
        movement_dist_m = 0.0
        step_conf = "HIGH"
        step_reason = "No forward movement during stop/turn"
        movement_steps = 0

        if action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
            movement_dist_m, step_conf, step_reason = self.estimate_movement_distance(
                distance_m=primary_dist,
                distance_category=primary_obstacle.get("distance_category", "NEAR") if primary_obstacle else "FAR",
                action=action,
                free_path=free_path,
            )
            movement_steps = self.estimate_walking_steps(movement_dist_m, self.step_length_m)
            self.last_steps = movement_steps
        elif action == "SLOW_DOWN":
            movement_dist_m, step_conf, step_reason = self.estimate_movement_distance(
                distance_m=primary_dist,
                distance_category=primary_obstacle.get("distance_category", "NEAR") if primary_obstacle else "FAR",
                action="SLOW_DOWN",
                free_path=free_path,
            )
            movement_steps = max(1, self.estimate_walking_steps(movement_dist_m, self.step_length_m))
            self.last_steps = movement_steps
        elif action in ("MOVE_LEFT", "MOVE_RIGHT"):
            movement_dist_m = 0.5
            movement_steps = 1
            step_conf = "HIGH"
            step_reason = f"Lateral step {direction.lower()}"
            self.last_steps = movement_steps
        elif action in ("BACK", "GO_BACK") and backward_path_clear:
            movement_dist_m = 0.75
            movement_steps = 1
            step_conf = "HIGH"
            step_reason = "Safe backward step"
            self.last_steps = movement_steps
        else:
            self.last_steps = 0

        steps = movement_steps

        return {
            "action": action,
            "direction": direction,
            "distance_m": primary_dist,
            "steps": steps,
            "movement_steps": movement_steps,
            "movement_distance_m": movement_dist_m,
            "step_confidence": step_conf,
            "step_reason": step_reason,
            "confidence": conf,
            "turn_required": turn_required,
            "movement_required": movement_required,
            "reason": reason,
            "turn_angle_deg": turn_angle_deg,
            "turn_direction": turn_direction,
            "movement_action": movement_action,
            "navigation_stage": navigation_stage,
            "turn_confidence": turn_confidence,
            "angle": turn_angle_deg,
        }

    # Backward compatibility alias
    def _compute_navigation(
        self,
        safety_level: str,
        emergency_stop: bool,
        primary_obstacle: Optional[Dict[str, Any]],
        free_path: Dict[str, Any],
    ) -> str:
        res = self._compute_navigation_candidate(
            safety_level=safety_level,
            emergency_stop=emergency_stop,
            primary_obstacle=primary_obstacle,
            free_path=free_path,
        )
        return res[0]

    # ========================================================
    # VOICE INSTRUCTION
    # ========================================================

    def generate_voice_instruction(
        self,
        safety_level: str,
        emergency_stop: bool,
        primary_obstacle: Optional[
            Dict[str, Any]
        ],
        navigation: str,
        objects: Optional[
            List[Dict[str, Any]]
        ] = None,
        steps: Optional[int] = None,
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate short, actionable English voice guidance.

        Strictly directive and concise:
            - Stop immediately.
            - Move left.
            - Move right.
            - Continue forward.
            - Slow down.
        """
        if emergency_stop or safety_level == "EMERGENCY":
            return "Stop immediately."

        nav = str(navigation).upper() if navigation else ""

        # Rapidly approaching check
        is_rapid = False
        if predictive_threat and isinstance(predictive_threat, dict):
            if str(predictive_threat.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING":
                is_rapid = True
        if primary_obstacle and isinstance(primary_obstacle, dict):
            if str(primary_obstacle.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING" or primary_obstacle.get("rapidly_approaching"):
                is_rapid = True

        if is_rapid and nav == "STOP":
            return "Stop immediately."

        # Navigation action-based directives
        if nav == "STOP":
            obs_dist = self._safe_float(primary_obstacle.get("distance_m", 99.0)) if primary_obstacle else 99.0
            obs_risk = self._safe_float(primary_obstacle.get("risk_score", 0.0)) if primary_obstacle else 0.0
            if obs_dist <= 0.8 or obs_risk >= 0.70 or safety_level in ("DANGER", "EMERGENCY") or emergency_stop:
                return "Stop immediately."
            return "Stop. No safe path."
            return "Stop."

        if nav == "TURN_LEFT":
            return "Turn left."
            return "Turn left about 45 degrees."

        if nav == "TURN_RIGHT":
            return "Turn right."
            return "Turn right about 45 degrees."

        if nav == "MOVE_LEFT":
            return "Move left."

        if nav == "MOVE_RIGHT":
            return "Move right."

        if nav == "SLOW_DOWN":
            return "Slow down."

        if nav in ("BACK", "GO_BACK"):
            return "Go back."

        if nav in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
            return "Go forward."
            if steps is not None and steps >= 2 and steps <= 8:
                return f"Move forward for about {steps} steps."
            return "Move forward."

        if nav in ("BACK", "GO_BACK"):
            return "Move back."

        # Safety-level fallback
        if safety_level == "DANGER":
            return "Slow down."

        return "Go forward."
        return "Move forward."

    # ========================================================
    # POSITION PHRASE
    # ========================================================

    @staticmethod
    def _position_phrase(
        position: str,
    ) -> str:
        """
        Convert position to natural voice wording.
        """

        position = str(
            position
        ).upper()

        if position == "LEFT":
            return "on your left"

        if position == "RIGHT":
            return "on your right"

        return "ahead"

    # ========================================================
    # MULTI-OBJECT REASONING
    # ========================================================

    def multi_object_reasoning(
        self,
        objects: List[
            Dict[str, Any]
        ],
        unknown_objects: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Analyze multiple objects.

        Produces:

            object_count
            important_objects
            center_obstacles
            human_detected
            vehicle_detected
        """

        all_items: List[Dict[str, Any]] = list(objects or [])
        if unknown_objects:
            all_items.extend(unknown_objects)

        if not all_items:

            return {
                "object_count": 0,
                "left_obstacles": 0,
                "center_obstacles": 0,
                "right_obstacles": 0,
                "human_detected": False,
                "vehicle_detected": False,
                "highest_risk_object_id": None,
                "closest_object_id": None,
            }

        sorted_objects = sorted(
            all_items,
            key=lambda item:
                self._safe_float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            reverse=True,
        )

        left_obstacles = sum(
            1 for obj in all_items
            if str(obj.get("position", "")).upper() == "LEFT"
        )

        center_obstacles = sum(
            1 for obj in all_items
            if str(obj.get("position", "")).upper() == "CENTER"
        )

        right_obstacles = sum(
            1 for obj in all_items
            if str(obj.get("position", "")).upper() == "RIGHT"
        )

        highest_risk_id = None
        if sorted_objects:
            highest_risk_id = sorted_objects[0].get("id")

        closest_id = None
        # Select closest obstacle using canonical distance_m (filter for valid positive distances)
        valid_dist_items = [
            item for item in all_items
            if self._safe_float(item.get("distance_m", 0.0)) > 0.1
            and item.get("distance_category") != "UNKNOWN_DISTANCE"
        ]
        if valid_dist_items:
            closest_obj = min(
                valid_dist_items,
                key=lambda item: self._safe_float(item.get("distance_m", 999.0))
            )
            closest_id = closest_obj.get("id")
        elif all_items:
            closest_id = all_items[0].get("id")

        class_names = {
            str(
                obj.get(
                    "class_name",
                    "",
                )
            ).lower()
            for obj in (objects or [])
        }

        return {
            "object_count":
                len(all_items),

            "left_obstacles":
                left_obstacles,

            "center_obstacles":
                center_obstacles,

            "right_obstacles":
                right_obstacles,

            "human_detected":
                bool(
                    class_names
                    & self.HUMAN_CLASSES
                ),

            "vehicle_detected":
                bool(
                    class_names
                    & self.VEHICLE_CLASSES
                ),

            "highest_risk_object_id":
                highest_risk_id,

            "closest_object_id":
                closest_id,
        }

    # ========================================================
    # STRUCTURED NAVIGATION & STEP BUILDERS
    # ========================================================

    def build_navigation_result(
        self,
        raw_direction: Any,
        safety_level: str = "SAFE",
        emergency_stop: bool = False,
        primary_obstacle: Optional[Dict[str, Any]] = None,
        free_path: Optional[Dict[str, Any]] = None,
        backward_path_clear: bool = False,
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Map current navigation decision into the canonical structured navigation contract.
        """
        if isinstance(raw_direction, dict) and "action" in raw_direction:
            return raw_direction

        return self.determine_navigation(
            safety_level=safety_level,
            emergency_stop=emergency_stop,
            primary_obstacle=primary_obstacle,
            free_path=free_path or {},
            backward_path_clear=backward_path_clear,
            predictive_threat=predictive_threat,
        )

    def build_step_guidance(
        self,
        navigation: Dict[str, Any],
        primary_obstacle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build step guidance contract.
        """
        act = str(navigation.get("action", "")).upper()
        nav_steps = int(navigation.get("movement_steps", navigation.get("steps", 0)))
        dist_m = float(navigation.get("movement_distance_m", navigation.get("distance_m", 0.0)))
        step_conf = str(navigation.get("step_confidence", navigation.get("confidence", "HIGH")))
        step_reason = str(navigation.get("step_reason", navigation.get("reason", "")))

        if act in ("STOP", "EMERGENCY_STOP"):
            sg_action = "STOP"
            sg_steps = 0
            sg_dist = 0.0
        elif act in ("TURN_LEFT", "TURN_RIGHT"):
            sg_action = "NONE"
            sg_steps = 0
            sg_dist = 0.0
        elif act == "MOVE_LEFT":
            sg_action = "WALK_LEFT"
            sg_steps = max(1, nav_steps)
            sg_dist = dist_m if dist_m > 0 else 0.5
        elif act == "MOVE_RIGHT":
            sg_action = "WALK_RIGHT"
            sg_steps = max(1, nav_steps)
            sg_dist = dist_m if dist_m > 0 else 0.5
        elif act in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
            sg_action = "WALK_FORWARD"
            sg_steps = nav_steps
            sg_dist = dist_m
        elif act == "SLOW_DOWN":
            sg_action = "WALK_FORWARD"
            sg_steps = max(1, nav_steps)
            sg_dist = dist_m
        elif act in ("BACK", "GO_BACK"):
            sg_action = "WALK_BACK"
            sg_steps = max(1, nav_steps)
            sg_dist = dist_m if dist_m > 0 else 0.75
        else:
            sg_action = "NONE"
            sg_steps = 0
            sg_dist = 0.0

        if sg_steps == 0:
            step_cat = "0 steps"
        elif sg_steps <= 1:
            step_cat = "0–1 steps"
        elif sg_steps <= 3:
            step_cat = "2–3 steps"
        elif sg_steps <= 6:
            step_cat = "4–6 steps"
        else:
            step_cat = "6+ steps"

        return {
            "action": sg_action,
            "steps": sg_steps,
            "step_category": step_cat,
            "distance_m": round(sg_dist, 2),
            "step_length_m": self.step_length_m,
            "safety_margin_m": self.safety_margin_m,
            "step_confidence": step_conf,
            "confidence": step_conf,
            "reason": step_reason,
        }

    # ========================================================
    # MAIN ANALYZE
    # ========================================================

    def analyze(
        self,
        objects: Optional[
            List[Dict[str, Any]]
        ] = None,
        scene_depth: Optional[
            Dict[str, Any]
        ] = None,
        frame_width: float = 640,
        frame_height: float = 480,
        unknown_objects: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Main decision-engine function.

        Input:

            objects:
                YOLO detections with depth.

            scene_depth:
                MiDaS scene depth information.

            frame_width:
                Camera frame width.

            frame_height:
                Camera frame height.

            unknown_objects:
                Candidate physical obstacles not classified by YOLO.

        Output:

            safety_level
            emergency_stop
            navigation
            step_guidance
            voice_instruction
            primary_obstacle
            objects
            unknown_objects
            free_path
            scene
            multiple_objects
            predictive_threat
            approaching
            approach_rate
            obstacle_stability
            emergency_alert
            navigation_confidence
        """

        if objects is None:
            objects = []

        if scene_depth is None:
            scene_depth = {}

        # ----------------------------------------------------
        # Normalize detections
        # ----------------------------------------------------

        normalized = self.normalize_objects(
            objects,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        # ----------------------------------------------------
        # Remove duplicates
        # ----------------------------------------------------

        filtered_objects = (
            self.remove_duplicates(
                normalized
            )
        )

        # ----------------------------------------------------
        # Sort by risk
        # ----------------------------------------------------

        filtered_objects.sort(
            key=lambda item:
                self._safe_float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            reverse=True,
        )

        # ----------------------------------------------------
        # Unknown obstacles
        # ----------------------------------------------------

        normalized_unknowns = self.normalize_unknown_objects(
            unknown_objects,
            frame_width=frame_width,
            frame_height=frame_height,
            known_objects=filtered_objects,
        )

        # ----------------------------------------------------
        # Multi-object temporal tracking and distance smoothing (Phase 5 & 11)
        # ----------------------------------------------------

        self.track_and_smooth_distances(
            objects=filtered_objects,
            unknown_objects=normalized_unknowns,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        # ----------------------------------------------------
        # Sort by risk
        # ----------------------------------------------------

        filtered_objects.sort(
            key=lambda item:
                self._safe_float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            reverse=True,
        )

        normalized_unknowns.sort(
            key=lambda item:
                self._safe_float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            reverse=True,
        )

        # Canonical contract list for response with Phase 11 fields
        canonical_unknowns = [
            {
                "id": u["id"],
                "class_name": "unknown_obstacle",
                "position": u["position"],
                "bbox": u["bbox"],
                "center_x": u.get("center_x", 0.0),
                "center_y": u.get("center_y", 0.0),
                "width": u.get("width", 0.0),
                "height": u.get("height", 0.0),
                "distance_m": u["distance_m"],
                "distance_category": u["distance_category"],
                "distance_confidence": u["distance_confidence"],
                "risk_score": u["risk_score"],
                "navigation_relevance": u["navigation_relevance"],
                "confidence": u["confidence"],
                "motion_state": u.get("motion_state", "UNKNOWN_MOTION"),
                "approach_rate_mps": u.get("approach_rate_mps", 0.0),
                "time_to_collision_s": u.get("time_to_collision_s"),
                "approach_score": u.get("approach_score", 0.0),
                "is_approaching": u.get("is_approaching", False),
                "approaching": u.get("approaching", False),
                "approach_rate": u.get("approach_rate", 0.0),
            }
            for u in normalized_unknowns
        ]

        # ----------------------------------------------------
        # Free path (Phase 7 Safe Walking Corridor Detection)
        # ----------------------------------------------------

        free_path = (
            self.analyze_free_path(
                scene_depth=scene_depth,
                objects=filtered_objects,
                unknown_objects=normalized_unknowns,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )

        # ----------------------------------------------------
        # Primary obstacle
        # ----------------------------------------------------

        all_candidates = list(filtered_objects)
        all_candidates.extend(normalized_unknowns)
        all_candidates.sort(
            key=lambda item:
                self._safe_float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            reverse=True,
        )

        primary_obstacle: Optional[
            Dict[str, Any]
        ] = None

        if all_candidates:
            primary_obstacle = (
                all_candidates[0]
            )

        tracking_info = self.update_tracking_and_approach(
            primary_obstacle
        )

        # ----------------------------------------------------
        # Multi-object reasoning
        # ----------------------------------------------------

        multi_object = (
            self.multi_object_reasoning(
                filtered_objects,
                unknown_objects=normalized_unknowns,
            )
        )

        # ----------------------------------------------------
        # Predictive threat assessment (Phase 11)
        # ----------------------------------------------------

        predictive_threat = self.compute_predictive_threat(
            objects=filtered_objects,
            unknown_objects=normalized_unknowns,
            primary_obstacle=primary_obstacle,
        )

        # ----------------------------------------------------
        # Emergency logic
        # ----------------------------------------------------

        emergency_stop = (
            self.should_emergency_stop(
                primary_obstacle,
                scene_depth,
                predictive_threat=predictive_threat,
            )
        )

        if predictive_threat.get("threat_level") == "EMERGENCY":
            emergency_stop = True

        # ----------------------------------------------------
        # Primary risk
        # ----------------------------------------------------

        if primary_obstacle:

            primary_risk = (
                self._safe_float(
                    primary_obstacle.get(
                        "risk_score",
                        0.0,
                    )
                )
            )

        else:

            primary_risk = 0.0

        # Multi-object collective center path reasoning:
        # If multiple obstacles occupy the center path, elevate risk so the system
        # does not conclude "center is safe" simply because each obstacle alone has moderate risk.
        center_count = int(multi_object.get("center_obstacles", 0))
        left_count = int(multi_object.get("left_obstacles", 0))
        right_count = int(multi_object.get("right_obstacles", 0))

        if center_count >= 2:
            collective_bonus = min(0.20, (center_count - 1) * 0.10)
            primary_risk = min(0.95, max(primary_risk + collective_bonus, self.aware_threshold + 0.05))
            if primary_risk >= 0.50 and isinstance(free_path, dict):
                free_path["is_center_blocked"] = True
                free_path["center_clear"] = False
                free_path["center_blocked"] = True
                if "CENTER" in free_path.get("safe_directions", []):
                    free_path["safe_directions"].remove("CENTER")
                free_path["no_safe_path"] = bool(len(free_path.get("safe_directions", [])) == 0)
                if free_path.get("best_direction") == "CENTER":
                    rem = free_path.get("safe_directions", [])
                    if "RIGHT" in rem and "LEFT" in rem:
                        free_path["best_direction"] = "RIGHT" if free_path.get("right_risk", 1.0) <= free_path.get("left_risk", 1.0) else "LEFT"
                    elif rem:
                        free_path["best_direction"] = rem[0]
                    else:
                        free_path["best_direction"] = "RIGHT" if free_path.get("right_risk", 1.0) <= free_path.get("left_risk", 1.0) else "LEFT"

        # Multi-obstacle all-corridor reasoning (Requirement 2):
        # If obstacles occupy center, left, and right simultaneously
        if center_count >= 1 and left_count >= 1 and right_count >= 1:
            if isinstance(free_path, dict):
                if free_path.get("left_risk", 0.0) >= 0.35 and free_path.get("right_risk", 0.0) >= 0.35:
                    free_path["left_clear"] = False
                    free_path["right_clear"] = False
                    free_path["center_clear"] = False
                    free_path["safe_directions"] = []
                    free_path["no_safe_path"] = True

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        safety_level = (
            self.get_safety_level(
                primary_risk,
                emergency_stop,
            )
        )

        # Predictive safety escalation (Phase 11)
        pt_pos = str(predictive_threat.get("position", "")).upper()
        pt_mot = str(predictive_threat.get("motion_state", "")).upper()
        pt_dist = self._safe_float(predictive_threat.get("distance_m", 99.0), 99.0)
        pt_threat = str(predictive_threat.get("threat_level", "NONE")).upper()

        if not emergency_stop:
            if pt_threat == "EMERGENCY":
                emergency_stop = True
                safety_level = "EMERGENCY"
            elif pt_threat == "DANGER" or (pt_pos == "CENTER" and pt_mot == "RAPIDLY_APPROACHING" and pt_dist <= 2.5):
                if safety_level in ("SAFE", "AWARE", "CAUTION"):
                    safety_level = "DANGER"
            elif pt_threat == "CAUTION" or (pt_pos == "CENTER" and pt_mot in ("APPROACHING", "RAPIDLY_APPROACHING") and pt_dist <= 2.2):
                if safety_level in ("SAFE", "AWARE"):
                    safety_level = "CAUTION"

        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        navigation = (
            self.determine_navigation(
                safety_level=safety_level,
                emergency_stop=emergency_stop,
                primary_obstacle=primary_obstacle,
                free_path=free_path,
                predictive_threat=predictive_threat,
            )
        )

        # ----------------------------------------------------
        # Voice
        # ----------------------------------------------------

        voice_instruction = (
            self.generate_voice_instruction(
                safety_level=safety_level,
                emergency_stop=emergency_stop,
                primary_obstacle=primary_obstacle,
                navigation=navigation.get("action", navigation) if isinstance(navigation, dict) else navigation,
                objects=filtered_objects,
                steps=navigation.get("movement_steps", navigation.get("steps", 0)) if isinstance(navigation, dict) else 0,
                predictive_threat=predictive_threat,
            )
        )

        # ----------------------------------------------------
        # Scene normalization
        # ----------------------------------------------------

        scene = {
            "average_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "average_depth",
                            1.0,
                        )
                    ),
                    4,
                ),

            "center_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "center_depth",
                            1.0,
                        )
                    ),
                    4,
                ),

            "left_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "left_depth",
                            1.0,
                        )
                    ),
                    4,
                ),

            "center_region_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "center_region_depth",
                            scene_depth.get(
                                "center_depth",
                                1.0,
                            ),
                        )
                    ),
                    4,
                ),

            "right_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "right_depth",
                            1.0,
                        )
                    ),
                    4,
                ),
        }

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        structured_navigation = self.build_navigation_result(
            raw_direction=navigation,
            safety_level=safety_level,
            emergency_stop=emergency_stop,
            primary_obstacle=primary_obstacle,
            predictive_threat=predictive_threat,
        )

        step_guidance = self.build_step_guidance(
            navigation=structured_navigation,
            primary_obstacle=primary_obstacle,
        )

        return {
            "safety_level":
                safety_level,

            "emergency_stop":
                bool(
                    emergency_stop
                ),

            "navigation":
                structured_navigation,

            "step_guidance":
                step_guidance,

            "voice_instruction":
                voice_instruction,

            "primary_obstacle":
                primary_obstacle,

            "objects":
                filtered_objects,

            "unknown_objects":
                canonical_unknowns,

            "free_path":
                free_path,

            "scene":
                scene,

            "multiple_objects":
                multi_object,

            "predictive_threat":
                predictive_threat,

            # Compatibility alias
            "multi_object":
                multi_object,

            "approaching":
                bool(
                    predictive_threat.get(
                        "is_approaching",
                        tracking_info.get(
                            "approaching",
                            False,
                        ),
                    )
                ),

            "approach_rate":
                float(
                    predictive_threat.get(
                        "approach_rate_mps",
                        tracking_info.get(
                            "approach_rate",
                            0.0,
                        ),
                    )
                ),

            "obstacle_stability":
                int(
                    tracking_info.get(
                        "obstacle_stability",
                        0,
                    )
                ),

            "emergency_alert":
                bool(
                    emergency_stop
                    or safety_level == "EMERGENCY"
                ),

            "navigation_confidence":
                0.92 if safety_level != "UNKNOWN" else 0.50,
        }


# ============================================================
# EXPORT
# ============================================================

__all__ = [
    "DecisionEngine",
]