from .detector import YOLOv8TFLiteDetector
from .decision_engine import DecisionEngine
from .proximity_engine import ProximityEngine
from .path_analyzer import PathAnalyzer
from .alert_manager import AlertManager
from .alert_selector import AlertSelector
from .beep_manager import BeepManager
from .safety_decision_engine import SafetyDecisionEngine
from .depth_estimator import DepthEstimator
from .direction_engine import DirectionEngine


class EdgeAIPipeline:
    """
    Complete Edge AI pipeline.

    Processing flow:

        Camera Frame
              ↓
        YOLOv8n TFLite
              ↓
        Confidence Filtering + NMS
              ↓
        3-Frame Confirmation
              ↓
        MiDaS Depth Estimation
        (updated every N frames)
              ↓
        Proximity Engine
        (BBox Area + Depth)
              ↓
        Direction Engine
        (Left / Center / Right)
              ↓
        Path Analyzer
              ↓
        Alert Selector
              ↓
        Safety Decision Engine
              ↓
        Alert Manager
              ↓
        Voice / Beep

    Performance optimization:
        YOLO runs on every frame.

        MiDaS runs only every `depth_update_interval`
        frames and the latest depth map is reused
        between updates.
    """

    def __init__(self):

        # --------------------------------------------------
        # FRAME / DEPTH STATE
        # --------------------------------------------------

        self.frame_count = 0

        self.depth_update_interval = 3

        self.last_depth_map = None

        # --------------------------------------------------
        # YOLO OBJECT DETECTOR
        # --------------------------------------------------

        self.detector = YOLOv8TFLiteDetector(
            model_path="models/yolov8n.tflite",
            confidence_threshold=0.50,
            iou_threshold=0.45
        )

        # --------------------------------------------------
        # MONOCULAR DEPTH ESTIMATION
        # --------------------------------------------------

        self.depth_estimator = DepthEstimator(
            model_path="models/midas_small.onnx"
        )

        # --------------------------------------------------
        # TEMPORAL CONFIRMATION + PRIORITY
        # --------------------------------------------------

        self.decision_engine = DecisionEngine(
            confirmation_frames=3
        )

        # --------------------------------------------------
        # PROXIMITY ESTIMATION
        # --------------------------------------------------

        self.proximity_engine = ProximityEngine(
            near_area=0.12,
            very_close_area=0.30,
            center_threshold=0.25,
            near_depth=0.65,
            very_close_depth=0.82
        )

        # --------------------------------------------------
        # DIRECTION ENGINE
        # --------------------------------------------------

        self.direction_engine = DirectionEngine()

        # --------------------------------------------------
        # PATH ANALYSIS
        # --------------------------------------------------

        self.path_analyzer = PathAnalyzer(
            minimum_proximity="near",
            multi_object_threshold=2
        )

        # --------------------------------------------------
        # ALERT SELECTION
        # --------------------------------------------------

        self.alert_selector = AlertSelector()

        # --------------------------------------------------
        # ALERT COOLDOWN
        # --------------------------------------------------

        self.alert_manager = AlertManager(
            cooldown_seconds=3.0
        )

        # --------------------------------------------------
        # SAFETY DECISION ENGINE
        # --------------------------------------------------

        self.safety_decision_engine = SafetyDecisionEngine()

        # --------------------------------------------------
        # BEEP MANAGER
        # --------------------------------------------------

        self.beep_manager = BeepManager()

    # ======================================================
    # PROCESS ONE FRAME
    # ======================================================

    def process_frame(self, frame):

        frame_height, frame_width = frame.shape[:2]

        # --------------------------------------------------
        # STEP 1: YOLO DETECTION
        # --------------------------------------------------

        detections = self.detector.detect(frame)

        self.frame_count += 1

        # --------------------------------------------------
        # STEP 2: DEPTH ESTIMATION
        # --------------------------------------------------
        #
        # MiDaS is intentionally not executed on every frame.
        # The latest depth map is reused between updates.
        #
        # Frames:
        #   1 → depth update
        #   2 → reuse
        #   3 → reuse
        #   4 → depth update
        #   5 → reuse
        #   6 → reuse
        #   ...
        # --------------------------------------------------

        if (
            self.last_depth_map is None
            or (self.frame_count - 1)
            % self.depth_update_interval == 0
        ):

            self.last_depth_map = (
                self.depth_estimator.estimate(frame)
            )

        depth_map = self.last_depth_map

        # --------------------------------------------------
        # STEP 3: 3-FRAME CONFIRMATION
        # --------------------------------------------------

        confirmed = self.decision_engine.process_frame(
            detections
        )

        # --------------------------------------------------
        # NO CONFIRMED OBJECTS
        # --------------------------------------------------

        if not confirmed:

            clear_path = {
                "path_blocked": False,
                "nearby_objects": 0,
                "multiple_objects": False,
                "beep_required": False,
                "highest_priority_obstacle": None
            }

            safety_decision = (
                self.safety_decision_engine.evaluate(
                    [],
                    clear_path,
                    None
                )
            )

            beep_action = self.beep_manager.update(
                clear_path
            )

            return {
                "detections": detections,
                "confirmed": [],
                "objects": [],
                "path": clear_path,
                "selected_alert": None,
                "alerts": [],
                "safety_decision": safety_decision,
                "beep_action": beep_action,
                "depth_map": depth_map,
                "frame_count": self.frame_count
            }

        # --------------------------------------------------
        # STEP 4: PROXIMITY ANALYSIS
        # --------------------------------------------------

        objects = self.proximity_engine.process(
            confirmed,
            frame_width,
            frame_height,
            depth_map
        )

        # --------------------------------------------------
        # STEP 5: DIRECTION ANALYSIS
        # --------------------------------------------------
        #
        # Adds:
        #
        #   direction = "left"
        #   direction = "center"
        #   direction = "right"
        #
        # to every detected object.
        # --------------------------------------------------

        objects = self.direction_engine.process(
            objects,
            frame_width
        )

        # --------------------------------------------------
        # STEP 6: PATH ANALYSIS
        # --------------------------------------------------

        path_result = self.path_analyzer.analyze(
            objects
        )

        # --------------------------------------------------
        # STEP 7: SELECT MOST IMPORTANT OBJECT
        # --------------------------------------------------

        selected_object = self.alert_selector.select(
            objects
        )

        # --------------------------------------------------
        # STEP 8: SAFETY DECISION
        # --------------------------------------------------

        safety_decision = (
            self.safety_decision_engine.evaluate(
                objects,
                path_result,
                selected_object
            )
        )

        # --------------------------------------------------
        # STEP 9: VOICE ALERT
        # --------------------------------------------------

        alerts = []

        if selected_object is not None:

            alert = self.alert_manager.create_alert(
                selected_object
            )

            if alert is not None:

                alerts.append(alert)

        # --------------------------------------------------
        # STEP 10: BEEP
        # --------------------------------------------------

        beep_action = self.beep_manager.update(
            path_result
        )

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        return {
            "detections": detections,
            "confirmed": confirmed,
            "objects": objects,
            "path": path_result,
            "selected_alert": selected_object,
            "alerts": alerts,
            "safety_decision": safety_decision,
            "beep_action": beep_action,
            "depth_map": depth_map,
            "frame_count": self.frame_count
        }

    # ======================================================
    # RESET PIPELINE
    # ======================================================

    def reset(self):

        self.frame_count = 0

        self.last_depth_map = None

        self.decision_engine.reset()

        self.alert_manager.reset()

        self.beep_manager.stop()