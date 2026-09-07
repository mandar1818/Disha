import cv2

from src.detector import YOLOv8TFLiteDetector
from src.depth_estimator import DepthEstimator
from src.proximity_engine import ProximityEngine


IMAGE_PATH = "tests/test.jpg"


print("=" * 70)
print("YOLO + MiDaS + PROXIMITY TEST")
print("=" * 70)


# ------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

image_height, image_width = frame.shape[:2]

print(
    f"Image size: {image_width} x {image_height}"
)


# ------------------------------------------------------
# LOAD YOLO
# ------------------------------------------------------

detector = YOLOv8TFLiteDetector(
    model_path="models/yolov8n.tflite",
    confidence_threshold=0.50
)


# ------------------------------------------------------
# LOAD DEPTH MODEL
# ------------------------------------------------------

depth_estimator = DepthEstimator(
    model_path="models/midas_small.onnx"
)


# ------------------------------------------------------
# DETECT OBJECTS
# ------------------------------------------------------

detections = detector.detect(frame)

print(
    f"\nYOLO detections: {len(detections)}"
)


# ------------------------------------------------------
# ESTIMATE DEPTH
# ------------------------------------------------------

depth_map = depth_estimator.estimate(frame)

print(
    f"Depth map: {depth_map.shape}"
)


# ------------------------------------------------------
# PROXIMITY ENGINE
# ------------------------------------------------------

proximity_engine = ProximityEngine(
    near_area=0.12,
    very_close_area=0.30,
    center_threshold=0.25,
    near_depth=0.65,
    very_close_depth=0.82
)


objects = proximity_engine.process(
    detections,
    image_width,
    image_height,
    depth_map
)


# ------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------

print("\nObject proximity:")
print("-" * 70)

for obj in objects:

    print(
        f"{obj['class_name']:<15}"
        f" confidence={obj['confidence']:.3f}"
        f" area={obj['area_ratio']:.3f}"
        f" depth={obj.get('relative_depth', 0.0):.3f}"
        f" depth_state={obj.get('depth_proximity', 'N/A'):<11}"
        f" proximity={obj['proximity']:<11}"
        f" in_path={obj['in_path']}"
    )


print("\n" + "=" * 70)
print("YOLO + MiDaS + PROXIMITY TEST COMPLETE")
print("=" * 70)