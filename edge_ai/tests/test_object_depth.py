import cv2

from src.detector import YOLOv8TFLiteDetector
from src.depth_estimator import DepthEstimator


print("=" * 70)
print("YOLO + DEPTH OBJECT TEST")
print("=" * 70)

# --------------------------------------------------
# Load image
# --------------------------------------------------

image_path = "tests/test.jpg"

frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

height, width = frame.shape[:2]

print("Image size:", width, "x", height)

# --------------------------------------------------
# Load YOLO
# --------------------------------------------------

detector = YOLOv8TFLiteDetector(
    model_path="models/yolov8n.tflite",
    confidence_threshold=0.50,
    iou_threshold=0.45
)

# --------------------------------------------------
# Load MiDaS
# --------------------------------------------------

depth_estimator = DepthEstimator(
    model_path="models/midas_small.onnx"
)

# --------------------------------------------------
# YOLO detection
# --------------------------------------------------

detections = detector.detect(frame)

# --------------------------------------------------
# Depth estimation
# --------------------------------------------------

depth_map = depth_estimator.estimate(frame)

print()
print("Depth map:", depth_map.shape)

# --------------------------------------------------
# Get depth for each detected object
# --------------------------------------------------

print()
print("Object depth:")
print("-" * 70)

for detection in detections:

    class_name = detection["class_name"]
    confidence = detection["confidence"]
    bbox = detection["box"]

    depth = depth_estimator.estimate_bbox_depth(
        depth_map,
        bbox,
        width,
        height
    )

    print(
        f"{class_name:<15}"
        f"confidence={confidence:.3f}  "
        f"relative_depth={depth:.3f}"
    )

print()
print("=" * 70)
print("YOLO + DEPTH OBJECT TEST COMPLETE")
print("=" * 70)