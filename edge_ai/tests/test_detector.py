import sys
import os

# Add the project root to Python's import path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import cv2

from src.detector import YOLOv8TFLiteDetector


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "yolov8n.tflite"
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "tests",
    "test.jpg"
)

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45


# --------------------------------------------------
# Load detector
# --------------------------------------------------

print("=" * 60)
print("YOLOv8n TFLite DETECTOR TEST")
print("=" * 60)

detector = YOLOv8TFLiteDetector(
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    iou_threshold=IOU_THRESHOLD
)


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"\nCould not load image:\n{IMAGE_PATH}\n"
        "\nMake sure test.jpg exists inside tests folder."
    )


height, width = image.shape[:2]

print("\nTest image:")
print(f"Width  : {width}")
print(f"Height : {height}")

print(f"\nConfidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"IoU threshold       : {IOU_THRESHOLD}")


# --------------------------------------------------
# Run detection
# --------------------------------------------------

print("\nRunning TFLite inference...")

detections = detector.detect(image)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL DETECTIONS")
print("=" * 60)

if len(detections) == 0:

    print("No objects detected above confidence threshold.")

else:

    for i, detection in enumerate(detections, start=1):

        class_name = detection["class_name"]
        confidence = detection["confidence"]
        box = detection["box"]

        x1, y1, x2, y2 = box

        print(
            f"{i:2d}. "
            f"{class_name:20s} "
            f"confidence = {confidence:.3f} "
            f"Box = "
            f"[{x1:.1f}, {y1:.1f}, "
            f"{x2:.1f}, {y2:.1f}]"
        )


print("-" * 60)
print(
    f"Total final detections: {len(detections)}"
)

print("=" * 60)
print("DETECTOR TEST COMPLETED")
print("=" * 60)