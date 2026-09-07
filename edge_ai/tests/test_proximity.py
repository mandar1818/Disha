import sys
import os
import cv2

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.detector import YOLOv8TFLiteDetector
from src.decision_engine import DecisionEngine
from src.proximity_engine import ProximityEngine


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------

image_path = "tests/test.jpg"

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image")
    sys.exit(1)


image_height, image_width = image.shape[:2]


# ---------------------------------------------------------
# CREATE COMPONENTS
# ---------------------------------------------------------

detector = YOLOv8TFLiteDetector(
    confidence_threshold=0.50,
    iou_threshold=0.45
)

decision_engine = DecisionEngine(
    confirmation_frames=3
)

proximity_engine = ProximityEngine()


print()
print("=" * 70)
print("PROXIMITY ENGINE TEST")
print("=" * 70)

print(
    f"Image size: "
    f"{image_width} x {image_height}"
)


# ---------------------------------------------------------
# SIMULATE 3 FRAMES
# ---------------------------------------------------------

for frame_number in range(1, 4):

    print()
    print("-" * 70)
    print(f"FRAME {frame_number}")
    print("-" * 70)

    # YOLO
    detections = detector.detect(
        image
    )

    # Decision engine
    confirmed = decision_engine.process_frame(
        detections
    )

    if not confirmed:

        print("No confirmed objects yet.")
        continue

    # Proximity engine
    results = proximity_engine.process(
        confirmed,
        image_width,
        image_height
    )

    print()

    for detection in results:

        print(
            f"{detection['class_name']:15s}"
            f" confidence={detection['confidence']:.3f}"
            f" priority={detection['priority']:7s}"
            f" proximity={detection['proximity']:10s}"
            f" in_path={detection['in_path']}"
            f" area={detection['area_ratio']:.3f}"
        )


print()
print("=" * 70)
print("PROXIMITY TEST COMPLETE")
print("=" * 70)