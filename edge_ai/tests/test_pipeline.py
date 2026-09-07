import sys
import os
import cv2
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.detector import YOLOv8TFLiteDetector
from src.decision_engine import DecisionEngine


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------

image_path = "tests/test.jpg"

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image")
    sys.exit(1)


# ---------------------------------------------------------
# CREATE DETECTOR
# ---------------------------------------------------------

detector = YOLOv8TFLiteDetector(
    confidence_threshold=0.50,
    iou_threshold=0.45
)


# ---------------------------------------------------------
# CREATE DECISION ENGINE
# ---------------------------------------------------------

decision_engine = DecisionEngine(
    confirmation_frames=3
)


print()
print("=" * 70)
print("YOLO + DECISION ENGINE PIPELINE TEST")
print("=" * 70)


# ---------------------------------------------------------
# SIMULATE 5 CONSECUTIVE FRAMES
#
# We use the same test image repeatedly.
# In the real system these will be different
# camera frames.
# ---------------------------------------------------------

for frame_number in range(1, 6):

    print()
    print("-" * 70)
    print(f"FRAME {frame_number}")
    print("-" * 70)

    # Run YOLO
    detections = detector.detect(image)

    print()
    print("YOLO detections:")

    for detection in detections:

        print(
            f"  {detection['class_name']:15s}"
            f" confidence={detection['confidence']:.3f}"
        )

    # Send YOLO results to Decision Engine
    confirmed = decision_engine.process_frame(
        detections
    )

    print()
    print("Confirmed objects:")

    if not confirmed:

        print("  None")

    else:

        for detection in confirmed:

            print(
                f"  {detection['class_name']:15s}"
                f" confidence={detection['confidence']:.3f}"
                f" priority={detection['priority']:7s}"
                f" frames={detection['confirmation_frames']}"
            )

    time.sleep(0.2)


print()
print("=" * 70)
print("PIPELINE TEST COMPLETE")
print("=" * 70)