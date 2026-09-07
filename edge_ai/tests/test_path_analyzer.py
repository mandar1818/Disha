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
from src.path_analyzer import PathAnalyzer


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------

image_path = "tests/test.jpg"

image = cv2.imread(
    image_path
)

if image is None:
    print(
        "ERROR: Could not load image"
    )
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

path_analyzer = PathAnalyzer()


print()
print("=" * 70)
print("PATH ANALYZER TEST")
print("=" * 70)


# ---------------------------------------------------------
# PROCESS 3 FRAMES
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

    # 3-frame confirmation
    confirmed = decision_engine.process_frame(
        detections
    )

    if not confirmed:

        print(
            "No confirmed objects yet."
        )
        continue

    # Proximity
    proximity_results = (
        proximity_engine.process(
            confirmed,
            image_width,
            image_height
        )
    )

    # Path analysis
    path_result = path_analyzer.analyze(
        proximity_results
    )

    print()

    print(
        "Path blocked:",
        path_result["path_blocked"]
    )

    print(
        "Nearby objects:",
        path_result["nearby_count"]
    )

    print(
        "Multiple objects:",
        path_result["multiple_objects"]
    )

    print(
        "Beep required:",
        path_result["beep_required"]
    )

    print()

    print("Nearby obstacles:")

    if not path_result[
        "nearby_objects"
    ]:

        print(
            "  None"
        )

    else:

        for obj in path_result[
            "nearby_objects"
        ]:

            print(
                f"  {obj['class_name']:15s}"
                f" priority={obj['priority']:7s}"
                f" proximity={obj['proximity']}"
            )

    if path_result[
        "highest_priority"
    ]:

        obj = path_result[
            "highest_priority"
        ]

        print()

        print(
            "Highest priority obstacle:",
            obj["class_name"]
        )


print()
print("=" * 70)
print("PATH ANALYZER TEST COMPLETE")
print("=" * 70)