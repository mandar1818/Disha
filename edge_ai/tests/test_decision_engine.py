import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.decision_engine import DecisionEngine


def create_detection(
    class_id,
    class_name,
    confidence
):
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "box": [100, 100, 300, 300]
    }


# ---------------------------------------------------------
# CREATE ENGINE
# ---------------------------------------------------------

engine = DecisionEngine(
    confirmation_frames=3
)


print("=" * 60)
print("DECISION ENGINE TEST")
print("=" * 60)


# ---------------------------------------------------------
# FRAME 1
# ---------------------------------------------------------

detections = [
    create_detection(
        0,
        "person",
        0.90
    ),
    create_detection(
        62,
        "laptop",
        0.87
    ),
    create_detection(
        56,
        "chair",
        0.50
    )
]

result = engine.process_frame(
    detections
)

print()
print("FRAME 1")
print("Confirmed:", result)


# ---------------------------------------------------------
# FRAME 2
# ---------------------------------------------------------

result = engine.process_frame(
    detections
)

print()
print("FRAME 2")
print("Confirmed:", result)


# ---------------------------------------------------------
# FRAME 3
# ---------------------------------------------------------

result = engine.process_frame(
    detections
)

print()
print("FRAME 3")

for detection in result:

    print(
        f"{detection['class_name']:15s}"
        f" confidence={detection['confidence']:.2f}"
        f" priority={detection['priority']}"
        f" frames={detection['confirmation_frames']}"
    )


# ---------------------------------------------------------
# PRIORITY TEST
# ---------------------------------------------------------

print()
print("=" * 60)
print("PRIORITY TEST")
print("=" * 60)

test_classes = [
    "car",
    "person",
    "chair",
    "laptop",
    "knife",
    "cup"
]

for class_name in test_classes:

    priority = engine.get_priority(
        class_name
    )

    print(
        f"{class_name:15s} -> {priority}"
    )


print()
print("=" * 60)
print("DECISION ENGINE TEST COMPLETE")
print("=" * 60)