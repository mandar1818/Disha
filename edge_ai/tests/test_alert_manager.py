import sys
import os
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.alert_manager import AlertManager


def create_detection(
    class_name,
    priority,
    proximity
):

    return {
        "class_name": class_name,
        "class_id": 56,
        "confidence": 0.80,
        "box": [100, 100, 500, 500],
        "priority": priority,
        "proximity": proximity
    }


# ---------------------------------------------------------
# CREATE ALERT MANAGER
# ---------------------------------------------------------

manager = AlertManager(
    cooldown_seconds=3.0
)


print("=" * 70)
print("ALERT MANAGER TEST")
print("=" * 70)


# ---------------------------------------------------------
# FIRST ALERT
# ---------------------------------------------------------

chair = create_detection(
    "chair",
    "warning",
    "near"
)

alert = manager.create_alert(
    chair
)

print()
print("ALERT 1:")
print(alert)


# ---------------------------------------------------------
# IMMEDIATE REPEAT
# ---------------------------------------------------------

alert = manager.create_alert(
    chair
)

print()
print("ALERT 2 (immediate repeat):")
print(alert)


# ---------------------------------------------------------
# SIGNIFICANT CHANGE
# ---------------------------------------------------------

chair_close = create_detection(
    "chair",
    "warning",
    "very_close"
)

alert = manager.create_alert(
    chair_close
)

print()
print("ALERT 3 (chair became very close):")
print(alert)


# ---------------------------------------------------------
# IMMEDIATE REPEAT OF VERY CLOSE
# ---------------------------------------------------------

alert = manager.create_alert(
    chair_close
)

print()
print("ALERT 4 (immediate repeat):")
print(alert)


# ---------------------------------------------------------
# WAIT FOR COOLDOWN
# ---------------------------------------------------------

print()
print("Waiting 3 seconds for cooldown...")

time.sleep(3.1)

alert = manager.create_alert(
    chair_close
)

print()
print("ALERT 5 (after cooldown):")
print(alert)


print()
print("=" * 70)
print("ALERT MANAGER TEST COMPLETE")
print("=" * 70)