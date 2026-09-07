import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.beep_manager import BeepManager


manager = BeepManager()


print("=" * 60)
print("BEEP MANAGER TEST")
print("=" * 60)


# ---------------------------------------------------------
# PATH CLEAR
# ---------------------------------------------------------

clear_path = {
    "beep_required": False
}

result = manager.update(
    clear_path
)

print()
print("PATH CLEAR:")
print(result)


# ---------------------------------------------------------
# OBSTACLE APPEARS
# ---------------------------------------------------------

blocked_path = {
    "beep_required": True
}

result = manager.update(
    blocked_path
)

print()
print("OBSTACLE APPEARS:")
print(result)


# ---------------------------------------------------------
# OBSTACLE REMAINS
# ---------------------------------------------------------

result = manager.update(
    blocked_path
)

print()
print("OBSTACLE REMAINS:")
print(result)


# ---------------------------------------------------------
# STILL BLOCKED
# ---------------------------------------------------------

result = manager.update(
    blocked_path
)

print()
print("STILL BLOCKED:")
print(result)


# ---------------------------------------------------------
# PATH CLEARS
# ---------------------------------------------------------

result = manager.update(
    clear_path
)

print()
print("PATH CLEARS:")
print(result)


# ---------------------------------------------------------
# REMAINS CLEAR
# ---------------------------------------------------------

result = manager.update(
    clear_path
)

print()
print("REMAINS CLEAR:")
print(result)


print()
print("=" * 60)
print("BEEP MANAGER TEST COMPLETE")
print("=" * 60)