from src.beep_manager import BeepManager


manager = BeepManager()


clear_path = {
    "beep_required": False
}

blocked_path = {
    "beep_required": True
}


print("=" * 60)
print("BEEP AUDIO TEST")
print("=" * 60)

print("\n1. Path clear")
print("Action:", manager.update(clear_path))

print("\n2. Obstacle appears")
print("Action:", manager.update(blocked_path))

print("\n3. Obstacle remains")
print("Action:", manager.update(blocked_path))

print("\n4. Path clears")
print("Action:", manager.update(clear_path))

print("\n5. Path remains clear")
print("Action:", manager.update(clear_path))

print("\n" + "=" * 60)
print("BEEP AUDIO TEST COMPLETE")
print("=" * 60)