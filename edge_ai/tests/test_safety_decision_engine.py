from src.safety_decision_engine import SafetyDecisionEngine


engine = SafetyDecisionEngine()


objects = [
    {
        "class_name": "person",
        "confidence": 0.90,
        "priority": "warning",
        "proximity": "very_close",
        "in_path": True
    },
    {
        "class_name": "chair",
        "confidence": 0.50,
        "priority": "warning",
        "proximity": "very_close",
        "in_path": True
    },
    {
        "class_name": "laptop",
        "confidence": 0.87,
        "priority": "normal",
        "proximity": "far",
        "in_path": False
    }
]


path_result = {
    "path_blocked": True,
    "nearby_count": 2,
    "multiple_objects": True,
    "beep_required": True
}


selected_object = objects[0]


result = engine.evaluate(
    objects,
    path_result,
    selected_object
)


print("=" * 70)
print("SAFETY DECISION ENGINE TEST")
print("=" * 70)

print()
print("Situation        :", result["situation"])
print("Risk level       :", result["risk_level"])
print("Path blocked     :", result["path_blocked"])
print("Multiple objects :", result["multiple_objects"])
print("Nearby objects   :", result["nearby_objects"])
print("Voice required   :", result["voice_required"])
print("Beep required    :", result["beep_required"])

if result["primary_object"]:
    print(
        "Primary object   :",
        result["primary_object"]["class_name"]
    )

print()
print("=" * 70)