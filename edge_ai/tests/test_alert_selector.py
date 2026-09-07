from src.alert_selector import AlertSelector


selector = AlertSelector()


test_cases = [
    {
        "name": "Danger overrides warning",
        "objects": [
            {
                "class_name": "person",
                "confidence": 0.95,
                "priority": "warning",
                "proximity": "very_close"
            },
            {
                "class_name": "car",
                "confidence": 0.60,
                "priority": "danger",
                "proximity": "near"
            }
        ],
        "expected": "car"
    },

    {
        "name": "Warning overrides normal",
        "objects": [
            {
                "class_name": "laptop",
                "confidence": 0.99,
                "priority": "normal",
                "proximity": "far"
            },
            {
                "class_name": "chair",
                "confidence": 0.55,
                "priority": "warning",
                "proximity": "near"
            }
        ],
        "expected": "chair"
    },

    {
        "name": "Very close wins within same priority",
        "objects": [
            {
                "class_name": "person",
                "confidence": 0.90,
                "priority": "warning",
                "proximity": "near"
            },
            {
                "class_name": "chair",
                "confidence": 0.60,
                "priority": "warning",
                "proximity": "very_close"
            }
        ],
        "expected": "chair"
    },

    {
        "name": "Higher confidence wins when priority and proximity are same",
        "objects": [
            {
                "class_name": "person",
                "confidence": 0.70,
                "priority": "warning",
                "proximity": "near"
            },
            {
                "class_name": "chair",
                "confidence": 0.90,
                "priority": "warning",
                "proximity": "near"
            }
        ],
        "expected": "chair"
    }
]


print("=" * 70)
print("ALERT SELECTOR PRIORITY TEST")
print("=" * 70)


all_passed = True


for test in test_cases:

    selected = selector.select(test["objects"])

    if selected:
        result = selected["class_name"]
    else:
        result = None

    passed = result == test["expected"]

    print()
    print(test["name"])
    print(f"Expected : {test['expected']}")
    print(f"Selected : {result}")

    if passed:
        print("Result   : PASS")
    else:
        print("Result   : FAIL")
        all_passed = False


print()
print("=" * 70)

if all_passed:
    print("ALL ALERT SELECTOR TESTS PASSED")
else:
    print("SOME TESTS FAILED")

print("=" * 70)