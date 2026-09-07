from src.integration_output import IntegrationOutput


def main():

    print("=" * 80)
    print("EDGE AI INTEGRATION OUTPUT TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Simulated internal Edge AI result
    # --------------------------------------------------

    pipeline_result = {

        "objects": [
            {
                "class_name": "person",
                "confidence": 0.901,
                "priority": "warning",
                "proximity": "very_close",
                "direction": "center",
                "in_path": True
            },
            {
                "class_name": "chair",
                "confidence": 0.505,
                "priority": "warning",
                "proximity": "very_close",
                "direction": "left",
                "in_path": True
            },
            {
                "class_name": "laptop",
                "confidence": 0.876,
                "priority": "normal",
                "proximity": "far",
                "direction": "right",
                "in_path": False
            }
        ],

        "path": {
            "path_blocked": True,
            "nearby_objects": 2,
            "multiple_objects": True
        },

        "safety_decision": {
            "risk_level": "warning",
            "situation": "multiple_obstacles",
            "beep_required": True
        },

        "alerts": [
            {
                "class_name": "person",
                "priority": "warning",
                "proximity": "very_close",
                "message": "person very close"
            }
        ]
    }

    # --------------------------------------------------
    # Convert to integration output
    # --------------------------------------------------

    output_engine = IntegrationOutput()

    output = output_engine.build(
        pipeline_result
    )

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    print()
    print("FINAL OUTPUT:")
    print()

    print(output)

    print()

    print("=" * 80)
    print("OBJECTS")
    print("=" * 80)

    for obj in output["objects"]:

        print(
            f"{obj['name']:10} | "
            f"confidence={obj['confidence']:.3f} | "
            f"priority={obj['priority']:7} | "
            f"proximity={obj['proximity']:10} | "
            f"direction={obj['direction']:6} | "
            f"in_path={obj['in_path']}"
        )

    print()

    print("=" * 80)
    print("SAFETY INFORMATION")
    print("=" * 80)

    print(
        "Path blocked:",
        output["path_blocked"]
    )

    print(
        "Multiple objects:",
        output["multiple_objects"]
    )

    print(
        "Nearby objects:",
        output["nearby_objects"]
    )

    print(
        "Risk level:",
        output["risk_level"]
    )

    print(
        "Situation:",
        output["situation"]
    )

    print(
        "Voice message:",
        output["voice_message"]
    )

    print(
        "Beep:",
        output["beep"]
    )

    print()

    # --------------------------------------------------
    # Basic validation
    # --------------------------------------------------

    assert "objects" in output
    assert "path_blocked" in output
    assert "multiple_objects" in output
    assert "nearby_objects" in output
    assert "risk_level" in output
    assert "situation" in output
    assert "voice_message" in output
    assert "beep" in output

    assert len(output["objects"]) == 3

    assert output["objects"][0]["name"] == "person"
    assert output["objects"][0]["direction"] == "center"

    assert output["objects"][1]["name"] == "chair"
    assert output["objects"][1]["direction"] == "left"

    assert output["objects"][2]["name"] == "laptop"
    assert output["objects"][2]["direction"] == "right"

    assert output["path_blocked"] is True
    assert output["multiple_objects"] is True
    assert output["nearby_objects"] == 2

    assert output["risk_level"] == "warning"

    assert (
        output["situation"]
        == "multiple_obstacles"
    )

    assert (
        output["voice_message"]
        == "person very close"
    )

    assert output["beep"] is True

    print("=" * 80)
    print("ALL INTEGRATION OUTPUT TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()