from src.direction_engine import DirectionEngine


def main():

    print("=" * 80)
    print("DIRECTION ENGINE TEST")
    print("=" * 80)

    image_width = 1300

    engine = DirectionEngine()

    test_objects = [
        {
            "class_name": "chair",
            "box": [50, 200, 300, 500]
        },
        {
            "class_name": "person",
            "box": [500, 200, 800, 700]
        },
        {
            "class_name": "car",
            "box": [950, 200, 1250, 700]
        }
    ]

    results = engine.process(
        test_objects,
        image_width
    )

    print()

    for result in results:

        print(
            f"{result['class_name']:10} "
            f"direction={result['direction']}"
        )

    print()
    print("=" * 80)
    print("DIRECTION ENGINE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()