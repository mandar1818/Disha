class IntegrationOutput:
    """
    Converts the internal Edge AI pipeline result into
    a clean structure that can later be passed to
    React Native.
    """

    def build(self, pipeline_result):

        # --------------------------------------------------
        # GET INTERNAL RESULTS
        # --------------------------------------------------

        objects = pipeline_result.get("objects", [])

        path = pipeline_result.get("path", {})

        safety = pipeline_result.get(
            "safety_decision",
            {}
        )

        alerts = pipeline_result.get(
            "alerts",
            []
        )

        # --------------------------------------------------
        # FORMAT OBJECTS
        # --------------------------------------------------

        output_objects = []

        for obj in objects:

            output_objects.append({
                "name": obj.get(
                    "class_name"
                ),

                "confidence": round(
                    float(
                        obj.get(
                            "confidence",
                            0.0
                        )
                    ),
                    3
                ),

                "priority": obj.get(
                    "priority",
                    "normal"
                ),

                "proximity": obj.get(
                    "proximity",
                    "far"
                ),

                "direction": obj.get(
                    "direction",
                    "center"
                ),

                "in_path": bool(
                    obj.get(
                        "in_path",
                        False
                    )
                )
            })

        # --------------------------------------------------
        # NEARBY OBJECTS
        # --------------------------------------------------

        nearby_objects = path.get(
            "nearby_objects",
            []
        )

        # The pipeline may provide either:
        #
        # 1. a list of nearby objects
        # 2. an integer count
        #
        # Handle both safely.

        if isinstance(
            nearby_objects,
            list
        ):

            nearby_count = len(
                nearby_objects
            )

        else:

            try:

                nearby_count = int(
                    nearby_objects
                )

            except (
                TypeError,
                ValueError
            ):

                nearby_count = 0

        # --------------------------------------------------
        # VOICE MESSAGE
        # --------------------------------------------------

        voice_message = None

        if alerts:

            voice_message = alerts[0].get(
                "message"
            )

        # --------------------------------------------------
        # FINAL OUTPUT
        # --------------------------------------------------

        return {

            "objects": output_objects,

            "path_blocked": bool(
                path.get(
                    "path_blocked",
                    False
                )
            ),

            "multiple_objects": bool(
                path.get(
                    "multiple_objects",
                    False
                )
            ),

            "nearby_objects": nearby_count,

            "risk_level": safety.get(
                "risk_level",
                "normal"
            ),

            "situation": safety.get(
                "situation",
                "clear"
            ),

            "voice_message": voice_message,

            "beep": bool(
                safety.get(
                    "beep_required",
                    False
                )
            )
        }