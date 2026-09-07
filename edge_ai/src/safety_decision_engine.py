class SafetyDecisionEngine:
    """
    Central safety decision-making layer.

    Considers:
    - Object priority
    - Proximity
    - Whether the object is in the walking path
    - Multiple nearby obstacles
    - Selected alert object
    """

    PRIORITY_RANK = {
        "danger": 3,
        "warning": 2,
        "normal": 1
    }

    PROXIMITY_RANK = {
        "very_close": 3,
        "near": 2,
        "far": 1
    }

    def evaluate(self, objects, path_result, selected_object=None):

        # --------------------------------------------------
        # NO OBJECTS
        # --------------------------------------------------

        if not objects:
            return {
                "situation": "clear",
                "risk_level": "normal",
                "path_blocked": False,
                "multiple_objects": False,
                "nearby_objects": 0,
                "primary_object": None,
                "voice_required": False,
                "beep_required": False
            }

        # --------------------------------------------------
        # RELEVANT OBJECTS
        # Only objects that are near AND in the path
        # are considered immediate obstacles.
        # --------------------------------------------------

        nearby_objects = [
            obj for obj in objects
            if (
                obj.get("in_path", False)
                and obj.get("proximity") in ("near", "very_close")
            )
        ]

        # --------------------------------------------------
        # DETERMINE PRIMARY SAFETY OBJECT
        # --------------------------------------------------

        if nearby_objects:

            highest_risk_object = max(
                nearby_objects,
                key=lambda obj: (
                    self.PRIORITY_RANK.get(
                        obj.get("priority", "normal"),
                        1
                    ),
                    self.PROXIMITY_RANK.get(
                        obj.get("proximity", "far"),
                        1
                    ),
                    obj.get("confidence", 0.0)
                )
            )

        else:

            # No immediate obstacle.
            # Use selected object only as contextual information.
            highest_risk_object = selected_object

        # --------------------------------------------------
        # RISK LEVEL
        # --------------------------------------------------

        if nearby_objects:

            risk_level = highest_risk_object.get(
                "priority",
                "normal"
            )

        else:

            risk_level = "normal"

        # --------------------------------------------------
        # SITUATION CLASSIFICATION
        # --------------------------------------------------

        if len(nearby_objects) >= 2:

            situation = "multiple_obstacles"

        elif path_result.get("path_blocked", False):

            situation = "obstacle_ahead"

        elif (
            highest_risk_object is not None
            and highest_risk_object.get("proximity") == "very_close"
        ):

            situation = "object_very_close"

        elif (
            highest_risk_object is not None
            and highest_risk_object.get("proximity") == "near"
        ):

            situation = "object_nearby"

        elif objects:

            situation = "objects_detected"

        else:

            situation = "clear"

        # --------------------------------------------------
        # VOICE
        # --------------------------------------------------

        voice_required = selected_object is not None

        # --------------------------------------------------
        # BEEP
        # --------------------------------------------------

        beep_required = path_result.get(
            "beep_required",
            False
        )

        # --------------------------------------------------
        # RETURN SAFETY DECISION
        # --------------------------------------------------

        return {
            "situation": situation,
            "risk_level": risk_level,
            "path_blocked": path_result.get(
                "path_blocked",
                False
            ),
            "multiple_objects": len(nearby_objects) >= 2,
            "nearby_objects": len(nearby_objects),
            "primary_object": (
                highest_risk_object
                if highest_risk_object is not None
                else selected_object
            ),
            "voice_required": voice_required,
            "beep_required": beep_required
        }