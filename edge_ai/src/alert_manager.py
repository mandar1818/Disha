import time


class AlertManager:

    def __init__(
        self,
        cooldown_seconds=3.0
    ):
        self.cooldown_seconds = cooldown_seconds

        # Stores the last alert for each object class
        #
        # Example:
        # {
        #     "chair": {
        #         "proximity": "near",
        #         "time": 123456789
        #     }
        # }
        self.last_alerts = {}

    # ------------------------------------------------------
    # CHECK WHETHER ALERT CAN BE SENT
    # ------------------------------------------------------

    def should_alert(self, detection):

        class_name = detection["class_name"]
        proximity = detection["proximity"]

        current_time = time.time()

        # No previous alert for this object
        if class_name not in self.last_alerts:

            return True

        previous = self.last_alerts[class_name]

        previous_time = previous["time"]
        previous_proximity = previous["proximity"]

        elapsed = (
            current_time -
            previous_time
        )

        # --------------------------------------------------
        # SIGNIFICANT CHANGE
        #
        # Example:
        #
        # near → very_close
        # far  → near
        # --------------------------------------------------

        if proximity != previous_proximity:

            return True

        # --------------------------------------------------
        # COOLDOWN
        # --------------------------------------------------

        if elapsed >= self.cooldown_seconds:

            return True

        return False

    # ------------------------------------------------------
    # CREATE ALERT
    # ------------------------------------------------------

    def create_alert(self, detection):

        if not self.should_alert(detection):

            return None

        class_name = detection["class_name"]
        proximity = detection["proximity"]

        current_time = time.time()

        # Store alert state
        self.last_alerts[class_name] = {
            "proximity": proximity,
            "time": current_time
        }

        # --------------------------------------------------
        # CREATE HUMAN-FRIENDLY MESSAGE
        # --------------------------------------------------

        if proximity == "very_close":

            message = (
                f"{class_name} very close"
            )

        elif proximity == "near":

            message = (
                f"{class_name} nearby"
            )

        else:

            message = (
                f"{class_name} ahead"
            )

        return {
            "class_name": class_name,
            "priority": detection["priority"],
            "proximity": proximity,
            "message": message
        }

    # ------------------------------------------------------
    # PROCESS DETECTIONS
    # ------------------------------------------------------

    def process(self, detections):

        alerts = []

        for detection in detections:

            alert = self.create_alert(
                detection
            )

            if alert is not None:

                alerts.append(alert)

        return alerts

    # ------------------------------------------------------
    # CLEAR OLD OBJECT
    # ------------------------------------------------------

    def remove_object(self, class_name):

        if class_name in self.last_alerts:

            del self.last_alerts[class_name]

    # ------------------------------------------------------
    # RESET
    # ------------------------------------------------------

    def reset(self):

        self.last_alerts.clear()