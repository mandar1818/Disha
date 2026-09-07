from .edge_ai_pipeline import EdgeAIPipeline
from .integration_output import IntegrationOutput


class EdgeAI:
    """
    Single public entry point for the Edge AI system.

    React Native integration will eventually interact
    with this class instead of directly interacting with
    YOLO, MiDaS, decision engines, or other internal modules.

    Usage:

        edge_ai = EdgeAI()

        result = edge_ai.process(frame)
    """

    def __init__(self):

        # --------------------------------------------------
        # INTERNAL PIPELINE
        # --------------------------------------------------

        self.pipeline = EdgeAIPipeline()

        # --------------------------------------------------
        # APP INTEGRATION OUTPUT
        # --------------------------------------------------

        self.output_formatter = IntegrationOutput()

    # ======================================================
    # PROCESS ONE CAMERA FRAME
    # ======================================================

    def process(self, frame):
        """
        Process one camera frame through the complete
        Edge AI pipeline.

        Args:
            frame:
                OpenCV BGR image/frame.

        Returns:
            Clean dictionary containing only the
            information required by the application.
        """

        pipeline_result = self.pipeline.process_frame(
            frame
        )

        output = self.output_formatter.build(
            pipeline_result
        )

        return output

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):
        """
        Reset all internal Edge AI state.

        This should be called when starting a new
        detection session or when the camera session
        is restarted.
        """

        self.pipeline.reset()