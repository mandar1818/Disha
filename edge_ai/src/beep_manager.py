import threading
import time
import winsound


class BeepManager:
    """
    Non-blocking beep manager for the Edge AI system.

    The beep runs in a background thread so that the
    main AI pipeline can continue processing frames.

    States:
        START_BEEP
        CONTINUE_BEEP
        STOP_BEEP
        NO_ACTION
    """

    def __init__(
        self,
        frequency=1000,
        duration_ms=250,
        interval_seconds=0.05
    ):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.interval_seconds = interval_seconds

        self.beeping = False
        self._stop_event = threading.Event()
        self._beep_thread = None

        self.lock = threading.Lock()

    def _beep_loop(self):
        """
        Background beep loop.

        Continues until the stop event is triggered.
        """

        while not self._stop_event.is_set():

            try:
                winsound.Beep(
                    self.frequency,
                    self.duration_ms
                )
            except RuntimeError:
                break

            if self._stop_event.wait(
                self.interval_seconds
            ):
                break

    def start(self):
        """
        Start the beep in a background thread.
        """

        with self.lock:

            if self.beeping:
                return "CONTINUE_BEEP"

            self._stop_event.clear()

            self.beeping = True

            self._beep_thread = threading.Thread(
                target=self._beep_loop,
                daemon=True
            )

            self._beep_thread.start()

            return "START_BEEP"

    def stop(self):
        """
        Stop the background beep.
        """

        with self.lock:

            if not self.beeping:
                return "NO_ACTION"

            self._stop_event.set()

            self.beeping = False

            thread = self._beep_thread
            self._beep_thread = None

        if thread is not None:
            thread.join(timeout=0.5)

        return "STOP_BEEP"

    def update(self, path_result):
        """
        Update beep state based on path analysis.

        Beep when the path is blocked or when multiple
        nearby objects are detected.
        """

        beep_required = path_result.get(
            "beep_required",
            False
        )

        if beep_required:

            if self.beeping:
                return "CONTINUE_BEEP"

            return self.start()

        else:

            if self.beeping:
                return self.stop()

            return "NO_ACTION"

    def is_beeping(self):
        """
        Return current beep state.
        """

        return self.beeping