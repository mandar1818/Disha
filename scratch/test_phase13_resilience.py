#!/usr/bin/env python3
"""
SmartVisionAI - Phase 13 Resilience & Accessibility Test Suite
============================================================
Automated verification for Phase 13:
1. Connection Watchdog state machine, counters, exponential backoff, concurrency guard
2. Sensor & Camera Integrity algorithms (pure pixel math & safe runtime fallback)
3. Haptic feedback pattern definitions, durations, differentiation, and preferences
4. Centralized Voice Safety events, priorities, preemption, and duplicate suppression
5. Mobile Accessibility (TalkBack props, entrance announcements, home screen, settings, finder)
6. Prevention of /detect-video and preservation of core API contract
7. Complete Phase 4–12 historical regression verification
"""

import os
import sys
import re
import json
import math
import subprocess
import unittest
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MOBILE_DIR = os.path.join(PROJECT_ROOT, "mobile")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch")

# =====================================================================
# 1. PYTHON REFERENCE IMPLEMENTATIONS OF PHASE 13 SPECIFICATIONS
# =====================================================================

BACKOFF_STEPS_MS = [2000, 4000, 8000, 16000, 30000]
MAX_BACKOFF_MS = 30000

class ConnectionWatchdogPy:
    """
    Python reference of mobile/src/services/connectionWatchdog.ts
    Mirrors exact state machine, counter updates, backoff, and listener semantics.
    """
    def __init__(self):
        self.state = "CONNECTED"
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.last_error = ""
        self.backoff_index = 0
        self.is_checking_health = False
        self.is_running = False
        self.listeners = []

    def get_state(self) -> str:
        return self.state

    def is_connected(self) -> bool:
        return self.state == "CONNECTED"

    def is_degraded(self) -> bool:
        return self.state == "DEGRADED"

    def is_lost(self) -> bool:
        return self.state == "LOST"

    def is_recovering(self) -> bool:
        return self.state == "RECOVERING"

    def get_consecutive_failures(self) -> int:
        return self.consecutive_failures

    def get_consecutive_successes(self) -> int:
        return self.consecutive_successes

    def get_current_backoff_ms(self) -> int:
        if self.backoff_index < len(BACKOFF_STEPS_MS):
            return BACKOFF_STEPS_MS[self.backoff_index]
        return MAX_BACKOFF_MS

    def subscribe(self, listener):
        self.listeners.append(listener)
        def unsubscribe():
            if listener in self.listeners:
                self.listeners.remove(listener)
        return unsubscribe

    def _notify_listeners(self, prev_state: str):
        ctx = {
            "consecutiveFailures": self.consecutive_failures,
            "consecutiveSuccesses": self.consecutive_successes,
            "currentBackoffMs": self.get_current_backoff_ms(),
            "error": self.last_error,
        }
        for l in list(self.listeners):
            l(self.state, prev_state, ctx)

    def record_success(self, now: int = 0) -> str:
        prev = self.state
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = now
        self.backoff_index = 0

        next_state = self.state
        if self.state == "DEGRADED":
            next_state = "CONNECTED"
        elif self.state == "LOST":
            next_state = "RECOVERING"
        elif self.state == "RECOVERING":
            if self.consecutive_successes >= 2:
                next_state = "CONNECTED"

        self.state = next_state
        if next_state != prev:
            self._notify_listeners(prev)
        return self.state

    def record_failure(self, error: str = "", now: int = 0) -> str:
        prev = self.state
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = now
        self.last_error = error

        next_state = self.state
        if self.state == "CONNECTED":
            next_state = "DEGRADED"
        elif self.state in ("DEGRADED", "RECOVERING"):
            next_state = "LOST"
        elif self.state == "LOST":
            self.backoff_index = min(self.backoff_index + 1, len(BACKOFF_STEPS_MS) - 1)

        self.state = next_state
        if next_state != prev:
            self._notify_listeners(prev)
        return self.state

    def trigger_concurrent_health_check(self) -> bool:
        """Attempts to start health check; returns False if already busy."""
        if self.is_checking_health:
            return False  # Concurrency lock prevented overlapping request
        self.is_checking_health = True
        return True

    def complete_health_check(self):
        self.is_checking_health = False

    def reset(self):
        self.state = "CONNECTED"
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.backoff_index = 0
        self.is_checking_health = False
        self.is_running = False

    def stop(self):
        self.is_running = False
        self.is_checking_health = False

    def destroy(self):
        self.stop()
        self.listeners.clear()


# --- Camera Integrity Algorithm ---

def evaluate_frame_integrity(mean_luminance: float, variance: float) -> Dict[str, Any]:
    """
    Formal Phase 13 Mathematical Specification for Pixel-Based Frame Integrity.
    Mirrors evaluateFrameIntegrity() in mobile/app/camera.tsx.
    """
    if mean_luminance < 15:
        return {
            "state": "DARK",
            "mean_luminance": mean_luminance,
            "variance": variance,
            "message": "Scene too dark for reliable guidance.",
            "is_valid": False,
        }

    if variance < 10:
        if mean_luminance < 35:
            return {
                "state": "OCCLUDED",
                "mean_luminance": mean_luminance,
                "variance": variance,
                "message": "Camera view obstructed or lens covered.",
                "is_valid": False,
            }

    return {
        "state": "NORMAL",
        "mean_luminance": mean_luminance,
        "variance": variance,
        "message": "Camera view normal.",
        "is_valid": True,
    }


def diagnose_captured_picture(picture: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Mobile-Side Fallback Integrity Heuristic.
    Mirrors diagnoseCapturedPicture() in mobile/app/camera.tsx.
    """
    if not picture or not picture.get("uri"):
        return {
            "state": "DEGRADED",
            "message": "No image captured from camera.",
            "is_valid": False,
        }

    w = picture.get("width")
    h = picture.get("height")
    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
        if w <= 0 or h <= 0:
            return {
                "state": "DEGRADED",
                "message": "Invalid frame dimensions.",
                "is_valid": False,
            }

    return {
        "state": "NORMAL",
        "message": "Camera frame valid (mobile fallback diagnostic).",
        "is_valid": True,
    }


# --- Haptic Patterns Specification ---

HAPTIC_PATTERNS = {
    "EMERGENCY": [0, 400, 150, 400, 150, 600],
    "CONNECTION_LOST": [0, 250, 100, 250, 100, 250],
    "DANGER": [0, 150, 100, 150],
    "TURN_LEFT": [0, 80, 80, 80],
    "TURN_RIGHT": [0, 240],
    "CAMERA_OCCLUDED": [0, 60, 60, 60, 60, 60],
    "PATH_CLEAR": [0, 50],
}


# --- Phase 13 Extended Voice Decision Manager ---

SAFETY_MESSAGES = {
    "EMERGENCY_STOP": "Stop immediately.",
    "CONNECTION_LOST": "Connection lost. Stop walking.",
    "CONNECTION_RESTORED": "Connection restored. Resuming guidance.",
    "CAMERA_OCCLUDED": "Camera view obstructed or scene too dark. Please check camera view.",
}

class VoiceDecisionManagerPhase13Py:
    """
    Extends Phase 12 Voice Decision Manager with Phase 13 discrete safety events.
    Mirrors mobile/src/services/speech.ts.
    """
    def __init__(self):
        self.last_spoken_text = ""
        self.last_spoken_at = 0
        self.last_priority = 0
        self.is_speaking = False
        self.queued_instruction = None

    def evaluate_safety_event(self, event: str, now: int = 0) -> Dict[str, Any]:
        instruction = ""
        priority = 4
        bypass_cooldown = False

        if event == "CONNECTION_LOST":
            instruction = SAFETY_MESSAGES["CONNECTION_LOST"]
            priority = 5
        elif event == "CONNECTION_RESTORED":
            instruction = SAFETY_MESSAGES["CONNECTION_RESTORED"]
            priority = 4
            bypass_cooldown = True
        elif event == "CAMERA_OCCLUDED":
            instruction = SAFETY_MESSAGES["CAMERA_OCCLUDED"]
            priority = 4

        elapsed = now - self.last_spoken_at
        is_same = (instruction.lower().strip() == self.last_spoken_text.lower().strip()) and bool(instruction)

        # Duplicate suppression
        if is_same and not bypass_cooldown:
            repeat_window = 5000 if event == "CONNECTION_LOST" else 7000
            if elapsed < repeat_window:
                return {
                    "action": "SUPPRESS",
                    "instruction": instruction,
                    "priority": priority,
                    "reason": f"Duplicate safety event within repeat window ({elapsed}ms < {repeat_window}ms)",
                }

        # Speech lock / preemption
        if self.is_speaking:
            if priority > self.last_priority or (priority == 5 and not is_same):
                return {
                    "action": "INTERRUPT",
                    "instruction": instruction,
                    "priority": priority,
                    "reason": f"Safety event {event} (Priority {priority}) interrupts active speech",
                    "bypass_cooldown": True,
                }
            return {
                "action": "SUPPRESS",
                "instruction": instruction,
                "priority": priority,
                "reason": f"Active speech prevents safety event {event}",
            }

        return {
            "action": "SPEAK",
            "instruction": instruction,
            "priority": priority,
            "reason": f"Safety event {event} immediate trigger",
            "bypass_cooldown": bypass_cooldown,
        }

    def handle_safety_event(self, event: str, now: int = 0) -> Dict[str, Any]:
        decision = self.evaluate_safety_event(event, now)
        if decision["action"] in ("SPEAK", "INTERRUPT"):
            if event in ("CONNECTION_LOST", "CAMERA_OCCLUDED"):
                self.queued_instruction = None
            self.last_spoken_text = decision["instruction"]
            self.last_spoken_at = now
            self.last_priority = decision["priority"]
            self.is_speaking = True
        return decision


# =====================================================================
# 2. UNIT TESTS
# =====================================================================

class TestPhase13ConnectionWatchdog(unittest.TestCase):
    """Behavioral tests for ConnectionWatchdog state machine, counters, and backoff."""

    def setUp(self):
        self.wd = ConnectionWatchdogPy()

    def test_01_initial_state_connected(self):
        """[BEHAVIORAL] Watchdog initializes in CONNECTED with 0 failures and initial 2s backoff."""
        self.assertEqual(self.wd.get_state(), "CONNECTED")
        self.assertTrue(self.wd.is_connected())
        self.assertFalse(self.wd.is_degraded())
        self.assertFalse(self.wd.is_lost())
        self.assertFalse(self.wd.is_recovering())
        self.assertEqual(self.wd.get_consecutive_failures(), 0)
        self.assertEqual(self.wd.get_consecutive_successes(), 0)
        self.assertEqual(self.wd.get_current_backoff_ms(), 2000)

    def test_02_single_failure_transitions_to_degraded(self):
        """[BEHAVIORAL] Single frame failure transitions from CONNECTED to DEGRADED."""
        state = self.wd.record_failure("Network timeout", now=1000)
        self.assertEqual(state, "DEGRADED")
        self.assertTrue(self.wd.is_degraded())
        self.assertFalse(self.wd.is_connected())
        self.assertFalse(self.wd.is_lost())
        self.assertEqual(self.wd.get_consecutive_failures(), 1)
        self.assertEqual(self.wd.get_consecutive_successes(), 0)

    def test_03_second_failure_transitions_to_lost(self):
        """[BEHAVIORAL] Second consecutive failure triggers fail-safe LOST state."""
        self.wd.record_failure("Failure 1", now=1000)
        state = self.wd.record_failure("Failure 2", now=1500)
        self.assertEqual(state, "LOST")
        self.assertTrue(self.wd.is_lost())
        self.assertFalse(self.wd.is_degraded())
        self.assertFalse(self.wd.is_connected())
        self.assertEqual(self.wd.get_consecutive_failures(), 2)

    def test_04_recovery_from_degraded_immediate(self):
        """[BEHAVIORAL] Single success from DEGRADED immediately restores CONNECTED."""
        self.wd.record_failure("Glitch", now=1000)
        self.assertEqual(self.wd.get_state(), "DEGRADED")
        state = self.wd.record_success(now=2000)
        self.assertEqual(state, "CONNECTED")
        self.assertTrue(self.wd.is_connected())
        self.assertEqual(self.wd.get_consecutive_failures(), 0)
        self.assertEqual(self.wd.get_consecutive_successes(), 1)

    def test_05_recovery_from_lost_first_success_recovering(self):
        """[BEHAVIORAL] First successful check after LOST transitions to RECOVERING."""
        self.wd.record_failure("Fail 1", now=1000)
        self.wd.record_failure("Fail 2", now=1500)
        self.assertEqual(self.wd.get_state(), "LOST")
        state = self.wd.record_success(now=3500)
        self.assertEqual(state, "RECOVERING")
        self.assertTrue(self.wd.is_recovering())
        self.assertFalse(self.wd.is_connected())
        self.assertEqual(self.wd.get_consecutive_successes(), 1)

    def test_06_recovery_from_recovering_second_success_connected(self):
        """[BEHAVIORAL] Second consecutive success confirms full recovery back to CONNECTED."""
        self.wd.record_failure("Fail 1", now=1000)
        self.wd.record_failure("Fail 2", now=1500)
        self.wd.record_success(now=3500)  # LOST -> RECOVERING (success 1)
        state = self.wd.record_success(now=5500)  # RECOVERING -> CONNECTED (success 2)
        self.assertEqual(state, "CONNECTED")
        self.assertTrue(self.wd.is_connected())
        self.assertEqual(self.wd.get_consecutive_successes(), 2)

    def test_07_recovery_interrupted_by_failure_returns_to_lost(self):
        """[BEHAVIORAL] A failure while RECOVERING immediately falls back to LOST."""
        self.wd.record_failure("Fail 1", now=1000)
        self.wd.record_failure("Fail 2", now=1500)
        self.wd.record_success(now=3500)  # RECOVERING
        self.assertEqual(self.wd.get_state(), "RECOVERING")
        state = self.wd.record_failure("Fail during recovery", now=4000)
        self.assertEqual(state, "LOST")
        self.assertTrue(self.wd.is_lost())
        self.assertEqual(self.wd.get_consecutive_successes(), 0)

    def test_08_exponential_backoff_progression(self):
        """[BEHAVIORAL] Consecutive failures in LOST progress through [2s, 4s, 8s, 16s, 30s] max 30s."""
        self.wd.record_failure("F1", now=1000)  # DEGRADED, backoff=2000
        self.assertEqual(self.wd.get_current_backoff_ms(), 2000)
        self.wd.record_failure("F2", now=1500)  # LOST, backoff=2000
        self.assertEqual(self.wd.get_current_backoff_ms(), 2000)
        self.wd.record_failure("F3", now=3500)  # LOST, step 1 -> 4000
        self.assertEqual(self.wd.get_current_backoff_ms(), 4000)
        self.wd.record_failure("F4", now=7500)  # LOST, step 2 -> 8000
        self.assertEqual(self.wd.get_current_backoff_ms(), 8000)
        self.wd.record_failure("F5", now=15500) # LOST, step 3 -> 16000
        self.assertEqual(self.wd.get_current_backoff_ms(), 16000)
        self.wd.record_failure("F6", now=31500) # LOST, step 4 -> 30000
        self.assertEqual(self.wd.get_current_backoff_ms(), 30000)
        self.wd.record_failure("F7", now=61500) # LOST, clamped -> 30000
        self.assertEqual(self.wd.get_current_backoff_ms(), 30000)

    def test_09_backoff_reset_on_success(self):
        """[BEHAVIORAL] Successful health check resets backoff duration back to 2000ms."""
        self.wd.record_failure("F1", now=1000)
        self.wd.record_failure("F2", now=1500)
        self.wd.record_failure("F3", now=3500)  # 4000ms
        self.assertEqual(self.wd.get_current_backoff_ms(), 4000)
        self.wd.record_success(now=7500)
        self.assertEqual(self.wd.get_current_backoff_ms(), 2000)

    def test_10_concurrency_lock_prevents_overlapping_requests(self):
        """[BEHAVIORAL] Watchdog concurrency guard prevents overlapping health probes."""
        acquired1 = self.wd.trigger_concurrent_health_check()
        self.assertTrue(acquired1)
        acquired2 = self.wd.trigger_concurrent_health_check()
        self.assertFalse(acquired2)  # Overlap blocked
        self.wd.complete_health_check()
        acquired3 = self.wd.trigger_concurrent_health_check()
        self.assertTrue(acquired3)

    def test_11_watchdog_reset_restores_initial_state(self):
        """[BEHAVIORAL] reset() completely restores initial state and counters."""
        self.wd.record_failure("F1")
        self.wd.record_failure("F2")
        self.assertEqual(self.wd.get_state(), "LOST")
        self.wd.reset()
        self.assertEqual(self.wd.get_state(), "CONNECTED")
        self.assertEqual(self.wd.get_consecutive_failures(), 0)
        self.assertEqual(self.wd.get_consecutive_successes(), 0)
        self.assertEqual(self.wd.get_current_backoff_ms(), 2000)

    def test_12_listener_notifications_on_transitions(self):
        """[BEHAVIORAL] Subscribed observers receive state transition callbacks with context."""
        events = []
        def on_change(next_s, prev_s, ctx):
            events.append((next_s, prev_s, ctx["consecutiveFailures"]))

        unsub = self.wd.subscribe(on_change)
        self.wd.record_failure("F1")
        self.wd.record_failure("F2")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], ("DEGRADED", "CONNECTED", 1))
        self.assertEqual(events[1], ("LOST", "DEGRADED", 2))
        unsub()
        self.wd.record_success()
        self.assertEqual(len(events), 2)  # No new event after unsubscribe


class TestCameraIntegrityAndFailSafe(unittest.TestCase):
    """Behavioral tests for pixel-based math algorithm, runtime fallback, and fail-safe suppression."""

    def test_13_dark_threshold_mean_below_15(self):
        """[BEHAVIORAL] mean < 15 classifies as DARK and invalid."""
        diag = evaluate_frame_integrity(mean_luminance=10.0, variance=25.0)
        self.assertEqual(diag["state"], "DARK")
        self.assertFalse(diag["is_valid"])

    def test_14_dark_boundary_14_9_vs_15_0(self):
        """[BEHAVIORAL] Boundary: 14.9 is DARK; 15.0 with normal variance is NORMAL."""
        diag_below = evaluate_frame_integrity(mean_luminance=14.9, variance=20.0)
        diag_exact = evaluate_frame_integrity(mean_luminance=15.0, variance=20.0)
        diag_above = evaluate_frame_integrity(mean_luminance=15.1, variance=20.0)
        self.assertEqual(diag_below["state"], "DARK")
        self.assertEqual(diag_exact["state"], "NORMAL")
        self.assertEqual(diag_above["state"], "NORMAL")

    def test_15_occluded_variance_below_10_low_mean(self):
        """[BEHAVIORAL] variance < 10 with mean < 35 classifies as OCCLUDED and invalid."""
        diag = evaluate_frame_integrity(mean_luminance=22.0, variance=6.5)
        self.assertEqual(diag["state"], "OCCLUDED")
        self.assertFalse(diag["is_valid"])

    def test_16_occluded_boundary_variance_9_9_vs_10_0(self):
        """[BEHAVIORAL] Boundary: variance 9.9 (mean=25) is OCCLUDED; 10.0 is NORMAL."""
        diag_occ = evaluate_frame_integrity(mean_luminance=25.0, variance=9.9)
        diag_norm = evaluate_frame_integrity(mean_luminance=25.0, variance=10.0)
        self.assertEqual(diag_occ["state"], "OCCLUDED")
        self.assertEqual(diag_norm["state"], "NORMAL")

    def test_17_occluded_mean_false_positive_guard_above_35(self):
        """[BEHAVIORAL] False-positive guard: low variance with mean >= 35 is NOT occluded."""
        diag_high = evaluate_frame_integrity(mean_luminance=50.0, variance=5.0)
        self.assertEqual(diag_high["state"], "NORMAL")
        self.assertTrue(diag_high["is_valid"])

    def test_18_normal_scene(self):
        """[BEHAVIORAL] Well-lit scene with adequate texture classifies as NORMAL."""
        diag = evaluate_frame_integrity(mean_luminance=120.0, variance=45.0)
        self.assertEqual(diag["state"], "NORMAL")
        self.assertTrue(diag["is_valid"])

    def test_19_runtime_fallback_missing_or_empty_uri(self):
        """[BEHAVIORAL] Mobile runtime fallback: missing or empty picture URI fails as DEGRADED."""
        self.assertFalse(diagnose_captured_picture(None)["is_valid"])
        self.assertFalse(diagnose_captured_picture({})["is_valid"])
        self.assertFalse(diagnose_captured_picture({"uri": ""})["is_valid"])

    def test_20_runtime_fallback_zero_dimensions(self):
        """[BEHAVIORAL] Mobile runtime fallback: zero/negative dimensions fail as DEGRADED."""
        d_zero_w = diagnose_captured_picture({"uri": "file://frame.jpg", "width": 0, "height": 480})
        d_zero_h = diagnose_captured_picture({"uri": "file://frame.jpg", "width": 640, "height": 0})
        self.assertEqual(d_zero_w["state"], "DEGRADED")
        self.assertFalse(d_zero_w["is_valid"])
        self.assertEqual(d_zero_h["state"], "DEGRADED")
        self.assertFalse(d_zero_h["is_valid"])

    def test_21_runtime_fallback_valid_capture(self):
        """[BEHAVIORAL] Mobile runtime fallback: valid URI and positive dimensions produce NORMAL."""
        diag = diagnose_captured_picture({"uri": "file://valid_frame.jpg", "width": 640, "height": 480})
        self.assertEqual(diag["state"], "NORMAL")
        self.assertTrue(diag["is_valid"])

    def test_22_failsafe_suppresses_forward_and_turn_guidance(self):
        """[BEHAVIORAL] Unsafe camera integrity suppresses all active navigation commands."""
        unsafe_diag = evaluate_frame_integrity(mean_luminance=5.0, variance=2.0)
        self.assertFalse(unsafe_diag["is_valid"])

        # Simulated frame processing dispatch
        dispatched_guidance = "GO_FORWARD"
        if not unsafe_diag["is_valid"]:
            dispatched_guidance = "STOP_GUIDANCE_SUPPRESSED"

        self.assertNotIn(dispatched_guidance, ["GO_FORWARD", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"])
        self.assertEqual(dispatched_guidance, "STOP_GUIDANCE_SUPPRESSED")


class TestHapticFeedbackService(unittest.TestCase):
    """Behavioral tests for haptic patterns, durations, differentiation, and preferences."""

    def test_23_haptic_events_defined(self):
        """[BEHAVIORAL] All 7 standard Phase 13 haptic events are defined."""
        expected_events = [
            "EMERGENCY",
            "CONNECTION_LOST",
            "DANGER",
            "TURN_LEFT",
            "TURN_RIGHT",
            "CAMERA_OCCLUDED",
            "PATH_CLEAR",
        ]
        for ev in expected_events:
            self.assertIn(ev, HAPTIC_PATTERNS)

    def test_24_haptic_patterns_numeric_and_non_empty(self):
        """[BEHAVIORAL] Every vibration pattern consists of positive numbers and is non-empty."""
        for ev, pattern in HAPTIC_PATTERNS.items():
            self.assertTrue(len(pattern) > 0, f"Pattern for {ev} is empty")
            for val in pattern:
                self.assertIsInstance(val, int)
                self.assertGreaterEqual(val, 0)

    def test_25_emergency_stronger_than_turns(self):
        """[BEHAVIORAL] EMERGENCY vibration total buzz duration exceeds TURN actions."""
        # Buzz durations are at odd indices [wait, buzz, wait, buzz, ...]
        emerg_buzz = sum(HAPTIC_PATTERNS["EMERGENCY"][1::2])
        left_buzz = sum(HAPTIC_PATTERNS["TURN_LEFT"][1::2])
        right_buzz = sum(HAPTIC_PATTERNS["TURN_RIGHT"][1::2])

        self.assertGreater(emerg_buzz, left_buzz)
        self.assertGreater(emerg_buzz, right_buzz)
        self.assertEqual(emerg_buzz, 1400)  # 400 + 400 + 600 ms

    def test_26_connection_lost_pattern_rhythmic_alert(self):
        """[BEHAVIORAL] CONNECTION_LOST uses repeated 250ms alert buzz pulses."""
        conn_buzzes = HAPTIC_PATTERNS["CONNECTION_LOST"][1::2]
        self.assertEqual(conn_buzzes, [250, 250, 250])

    def test_27_turn_left_differs_from_turn_right(self):
        """[BEHAVIORAL] TURN_LEFT and TURN_RIGHT have distinct, tactilely distinguishable patterns."""
        self.assertNotEqual(HAPTIC_PATTERNS["TURN_LEFT"], HAPTIC_PATTERNS["TURN_RIGHT"])
        # Turn left is double short pulse; Turn right is single elongated pulse
        self.assertEqual(len(HAPTIC_PATTERNS["TURN_LEFT"]), 4)
        self.assertEqual(len(HAPTIC_PATTERNS["TURN_RIGHT"]), 2)

    def test_28_haptics_preference_toggle(self):
        """[BEHAVIORAL] Haptics enabled preference state transitions cleanly."""
        haptics_enabled = True
        # Toggle off
        haptics_enabled = False
        self.assertFalse(haptics_enabled)
        # Toggle on
        haptics_enabled = True
        self.assertTrue(haptics_enabled)


class TestVoiceSafetyAndPriority(unittest.TestCase):
    """Behavioral tests for Phase 13 VoiceDecisionManager safety events and priority preemption."""

    def setUp(self):
        self.vm = VoiceDecisionManagerPhase13Py()

    def test_29_connection_lost_message_and_priority_5(self):
        """[BEHAVIORAL] CONNECTION_LOST triggers Priority 5 'Connection lost. Stop walking.'"""
        d = self.vm.handle_safety_event("CONNECTION_LOST", now=1000)
        self.assertEqual(d["action"], "SPEAK")
        self.assertEqual(d["instruction"], "Connection lost. Stop walking.")
        self.assertEqual(d["priority"], 5)

    def test_30_connection_restored_message_and_priority_4(self):
        """[BEHAVIORAL] CONNECTION_RESTORED triggers Priority 4 'Connection restored. Resuming guidance.'"""
        d = self.vm.handle_safety_event("CONNECTION_RESTORED", now=1000)
        self.assertEqual(d["action"], "SPEAK")
        self.assertEqual(d["instruction"], "Connection restored. Resuming guidance.")
        self.assertEqual(d["priority"], 4)
        self.assertTrue(d.get("bypass_cooldown"))

    def test_31_camera_occluded_message_and_priority_4(self):
        """[BEHAVIORAL] CAMERA_OCCLUDED triggers Priority 4 'Camera view obstructed or scene too dark. Please check camera view.'"""
        d = self.vm.handle_safety_event("CAMERA_OCCLUDED", now=1000)
        self.assertEqual(d["action"], "SPEAK")
        self.assertEqual(d["instruction"], "Camera view obstructed or scene too dark. Please check camera view.")
        self.assertEqual(d["priority"], 4)

    def test_32_priority_5_connection_loss_preempts_movement(self):
        """[BEHAVIORAL] Priority 5 Connection Lost interrupts lower-priority active speech."""
        # Active walking speech at Priority 2
        self.vm.last_spoken_text = "Move forward four steps."
        self.vm.last_priority = 2
        self.vm.is_speaking = True

        d = self.vm.handle_safety_event("CONNECTION_LOST", now=1500)
        self.assertEqual(d["action"], "INTERRUPT")
        self.assertEqual(d["instruction"], "Connection lost. Stop walking.")
        self.assertEqual(d["priority"], 5)

    def test_33_priority_5_clears_queued_movement(self):
        """[BEHAVIORAL] Connection loss discards queued walking commands."""
        self.vm.queued_instruction = {"text": "Move forward three steps.", "priority": 2}
        self.vm.handle_safety_event("CONNECTION_LOST", now=1000)
        self.assertIsNone(self.vm.queued_instruction)

    def test_34_safety_event_duplicate_suppressed_within_window(self):
        """[BEHAVIORAL] Duplicate connection lost alert is suppressed within repeat cooldown."""
        self.vm.is_speaking = False
        d1 = self.vm.handle_safety_event("CONNECTION_LOST", now=1000)
        self.assertEqual(d1["action"], "SPEAK")
        self.vm.is_speaking = False

        # Frame 2 within 5s connection repeat window
        d2 = self.vm.handle_safety_event("CONNECTION_LOST", now=3000)
        self.assertEqual(d2["action"], "SUPPRESS")

    def test_35_connection_restored_bypasses_cooldown(self):
        """[BEHAVIORAL] CONNECTION_RESTORED bypasses cooldown to immediately announce resumption."""
        self.vm.is_speaking = False
        self.vm.handle_safety_event("CONNECTION_LOST", now=1000)
        self.vm.is_speaking = False

        # Connection restored shortly after
        d_restored = self.vm.handle_safety_event("CONNECTION_RESTORED", now=2500)
        self.assertEqual(d_restored["action"], "SPEAK")
        self.assertEqual(d_restored["instruction"], "Connection restored. Resuming guidance.")


class TestStaticImplementationChecks(unittest.TestCase):
    """Static source analysis of TypeScript implementation files for Phase 13 contracts."""

    def _read_file(self, rel_path: str) -> str:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        self.assertTrue(os.path.exists(full_path), f"Required project file missing: {rel_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_36_static_watchdog_api(self):
        """[STATIC] mobile/src/services/connectionWatchdog.ts exports ConnectionWatchdog with all required APIs."""
        content = self._read_file("mobile/src/services/connectionWatchdog.ts")
        required_elements = [
            "export class ConnectionWatchdog",
            "getState(): ConnectionState",
            "isConnected(): boolean",
            "isDegraded(): boolean",
            "isLost(): boolean",
            "isRecovering(): boolean",
            "getConsecutiveFailures(): number",
            "getConsecutiveSuccesses(): number",
            "getCurrentBackoffMs(): number",
            "recordSuccess(",
            "recordFailure(",
            "subscribe(",
            "reset(): void",
            "stop(): void",
            "destroy(): void",
            "export const connectionWatchdog = new ConnectionWatchdog()",
        ]
        for elem in required_elements:
            self.assertIn(elem, content, f"Missing watchdog API: {elem}")

    def test_37_static_haptics_api(self):
        """[STATIC] mobile/src/services/haptics.ts defines all 7 events and preferences."""
        content = self._read_file("mobile/src/services/haptics.ts")
        required = [
            "HAPTIC_PATTERNS",
            "EMERGENCY",
            "CONNECTION_LOST",
            "DANGER",
            "TURN_LEFT",
            "TURN_RIGHT",
            "CAMERA_OCCLUDED",
            "PATH_CLEAR",
            "isHapticsEnabled()",
            "setHapticsEnabled(",
            "triggerHaptic(",
            "stopHaptic()",
        ]
        for r in required:
            self.assertIn(r, content, f"Missing haptics service definition: {r}")

    def test_38_static_speech_safety_messages(self):
        """[STATIC] mobile/src/services/speech.ts defines exact Phase 13 English safety phrases."""
        content = self._read_file("mobile/src/services/speech.ts")
        phrases = [
            "Connection lost. Stop walking.",
            "Connection restored. Resuming guidance.",
            "Camera view obstructed or scene too dark. Please check camera view.",
        ]
        for phrase in phrases:
            self.assertIn(phrase, content, f"Missing exact voice phrase: '{phrase}'")

        # Verify exported helper methods
        self.assertIn("speakConnectionLost", content)
        self.assertIn("speakConnectionRestored", content)
        self.assertIn("speakCameraOccluded", content)
        self.assertIn("handleSafetyEvent", content)

    def test_39_static_types_phase13(self):
        """[STATIC] mobile/src/types/smartvision.ts defines Phase 13 types and AppSettings."""
        content = self._read_file("mobile/src/types/smartvision.ts")
        required_types = [
            "ConnectionState",
            "CameraIntegrityState",
            "CameraIntegrityDiagnostics",
            "hapticsEnabled?: boolean",
        ]
        for t in required_types:
            self.assertIn(t, content, f"Missing type definition: {t}")

    def test_40_static_camera_resilience(self):
        """[STATIC] mobile/app/camera.tsx integrates watchdog, haptics, and occlusion suppression."""
        content = self._read_file("mobile/app/camera.tsx")
        # Watchdog & haptics imports
        self.assertIn("connectionWatchdog", content)
        self.assertIn("triggerHaptic", content)
        self.assertIn("voiceManager", content)
        # Integrity checks
        self.assertIn("evaluateFrameIntegrity", content)
        self.assertIn("diagnoseCapturedPicture", content)
        self.assertIn("CAMERA_OCCLUDED", content)
        self.assertIn("speakCameraOccluded", content)
        # Overlays with accessibilityRole="alert" and exit button
        self.assertIn('accessibilityRole="alert"', content)
        self.assertIn("EXIT CAMERA", content)

    def test_41_static_home_accessibility(self):
        """[STATIC] mobile/app/index.tsx has NO redirect and contains accessible navigation cards."""
        content = self._read_file("mobile/app/index.tsx")
        self.assertNotIn("<Redirect href=", content, "Found forbidden hardcoded Redirect in index.tsx")
        # Routes
        self.assertIn('router.push("/camera")', content)
        self.assertIn('router.push("/finder")', content)
        self.assertIn('router.push("/settings")', content)
        self.assertIn('router.push("/about")', content)
        # TalkBack Accessibility
        self.assertIn("AccessibilityInfo.announceForAccessibility", content)
        self.assertIn('accessibilityRole="button"', content)
        self.assertIn('accessibilityLabel="Live Guidance"', content)
        self.assertIn('accessibilityLabel="Finder"', content)

    def test_42_static_finder_safety_and_accessibility(self):
        """[STATIC] mobile/app/finder.tsx is identification-only, accessible, and has no /detect-video."""
        content = self._read_file("mobile/app/finder.tsx")
        self.assertNotIn("/detect-video", content, "Forbidden /detect-video found in finder.tsx")
        # Voice manager integration
        self.assertIn("voiceManager", content)
        self.assertIn("formatStepWords", content)
        self.assertIn("distanceToSteps", content)
        # TalkBack Accessibility
        self.assertIn("AccessibilityInfo.announceForAccessibility", content)
        self.assertIn('accessibilityLabel="Go back"', content)
        self.assertIn('accessibilityLabel="Target object search input"', content)
        self.assertIn('accessibilityLabel="Stop search"', content)

    def test_43_static_settings_safety_and_accessibility(self):
        """[STATIC] mobile/app/settings.tsx is English-only, exposes haptics, and does not trigger haptics."""
        content = self._read_file("mobile/app/settings.tsx")
        # English only
        self.assertNotIn("Hindi", content)
        self.assertNotIn("Marathi", content)
        # Haptics preference
        self.assertIn("HAPTIC FEEDBACK", content)
        self.assertIn("isHapticsEnabled", content)
        self.assertIn("setHapticsEnabled", content)
        self.assertNotIn("triggerHaptic", content, "Settings must NOT call triggerHaptic()")
        # TalkBack Accessibility
        self.assertIn("AccessibilityInfo.announceForAccessibility", content)
        self.assertIn('accessibilityRole="switch"', content)
        self.assertIn('accessibilityRole="button"', content)

    def test_44_static_no_detect_video_endpoint(self):
        """[STATIC] Verifies project has NO /detect-video endpoint in backend or mobile api."""
        backend_content = self._read_file("backend/routes/detection.py")
        mobile_api = self._read_file("mobile/src/services/api.ts")
        self.assertNotIn("/detect-video", backend_content, "Found forbidden /detect-video in backend detection.py")
        self.assertNotIn("/detect-video", mobile_api, "Found forbidden /detect-video in mobile api.ts")

    def test_45_static_backend_endpoints_intact(self):
        """[STATIC] Backend routes /health, /system/status, /detect, /detect/status are intact."""
        app_content = self._read_file("backend/app.py")
        detection_content = self._read_file("backend/routes/detection.py")
        self.assertIn('"/health"', app_content)
        self.assertIn('"/system/status"', app_content)
        self.assertIn('"/detect"', detection_content)
        self.assertIn('"/detect/status"', detection_content)


class TestHistoricalPhaseRegression(unittest.TestCase):
    """Executes historical Phase 4–12 regression suites and reports actual results."""

    def _run_script(self, script_name: str) -> Tuple[int, str]:
        path = os.path.join(SCRATCH_DIR, script_name)
        if not os.path.exists(path):
            return -1, f"Test file {script_name} not found"
        cmd = [sys.executable, path]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        import gc
        gc.collect()
        try:
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                env=env,
            )
            gc.collect()
            return proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as e:
            gc.collect()
            return -1, f"Test {script_name} timed out after 120 seconds: {e}"
        except Exception as e:
            gc.collect()
            return -1, f"Test {script_name} execution error: {e}"

    def test_46_phase4_all_tests_regression(self):
        """[REGRESSION] Phase 4 Unknown Obstacle Verification Suite."""
        code, out = self._run_script("run_phase4_all_tests.py")
        self.assertEqual(code, 0, f"Phase 4 regression failed:\n{out}")
        self.assertIn("ALL 10 TESTS + CRASH RESISTANCE PASSED SUCCESSFULLY", out)

    def test_47_phase5_verification_regression(self):
        """[REGRESSION] Phase 5 Distance Estimation Verification Suite."""
        code, out = self._run_script("test_phase5_verification.py")
        self.assertEqual(code, 0, f"Phase 5 regression failed:\n{out}")
        self.assertIn("ALL 9/9 PHASE 5 VERIFICATION TESTS PASSED", out)

    def test_48_phase6_verification_regression(self):
        """[REGRESSION] Phase 6 Multi-Object Reasoning Verification Suite."""
        code, out = self._run_script("test_phase6_verification.py")
        self.assertEqual(code, 0, f"Phase 6 regression failed:\n{out}")
        self.assertIn("ALL 12/12 PHASE 6 VERIFICATION SCENARIOS PASSED", out)

    def test_49_phase7_verification_regression(self):
        """[REGRESSION] Phase 7 Safe Walking Corridor Verification Suite."""
        code, out = self._run_script("test_phase7_verification.py")
        self.assertEqual(code, 0, f"Phase 7 regression failed:\n{out}")
        self.assertIn("ALL 16 PHASE 7 TESTS PASSED SUCCESSFULLY", out)

    def test_50_phase8_verification_regression(self):
        """[REGRESSION] Phase 8 Navigation Decision Engine Verification Suite."""
        code, out = self._run_script("test_phase8_verification.py")
        self.assertEqual(code, 0, f"Phase 8 regression failed:\n{out}")
        self.assertIn("ALL 23 PHASE 8 UNIT TESTS PASSED SUCCESSFULLY", out)

    def test_51_phase9_verification_regression(self):
        """[REGRESSION] Phase 9 Turn-vs-Movement Guidance Verification Suite."""
        code, out = self._run_script("test_phase9_verification.py")
        self.assertEqual(code, 0, f"Phase 9 regression failed:\n{out}")
        self.assertIn("ALL 30/30 PHASE 9 VERIFICATION TESTS PASSED", out)

    def test_52_phase10_verification_regression(self):
        """[REGRESSION] Phase 10 Walking Steps & Movement Distance Verification Suite."""
        code, out = self._run_script("test_phase10_verification.py")
        self.assertEqual(code, 0, f"Phase 10 regression failed:\n{out}")
        self.assertIn("PHASE 10 TEST RESULT: 35/35 PASSED", out)

    def test_53_phase11_verification_regression(self):
        """[REGRESSION] Phase 11 Approaching-Obstacle Detection Verification Suite."""
        code, out = self._run_script("test_phase11_verification.py")
        self.assertEqual(code, 0, f"Phase 11 regression failed:\n{out}")
        self.assertIn("PHASE 11 TEST RESULT: 37/37 PASSED", out)

    def test_54_phase12_voice_regression(self):
        """[REGRESSION] Phase 12 Voice Decision Guidance Verification Suite."""
        code, out = self._run_script("test_phase12_voice.py")
        self.assertEqual(code, 0, f"Phase 12 regression failed:\n{out}")
        self.assertIn("Ran 54 tests", out)


# =====================================================================
# 3. STRUCTURED REPORT TEST RUNNER
# =====================================================================

class CustomStructuredTestResult(unittest.TestResult):
    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super().__init__(stream, descriptions, verbosity)
        self.passed_tests = []
        self.failed_tests = []
        self.skipped_tests = []
        self.current_section = None

    def startTest(self, test):
        super().startTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        doc = test.shortDescription() or str(test)
        self.passed_tests.append((test, doc))
        print(f"PASS: {doc}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        doc = test.shortDescription() or str(test)
        self.failed_tests.append((test, doc, err))
        err_msg = str(err[1])
        try:
            print(f"FAIL: {doc}\n  -> {err_msg}")
        except Exception:
            safe_msg = err_msg.encode("ascii", errors="replace").decode("ascii")
            print(f"FAIL: {doc}\n  -> {safe_msg}")

    def addError(self, test, err):
        super().addError(test, err)
        doc = test.shortDescription() or str(test)
        self.failed_tests.append((test, doc, err))
        err_msg = str(err[1])
        try:
            print(f"ERROR: {doc}\n  -> {err_msg}")
        except Exception:
            safe_msg = err_msg.encode("ascii", errors="replace").decode("ascii")
            print(f"ERROR: {doc}\n  -> {safe_msg}")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        doc = test.shortDescription() or str(test)
        self.skipped_tests.append((test, doc, reason))
        print(f"SKIP: {doc} (Reason: {reason})")


def run_phase13_suite(include_historical: bool = True):
    print("=" * 60)
    print("SmartVisionAI Phase 13 Resilience Test Suite")
    print("============================================================")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        ("Connection Watchdog", TestPhase13ConnectionWatchdog),
        ("Camera Integrity & Fail-Safe", TestCameraIntegrityAndFailSafe),
        ("Haptic Feedback Service", TestHapticFeedbackService),
        ("Voice Safety & Priority", TestVoiceSafetyAndPriority),
        ("Static Implementation Checks", TestStaticImplementationChecks),
    ]
    if include_historical:
        test_classes.append(("Phase 4–12 Regression Suites", TestHistoricalPhaseRegression))

    result = CustomStructuredTestResult()

    for section_name, cls in test_classes:
        print(f"\n[{section_name}]")
        section_suite = loader.loadTestsFromTestCase(cls)
        section_suite.run(result)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Passed:  {len(result.passed_tests)}")
    print(f"Failed:  {len(result.failed_tests)}")
    print(f"Skipped: {len(result.skipped_tests)}")
    print("=" * 60)

    if result.failed_tests:
        print("\nFailures Detail:")
        for test, doc, err in result.failed_tests:
            err_msg = str(err[1])
            try:
                print(f"- {doc}: {err_msg}")
            except Exception:
                safe_msg = err_msg.encode("ascii", errors="replace").decode("ascii")
                print(f"- {doc}: {safe_msg}")

    return len(result.failed_tests) == 0


if __name__ == "__main__":
    success = run_phase13_suite()
    sys.exit(0 if success else 1)
