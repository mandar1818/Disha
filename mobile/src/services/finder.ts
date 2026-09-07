/**
 * SmartVisionAI
 * Voice Object Finder Service
 *
 * Responsibilities:
 * - Convert spoken commands into a target object
 * - Normalize object names
 * - Check backend detections for the requested object
 * - Determine approximate direction
 * - Generate Finder voice messages
 *
 * Example:
 *
 * User:
 *   "Find my bottle"
 *
 * Finder:
 *   target = "bottle"
 *
 * Camera frame
 *      ↓
 * api.detectImage()
 *      ↓
 * YOLO detections
 *      ↓
 * findTargetInResponse()
 *      ↓
 * "Bottle found on your left."
 */

import {
  getDetections,
} from "src/services/api";

import {
  speak,
  speakEmergency,
  stopSpeech,
} from "src/services/speech";

import {
  FINDER_INTERVAL_MS,
  MIN_CONFIDENCE,
} from "src/constants/config";

import type {
  DetectionObject,
  SmartVisionResponse,
} from "src/types/smartvision";

/* =========================================================
   TYPES
   ========================================================= */

export interface FinderTarget {
  original: string;
  normalized: string;
}

export type FinderPosition =
  | "LEFT"
  | "CENTER"
  | "RIGHT"
  | "UNKNOWN";

export interface FinderMatch {
  found: boolean;
  target: string;
  detection: DetectionObject | null;
  position: FinderPosition;
  direction: string;
  confidence: number;
  message: string;
}

/* =========================================================
   CONSTANTS
   ========================================================= */

/**
 * Re-exported so Finder screen can use the same
 * scanning interval as the service.
 */
export const finderScanInterval =
  FINDER_INTERVAL_MS;

/* =========================================================
   TARGET NORMALIZATION
   ========================================================= */

/**
 * Normalize text coming from speech recognition.
 */
export function normalizeTarget(
  value: string
): string {
  return value
    .toLowerCase()
    .replace(/[?.!,;:]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Remove common Finder command words.
 *
 * Examples:
 *
 * "find my bottle"
 *       ↓
 * "bottle"
 *
 * "where is my phone"
 *       ↓
 * "phone"
 */
export function extractTarget(
  transcript: string
): string {
  const text = normalizeTarget(transcript);

  if (!text) {
    return "";
  }

  const patterns = [
    /^find\s+(?:my\s+)?(.+)$/i,
    /^where\s+is\s+(?:my\s+)?(.+)$/i,
    /^where's\s+(?:my\s+)?(.+)$/i,
    /^locate\s+(?:my\s+)?(.+)$/i,
    /^search\s+for\s+(?:my\s+)?(.+)$/i,
    /^look\s+for\s+(?:my\s+)?(.+)$/i,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);

    if (match?.[1]) {
      return normalizeTarget(match[1]);
    }
  }

  return text;
}

/**
 * Create a structured Finder target.
 */
export function createFinderTarget(
  transcript: string
): FinderTarget | null {
  const target = extractTarget(transcript);

  if (!target) {
    return null;
  }

  return {
    original: transcript.trim(),
    normalized: target,
  };
}

/* =========================================================
   OBJECT NAME MATCHING
   ========================================================= */

/**
 * Basic aliases for common objects.
 *
 * YOLO/COCO may use singular class names while users
 * naturally speak plural or alternative names.
 */
const OBJECT_ALIASES = {
  person: ["person", "human", "man", "woman", "people"],
  bottle: ["bottle", "water bottle", "bottles"],
  cup: ["cup", "mug"],
  cell_phone: [
    "cell phone",
    "cellphone",
    "phone",
    "mobile",
    "smartphone",
  ],
  laptop: ["laptop", "notebook computer"],
  tv: ["tv", "television"],
  chair: ["chair", "seat"],
  couch: ["couch", "sofa"],
  backpack: ["backpack", "bag"],
  handbag: ["handbag", "purse"],
  suitcase: ["suitcase", "luggage"],
  book: ["book"],
  keyboard: ["keyboard"],
  mouse: ["mouse", "computer mouse"],
  remote: ["remote", "remote control"],
  umbrella: ["umbrella"],
  bicycle: ["bicycle", "bike"],
  motorcycle: ["motorcycle", "motorbike"],
  car: ["car", "automobile"],
  bus: ["bus"],
  truck: ["truck"],
  "traffic light": ["traffic light", "signal"],
  "stop sign": ["stop sign"],
} as const satisfies Record<string, readonly string[]>;

/**
 * Convert an object name into a normalized comparison form.
 */
function normalizeObjectName(
  value: string | undefined
): string {
  return normalizeTarget(value ?? "")
    .replace(/_/g, " ");
}

/**
 * Check whether detected object matches requested target.
 */
export function objectMatchesTarget(
  detection: DetectionObject,
  target: string
): boolean {
  const requested = normalizeObjectName(target);

  if (!requested) {
    return false;
  }

  const detected = normalizeObjectName(
    detection.class_name ??
      detection.name ??
      detection.label
  );

  if (!detected) {
    return false;
  }

  // Exact match.
  if (detected === requested) {
    return true;
  }

  // Requested text contains detected class.
  if (
    requested.includes(detected) ||
    detected.includes(requested)
  ) {
    return true;
  }

  // Alias matching.
  for (const [canonical, aliases] of Object.entries(
    OBJECT_ALIASES
  )) {
    const normalizedAliases = aliases.map(
      normalizeObjectName
    );

    const requestMatches =
      canonical === requested ||
      normalizedAliases.includes(requested);

    const detectionMatches =
      canonical === detected ||
      normalizedAliases.includes(detected);

    if (requestMatches && detectionMatches) {
      return true;
    }
  }

  return false;
}

/* =========================================================
   POSITION
   ========================================================= */

/**
 * Determine horizontal position from backend position
 * or bounding-box center.
 */
export function getFinderPosition(
  detection: DetectionObject
): FinderPosition {
  const backendPosition =
    detection.position?.toString().toUpperCase();

  if (
    backendPosition === "LEFT" ||
    backendPosition === "CENTER" ||
    backendPosition === "RIGHT"
  ) {
    return backendPosition;
  }

  const box = detection.bbox;

  if (Array.isArray(box) && box.length >= 4) {
    const x1 = Number(box[0]);
    const x2 = Number(box[2]);

    if (
      Number.isFinite(x1) &&
      Number.isFinite(x2)
    ) {
      const centerX = (x1 + x2) / 2;

      /**
       * These thresholds work with normalized or
       * approximately normalized coordinates.
       *
       * For pixel coordinates, the backend position
       * is preferred.
       */
      if (centerX < 0.4) {
        return "LEFT";
      }

      if (centerX > 0.6) {
        return "RIGHT";
      }

      return "CENTER";
    }
  }

  return "UNKNOWN";
}

/**
 * Convert Finder position into natural language.
 */
export function positionToSpeech(
  position: FinderPosition
): string {
  switch (position) {
    case "LEFT":
      return "on your left";

    case "RIGHT":
      return "on your right";

    case "CENTER":
      return "ahead";

    default:
      return "nearby";
  }
}

/* =========================================================
   DISTANCE
   ========================================================= */

/**
 * Convert backend distance into readable text.
 */
export function getDistanceText(
  detection: DetectionObject
): string {
  const value =
    detection.estimated_distance ??
    detection.depth;

  if (value === undefined || value === null) {
    return "";
  }

  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "";
  }

  /**
   * If backend already provides a reasonable meter value.
   */
  if (number > 0 && number < 100) {
    return `about ${number.toFixed(1)} meters away`;
  }

  return "";
}

/* =========================================================
   FIND TARGET
   ========================================================= */

/**
 * Find the requested object in the backend response.
 */
export function findTargetInResponse(
  response: SmartVisionResponse,
  target: string
): FinderMatch {
  const requestedTarget =
    extractTarget(target);

  if (!requestedTarget) {
    return {
      found: false,
      target: "",
      detection: null,
      position: "UNKNOWN",
      direction: "nearby",
      confidence: 0,
      message: "Please tell me what object to find.",
    };
  }

  if (!response.success) {
    return {
      found: false,
      target: requestedTarget,
      detection: null,
      position: "UNKNOWN",
      direction: "nearby",
      confidence: 0,
      message:
        response.error ??
        "Unable to analyze the camera image.",
    };
  }

  const detections = getDetections(response);

  const matches = detections.filter(
    (detection) =>
      objectMatchesTarget(
        detection,
        requestedTarget
      ) &&
      Number(detection.confidence ?? 0) >=
        MIN_CONFIDENCE
  );

  if (matches.length === 0) {
    return {
      found: false,
      target: requestedTarget,
      detection: null,
      position: "UNKNOWN",
      direction: "nearby",
      confidence: 0,
      message: `Looking for ${requestedTarget}.`,
    };
  }

  /**
   * If multiple matching objects exist,
   * select the highest-confidence detection.
   */
  const bestMatch = [...matches].sort(
    (a, b) =>
      Number(b.confidence ?? 0) -
      Number(a.confidence ?? 0)
  )[0];

  const position =
    getFinderPosition(bestMatch);

  const direction =
    positionToSpeech(position);

  const distance =
    getDistanceText(bestMatch);

  const label =
    bestMatch.class_name ??
    bestMatch.name ??
    bestMatch.label ??
    requestedTarget;

  let message =
    `${label} found ${direction}`;

  if (distance) {
    message += `, ${distance}`;
  }

  message += ".";

  return {
    found: true,
    target: requestedTarget,
    detection: bestMatch,
    position,
    direction,
    confidence: Number(
      bestMatch.confidence ?? 0
    ),
    message,
  };
}

/* =========================================================
   SPEECH
   ========================================================= */

/**
 * Announce that Finder has started.
 */
export async function announceFinderStart(
  target: string
): Promise<void> {
  const cleanTarget =
    extractTarget(target);

  if (!cleanTarget) {
    return;
  }

  await speak(
    `Searching for ${cleanTarget}.`
  );
}

/**
 * Announce that the target has been found.
 */
export async function announceFinderResult(
  result: FinderMatch
): Promise<void> {
  if (!result.found) {
    return;
  }

  await speak(result.message);
}

/**
 * Announce Finder error.
 */
export async function announceFinderError(
  message: string
): Promise<void> {
  if (!message.trim()) {
    return;
  }

  await speak(message);
}

/**
 * Emergency Finder announcement.
 */
export async function announceFinderEmergency(
  message: string = "Emergency. Stop immediately."
): Promise<void> {
  await speakEmergency(message);
}

/**
 * Stop Finder speech.
 */
export async function stopFinderSpeech(): Promise<void> {
  await stopSpeech();
}

/* =========================================================
   SEARCH STATE HELPERS
   ========================================================= */

/**
 * Check whether a Finder target is valid.
 */
export function isValidFinderTarget(
  target: string
): boolean {
  return extractTarget(target).length > 0;
}

/**
 * Create a user-friendly searching message.
 */
export function getSearchingMessage(
  target: string
): string {
  const cleanTarget =
    extractTarget(target);

  if (!cleanTarget) {
    return "Tell me what object to find.";
  }

  return `Looking for ${cleanTarget}...`;
}

/**
 * Create a user-friendly not-found message.
 */
export function getNotFoundMessage(
  target: string
): string {
  const cleanTarget =
    extractTarget(target);

  if (!cleanTarget) {
    return "No target selected.";
  }

  return `I cannot see the ${cleanTarget} yet. Keep scanning.`;
}

/* =========================================================
   DEFAULT EXPORT
   ========================================================= */

export default {
  extractTarget,
  createFinderTarget,
  normalizeTarget,
  objectMatchesTarget,
  getFinderPosition,
  positionToSpeech,
  getDistanceText,
  findTargetInResponse,
  announceFinderStart,
  announceFinderResult,
  announceFinderError,
  announceFinderEmergency,
  stopFinderSpeech,
  isValidFinderTarget,
  getSearchingMessage,
  getNotFoundMessage,
  finderScanInterval,
};