import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  AccessibilityInfo,
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { router } from "expo-router";
import { Camera, type PermissionResponse } from "expo-camera";
import { Ionicons } from "@expo/vector-icons";

import CameraView from "src/components/CameraView";
import {
  COLORS,
  FINDER_INTERVAL_MS,
  IMAGE_QUALITY,
  MIN_CONFIDENCE,
  distanceToSteps,
} from "src/constants/config";
import { detectImage } from "src/services/api";
import { formatStepWords, voiceManager } from "src/services/speech";
import type {
  DetectionObject,
  SmartVisionResponse,
} from "src/types/smartvision";

/* -------------------------------------------------------------------------- */
/* Types                                                                      */
/* -------------------------------------------------------------------------- */

type FinderState =
  | "idle"
  | "searching"
  | "found"
  | "not-found"
  | "error";

interface FinderResult {
  object: DetectionObject | null;
  message: string;
  direction: string;
}

/* -------------------------------------------------------------------------- */
/* Helper functions                                                           */
/* -------------------------------------------------------------------------- */

function cleanTarget(value: string): string {
  return value
    .toLowerCase()
    .replace(/[?.!,]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function extractTargetFromCommand(command: string): string {
  const text = cleanTarget(command);

  if (!text) {
    return "";
  }

  const patterns = [
    /^find\s+(?:my\s+)?(.+)$/i,
    /^where\s+is\s+(?:my\s+)?(.+)$/i,
    /^locate\s+(?:my\s+)?(.+)$/i,
    /^search\s+for\s+(?:my\s+)?(.+)$/i,
    /^look\s+for\s+(?:my\s+)?(.+)$/i,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);

    if (match?.[1]) {
      return cleanTarget(match[1]);
    }
  }

  return text;
}

function getDetectionLabel(object: DetectionObject): string {
  return (
    object.class_name ??
    object.name ??
    object.label ??
    "object"
  ).toString();
}

function getDirection(object: DetectionObject): string {
  const position = String(
    object.position ?? object.direction ?? "CENTER",
  ).toUpperCase();

  if (position.includes("LEFT")) {
    return "on your left";
  }

  if (position.includes("RIGHT")) {
    return "on your right";
  }

  return "ahead";
}

function getObjects(
  response: SmartVisionResponse,
): DetectionObject[] {
  const objects =
    response.objects ??
    response.detections ??
    [];

  return Array.isArray(objects) ? objects : [];
}

function matchesTarget(
  object: DetectionObject,
  target: string,
): boolean {
  const wanted = cleanTarget(target);

  if (!wanted) {
    return false;
  }

  const label = cleanTarget(getDetectionLabel(object));

  if (!label) {
    return false;
  }

  if (label === wanted) {
    return true;
  }

  if (label.includes(wanted)) {
    return true;
  }

  if (wanted.includes(label)) {
    return true;
  }

  return false;
}

function findTarget(
  response: SmartVisionResponse,
  target: string,
): DetectionObject | null {
  const objects = getObjects(response);

  const candidates = objects
    .filter((object) => matchesTarget(object, target))
    .filter(
      (object) =>
        Number(object.confidence ?? 0) >= MIN_CONFIDENCE,
    )
    .sort(
      (a, b) =>
        Number(b.confidence ?? 0) -
        Number(a.confidence ?? 0),
    );

  return candidates[0] ?? null;
}

/* -------------------------------------------------------------------------- */
/* Screen                                                                     */
/* -------------------------------------------------------------------------- */

export default function Finder() {
  const [permission, setPermission] =
    useState<PermissionResponse | null>(
      null,
    );

  const requestPermission = useCallback(
    async () => {
      const result =
        await Camera.requestCameraPermissionsAsync();
      setPermission(result);
      return result;
    },
    [],
  );

  useEffect(() => {
    let isMounted = true;

    const loadPermission = async () => {
      const result =
        await Camera.getCameraPermissionsAsync();

      if (isMounted) {
        setPermission(result);
      }
    };

    void loadPermission();

    return () => {
      isMounted = false;
    };
  }, []);

  const [target, setTarget] = useState("");
  const [command, setCommand] = useState("");
  const [message, setMessage] = useState(
    'Enter an object, for example "bottle".',
  );

  const [state, setState] =
    useState<FinderState>("idle");

  const [lastResult, setLastResult] =
    useState<FinderResult>({
      object: null,
      message: "",
      direction: "",
    });

  const [processing, setProcessing] =
    useState(false);

  const targetRef = useRef("");
  const processingRef = useRef(false);
  const mountedRef = useRef(true);

  /* ------------------------------------------------------------------------ */
  /* Cleanup & Screen Reader Announcement                                     */
  /* ------------------------------------------------------------------------ */

  useEffect(() => {
    mountedRef.current = true;
    AccessibilityInfo.announceForAccessibility(
      "Object Finder screen. Enter an object to find.",
    );

    return () => {
      mountedRef.current = false;
    };
  }, []);

  /* ------------------------------------------------------------------------ */
  /* Target                                                                   */
  /* ------------------------------------------------------------------------ */

  const setSearchTarget = useCallback(
    (value: string) => {
      const cleaned = cleanTarget(value);

      setTarget(cleaned);
      targetRef.current = cleaned;
    },
    [],
  );

  /* ------------------------------------------------------------------------ */
  /* Process detection result                                                 */
  /* ------------------------------------------------------------------------ */

  const processDetectionResult = useCallback(
    (response: SmartVisionResponse, wanted: string) => {
      const object = findTarget(
        response,
        wanted,
      );

      if (object) {
        const label = getDetectionLabel(object);
        const direction = getDirection(object);

        const confidence = Math.round(
          Number(object.confidence ?? 0) * 100,
        );

        // Compute step & distance guidance from existing DetectionObject fields
        let steps = object.estimated_steps;
        if (
          (typeof steps !== "number" || steps <= 0) &&
          typeof object.distance_m === "number" &&
          object.distance_m > 0
        ) {
          steps = distanceToSteps(object.distance_m);
        }

        const distM =
          typeof object.distance_m === "number" && object.distance_m > 0
            ? object.distance_m
            : typeof object.estimated_distance_m === "number" &&
              object.estimated_distance_m > 0
            ? object.estimated_distance_m
            : null;

        let locationPhrase = direction;
        if (typeof steps === "number" && steps > 0) {
          const stepWord = formatStepWords(steps);
          if (direction === "ahead") {
            locationPhrase = `${stepWord} ahead`;
          } else {
            locationPhrase = `${stepWord} ahead ${direction}`;
          }
        } else if (distM !== null) {
          if (direction === "ahead") {
            locationPhrase = `${distM.toFixed(1)}m ahead`;
          } else {
            locationPhrase = `${distM.toFixed(1)}m ahead ${direction}`;
          }
        } else if (
          object.distance_category &&
          object.distance_category !== "UNKNOWN_DISTANCE"
        ) {
          const cat = object.distance_category.toLowerCase().replace(/_/g, " ");
          locationPhrase = `${cat} ${direction}`;
        }

        const spokenText = `${label} found ${locationPhrase}.`;
        const resultMessage = `${label} found ${locationPhrase}. Confidence ${confidence}%.`;

        setLastResult({
          object,
          message: resultMessage,
          direction: locationPhrase,
        });

        setState("found");
        setMessage(resultMessage);

        void voiceManager.speak(spokenText, { priority: 1 });
        return;
      }

      setLastResult({
        object: null,
        message: "",
        direction: "",
      });

      setState("not-found");
      setMessage(`Looking for ${wanted}...`);
    },
    [],
  );

  /* ------------------------------------------------------------------------ */
  /* Scan                                                                     */
  /* ------------------------------------------------------------------------ */

  const scan = useCallback(async () => {
    if (!targetRef.current) {
      return;
    }

    if (processingRef.current) {
      return;
    }

    processingRef.current = true;

    if (mountedRef.current) {
      setProcessing(true);
      setState("searching");
      setMessage(
        `Looking for ${targetRef.current}...`,
      );
    }

    try {
      setMessage(
        `Scanning for ${targetRef.current}...`,
      );
    } catch {
      if (!mountedRef.current) {
        return;
      }

      setState("error");
      setMessage(
        "Finder scan failed. Please try again.",
      );
    } finally {
      processingRef.current = false;

      if (mountedRef.current) {
        setProcessing(false);
      }
    }
  }, []);

  /* ------------------------------------------------------------------------ */
  /* Process captured image                                                   */
  /* ------------------------------------------------------------------------ */

  const handleCapturedImage = useCallback(
    async (uri: string) => {
      const wanted = targetRef.current;

      if (!wanted) {
        return;
      }

      if (processingRef.current) {
        return;
      }

      processingRef.current = true;

      if (mountedRef.current) {
        setProcessing(true);
        setState("searching");
        setMessage(`Searching for ${wanted}...`);
      }

      try {
        const response = await detectImage(uri);

        if (!mountedRef.current) {
          return;
        }

        if (!response.success) {
          setState("error");

          setMessage(
            response.error ??
              "Backend detection failed.",
          );

          return;
        }

        processDetectionResult(response, wanted);
      } catch (error) {
        if (!mountedRef.current) {
          return;
        }

        setState("error");

        const text =
          error instanceof Error
            ? error.message
            : "Unknown finder error.";

        setMessage(
          `Finder error: ${text}`,
        );
      } finally {
        processingRef.current = false;

        if (mountedRef.current) {
          setProcessing(false);
        }
      }
    },
    [processDetectionResult],
  );

  /* ------------------------------------------------------------------------ */
  /* Start search                                                             */
  /* ------------------------------------------------------------------------ */

  const startSearch = useCallback(() => {
    const extracted =
      extractTargetFromCommand(command);

    if (!extracted) {
      setMessage(
        "Please enter an object to find.",
      );
      void voiceManager.speak(
        "Please enter an object to find.",
        { priority: 1 },
      );
      return;
    }

    setSearchTarget(extracted);

    setState("searching");

    setMessage(
      `Searching for ${extracted}...`,
    );

    void voiceManager.speak(
      `Searching for ${extracted}.`,
      { priority: 1 },
    );
  }, [command, setSearchTarget]);

  /* ------------------------------------------------------------------------ */
  /* Stop search                                                              */
  /* ------------------------------------------------------------------------ */

  const stopSearch = useCallback(() => {
    targetRef.current = "";

    setTarget("");
    setState("idle");
    setProcessing(false);

    setMessage(
      "Finder stopped.",
    );

    void voiceManager.speak(
      "Finder stopped.",
      { priority: 1 },
    );
  }, []);

  /* ------------------------------------------------------------------------ */
  /* Permission                                                               */
  /* ------------------------------------------------------------------------ */

  if (!permission) {
    return (
      <View
        accessible={true}
        accessibilityRole="progressbar"
        accessibilityLabel="Checking camera permission"
        style={styles.center}
      >
        <ActivityIndicator
          size="large"
          color={COLORS.primary}
        />

        <Text style={styles.loadingText}>
          Checking camera permission...
        </Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionScreen}>
        <View
          accessible={true}
          accessibilityRole="alert"
          accessibilityLabel="Camera Permission Required. SmartVisionAI Finder needs access to your camera to locate objects around you."
          style={styles.permissionCard}
        >
          <View style={styles.iconCircle} importantForAccessibility="no">
            <Ionicons
              name="camera"
              size={38}
              color={COLORS.primary}
            />
          </View>

          <Text style={styles.title}>
            Camera Permission Required
          </Text>

          <Text style={styles.description}>
            SmartVisionAI Finder needs access to
            your camera to locate objects around
            you.
          </Text>

          <Pressable
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Allow camera access"
            accessibilityHint="Requests camera permission required for object detection"
            style={styles.primaryButton}
            onPress={requestPermission}
          >
            <Ionicons
              name="camera-outline"
              size={20}
              color={COLORS.white}
            />

            <Text style={styles.buttonText}>
              ALLOW CAMERA
            </Text>
          </Pressable>

          <Pressable
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Go back"
            accessibilityHint="Returns to home screen"
            style={styles.secondaryButton}
            onPress={() => router.back()}
          >
            <Text style={styles.buttonText}>
              GO BACK
            </Text>
          </Pressable>
        </View>
      </View>
    );
  }

  /* ------------------------------------------------------------------------ */
  /* Main UI                                                                  */
  /* ------------------------------------------------------------------------ */

  return (
    <View style={styles.screen}>
      <View style={StyleSheet.absoluteFill}>
        <CameraView
          detectionIntervalMs={FINDER_INTERVAL_MS}
          imageQuality={IMAGE_QUALITY}
          showOverlay={false}
          autoDetect={state === "searching"}
          onDetection={(response) => {
            const wanted = targetRef.current;
            if (wanted && !processingRef.current) {
              processDetectionResult(response, wanted);
            }
          }}
          {...({
            quality: IMAGE_QUALITY,
            intervalMs: FINDER_INTERVAL_MS,
            onCapture: handleCapturedImage,
          } as any)}
        />
      </View>

      {/* Dark overlay */}
      <View
        pointerEvents="box-none"
        style={styles.overlay}
      >
        {/* Header */}
        <View style={styles.header}>
          <Pressable
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Go back"
            accessibilityHint="Returns to home screen"
            style={styles.headerButton}
            onPress={() => router.back()}
          >
            <Ionicons
              name="arrow-back"
              size={24}
              color={COLORS.white}
            />
          </Pressable>

          <View
            accessible={true}
            accessibilityRole="header"
            accessibilityLabel={`Object Finder. Status: ${state}`}
            style={styles.headerTitleBox}
          >
            <Text style={styles.headerTitle}>
              Object Finder
            </Text>

            <Text style={styles.headerSubtitle}>
              Find objects using AI
            </Text>
          </View>

          <View
            importantForAccessibility="no"
            accessibilityElementsHidden={true}
            style={[
              styles.statusDot,
              state === "found"
                ? styles.statusFound
                : state === "error"
                  ? styles.statusError
                  : styles.statusSearching,
            ]}
          />
        </View>

        {/* Center target */}
        <View
          pointerEvents="none"
          importantForAccessibility="no-hide-descendants"
          accessibilityElementsHidden={true}
          style={styles.targetArea}
        >
          <View style={styles.cornerTL} />
          <View style={styles.cornerTR} />
          <View style={styles.cornerBL} />
          <View style={styles.cornerBR} />

          {processing && (
            <ActivityIndicator
              size="large"
              color={COLORS.primary}
            />
          )}
        </View>

        {/* Bottom panel */}
        <View style={styles.bottomPanel}>
          <Text
            accessible={true}
            accessibilityRole="header"
            style={styles.panelTitle}
          >
            {target
              ? `Finding: ${target}`
              : "What do you want to find?"}
          </Text>

          <Text
            accessible={true}
            accessibilityRole="text"
            style={styles.panelHint}
          >
            Type an object name or command
          </Text>

          <TextInput
            accessible={true}
            accessibilityRole="search"
            accessibilityLabel="Target object search input"
            accessibilityHint="Enter name of object to find, such as bottle or chair"
            value={command}
            onChangeText={setCommand}
            placeholder='Example: "find my bottle"'
            placeholderTextColor={
              COLORS.secondaryText
            }
            style={styles.input}
            autoCapitalize="none"
            autoCorrect={false}
            returnKeyType="search"
            onSubmitEditing={startSearch}
          />

          <View style={styles.buttonRow}>
            <Pressable
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel={processing ? "Searching for target" : "Start search"}
              accessibilityHint="Starts searching for the specified object"
              accessibilityState={{ disabled: processing, busy: processing }}
              style={styles.searchButton}
              onPress={startSearch}
              disabled={processing}
            >
              {processing ? (
                <ActivityIndicator
                  color={COLORS.white}
                />
              ) : (
                <Ionicons
                  name="search"
                  size={21}
                  color={COLORS.white}
                />
              )}

              <Text style={styles.buttonText}>
                {processing
                  ? "SEARCHING"
                  : "START SEARCH"}
              </Text>
            </Pressable>

            <Pressable
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Stop search"
              accessibilityHint="Stops searching for objects"
              style={styles.stopButton}
              onPress={stopSearch}
            >
              <Ionicons
                name="stop"
                size={20}
                color={COLORS.white}
              />

              <Text style={styles.buttonText}>
                STOP
              </Text>
            </Pressable>
          </View>

          {/* Message */}
          <View
            accessible={true}
            accessibilityRole="alert"
            accessibilityLiveRegion="polite"
            accessibilityLabel={`Finder Status: ${message}`}
            style={styles.messageCard}
          >
            <View style={styles.messageIcon} importantForAccessibility="no">
              <Ionicons
                name={
                  state === "found"
                    ? "checkmark-circle"
                    : state === "error"
                      ? "warning"
                      : "scan"
                }
                size={23}
                color={
                  state === "found"
                    ? COLORS.success
                    : state === "error"
                      ? COLORS.danger
                      : COLORS.primary
                }
              />
            </View>

            <View style={styles.messageContent}>
              <Text style={styles.messageLabel}>
                Finder Status
              </Text>

              <Text style={styles.messageText}>
                {message}
              </Text>
            </View>
          </View>

          {/* Found object */}
          {lastResult.object && (
            <View
              accessible={true}
              accessibilityRole="summary"
              accessibilityLiveRegion="assertive"
              accessibilityLabel={lastResult.message || `Object found: ${getDetectionLabel(lastResult.object)}`}
              style={styles.resultCard}
            >
              <View style={styles.resultHeader}>
                <Ionicons
                  name="location"
                  size={22}
                  color={COLORS.success}
                />

                <Text style={styles.resultTitle}>
                  Object Found
                </Text>
              </View>

              <Text style={styles.resultObject}>
                {getDetectionLabel(
                  lastResult.object,
                )}
              </Text>

              <Text style={styles.resultDirection}>
                {lastResult.direction}
              </Text>

              <Text style={styles.resultConfidence}>
                Confidence:{" "}
                {Math.round(
                  Number(
                    lastResult.object
                      .confidence ?? 0,
                  ) * 100,
                )}
                %
              </Text>
            </View>
          )}

          <Text
            accessible={true}
            accessibilityRole="text"
            style={styles.exampleText}
          >
            Try: bottle • chair • person • car •
            backpack
          </Text>
        </View>
      </View>
    </View>
  );
}

/* -------------------------------------------------------------------------- */
/* Styles                                                                     */
/* -------------------------------------------------------------------------- */

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },

  center: {
    flex: 1,
    backgroundColor: COLORS.bg,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },

  loadingText: {
    marginTop: 14,
    color: COLORS.secondaryText,
    fontSize: 15,
  },

  permissionScreen: {
    flex: 1,
    backgroundColor: COLORS.bg,
    alignItems: "center",
    justifyContent: "center",
    padding: 22,
  },

  permissionCard: {
    width: "100%",
    backgroundColor: COLORS.panel,
    borderRadius: 24,
    padding: 24,
    alignItems: "center",
  },

  iconCircle: {
    width: 82,
    height: 82,
    borderRadius: 41,
    backgroundColor: COLORS.surfaceLight,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 20,
  },

  title: {
    color: COLORS.white,
    fontSize: 25,
    fontWeight: "900",
    textAlign: "center",
    marginBottom: 12,
  },

  description: {
    color: COLORS.secondaryText,
    fontSize: 15,
    lineHeight: 22,
    textAlign: "center",
    marginBottom: 24,
  },

  overlay: {
    flex: 1,
    justifyContent: "space-between",
  },

  header: {
    paddingTop: 54,
    paddingHorizontal: 16,
    flexDirection: "row",
    alignItems: "center",
  },

  headerButton: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: "rgba(7,16,28,0.82)",
    alignItems: "center",
    justifyContent: "center",
  },

  headerTitleBox: {
    flex: 1,
    marginLeft: 13,
  },

  headerTitle: {
    color: COLORS.white,
    fontSize: 20,
    fontWeight: "900",
  },

  headerSubtitle: {
    color: COLORS.secondaryText,
    fontSize: 12,
    marginTop: 2,
  },

  statusDot: {
    width: 13,
    height: 13,
    borderRadius: 7,
    marginLeft: 10,
  },

  statusSearching: {
    backgroundColor: COLORS.primary,
  },

  statusFound: {
    backgroundColor: COLORS.success,
  },

  statusError: {
    backgroundColor: COLORS.danger,
  },

  targetArea: {
    position: "absolute",
    top: "35%",
    left: "50%",
    width: 190,
    height: 190,
    marginLeft: -95,
    marginTop: -95,
    alignItems: "center",
    justifyContent: "center",
  },

  cornerTL: {
    position: "absolute",
    left: 0,
    top: 0,
    width: 38,
    height: 38,
    borderLeftWidth: 4,
    borderTopWidth: 4,
    borderColor: COLORS.primary,
    borderTopLeftRadius: 10,
  },

  cornerTR: {
    position: "absolute",
    right: 0,
    top: 0,
    width: 38,
    height: 38,
    borderRightWidth: 4,
    borderTopWidth: 4,
    borderColor: COLORS.primary,
    borderTopRightRadius: 10,
  },

  cornerBL: {
    position: "absolute",
    left: 0,
    bottom: 0,
    width: 38,
    height: 38,
    borderLeftWidth: 4,
    borderBottomWidth: 4,
    borderColor: COLORS.primary,
    borderBottomLeftRadius: 10,
  },

  cornerBR: {
    position: "absolute",
    right: 0,
    bottom: 0,
    width: 38,
    height: 38,
    borderRightWidth: 4,
    borderBottomWidth: 4,
    borderColor: COLORS.primary,
    borderBottomRightRadius: 10,
  },

  bottomPanel: {
    backgroundColor: "rgba(5,12,22,0.96)",
    borderTopLeftRadius: 26,
    borderTopRightRadius: 26,
    paddingHorizontal: 17,
    paddingTop: 18,
    paddingBottom: 25,
  },

  panelTitle: {
    color: COLORS.white,
    fontSize: 20,
    fontWeight: "900",
  },

  panelHint: {
    color: COLORS.secondaryText,
    fontSize: 12,
    marginTop: 3,
    marginBottom: 10,
  },

  input: {
    height: 50,
    borderRadius: 13,
    backgroundColor: COLORS.surfaceLight,
    color: COLORS.white,
    paddingHorizontal: 15,
    fontSize: 15,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  buttonRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 10,
  },

  primaryButton: {
    minHeight: 52,
    width: "100%",
    borderRadius: 14,
    backgroundColor: COLORS.primary,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 9,
  },

  secondaryButton: {
    minHeight: 50,
    width: "100%",
    borderRadius: 14,
    backgroundColor: COLORS.surfaceLight,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 10,
  },

  searchButton: {
    flex: 1,
    minHeight: 50,
    borderRadius: 13,
    backgroundColor: COLORS.primary,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
  },

  stopButton: {
    width: 95,
    minHeight: 50,
    borderRadius: 13,
    backgroundColor: COLORS.danger,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 7,
  },

  buttonText: {
    color: COLORS.white,
    fontSize: 13,
    fontWeight: "900",
  },

  messageCard: {
    marginTop: 11,
    borderRadius: 14,
    backgroundColor: COLORS.panel2,
    padding: 12,
    flexDirection: "row",
    alignItems: "center",
  },

  messageIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.surfaceLight,
    alignItems: "center",
    justifyContent: "center",
  },

  messageContent: {
    flex: 1,
    marginLeft: 10,
  },

  messageLabel: {
    color: COLORS.secondaryText,
    fontSize: 10,
    fontWeight: "800",
    textTransform: "uppercase",
  },

  messageText: {
    color: COLORS.white,
    fontSize: 14,
    fontWeight: "700",
    marginTop: 2,
  },

  resultCard: {
    marginTop: 10,
    backgroundColor: "rgba(28,132,84,0.18)",
    borderWidth: 1,
    borderColor: COLORS.success,
    borderRadius: 14,
    padding: 13,
  },

  resultHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 7,
  },

  resultTitle: {
    color: COLORS.success,
    fontSize: 13,
    fontWeight: "900",
  },

  resultObject: {
    color: COLORS.white,
    fontSize: 20,
    fontWeight: "900",
    marginTop: 6,
  },

  resultDirection: {
    color: COLORS.secondaryText,
    fontSize: 14,
    marginTop: 2,
  },

  resultConfidence: {
    color: COLORS.success,
    fontSize: 12,
    fontWeight: "800",
    marginTop: 5,
  },

  exampleText: {
    color: COLORS.secondaryText,
    fontSize: 11,
    textAlign: "center",
    marginTop: 9,
  },
});