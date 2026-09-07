import React, { useCallback, useEffect, useState } from "react";
import {
  AccessibilityInfo,
  ActivityIndicator,
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from "react-native";
import { router } from "expo-router";

import {
  COLORS,
  DETECTION_INTERVAL_MS,
  FINDER_INTERVAL_MS,
  IMAGE_QUALITY,
  MIN_CONFIDENCE,
  SPEECH_COOLDOWN_MS,
  getBackendUrl,
  setBackendUrl,
} from "src/constants/config";

import {
  checkBackendHealth,
  getSystemStatus,
  testConnection,
} from "src/services/api";

import { isHapticsEnabled, setHapticsEnabled } from "src/services/haptics";

import type { SystemStatusResponse } from "src/types/smartvision";

/* -------------------------------------------------------------------------- */
/* Types                                                                      */
/* -------------------------------------------------------------------------- */

type ToggleRowProps = {
  title: string;
  description: string;
  value: boolean;
  onValueChange: (value: boolean) => void;
  accessibilityHint?: string;
};

type SettingRowProps = {
  title: string;
  value: string;
  description?: string;
};

/* -------------------------------------------------------------------------- */
/* Reusable UI                                                                */
/* -------------------------------------------------------------------------- */

function ToggleRow({
  title,
  description,
  value,
  onValueChange,
  accessibilityHint,
}: ToggleRowProps) {
  return (
    <Pressable
      style={styles.settingRow}
      onPress={() => onValueChange(!value)}
      accessible={true}
      accessibilityRole="switch"
      accessibilityLabel={title}
      accessibilityHint={
        accessibilityHint ??
        `Double tap to turn ${value ? "off" : "on"} ${title}.`
      }
      accessibilityState={{ checked: value }}
    >
      <View style={styles.settingTextContainer} importantForAccessibility="no">
        <Text style={styles.settingTitle}>{title}</Text>

        <Text style={styles.settingDescription}>{description}</Text>
      </View>

      <Switch
        value={value}
        onValueChange={onValueChange}
        accessible={false}
        importantForAccessibility="no"
        trackColor={{
          false: COLORS.panel2,
          true: COLORS.primaryDark,
        }}
        thumbColor={value ? COLORS.primary : COLORS.muted}
      />
    </Pressable>
  );
}

function SettingRow({
  title,
  value,
  description,
}: SettingRowProps) {
  const rowLabel = description
    ? `${title}: ${value}. ${description}`
    : `${title}: ${value}`;

  return (
    <View
      style={styles.settingRow}
      accessible={true}
      accessibilityRole="text"
      accessibilityLabel={rowLabel}
    >
      <View style={styles.settingTextContainer} importantForAccessibility="no">
        <Text style={styles.settingTitle}>{title}</Text>

        {description ? (
          <Text style={styles.settingDescription}>
            {description}
          </Text>
        ) : null}
      </View>

      <Text style={styles.settingValue} importantForAccessibility="no">
        {value}
      </Text>
    </View>
  );
}

/* -------------------------------------------------------------------------- */
/* Main Screen                                                                */
/* -------------------------------------------------------------------------- */

export default function SettingsScreen() {
  const [backendUrl, setBackendUrlState] = useState("");

  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [emergencyVoiceEnabled, setEmergencyVoiceEnabled] = useState(true);
  const [finderVoiceEnabled, setFinderVoiceEnabled] = useState(true);
  const [autoDetectionEnabled, setAutoDetectionEnabled] = useState(true);
  const [hapticsEnabled, setHapticsEnabledState] = useState(isHapticsEnabled());

  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<
    "unknown" | "connected" | "failed"
  >("unknown");

  const [systemStatus, setSystemStatus] =
    useState<SystemStatusResponse | null>(null);

  const [loadingStatus, setLoadingStatus] = useState(false);

  /* ------------------------------------------------------------------------ */
  /* Load configuration & Screen Entrance Announcement                       */
  /* ------------------------------------------------------------------------ */

  useEffect(() => {
    setBackendUrlState(getBackendUrl());
    AccessibilityInfo.announceForAccessibility(
      "Settings screen. Disha accessibility and guidance preferences."
    );
  }, []);

  const handleHapticsToggle = useCallback((value: boolean) => {
    setHapticsEnabledState(value);
    setHapticsEnabled(value);
  }, []);

  /* ------------------------------------------------------------------------ */
  /* Backend connection                                                       */
  /* ------------------------------------------------------------------------ */

  const handleTestConnection = useCallback(async () => {
    setTestingConnection(true);
    setConnectionStatus("unknown");

    try {
      const result = await testConnection();

      setConnectionStatus(result ? "connected" : "failed");

      if (result) {
        Alert.alert(
          "Backend Connected",
          "Disha successfully connected to the FastAPI backend."
        );
      } else {
        Alert.alert(
          "Connection Failed",
          "The mobile app could not connect to the Disha backend.\n\nCheck that:\n• FastAPI is running\n• Your phone and PC are on the same network\n• The backend IP address is correct\n• Port 8000 is allowed"
        );
      }
    } catch {
      setConnectionStatus("failed");

      Alert.alert(
        "Connection Error",
        "Unable to test the backend connection."
      );
    } finally {
      setTestingConnection(false);
    }
  }, []);

  /* ------------------------------------------------------------------------ */
  /* Save backend URL                                                         */
  /* ------------------------------------------------------------------------ */

  const handleSaveBackendUrl = useCallback(() => {
    const cleaned = backendUrl.trim().replace(/\/+$/, "");

    if (!cleaned) {
      Alert.alert(
        "Invalid URL",
        "Please enter a valid backend URL."
      );
      return;
    }

    if (
      !cleaned.startsWith("http://") &&
      !cleaned.startsWith("https://")
    ) {
      Alert.alert(
        "Invalid URL",
        "Backend URL must start with http:// or https://."
      );
      return;
    }

    setBackendUrl(cleaned);
    setBackendUrlState(cleaned);
    setConnectionStatus("unknown");

    Alert.alert(
      "Backend URL Saved",
      `New backend URL:\n${cleaned}\n\nTest the connection to verify it.`
    );
  }, [backendUrl]);

  /* ------------------------------------------------------------------------ */
  /* System status                                                            */
  /* ------------------------------------------------------------------------ */

  const handleSystemStatus = useCallback(async () => {
    setLoadingStatus(true);

    try {
      const result = await getSystemStatus();

      setSystemStatus(result);

      if (!result.success) {
        Alert.alert(
          "System Status",
          result.error ?? "Unable to get backend system status."
        );
      }
    } catch {
      Alert.alert(
        "System Status",
        "Unable to communicate with the backend."
      );
    } finally {
      setLoadingStatus(false);
    }
  }, []);

  /* ------------------------------------------------------------------------ */
  /* Health check                                                             */
  /* ------------------------------------------------------------------------ */

  const handleHealthCheck = useCallback(async () => {
    setTestingConnection(true);

    try {
      const healthy = await checkBackendHealth();

      setConnectionStatus(
        healthy ? "connected" : "failed"
      );

      Alert.alert(
        healthy ? "Backend Healthy" : "Backend Unavailable",
        healthy
          ? "FastAPI backend is healthy and ready."
          : "FastAPI backend is not responding."
      );
    } catch {
      setConnectionStatus("failed");

      Alert.alert(
        "Health Check Failed",
        "Could not check backend health."
      );
    } finally {
      setTestingConnection(false);
    }
  }, []);

  /* ------------------------------------------------------------------------ */
  /* Backend status text                                                      */
  /* ------------------------------------------------------------------------ */

  const connectionText =
    connectionStatus === "connected"
      ? "CONNECTED"
      : connectionStatus === "failed"
        ? "OFFLINE"
        : "NOT TESTED";

  const connectionColor =
    connectionStatus === "connected"
      ? COLORS.success
      : connectionStatus === "failed"
        ? COLORS.danger
        : COLORS.muted;

  const processorLoaded =
    systemStatus?.components?.loaded === true;

  /* ------------------------------------------------------------------------ */
  /* Render                                                                   */
  /* ------------------------------------------------------------------------ */

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="Go back"
          accessibilityHint="Returns to the Disha home screen"
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <Text style={styles.backButtonText} importantForAccessibility="no">‹</Text>
        </Pressable>

        <View
          accessible={true}
          accessibilityRole="header"
          accessibilityLabel="Settings. Disha configuration"
          style={styles.headerText}
        >
          <Text style={styles.headerTitle}>Settings</Text>

          <Text style={styles.headerSubtitle}>
            Disha configuration
          </Text>
        </View>
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* ---------------------------------------------------------------- */}
        {/* Backend                                                          */}
        {/* ---------------------------------------------------------------- */}

        <Text
          accessible={true}
          accessibilityRole="header"
          style={styles.sectionTitle}
        >
          BACKEND CONNECTION
        </Text>

        <View style={styles.card}>
          <Text
            accessible={true}
            accessibilityRole="text"
            style={styles.inputLabel}
          >
            FastAPI Backend URL
          </Text>

          <TextInput
            accessible={true}
            accessibilityRole="text"
            accessibilityLabel="FastAPI Backend URL input"
            accessibilityHint="Enter backend URL such as http://192.168.1.100:8000"
            value={backendUrl}
            onChangeText={setBackendUrlState}
            placeholder="http://192.168.x.x:8000"
            placeholderTextColor={COLORS.muted}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
            style={styles.input}
          />

          <Pressable
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Save Backend URL"
            accessibilityHint="Saves the entered backend URL address"
            style={styles.primaryButton}
            onPress={handleSaveBackendUrl}
          >
            <Text style={styles.primaryButtonText}>
              SAVE BACKEND URL
            </Text>
          </Pressable>

          <View
            accessible={true}
            accessibilityRole="text"
            accessibilityLabel={`Connection status: ${connectionText}`}
            style={styles.connectionRow}
          >
            <View
              importantForAccessibility="no"
              accessibilityElementsHidden={true}
              style={[
                styles.connectionDot,
                {
                  backgroundColor: connectionColor,
                },
              ]}
            />

            <Text
              importantForAccessibility="no"
              style={[
                styles.connectionText,
                {
                  color: connectionColor,
                },
              ]}
            >
              {connectionText}
            </Text>
          </View>

          <View style={styles.buttonRow}>
            <Pressable
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Test connection"
              accessibilityHint="Tests network communication with the FastAPI backend"
              accessibilityState={{ disabled: testingConnection, busy: testingConnection }}
              style={styles.secondaryButton}
              onPress={handleTestConnection}
              disabled={testingConnection}
            >
              {testingConnection ? (
                <ActivityIndicator
                  size="small"
                  color={COLORS.primary}
                />
              ) : (
                <Text style={styles.secondaryButtonText}>
                  TEST CONNECTION
                </Text>
              )}
            </Pressable>

            <Pressable
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Health check"
              accessibilityHint="Checks the FastAPI health endpoint status"
              accessibilityState={{ disabled: testingConnection, busy: testingConnection }}
              style={styles.secondaryButton}
              onPress={handleHealthCheck}
              disabled={testingConnection}
            >
              <Text style={styles.secondaryButtonText}>
                HEALTH
              </Text>
            </Pressable>
          </View>
        </View>

        {/* ---------------------------------------------------------------- */}
        {/* Detection                                                         */}
        {/* ---------------------------------------------------------------- */}

        <Text
          accessible={true}
          accessibilityRole="header"
          style={styles.sectionTitle}
        >
          DETECTION
        </Text>

        <View style={styles.card}>
          <ToggleRow
            title="Live Detection"
            description="Continuously process camera frames."
            value={autoDetectionEnabled}
            onValueChange={setAutoDetectionEnabled}
            accessibilityHint="Toggles continuous camera frame processing."
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <SettingRow
            title="Detection Interval"
            value={`${DETECTION_INTERVAL_MS} ms`}
            description="Minimum time between live detection frames."
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <SettingRow
            title="Finder Interval"
            value={`${FINDER_INTERVAL_MS} ms`}
            description="Interval used while searching for an object."
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <SettingRow
            title="Confidence Threshold"
            value={`${Math.round(MIN_CONFIDENCE * 100)}%`}
            description="Minimum YOLO confidence accepted by the app."
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <SettingRow
            title="Image Quality"
            value={IMAGE_QUALITY.toFixed(2)}
            description="JPEG quality used when sending camera frames."
          />
        </View>

        {/* ---------------------------------------------------------------- */}
        {/* Voice                                                             */}
        {/* ---------------------------------------------------------------- */}

        <Text
          accessible={true}
          accessibilityRole="header"
          style={styles.sectionTitle}
        >
          VOICE GUIDANCE
        </Text>

        <View style={styles.card}>
          <ToggleRow
            title="Voice Guidance"
            description="Speak navigation and scene instructions."
            value={voiceEnabled}
            onValueChange={setVoiceEnabled}
            accessibilityHint="Toggles spoken navigation and scene guidance."
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <ToggleRow
            title="Emergency Voice"
            description="Allow urgent emergency-stop announcements."
            value={emergencyVoiceEnabled}
            onValueChange={setEmergencyVoiceEnabled}
            accessibilityHint="Toggles high-priority emergency stop announcements."
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <ToggleRow
            title="Finder Voice"
            description="Speak the location of requested objects."
            value={finderVoiceEnabled}
            onValueChange={setFinderVoiceEnabled}
            accessibilityHint="Toggles spoken target location announcements."
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <SettingRow
            title="Speech Cooldown"
            value={`${SPEECH_COOLDOWN_MS} ms`}
            description="Prevents repeated identical announcements."
          />
        </View>

        {/* ---------------------------------------------------------------- */}
        {/* Haptic Feedback                                                  */}
        {/* ---------------------------------------------------------------- */}

        <Text
          accessible={true}
          accessibilityRole="header"
          style={styles.sectionTitle}
        >
          HAPTIC FEEDBACK
        </Text>

        <View style={styles.card}>
          <ToggleRow
            title="Haptic Feedback"
            description="Provides distinct vibration pulses for emergency stop, connection loss, approaching obstacles, turn directions, and camera obstruction."
            value={hapticsEnabled}
            onValueChange={handleHapticsToggle}
            accessibilityHint="Turns vibration feedback on or off."
          />
        </View>

        {/* ---------------------------------------------------------------- */}
        {/* System status                                                     */}
        {/* ---------------------------------------------------------------- */}

        <Text
          accessible={true}
          accessibilityRole="header"
          style={styles.sectionTitle}
        >
          SYSTEM STATUS
        </Text>

        <View style={styles.card}>
          <View style={styles.statusHeader}>
            <View>
              <Text
                accessible={true}
                accessibilityRole="text"
                style={styles.statusTitle}
              >
                Backend Components
              </Text>

              <Text
                accessible={true}
                accessibilityRole="text"
                style={styles.statusDescription}
              >
                YOLO, MiDaS and Decision Engine status.
              </Text>
            </View>

            <Pressable
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Check system status"
              accessibilityHint="Retrieves YOLO, MiDaS, and Decision Engine backend component status"
              accessibilityState={{ disabled: loadingStatus, busy: loadingStatus }}
              style={styles.statusButton}
              onPress={handleSystemStatus}
              disabled={loadingStatus}
            >
              {loadingStatus ? (
                <ActivityIndicator
                  size="small"
                  color={COLORS.white}
                />
              ) : (
                <Text style={styles.statusButtonText}>
                  CHECK
                </Text>
              )}
            </Pressable>
          </View>

          {systemStatus ? (
            <View style={styles.systemStatusContainer}>
              <View
                accessible={true}
                accessibilityRole="text"
                accessibilityLabel={`Backend: ${systemStatus.success ? "READY" : "ERROR"}`}
                style={styles.statusLine}
              >
                <Text style={styles.statusLabel} importantForAccessibility="no">Backend</Text>

                <Text
                  importantForAccessibility="no"
                  style={[
                    styles.statusValue,
                    {
                      color: systemStatus.success
                        ? COLORS.success
                        : COLORS.danger,
                    },
                  ]}
                >
                  {systemStatus.success
                    ? "READY"
                    : "ERROR"}
                </Text>
              </View>

              <View
                accessible={true}
                accessibilityRole="text"
                accessibilityLabel={`Processor: ${processorLoaded ? "LOADED" : "NOT LOADED"}`}
                style={styles.statusLine}
              >
                <Text style={styles.statusLabel} importantForAccessibility="no">
                  Processor
                </Text>

                <Text
                  importantForAccessibility="no"
                  style={[
                    styles.statusValue,
                    {
                      color: processorLoaded
                        ? COLORS.success
                        : COLORS.warning,
                    },
                  ]}
                >
                  {processorLoaded
                    ? "LOADED"
                    : "NOT LOADED"}
                </Text>
              </View>

              <View
                accessible={true}
                accessibilityRole="text"
                accessibilityLabel={`YOLO: ${systemStatus.components?.yolo?.loaded ?? systemStatus.components?.yolo?.is_loaded ? "LOADED" : "UNKNOWN"}`}
                style={styles.statusLine}
              >
                <Text style={styles.statusLabel} importantForAccessibility="no">
                  YOLO
                </Text>

                <Text style={styles.statusValue} importantForAccessibility="no">
                  {systemStatus.components?.yolo?.loaded ??
                  systemStatus.components?.yolo?.is_loaded
                    ? "LOADED"
                    : "UNKNOWN"}
                </Text>
              </View>

              <View
                accessible={true}
                accessibilityRole="text"
                accessibilityLabel={`MiDaS: ${systemStatus.components?.midas?.loaded ?? systemStatus.components?.midas?.is_loaded ? "LOADED" : "UNKNOWN"}`}
                style={styles.statusLine}
              >
                <Text style={styles.statusLabel} importantForAccessibility="no">
                  MiDaS
                </Text>

                <Text style={styles.statusValue} importantForAccessibility="no">
                  {systemStatus.components?.midas?.loaded ??
                  systemStatus.components?.midas?.is_loaded
                    ? "LOADED"
                    : "UNKNOWN"}
                </Text>
              </View>

              <View
                accessible={true}
                accessibilityRole="text"
                accessibilityLabel={`Decision Engine: ${systemStatus.components?.decision_engine?.loaded ? "LOADED" : "UNKNOWN"}`}
                style={styles.statusLine}
              >
                <Text style={styles.statusLabel} importantForAccessibility="no">
                  Decision Engine
                </Text>

                <Text style={styles.statusValue} importantForAccessibility="no">
                  {systemStatus.components?.decision_engine?.loaded
                    ? "LOADED"
                    : "UNKNOWN"}
                </Text>
              </View>
            </View>
          ) : (
            <Text
              accessible={true}
              accessibilityRole="text"
              style={styles.noStatusText}
            >
              Press CHECK to retrieve backend component status.
            </Text>
          )}
        </View>

        {/* ---------------------------------------------------------------- */}
        {/* Information                                                       */}
        {/* ---------------------------------------------------------------- */}

        <Text
          accessible={true}
          accessibilityRole="header"
          style={styles.sectionTitle}
        >
          APPLICATION
        </Text>

        <View style={styles.card}>
          <SettingRow
            title="Detection Engine"
            value="YOLOv8"
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <SettingRow
            title="Depth Engine"
            value="MiDaS"
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <SettingRow
            title="Backend"
            value="FastAPI"
          />

          <View style={styles.divider} importantForAccessibility="no" />

          <SettingRow
            title="Platform"
            value="Android"
          />
        </View>

        {/* ---------------------------------------------------------------- */}
        {/* Warning                                                           */}
        {/* ---------------------------------------------------------------- */}

        <View
          accessible={true}
          accessibilityRole="alert"
          accessibilityLabel="Important notice: Disha navigation guidance is an assistance system. Always remain aware of your surroundings and do not depend exclusively on the application for personal safety."
          style={styles.warningCard}
        >
          <Text style={styles.warningTitle} importantForAccessibility="no">
            IMPORTANT
          </Text>

          <Text style={styles.warningText} importantForAccessibility="no">
            Disha navigation guidance is an assistance
            system. Always remain aware of your surroundings and
            do not depend exclusively on the application for
            personal safety.
          </Text>
        </View>

        {/* Bottom spacing */}
        <View style={styles.bottomSpace} importantForAccessibility="no" />
      </ScrollView>
    </View>
  );
}

/* -------------------------------------------------------------------------- */
/* Styles                                                                     */
/* -------------------------------------------------------------------------- */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },

  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 18,
    paddingTop: 18,
    paddingBottom: 16,
    backgroundColor: COLORS.background,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },

  backButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: COLORS.panel,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },

  backButtonText: {
    color: COLORS.white,
    fontSize: 34,
    lineHeight: 36,
    fontWeight: "300",
  },

  headerText: {
    flex: 1,
  },

  headerTitle: {
    color: COLORS.white,
    fontSize: 23,
    fontWeight: "800",
  },

  headerSubtitle: {
    color: COLORS.secondaryText,
    fontSize: 13,
    marginTop: 3,
  },

  scroll: {
    flex: 1,
  },

  content: {
    padding: 16,
  },

  sectionTitle: {
    color: COLORS.accent,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 1.2,
    marginTop: 10,
    marginBottom: 9,
    marginLeft: 4,
  },

  card: {
    backgroundColor: COLORS.panel,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 16,
    marginBottom: 10,
  },

  inputLabel: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "700",
    marginBottom: 9,
  },

  input: {
    height: 48,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 10,
    backgroundColor: COLORS.background,
    color: COLORS.white,
    paddingHorizontal: 13,
    fontSize: 14,
  },

  primaryButton: {
    height: 46,
    borderRadius: 10,
    backgroundColor: COLORS.primary,
    justifyContent: "center",
    alignItems: "center",
    marginTop: 11,
  },

  primaryButtonText: {
    color: COLORS.white,
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 0.5,
  },

  connectionRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 14,
  },

  connectionDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    marginRight: 8,
  },

  connectionText: {
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0.8,
  },

  buttonRow: {
    flexDirection: "row",
    gap: 9,
    marginTop: 12,
  },

  secondaryButton: {
    flex: 1,
    minHeight: 43,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.primary,
    backgroundColor: COLORS.panel2,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 8,
  },

  secondaryButtonText: {
    color: COLORS.accent,
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 0.3,
  },

  settingRow: {
    minHeight: 48,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },

  settingTextContainer: {
    flex: 1,
    paddingRight: 12,
  },

  settingTitle: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "700",
  },

  settingDescription: {
    color: COLORS.secondaryText,
    fontSize: 11,
    lineHeight: 16,
    marginTop: 3,
  },

  settingValue: {
    color: COLORS.accent,
    fontSize: 12,
    fontWeight: "800",
    maxWidth: 125,
    textAlign: "right",
  },

  divider: {
    height: 1,
    backgroundColor: COLORS.border,
    marginVertical: 8,
  },

  statusHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },

  statusTitle: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "800",
  },

  statusDescription: {
    color: COLORS.secondaryText,
    fontSize: 11,
    marginTop: 3,
  },

  statusButton: {
    minWidth: 72,
    height: 38,
    borderRadius: 9,
    backgroundColor: COLORS.primary,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 12,
  },

  statusButtonText: {
    color: COLORS.white,
    fontSize: 11,
    fontWeight: "800",
  },

  systemStatusContainer: {
    marginTop: 14,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingTop: 8,
  },

  statusLine: {
    minHeight: 38,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },

  statusLabel: {
    color: COLORS.secondaryText,
    fontSize: 12,
  },

  statusValue: {
    color: COLORS.accent,
    fontSize: 11,
    fontWeight: "800",
  },

  noStatusText: {
    color: COLORS.muted,
    fontSize: 11,
    lineHeight: 16,
    marginTop: 13,
  },

  warningCard: {
    backgroundColor: COLORS.panel,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.warning,
    padding: 15,
    marginTop: 8,
  },

  warningTitle: {
    color: COLORS.warning,
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 1,
    marginBottom: 6,
  },

  warningText: {
    color: COLORS.secondaryText,
    fontSize: 11,
    lineHeight: 17,
  },

  bottomSpace: {
    height: 35,
  },
});