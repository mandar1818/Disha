/**
 * SmartVisionAI - Accessible Home Screen (Phase 13)
 *
 * Responsibilities:
 * - Clean, accessible navigation hub for visually impaired users.
 * - TalkBack screen-reader compliance with accessible={true}, accessibilityLabel,
 *   accessibilityHint, and accessibilityRole="button" across all controls.
 * - Single mount announcement via AccessibilityInfo.announceForAccessibility().
 * - Direct deterministic routing to:
 *     1. Live Guidance -> /camera
 *     2. Finder        -> /finder
 *     3. Settings      -> /settings
 *     4. About         -> /about
 * - Zero background camera detection, zero watchdog loops, zero speech manager
 *   instantiation on the home screen.
 */

import React, { useCallback, useEffect, useState } from "react";
import {
  AccessibilityInfo,
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { router } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";

import {
  APP_NAME,
  APP_VERSION,
  COLORS,
} from "src/constants/config";

import { checkBackendHealth } from "src/services/api";

type BackendState = "checking" | "online" | "offline";

export default function HomeScreen() {
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [checking, setChecking] = useState(false);

  /* =======================================================
     AUTOMATIC CAMERA LAUNCH (NO DASHBOARD DELAY)
  ======================================================= */

  useEffect(() => {
    // Auto-launch camera assist mode immediately on open
    router.replace("/camera");
  }, []);

  /* =======================================================
     ACCESSIBILITY MOUNT ANNOUNCEMENT
  ======================================================= */

  useEffect(() => {
    AccessibilityInfo.announceForAccessibility(
      "SmartVisionAI home. Choose Live Guidance, Finder, Settings, or About."
    );
  }, []);

  /* =======================================================
     BACKEND HEALTH STATUS
  ======================================================= */

  const checkConnection = useCallback(async () => {
    setChecking(true);
    setBackendState("checking");

    try {
      const healthy = await checkBackendHealth();
      setBackendState(healthy ? "online" : "offline");
    } catch {
      setBackendState("offline");
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void checkConnection();
  }, [checkConnection]);

  const backendLabel =
    backendState === "online"
      ? "Backend Online"
      : backendState === "offline"
      ? "Backend Offline"
      : "Checking Backend...";

  const backendColor =
    backendState === "online"
      ? COLORS.safe
      : backendState === "offline"
      ? COLORS.danger
      : COLORS.caution;

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView
        contentContainerStyle={styles.container}
        showsVerticalScrollIndicator={false}
      >
        {/* ================================================= */}
        {/* HEADER                                            */}
        {/* ================================================= */}

        <View style={styles.header} accessible={true}>
          <View style={styles.headerTextContainer}>
            <Text style={styles.appName}>{APP_NAME}</Text>
            <Text style={styles.subtitle}>
              Accessible real-time obstacle detection and navigation assistance.
            </Text>
          </View>

          <View style={styles.versionBadge} accessible={false}>
            <Text style={styles.versionText}>v{APP_VERSION}</Text>
          </View>
        </View>

        {/* ================================================= */}
        {/* BACKEND STATUS CARD                               */}
        {/* ================================================= */}

        <View
          style={styles.statusCard}
          accessible={true}
          accessibilityRole="summary"
          accessibilityLabel={`System Status: ${backendLabel}`}
          accessibilityHint="Double tap the refresh button on the right to re-check backend status."
        >
          <View
            style={[
              styles.statusDot,
              { backgroundColor: backendColor },
            ]}
          />

          <View style={styles.statusContent}>
            <Text style={styles.statusTitle}>System Status</Text>
            <Text style={[styles.statusValue, { color: backendColor }]}>
              {backendLabel}
            </Text>
          </View>

          <Pressable
            style={styles.refreshButton}
            onPress={() => void checkConnection()}
            disabled={checking}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Check backend status"
            accessibilityHint="Probes backend server connection"
          >
            {checking ? (
              <ActivityIndicator size="small" color={COLORS.white} />
            ) : (
              <Text style={styles.refreshText}>↻</Text>
            )}
          </Pressable>
        </View>

        {/* ================================================= */}
        {/* PRIMARY NAVIGATION CONTROLS (4 CARDS)             */}
        {/* ================================================= */}

        <Text style={styles.sectionTitle} accessibilityRole="header">
          Main Navigation
        </Text>

        {/* 1. LIVE GUIDANCE */}
        <Pressable
          style={({ pressed }) => [
            styles.navCard,
            styles.primaryNavCard,
            pressed && styles.cardPressed,
          ]}
          onPress={() => router.push("/camera")}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="Live Guidance"
          accessibilityHint="Open continuous camera guidance and obstacle detection."
        >
          <View style={styles.navIconBox}>
            <Text style={styles.navIcon}>📷</Text>
          </View>

          <View style={styles.navContent}>
            <Text style={styles.navTitle}>Live Guidance</Text>
            <Text style={styles.navDescription}>
              Continuous camera guidance, obstacle tracking, and walking step instructions.
            </Text>
          </View>

          <Text style={styles.navArrow}>›</Text>
        </Pressable>

        {/* 2. FINDER */}
        <Pressable
          style={({ pressed }) => [
            styles.navCard,
            pressed && styles.cardPressed,
          ]}
          onPress={() => router.push("/finder")}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="Finder"
          accessibilityHint="Find and locate detected objects."
        >
          <View style={styles.navIconBox}>
            <Text style={styles.navIcon}>🎙️</Text>
          </View>

          <View style={styles.navContent}>
            <Text style={styles.navTitle}>Finder</Text>
            <Text style={styles.navDescription}>
              Search for specific target objects with voice input and step guidance.
            </Text>
          </View>

          <Text style={styles.navArrow}>›</Text>
        </Pressable>

        {/* 3. SETTINGS */}
        <Pressable
          style={({ pressed }) => [
            styles.navCard,
            pressed && styles.cardPressed,
          ]}
          onPress={() => router.push("/settings")}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="Settings"
          accessibilityHint="Open SmartVisionAI settings."
        >
          <View style={styles.navIconBox}>
            <Text style={styles.navIcon}>⚙️</Text>
          </View>

          <View style={styles.navContent}>
            <Text style={styles.navTitle}>Settings</Text>
            <Text style={styles.navDescription}>
              Configure server connection URL, haptic feedback, and speech options.
            </Text>
          </View>

          <Text style={styles.navArrow}>›</Text>
        </Pressable>

        {/* 4. ABOUT */}
        <Pressable
          style={({ pressed }) => [
            styles.navCard,
            pressed && styles.cardPressed,
          ]}
          onPress={() => router.push("/about")}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="About SmartVisionAI"
          accessibilityHint="View information about SmartVisionAI."
        >
          <View style={styles.navIconBox}>
            <Text style={styles.navIcon}>ℹ️</Text>
          </View>

          <View style={styles.navContent}>
            <Text style={styles.navTitle}>About SmartVisionAI</Text>
            <Text style={styles.navDescription}>
              System architecture, model specifications, and project background.
            </Text>
          </View>

          <Text style={styles.navArrow}>›</Text>
        </Pressable>

        {/* ================================================= */}
        {/* FOOTER                                           */}
        {/* ================================================= */}

        <Text style={styles.footer} accessible={true}>
          SmartVisionAI • Assistive Computer Vision
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

/* =========================================================
   STYLES
   ========================================================= */

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },

  container: {
    padding: 18,
    paddingBottom: 35,
  },

  /* Header */
  header: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    marginBottom: 20,
  },

  headerTextContainer: {
    flex: 1,
    marginRight: 12,
  },

  appName: {
    color: COLORS.white,
    fontSize: 28,
    fontWeight: "900",
    letterSpacing: 0.5,
  },

  subtitle: {
    color: COLORS.muted,
    fontSize: 13,
    lineHeight: 18,
    marginTop: 4,
  },

  versionBadge: {
    backgroundColor: COLORS.panel2,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  versionText: {
    color: COLORS.secondaryText,
    fontSize: 11,
    fontWeight: "800",
  },

  /* Status */
  statusCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: COLORS.panel,
    borderRadius: 16,
    padding: 14,
    marginBottom: 22,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 12,
  },

  statusContent: {
    flex: 1,
  },

  statusTitle: {
    color: COLORS.muted,
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },

  statusValue: {
    fontSize: 14,
    fontWeight: "900",
    marginTop: 2,
  },

  refreshButton: {
    width: 38,
    height: 38,
    borderRadius: 12,
    backgroundColor: COLORS.panel2,
    alignItems: "center",
    justifyContent: "center",
  },

  refreshText: {
    color: COLORS.white,
    fontSize: 22,
    fontWeight: "700",
  },

  /* Section Title */
  sectionTitle: {
    color: COLORS.white,
    fontSize: 16,
    fontWeight: "900",
    marginBottom: 12,
    letterSpacing: 0.5,
  },

  /* Navigation Cards */
  navCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: COLORS.panel,
    borderRadius: 18,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  primaryNavCard: {
    borderColor: COLORS.primary,
    backgroundColor: "rgba(22, 119, 255, 0.08)",
  },

  cardPressed: {
    opacity: 0.8,
    transform: [{ scale: 0.99 }],
  },

  navIconBox: {
    width: 52,
    height: 52,
    borderRadius: 14,
    backgroundColor: COLORS.surfaceLight,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 14,
  },

  navIcon: {
    fontSize: 26,
  },

  navContent: {
    flex: 1,
  },

  navTitle: {
    color: COLORS.white,
    fontSize: 17,
    fontWeight: "900",
    marginBottom: 4,
  },

  navDescription: {
    color: COLORS.muted,
    fontSize: 12,
    lineHeight: 17,
  },

  navArrow: {
    color: COLORS.secondaryText,
    fontSize: 26,
    marginLeft: 8,
    fontWeight: "300",
  },

  /* Footer */
  footer: {
    color: COLORS.muted,
    textAlign: "center",
    fontSize: 11,
    marginTop: 24,
  },
});