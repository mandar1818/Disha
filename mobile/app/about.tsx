import React from "react";
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { router } from "expo-router";

import {
  APP_NAME,
  APP_VERSION,
  COLORS,
} from "src/constants/config";

const palette = {
  ...COLORS,
  primaryLight: COLORS.accent,
  textSecondary: COLORS.secondaryText,
  textPrimary: COLORS.text,
} as const;

export default function AboutScreen() {
  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <Text style={styles.backButtonText}>‹</Text>
        </Pressable>

        <View style={styles.headerText}>
          <Text style={styles.headerTitle}>About</Text>

          <Text style={styles.headerSubtitle}>
            {APP_NAME}
          </Text>
        </View>
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* App Identity */}
        <View style={styles.identityCard}>
          <View style={styles.logo}>
            <Text style={styles.logoText}>SV</Text>
          </View>

          <Text style={styles.appName}>
            {APP_NAME}
          </Text>

          <Text style={styles.version}>
            Version {APP_VERSION}
          </Text>

          <Text style={styles.tagline}>
            Intelligent Vision-Based Assistance
          </Text>
        </View>

        {/* About Project */}
        <Text style={styles.sectionTitle}>
          ABOUT THE PROJECT
        </Text>

        <View style={styles.card}>
          <Text style={styles.cardText}>
            Disha is an intelligent computer-vision
            assistance system designed to analyze the user's
            surroundings and provide real-time object detection,
            depth estimation, obstacle awareness, navigation
            guidance, and voice instructions.
          </Text>

          <Text style={[styles.cardText, styles.paragraph]}>
            The mobile application captures camera frames and
            communicates with the Disha FastAPI backend.
            The backend processes the frames using YOLOv8 object
            detection, MiDaS depth estimation, and a decision
            engine to determine safety and navigation information.
          </Text>
        </View>

        {/* Pipeline */}
        <Text style={styles.sectionTitle}>
          SYSTEM PIPELINE
        </Text>

        <View style={styles.card}>
          <PipelineItem
            number="01"
            title="Camera"
            description="Captures the surrounding scene."
          />

          <PipelineLine />

          <PipelineItem
            number="02"
            title="YOLOv8"
            description="Detects and classifies objects."
          />

          <PipelineLine />

          <PipelineItem
            number="03"
            title="MiDaS"
            description="Estimates scene and object depth."
          />

          <PipelineLine />

          <PipelineItem
            number="04"
            title="Decision Engine"
            description="Evaluates obstacle risk and safety."
          />

          <PipelineLine />

          <PipelineItem
            number="05"
            title="Navigation"
            description="Determines an appropriate movement direction."
          />

          <PipelineLine />

          <PipelineItem
            number="06"
            title="Voice Guidance"
            description="Provides spoken instructions."
          />
        </View>

        {/* Technology */}
        <Text style={styles.sectionTitle}>
          TECHNOLOGY
        </Text>

        <View style={styles.card}>
          <TechnologyRow
            name="Mobile"
            value="Expo React Native + TypeScript"
          />

          <Divider />

          <TechnologyRow
            name="Navigation"
            value="Expo Router"
          />

          <Divider />

          <TechnologyRow
            name="Computer Vision"
            value="YOLOv8"
          />

          <Divider />

          <TechnologyRow
            name="Depth Estimation"
            value="MiDaS"
          />

          <Divider />

          <TechnologyRow
            name="Backend"
            value="FastAPI + Python"
          />

          <Divider />

          <TechnologyRow
            name="Voice"
            value="Expo Speech"
          />
        </View>

        {/* Features */}
        <Text style={styles.sectionTitle}>
          KEY FEATURES
        </Text>

        <View style={styles.card}>
          <FeatureItem text="Real-time object detection" />
          <FeatureItem text="Depth and distance estimation" />
          <FeatureItem text="Obstacle risk assessment" />
          <FeatureItem text="Safety-level classification" />
          <FeatureItem text="Navigation direction guidance" />
          <FeatureItem text="Emergency stop detection" />
          <FeatureItem text="Voice navigation instructions" />
          <FeatureItem text="Voice-based object finder" />
          <FeatureItem text="Photo-based analysis" />
          <FeatureItem text="Video analysis support" />
        </View>

        {/* Safety */}
        <Text style={styles.sectionTitle}>
          SAFETY NOTICE
        </Text>

        <View style={styles.warningCard}>
          <Text style={styles.warningTitle}>
            IMPORTANT
          </Text>

          <Text style={styles.warningText}>
            Disha is an assistance and research
            application. Computer-vision predictions can be
            incorrect or delayed. Users should remain aware of
            their surroundings and must not rely exclusively on
            the application for personal safety or navigation.
          </Text>
        </View>

        {/* Project Architecture */}
        <Text style={styles.sectionTitle}>
          ARCHITECTURE
        </Text>

        <View style={styles.architectureCard}>
          <ArchitectureBlock
            title="MOBILE APP"
            items={[
              "Camera interface",
              "Live detection",
              "Object finder",
              "Navigation UI",
              "Voice guidance",
            ]}
          />

          <View style={styles.architectureArrow}>
            <Text style={styles.arrowText}>↕</Text>
          </View>

          <ArchitectureBlock
            title="FASTAPI BACKEND"
            items={[
              "Image processing",
              "YOLOv8 detection",
              "MiDaS depth",
              "Decision engine",
              "Navigation response",
            ]}
          />
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerTitle}>
            {APP_NAME}
          </Text>

          <Text style={styles.footerText}>
            Intelligent vision assistance through computer
            vision, depth estimation, and navigation reasoning.
          </Text>

          <Text style={styles.footerVersion}>
            v{APP_VERSION}
          </Text>
        </View>

        <View style={styles.bottomSpace} />
      </ScrollView>
    </View>
  );
}

/* -------------------------------------------------------------------------- */
/* Pipeline Component                                                         */
/* -------------------------------------------------------------------------- */

function PipelineItem({
  number,
  title,
  description,
}: {
  number: string;
  title: string;
  description: string;
}) {
  return (
    <View style={styles.pipelineItem}>
      <View style={styles.numberCircle}>
        <Text style={styles.numberText}>{number}</Text>
      </View>

      <View style={styles.pipelineText}>
        <Text style={styles.pipelineTitle}>
          {title}
        </Text>

        <Text style={styles.pipelineDescription}>
          {description}
        </Text>
      </View>
    </View>
  );
}

function PipelineLine() {
  return <View style={styles.pipelineLine} />;
}

/* -------------------------------------------------------------------------- */
/* Technology Component                                                       */
/* -------------------------------------------------------------------------- */

function TechnologyRow({
  name,
  value,
}: {
  name: string;
  value: string;
}) {
  return (
    <View style={styles.technologyRow}>
      <Text style={styles.technologyName}>
        {name}
      </Text>

      <Text style={styles.technologyValue}>
        {value}
      </Text>
    </View>
  );
}

function Divider() {
  return <View style={styles.divider} />;
}

/* -------------------------------------------------------------------------- */
/* Feature Component                                                          */
/* -------------------------------------------------------------------------- */

function FeatureItem({ text }: { text: string }) {
  return (
    <View style={styles.featureItem}>
      <View style={styles.featureDot} />

      <Text style={styles.featureText}>
        {text}
      </Text>
    </View>
  );
}

/* -------------------------------------------------------------------------- */
/* Architecture Component                                                     */
/* -------------------------------------------------------------------------- */

function ArchitectureBlock({
  title,
  items,
}: {
  title: string;
  items: string[];
}) {
  return (
    <View style={styles.architectureBlock}>
      <Text style={styles.architectureTitle}>
        {title}
      </Text>

      {items.map((item) => (
        <View
          key={item}
          style={styles.architectureItem}
        >
          <View style={styles.smallDot} />

          <Text style={styles.architectureItemText}>
            {item}
          </Text>
        </View>
      ))}
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

  identityCard: {
    backgroundColor: COLORS.panel,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 18,
    padding: 24,
    alignItems: "center",
    marginBottom: 10,
  },

  logo: {
    width: 82,
    height: 82,
    borderRadius: 22,
    backgroundColor: COLORS.primary,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 13,
  },

  logoText: {
    color: COLORS.white,
    fontSize: 25,
    fontWeight: "900",
    letterSpacing: 1,
  },

  appName: {
    color: COLORS.white,
    fontSize: 24,
    fontWeight: "900",
    textAlign: "center",
  },

  version: {
    color: palette.primaryLight,
    fontSize: 12,
    fontWeight: "700",
    marginTop: 5,
  },

  tagline: {
    color: palette.textSecondary,
    fontSize: 12,
    textAlign: "center",
    marginTop: 9,
  },

  sectionTitle: {
    color: palette.primaryLight,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 1.2,
    marginTop: 12,
    marginBottom: 9,
    marginLeft: 4,
  },

  card: {
    backgroundColor: COLORS.panel,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 16,
    marginBottom: 4,
  },

  cardText: {
    color: palette.textSecondary,
    fontSize: 12,
    lineHeight: 19,
  },

  paragraph: {
    marginTop: 13,
  },

  pipelineItem: {
    flexDirection: "row",
    alignItems: "center",
    minHeight: 52,
  },

  numberCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: COLORS.primary,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },

  numberText: {
    color: COLORS.white,
    fontSize: 10,
    fontWeight: "900",
  },

  pipelineText: {
    flex: 1,
  },

  pipelineTitle: {
    color: palette.textPrimary,
    fontSize: 13,
    fontWeight: "800",
  },

  pipelineDescription: {
    color: palette.textSecondary,
    fontSize: 11,
    marginTop: 3,
    lineHeight: 16,
  },

  pipelineLine: {
    width: 2,
    height: 14,
    backgroundColor: COLORS.border,
    marginLeft: 17,
  },

  technologyRow: {
    minHeight: 42,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },

  technologyName: {
    color: palette.textSecondary,
    fontSize: 12,
  },

  technologyValue: {
    color: palette.textPrimary,
    fontSize: 12,
    fontWeight: "700",
    maxWidth: "58%",
    textAlign: "right",
  },

  divider: {
    height: 1,
    backgroundColor: COLORS.border,
  },

  featureItem: {
    flexDirection: "row",
    alignItems: "center",
    minHeight: 36,
  },

  featureDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: palette.primaryLight,
    marginRight: 11,
  },

  featureText: {
    color: palette.textSecondary,
    fontSize: 12,
  },

  warningCard: {
    backgroundColor: COLORS.panel,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: COLORS.warning,
    padding: 16,
  },

  warningTitle: {
    color: COLORS.warning,
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 1,
    marginBottom: 7,
  },

  warningText: {
    color: palette.textSecondary,
    fontSize: 11,
    lineHeight: 18,
  },

  architectureCard: {
    backgroundColor: COLORS.panel,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 16,
  },

  architectureBlock: {
    backgroundColor: COLORS.panel2,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 14,
  },

  architectureTitle: {
    color: palette.primaryLight,
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 0.8,
    marginBottom: 10,
  },

  architectureItem: {
    flexDirection: "row",
    alignItems: "center",
    minHeight: 28,
  },

  smallDot: {
    width: 5,
    height: 5,
    borderRadius: 3,
    backgroundColor: COLORS.primary,
    marginRight: 9,
  },

  architectureItemText: {
    color: palette.textSecondary,
    fontSize: 11,
  },

  architectureArrow: {
    alignItems: "center",
    justifyContent: "center",
    height: 34,
  },

  arrowText: {
    color: palette.primaryLight,
    fontSize: 22,
    fontWeight: "800",
  },

  footer: {
    alignItems: "center",
    paddingHorizontal: 18,
    paddingTop: 28,
  },

  footerTitle: {
    color: COLORS.white,
    fontSize: 16,
    fontWeight: "900",
  },

  footerText: {
    color: COLORS.muted,
    fontSize: 10,
    lineHeight: 16,
    textAlign: "center",
    marginTop: 7,
  },

  footerVersion: {
    color: palette.primaryLight,
    fontSize: 10,
    fontWeight: "700",
    marginTop: 8,
  },

  bottomSpace: {
    height: 30,
  },
});