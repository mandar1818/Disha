import React from "react";
import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { SafeAreaProvider } from "react-native-safe-area-context";

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" />

      <Stack
        initialRouteName="camera"
        screenOptions={{
          headerShown: false,
          animation: "slide_from_right",
          contentStyle: {
            backgroundColor: "#07101C",
          },
        }}
      >
        <Stack.Screen name="camera" />
        <Stack.Screen name="index" />
        <Stack.Screen name="finder" />
        <Stack.Screen name="settings" />
        <Stack.Screen name="about" />
      </Stack>
    </SafeAreaProvider>
  );
}