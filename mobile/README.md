# SmartVisionAI Mobile

Complete Expo/React Native client for the existing SmartVisionAI FastAPI backend.

Expected endpoints:
- GET /health
- GET /system/status
- POST /detect (multipart field: file)

The client supports the FrameProcessor response fields used by the backend, including objects, safety_level, emergency_stop, navigation, voice_instruction, primary_obstacle and processing_time_ms.

## Network
Edit `.env` when the PC Wi-Fi address changes:
`EXPO_PUBLIC_BACKEND_URL=http://YOUR_PC_WIFI_IP:8000`

Current setup uses `172.20.10.2`.

## Install
`npm install`

## Android development build
Speech recognition is a native module, so ordinary Expo Go is not sufficient for Finder. Use:

`npx expo prebuild --clean`
`npx expo run:android`

After the first native build, normal JS/TS development uses `npx expo start`.

## iOS
Local iOS compilation requires macOS/Xcode. On Windows, use an EAS development build for iOS.

## Typecheck
`npm run typecheck`
