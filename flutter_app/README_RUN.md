# Flutter App - Run Instructions

This is a minimal Flutter frontend for the Personalized Anemia Detection system.

Quick start:

1. Ensure Flutter SDK is installed and available on PATH.
2. From this folder run:

   flutter pub get
   flutter run

Notes:
- The app posts multipart/form-data to a backend endpoint defined in `lib/services/api_service.dart`.
- By default the endpoint is `http://10.0.2.2:5000/predict` which routes to localhost on Android emulators. Change to your server address (ngrok/local network) if needed.
- Permissions: the app uses `image_picker` which needs camera and storage permissions configured for Android/iOS when building for device.

Files added:
- `lib/main.dart` - app entry + green theme
- `lib/screens/` - home, personalization form, camera, result screens
- `lib/services/api_service.dart` - simple HTTP upload service
- `lib/widgets/result_card.dart` - displays returned prediction
