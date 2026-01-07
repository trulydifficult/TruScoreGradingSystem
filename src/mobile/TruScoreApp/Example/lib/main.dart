import 'package:flutter/material.dart';
import 'mobile_wrapper.dart';
import 'screens/home_screen.dart';
import 'screens/camera_screen.dart';
import 'screens/analyzing_screen.dart';
import 'screens/results_screen.dart';

/// Enumeration describing the different screens within the app. We mirror the
/// React implementation where the app is a simple state machine that flips
/// between home, camera (front/back), analyzing, and results.
enum AppScreen { home, cameraFront, cameraBack, analyzing, results }

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  AppScreen _screen = AppScreen.home;
  String? _frontImage;
  String? _backImage;

  /// Resets the submission and navigates back to the home screen.
  void _reset() {
    setState(() {
      _frontImage = null;
      _backImage = null;
      _screen = AppScreen.home;
    });
  }

  /// Handles a photo capture from the camera screen. Depending on whether
  /// `isFront` is true or false we assign the image to the corresponding
  /// property. After capture we return to the home screen so the user can
  /// proceed or take the next picture.
  void _onCapture(String image, bool isFront) {
    setState(() {
      if (isFront) {
        _frontImage = image;
      } else {
        _backImage = image;
      }
      _screen = AppScreen.home;
    });
  }

  /// Starts the analysis routine. In the current prototype this simply
  /// navigates to the analyzing screen; the analyzing screen itself will
  /// transition to the results screen when its internal timer completes.
  void _onAnalyze() {
    setState(() {
      _screen = AppScreen.analyzing;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sports Card Grading',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: Colors.black,
        fontFamily: 'Roboto',
        // Setting primary swatch doesn't hurt; helps with built‑in components.
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.cyan).copyWith(
          brightness: Brightness.dark,
        ),
      ),
      home: Stack(
        children: [
          // Background image with overlay to darken it. We use the same
          // Unsplash image from the original app and blend it with a dark
          // color to keep the focus on the foreground content.
          Positioned.fill(
            child: Image.network(
              'https://images.unsplash.com/photo-1534068590799-09895a701e3e?auto=format&fit=crop&w=1280',
              fit: BoxFit.cover,
              color: Colors.black.withOpacity(0.5),
              colorBlendMode: BlendMode.darken,
            ),
          ),
          // The mobile frame containing our screens.
          MobileWrapper(
            child: Builder(
              builder: (context) {
                switch (_screen) {
                  case AppScreen.home:
                    return HomeScreen(
                      frontImage: _frontImage,
                      backImage: _backImage,
                      onScanFront: () => setState(() => _screen = AppScreen.cameraFront),
                      onScanBack: () => setState(() => _screen = AppScreen.cameraBack),
                      onAnalyze: _onAnalyze,
                    );
                  case AppScreen.cameraFront:
                    return CameraScreen(
                      side: 'front',
                      onCapture: (img) => _onCapture(img, true),
                      onBack: () => setState(() => _screen = AppScreen.home),
                    );
                  case AppScreen.cameraBack:
                    return CameraScreen(
                      side: 'back',
                      onCapture: (img) => _onCapture(img, false),
                      onBack: () => setState(() => _screen = AppScreen.home),
                    );
                  case AppScreen.analyzing:
                    return AnalyzingScreen(
                      onComplete: () => setState(() => _screen = AppScreen.results),
                    );
                  case AppScreen.results:
                    return ResultsScreen(
                      frontImage: _frontImage,
                      onHome: _reset,
                    );
                }
              },
            ),
          ),
        ],
      ),
    );
  }
}