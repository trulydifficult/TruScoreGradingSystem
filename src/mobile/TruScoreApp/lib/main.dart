// ignore_for_file: deprecated_member_use

import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart'; // Local Sizer package
import 'package:introduction_screen/introduction_screen.dart'; // Local Intro package
import 'package:shared_preferences/shared_preferences.dart'; // For persistent settings

import 'mobile_wrapper.dart';
import 'models/captured_image.dart';
import 'models/grading_result.dart';
import 'screens/camera_screen.dart';
import 'screens/analyzing_screen.dart';
import 'screens/results_screen.dart';
import 'screens/home_screen.dart';
import 'screens/market_analysis_screen.dart';
import 'screens/analysis_details_screen.dart';
import 'screens/settings_screen.dart';
import 'services/api_client.dart';

/// Enumeration describing the different screens within the app.
enum AppScreen { home, cameraFront, cameraBack, analyzing, results, market, analysisDetails, settings }

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  AppScreen _screen = AppScreen.home;
  final ApiClient _apiClient = ApiClient();
  CapturedImage? _frontImage;
  CapturedImage? _backImage;
  GradingResult? _gradingResult;
  String? _lastJobId;
  final _scaffoldMessengerKey = GlobalKey<ScaffoldMessengerState>();

  bool _showIntro = false;
  bool _initialized = false;

  @override
  void initState() {
    super.initState();
    _checkIntro();
  }

  Future<void> _checkIntro() async {
    final prefs = await SharedPreferences.getInstance();
    final seen = prefs.getBool('seen_intro') ?? false;
    setState(() {
      _showIntro = !seen;
      _initialized = true;
    });
  }

  Future<void> _onIntroDone() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('seen_intro', true);
    setState(() {
      _showIntro = false;
    });
  }

  /// Resets the submission and navigates back to the home screen.
  void _reset() {
    setState(() {
      _frontImage = null;
      _backImage = null;
      _gradingResult = null;
      _lastJobId = null;
      _screen = AppScreen.home;
    });
  }

  /// Handles a photo capture from the camera screen.
  void _onCapture(CapturedImage image, bool isFront) {
    setState(() {
      if (isFront) {
        _frontImage = image;
      } else {
        _backImage = image;
      }
      _screen = AppScreen.home;
    });
  }

  /// Starts the analysis routine.
  void _onAnalyze() {
    if (_frontImage == null || _backImage == null) {
      _onError('Please capture both sides before analyzing.');
      return;
    }
    setState(() {
      _screen = AppScreen.analyzing;
    });
  }

  void _onResult(GradingResult result, String jobId) {
    setState(() {
      _gradingResult = result;
      _lastJobId = jobId;
      _screen = AppScreen.results;
    });
  }

  void _onError(String message) {
    _scaffoldMessengerKey.currentState?.showSnackBar(
      SnackBar(content: Text(message)),
    );
    setState(() {
      _screen = AppScreen.home;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_initialized) {
      return const MaterialApp(home: Scaffold(backgroundColor: Colors.black));
    }

    return Sizer(
      builder: (context, orientation, deviceType) {
        return MaterialApp(
          title: 'Sports Card Grading',
          debugShowCheckedModeBanner: false,
          scaffoldMessengerKey: _scaffoldMessengerKey,
          theme: ThemeData(
            brightness: Brightness.dark,
            scaffoldBackgroundColor: Colors.black,
            fontFamily: 'Roboto',
            colorScheme: ColorScheme.fromSeed(seedColor: Colors.cyan).copyWith(
              brightness: Brightness.dark,
            ),
          ),
          builder: (context, child) {
            final baseStyle = DefaultTextStyle.of(context).style;
            return DefaultTextStyle(
              style: baseStyle.copyWith(decoration: TextDecoration.none),
              child: child ?? const SizedBox.shrink(),
            );
          },
          home: _showIntro ? _buildIntro() : _buildMainStack(),
        );
      },
    );
  }

  Widget _buildIntro() {
    return IntroductionScreen(
      pages: [
        PageViewModel(
          title: "Professional Grading",
          body: "AI-powered grading for your sports cards.",
          image: const Icon(Icons.camera_alt, size: 100, color: Colors.cyan),
          decoration: const PageDecoration(
            pageColor: Colors.black,
            bodyTextStyle: TextStyle(color: Colors.white70),
            titleTextStyle: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
          ),
        ),
        PageViewModel(
          title: "Instant Analysis",
          body: "Get instant feedback and market data.",
          image: const Icon(Icons.analytics, size: 100, color: Colors.purple),
          decoration: const PageDecoration(
            pageColor: Colors.black,
            bodyTextStyle: TextStyle(color: Colors.white70),
            titleTextStyle: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
          ),
        ),
      ],
      onDone: _onIntroDone,
      onSkip: _onIntroDone,
      showSkipButton: true,
      skip: const Text("Skip", style: TextStyle(color: Colors.white)),
      next: const Icon(Icons.arrow_forward, color: Colors.white),
      done: const Text("Done", style: TextStyle(fontWeight: FontWeight.w600, color: Colors.cyan)),
    );
  }

  Widget _buildMainStack() {
    return Stack(
      children: [
        // Background image
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
                    onSettings: () => setState(() => _screen = AppScreen.settings),
                    apiClient: _apiClient,
                    onError: _onError,
                  );
                case AppScreen.settings:
                  return Stack(
                    children: [
                      const SettingsScreen(),
                      Positioned(
                        top: 40,
                        left: 10,
                        child: IconButton(
                          icon: const Icon(Icons.arrow_back, color: Colors.white),
                          onPressed: () => setState(() => _screen = AppScreen.home),
                        ),
                      )
                    ],
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
                    front: _frontImage!,
                    back: _backImage,
                    apiClient: _apiClient,
                    lastJobId: _lastJobId,
                    onResult: _onResult,
                    onError: _onError,
                    onCancel: _reset,
                  );
                case AppScreen.results:
                  return ResultsScreen(
                    frontImage: _frontImage,
                    backImage: _backImage,
                    result: _gradingResult,
                    jobId: _lastJobId,
                    onHome: _reset,
                    onViewMarket: _lastJobId != null
                    ? () => setState(() => _screen = AppScreen.market)
                    : null,
                    onViewAnalysis: (_gradingResult?.visualizations.isNotEmpty ?? false) &&
                    _lastJobId != null
                    ? () => setState(() => _screen = AppScreen.analysisDetails)
                    : null,
                  );
                case AppScreen.market:
                  return MarketAnalysisScreen(
                    jobId: _lastJobId ?? '',
                    result: _gradingResult,
                    frontImage: _frontImage,
                    backImage: _backImage,
                    apiClient: _apiClient,
                    onBack: () => setState(() => _screen = AppScreen.results),
                    onHome: _reset,
                    onError: _onError,
                  );
                case AppScreen.analysisDetails:
                  return AnalysisDetailsScreen(
                    jobId: _lastJobId ?? '',
                    result: _gradingResult!,
                    onBack: () => setState(() => _screen = AppScreen.results),
                  );
              }
              // Fallback
              return const SizedBox.shrink();
            },
          ),
        ),
      ],
    );
  }
}
