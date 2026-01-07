import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

/// The analyzing screen displays a progress indicator while the system
/// "processes" the scanned images. It rotates two rings in opposite
/// directions and updates a text description based on the current step.
class AnalyzingScreen extends StatefulWidget {
  final VoidCallback onComplete;

  const AnalyzingScreen({super.key, required this.onComplete});

  @override
  State<AnalyzingScreen> createState() => _AnalyzingScreenState();
}

class _AnalyzingScreenState extends State<AnalyzingScreen>
    with TickerProviderStateMixin {
  late final AnimationController _outerController;
  late final AnimationController _innerController;
  double _progress = 0;
  int _stepIndex = 0;
  late Timer _timer;

  final List<_AnalysisStep> _steps = [
    _AnalysisStep('Scanning surface topography...', Icons.layers),
    _AnalysisStep('Analyzing centering geometry...', Icons.center_focus_strong),
    _AnalysisStep('Detecting edge imperfections...', Icons.search),
    _AnalysisStep('Querying global market database...', Icons.cloud_download),
    _AnalysisStep('Calculating final TruScore...', Icons.computer),
  ];

  @override
  void initState() {
    super.initState();
    _outerController =
        AnimationController(vsync: this, duration: const Duration(seconds: 8))
          ..repeat();
    _innerController =
        AnimationController(vsync: this, duration: const Duration(seconds: 4))
          ..repeat(reverse: false);
    _timer = Timer.periodic(const Duration(milliseconds: 50), (timer) {
      setState(() {
        _progress = math.min(_progress + 1.5, 100);
        final newIndex =
            (_progress / 100 * _steps.length).floor().clamp(0, _steps.length - 1);
        _stepIndex = newIndex;
      });
      if (_progress >= 100) {
        _timer.cancel();
        Future.delayed(const Duration(milliseconds: 800), widget.onComplete);
      }
    });
  }

  @override
  void dispose() {
    _timer.cancel();
    _outerController.dispose();
    _innerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final _AnalysisStep current = _steps[_stepIndex];
    return Scaffold(
      backgroundColor: Colors.blueGrey.shade900,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Rotating ring indicator
              SizedBox(
                width: 160,
                height: 160,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    // Outer ring
                    RotationTransition(
                      turns: _outerController,
                      child: CustomPaint(
                        size: const Size(160, 160),
                        painter: _RingPainter(
                          color: const Color(0xFF06B6D4),
                          strokeWidth: 4,
                          startAngle: 0,
                          sweepAngle: math.pi * 2,
                        ),
                      ),
                    ),
                    // Inner ring rotates opposite direction
                    RotationTransition(
                      turns: Tween<double>(begin: 1, end: 0).animate(_innerController),
                      child: CustomPaint(
                        size: const Size(120, 120),
                        painter: _RingPainter(
                          color: const Color(0xFFA855F7),
                          strokeWidth: 4,
                          startAngle: 0,
                          sweepAngle: math.pi * 2,
                        ),
                      ),
                    ),
                    // Center glowing core
                    Container(
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient: RadialGradient(
                          colors: [
                            const Color(0x3306B6D4),
                            Colors.transparent,
                          ],
                          stops: const [0.0, 1.0],
                        ),
                      ),
                    ),
                    // Current step icon
                    Icon(
                      current.icon,
                      size: 40,
                      color: Colors.white,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 32),
              const Text(
                'System Processing',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                height: 20,
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 300),
                  transitionBuilder: (child, anim) {
                    return FadeTransition(
                      opacity: anim,
                      child: SlideTransition(
                        position: Tween<Offset>(
                          begin: const Offset(0, 0.3),
                          end: Offset.zero,
                        ).animate(anim),
                        child: child,
                      ),
                    );
                  },
                  child: Text(
                    current.label,
                    key: ValueKey<String>(current.label),
                    style: const TextStyle(
                      color: Color(0xFF06B6D4),
                      fontSize: 14,
                      fontFamily: 'monospace',
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 32),
              // Linear progress bar
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: _progress / 100,
                  minHeight: 6,
                  backgroundColor: Colors.white.withOpacity(0.05),
                  valueColor: AlwaysStoppedAnimation(
                    const Color(0xFF06B6D4),
                  ),
                ),
              ),
              const SizedBox(height: 4),
              Align(
                alignment: Alignment.centerRight,
                child: Text(
                  '${_progress.toInt()}% COMPLETE',
                  style: const TextStyle(
                    fontSize: 10,
                    color: Colors.white54,
                    fontFamily: 'monospace',
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Simple container for the descriptive text and icon used in the analysis
/// steps.
class _AnalysisStep {
  final String label;
  final IconData icon;
  _AnalysisStep(this.label, this.icon);
}

/// Custom painter used to draw a full circular ring. The React version used
/// CSS borders with custom colours on specific quadrants – here we keep it
/// simple and draw a full circle in one colour.
class _RingPainter extends CustomPainter {
  final Color color;
  final double strokeWidth;
  final double startAngle;
  final double sweepAngle;

  _RingPainter({
    required this.color,
    required this.strokeWidth,
    required this.startAngle,
    required this.sweepAngle,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(
      rect.deflate(strokeWidth / 2),
      startAngle,
      sweepAngle,
      false,
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant _RingPainter oldDelegate) {
    return oldDelegate.color != color ||
        oldDelegate.strokeWidth != strokeWidth ||
        oldDelegate.startAngle != startAngle ||
        oldDelegate.sweepAngle != sweepAngle;
  }
}