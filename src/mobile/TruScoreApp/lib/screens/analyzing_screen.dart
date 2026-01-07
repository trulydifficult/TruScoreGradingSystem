// ignore_for_file: deprecated_member_use

import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/captured_image.dart';
import '../models/grading_result.dart';
import '../services/api_client.dart';

/// The analyzing screen now performs the real upload + polling against the
/// FastAPI bridge. It keeps the animated rings from the prototype but ties
/// progress to network activity.
class AnalyzingScreen extends StatefulWidget {
  final CapturedImage front;
  final CapturedImage? back;
  final ApiClient apiClient;
  final String? lastJobId;
  final void Function(GradingResult, String jobId) onResult;
  final void Function(String message) onError;
  final VoidCallback onCancel;

  const AnalyzingScreen({
    super.key,
    required this.front,
    required this.back,
    required this.apiClient,
    required this.onResult,
    required this.onError,
    required this.onCancel,
    this.lastJobId,
  });

  @override
  State<AnalyzingScreen> createState() => _AnalyzingScreenState();
}

class _AnalyzingScreenState extends State<AnalyzingScreen>
    with TickerProviderStateMixin {
  late final AnimationController _outerController;
  late final AnimationController _innerController;
  double _progress = 0.08;
  int _stepIndex = 0;
  String _status = 'Uploading photos...';
  bool _cancelled = false;

  final List<_AnalysisStep> _steps = const [
    _AnalysisStep('Uploading images securely...', Icons.cloud_upload_outlined),
    _AnalysisStep('Queued with TruScore engine...', Icons.schedule),
    _AnalysisStep('Analyzing surface + centering...', Icons.graphic_eq),
    _AnalysisStep('Crunching subgrades...', Icons.auto_graph),
    _AnalysisStep('Finalizing TruScore...', Icons.star_rate_rounded),
  ];

  @override
  void initState() {
    super.initState();
    _outerController =
        AnimationController(vsync: this, duration: const Duration(seconds: 8))
          ..repeat();
    _innerController =
        AnimationController(vsync: this, duration: const Duration(seconds: 4))
          ..repeat();
    _startJob();
  }

  Future<void> _startJob() async {
    if (widget.lastJobId != null) {
      await _poll(widget.lastJobId!);
      return;
    }
    try {
      final frontFile = File(widget.front.path);
      final backFile = widget.back != null ? File(widget.back!.path) : null;
      final jobId = await widget.apiClient.submitForGrading(
        front: frontFile,
        back: backFile,
        metadata: {
          'source': 'flutter-mobile',
          'front_from_gallery': widget.front.fromGallery,
          'back_from_gallery': widget.back?.fromGallery ?? false,
        },
      );
      if (!mounted) return;
      setState(() {
        _status = 'Waiting for analysis...';
        _progress = 0.3;
        _stepIndex = 1;
      });
      await _poll(jobId);
    } catch (e) {
      widget.onError('Upload failed: $e');
    }
  }

  Future<void> _poll(String jobId) async {
    while (mounted && !_cancelled) {
      try {
        final status = await widget.apiClient.fetchJob(jobId);
        if (status.isCompleted && status.result != null) {
          setState(() {
            _status = 'Analysis complete';
            _progress = 1.0;
            _stepIndex = _steps.length - 1;
          });
          await Future.delayed(const Duration(milliseconds: 400));
          widget.onResult(status.result!, jobId);
          return;
        } else if (status.isFailed) {
          widget.onError(status.error ?? 'Analysis failed');
          return;
        } else {
          setState(() {
            _status =
                status.status == 'running' ? 'Analyzing card...' : 'Queued...';
            _progress = math.min(_progress + 0.08, 0.88);
            _stepIndex = (_stepIndex + 1).clamp(0, _steps.length - 1);
          });
        }
      } catch (e) {
        widget.onError('Lost connection to grading service: $e');
        return;
      }
      await Future.delayed(const Duration(seconds: 1));
    }
  }

  @override
  void dispose() {
    _cancelled = true;
    _outerController.dispose();
    _innerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final _AnalysisStep current = _steps[_stepIndex];
    return Scaffold(
      backgroundColor: const Color(0xFF0B1224),
      body: Stack(
        children: [
          Positioned.fill(
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Colors.cyan.withOpacity(0.12),
                    Colors.purple.withOpacity(0.1),
                    Colors.black,
                  ],
                ),
              ),
            ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
              child: Column(
                children: [
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton.icon(
                      onPressed: () {
                        _cancelled = true;
                        widget.onCancel();
                      },
                      icon: const Icon(Icons.close, color: Colors.white70),
                      label: const Text(
                        'Cancel',
                        style: TextStyle(color: Colors.white70),
                      ),
                    ),
                  ),
                  const Spacer(),
                  // Rotating ring indicator
                  SizedBox(
                    width: 190,
                    height: 190,
                    child: Stack(
                      alignment: Alignment.center,
                      children: [
                        // Outer ring
                        RotationTransition(
                          turns: _outerController,
                          child: CustomPaint(
                            size: const Size(190, 190),
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
                          turns:
                              Tween<double>(begin: 1, end: 0).animate(_innerController),
                          child: CustomPaint(
                            size: const Size(140, 140),
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
                          width: 90,
                          height: 90,
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
                          size: 44,
                          color: Colors.white,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 32),
                  const Text(
                    'Analyzing Your Card',
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
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: _progress,
                      minHeight: 10,
                      backgroundColor: Colors.white.withOpacity(0.06),
                      valueColor: const AlwaysStoppedAnimation(
                        Color(0xFF06B6D4),
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  Align(
                    alignment: Alignment.centerRight,
                    child: Text(
                      '${(_progress * 100).toInt()}% • $_status',
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.white70,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ),
                  const Spacer(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Simple container for the descriptive text and icon used in the analysis
/// steps.
class _AnalysisStep {
  final String label;
  final IconData icon;
  const _AnalysisStep(this.label, this.icon);
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
