import 'dart:async';

import 'package:flutter/material.dart';

/// Camera screen that allows the user to capture either the front or back of a
/// card. A simple framing guide is drawn on top of a dark placeholder area
/// because we do not have access to the actual device camera in this
/// environment. When the capture button is pressed the app flashes and a
/// sample image is returned via the [onCapture] callback.
class CameraScreen extends StatefulWidget {
  final String side; // 'front' or 'back'
  final ValueChanged<String> onCapture;
  final VoidCallback onBack;

  const CameraScreen({
    super.key,
    required this.side,
    required this.onCapture,
    required this.onBack,
  });

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  bool _isFlashing = false;

  static const String _sampleFront =
      'https://images.unsplash.com/photo-1727157540259-51c72b9ac315?auto=format&fit=max&w=1080';
  static const String _sampleBack =
      'https://images.unsplash.com/photo-1600196024905-e0cd65ddc6f1?auto=format&fit=max&w=1080';

  /// Simulates taking a photo by briefly showing a white overlay and then
  /// calling the onCapture callback with a sample image URL.
  void _handleCapture() {
    setState(() {
      _isFlashing = true;
    });
    Timer(const Duration(milliseconds: 300), () {
      setState(() {
        _isFlashing = false;
      });
      widget.onCapture(widget.side == 'front' ? _sampleFront : _sampleBack);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        bottom: false,
        child: Stack(
          children: [
            Column(
              children: [
                // Top bar with back button and side indicator
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      IconButton(
                        onPressed: widget.onBack,
                        icon: const Icon(Icons.close),
                        color: Colors.white,
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.6),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: Colors.white.withOpacity(0.1),
                          ),
                        ),
                        child: Text(
                          '${widget.side[0].toUpperCase()}${widget.side.substring(1)} of Card',
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                            letterSpacing: 1.0,
                          ),
                        ),
                      ),
                      IconButton(
                        onPressed: () {},
                        icon: const Icon(Icons.flash_on),
                        color: Colors.amber,
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: Stack(
                    children: [
                      // Placeholder for camera feed
                      Container(
                        color: Colors.black,
                        alignment: Alignment.center,
                        child: Opacity(
                          opacity: 0.3,
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: const [
                              Text(
                                'Live Feed',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                  letterSpacing: 1.5,
                                ),
                              ),
                              SizedBox(height: 4),
                              Icon(
                                Icons.fiber_manual_record,
                                color: Colors.red,
                                size: 8,
                              ),
                            ],
                          ),
                        ),
                      ),
                      // Framing guide: a 5:7 aspect ratio box with corner markers and grid lines
                      Center(
                        child: AspectRatio(
                          aspectRatio: 5 / 7,
                          child: Container(
                            decoration: BoxDecoration(
                              border: Border.all(
                                color: Colors.white.withOpacity(0.3),
                              ),
                              borderRadius: BorderRadius.circular(12),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.85),
                                  spreadRadius: 1000,
                                  blurRadius: 0,
                                ),
                              ],
                            ),
                            child: Stack(
                              children: [
                                // Corner guides
                                Positioned(
                                  top: 0,
                                  left: 0,
                                  child: _cornerGuide(top: true, left: true),
                                ),
                                Positioned(
                                  top: 0,
                                  right: 0,
                                  child: _cornerGuide(top: true, left: false),
                                ),
                                Positioned(
                                  bottom: 0,
                                  left: 0,
                                  child: _cornerGuide(top: false, left: true),
                                ),
                                Positioned(
                                  bottom: 0,
                                  right: 0,
                                  child: _cornerGuide(top: false, left: false),
                                ),
                                // Center alignment text
                                Center(
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 12, vertical: 6),
                                    decoration: BoxDecoration(
                                      color: Colors.black.withOpacity(0.6),
                                      borderRadius: BorderRadius.circular(20),
                                      border: Border.all(
                                        color: Colors.white.withOpacity(0.1),
                                      ),
                                    ),
                                    child: Text(
                                      'Align ${widget.side}',
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontSize: 10,
                                        letterSpacing: 1.5,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  ),
                                ),
                                // Grid lines
                                Positioned.fill(
                                  child: IgnorePointer(
                                    child: Opacity(
                                      opacity: 0.15,
                                      child: Column(
                                        children: [
                                          Expanded(
                                            child: Container(
                                              decoration: const BoxDecoration(
                                                border: Border(
                                                  bottom: BorderSide(
                                                    color: Colors.white,
                                                    width: 1,
                                                  ),
                                                ),
                                              ),
                                            ),
                                          ),
                                          Expanded(
                                            child: Container(
                                              decoration: const BoxDecoration(
                                                border: Border(
                                                  bottom: BorderSide(
                                                    color: Colors.white,
                                                    width: 1,
                                                  ),
                                                ),
                                              ),
                                            ),
                                          ),
                                          Expanded(
                                            child: Container(
                                              decoration: const BoxDecoration(
                                                border: Border(
                                                  bottom: BorderSide(
                                                    color: Colors.white,
                                                    width: 1,
                                                  ),
                                                ),
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                ),
                                Positioned.fill(
                                  child: IgnorePointer(
                                    child: Opacity(
                                      opacity: 0.15,
                                      child: Row(
                                        children: [
                                          Expanded(
                                            child: Container(
                                              decoration: const BoxDecoration(
                                                border: Border(
                                                  right: BorderSide(
                                                    color: Colors.white,
                                                    width: 1,
                                                  ),
                                                ),
                                              ),
                                            ),
                                          ),
                                          Expanded(
                                            child: Container(
                                              decoration: const BoxDecoration(
                                                border: Border(
                                                  right: BorderSide(
                                                    color: Colors.white,
                                                    width: 1,
                                                  ),
                                                ),
                                              ),
                                            ),
                                          ),
                                          Expanded(
                                            child: Container(
                                              decoration: const BoxDecoration(
                                                border: Border(
                                                  right: BorderSide(
                                                    color: Colors.white,
                                                    width: 1,
                                                  ),
                                                ),
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                ),
                                // Dimension labels
                                Positioned(
                                  left: -32,
                                  top: 0,
                                  bottom: 0,
                                  child: Center(
                                    child: Transform.rotate(
                                      angle: -1.5708, // -90 degrees in radians
                                      child: const Text(
                                        '3.5"',
                                        style: TextStyle(
                                          color: Color(0xFF06B6D4),
                                          fontSize: 9,
                                          letterSpacing: 1.5,
                                          fontFamily: 'monospace',
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                                Positioned(
                                  bottom: -24,
                                  left: 0,
                                  right: 0,
                                  child: Center(
                                    child: const Text(
                                      '2.5"',
                                      style: TextStyle(
                                        color: Color(0xFF06B6D4),
                                        fontSize: 9,
                                        letterSpacing: 1.5,
                                        fontFamily: 'monospace',
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                // Bottom controls: preview placeholder, shutter button, rotate
                Container(
                  height: 120,
                  padding: const EdgeInsets.only(bottom: 24),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Preview placeholder
                      IconButton(
                        onPressed: () {},
                        iconSize: 32,
                        icon: const Icon(Icons.photo_library),
                        color: Colors.white24,
                      ),
                      const SizedBox(width: 40),
                      // Shutter button
                      GestureDetector(
                        onTap: _handleCapture,
                        child: Container(
                          width: 80,
                          height: 80,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(
                                color: Colors.white.withOpacity(0.2), width: 2),
                          ),
                          alignment: Alignment.center,
                          child: Container(
                            width: 64,
                            height: 64,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 40),
                      // Rotate camera button (non-functional in this mockup)
                      IconButton(
                        onPressed: () {},
                        iconSize: 32,
                        icon: const Icon(Icons.refresh),
                        color: Colors.white24,
                      ),
                    ],
                  ),
                ),
              ],
            ),
            // Flash overlay
            if (_isFlashing)
              Positioned.fill(
                child: Container(
                  color: Colors.white,
                ),
              ),
          ],
        ),
      ),
    );
  }

  /// Draws a corner guide used in the viewfinder. The `top` and `left`
  /// parameters determine which corners get a border.
  Widget _cornerGuide({required bool top, required bool left}) {
    return SizedBox(
      width: 32,
      height: 32,
      child: CustomPaint(
        painter: _CornerPainter(top: top, left: left),
      ),
    );
  }
}

/// A custom painter that draws L‑shaped corners for the viewfinder. The
/// painter only draws the top/left or bottom/right borders depending on the
/// booleans provided. This is a small detail from the original design that
/// helps orient the card within the frame.
class _CornerPainter extends CustomPainter {
  final bool top;
  final bool left;

  _CornerPainter({required this.top, required this.left});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF06B6D4)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final path = Path();
    if (top) {
      path.moveTo(0, 0);
      path.lineTo(size.width, 0);
    } else {
      path.moveTo(0, size.height);
      path.lineTo(size.width, size.height);
    }
    if (left) {
      path.moveTo(0, 0);
      path.lineTo(0, size.height);
    } else {
      path.moveTo(size.width, 0);
      path.lineTo(size.width, size.height);
    }
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _CornerPainter oldDelegate) {
    return oldDelegate.top != top || oldDelegate.left != left;
  }
}