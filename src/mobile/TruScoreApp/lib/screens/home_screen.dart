// ignore_for_file: deprecated_member_use

import 'dart:ui';

import 'package:flutter/material.dart';
import '../models/captured_image.dart';
import '../models/recent_scan.dart';
import '../services/api_client.dart';

/// The home screen is the landing page of the app. It shows a greeting,
/// allows the user to scan the front and back of a card, and displays a
/// list of recent scans. Once both sides are captured the "Analyze" button
/// becomes active.
class HomeScreen extends StatefulWidget {
  final CapturedImage? frontImage;
  final CapturedImage? backImage;
  final VoidCallback onScanFront;
  final VoidCallback onScanBack;
  final VoidCallback onAnalyze;
  final VoidCallback? onSettings; // Add callback
  final ApiClient apiClient;
  final void Function(String message)? onError;

  const HomeScreen({
    super.key,
    required this.frontImage,
    required this.backImage,
    required this.onScanFront,
    required this.onScanBack,
    required this.onAnalyze,
    required this.apiClient,
    this.onSettings, // Add to constructor
    this.onError,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _loading = true;
  List<RecentScan> _recentScans = [];
  String? _recentsError;

  @override
  void initState() {
    super.initState();
    _loadRecents();
  }

  bool get _readyToAnalyze =>
      widget.frontImage != null && widget.backImage != null;

  Future<void> _loadRecents() async {
    setState(() {
      _loading = true;
    });
    try {
      final items = await widget.apiClient.fetchRecentScans(limit: 10);
      if (!mounted) return;
      setState(() {
        _recentScans = items;
        _loading = false;
        _recentsError = null;
      });
    } catch (e) {
      // Keep silent in UI; show a small hint instead of full error pop-up.
      if (!mounted) return;
      setState(() {
        _loading = false;
        _recentsError = '$e';
      });
    }
  }

  String _timeAgo(DateTime? date) {
    if (date == null) return '—';
    final diff = DateTime.now().difference(date);
    if (diff.inMinutes < 1) return 'just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    final weeks = (diff.inDays / 7).floor();
    return '${weeks}w ago';
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header section
        _glassSurface(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: const [
                  Text(
                    'Welcome Back',
                    style: TextStyle(
                      color: Color(0xFF06B6D4),
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 1.4,
                    ),
                  ),
                  SizedBox(height: 6),
                  Text.rich(
                    TextSpan(
                      text: 'TruScore ',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                      children: [
                        TextSpan(
                          text: 'System',
                          style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.w300,
                            color: Colors.white54,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              // Profile icon button replaced with Settings
              IconButton(
                onPressed: widget.onSettings ?? () {},
                icon: const Icon(Icons.settings),
                color: Colors.white,
                tooltip: 'Settings',
              ),
            ],
          ),
        ),
        // Scrollable content area
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // New Submission label
                Row(
                  children: const [
                    Icon(
                      Icons.fiber_manual_record,
                      size: 8,
                      color: Color(0xFF06B6D4),
                    ),
                    SizedBox(width: 6),
                    Text(
                      'New Submission',
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1.0,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                // Scan buttons for front and back
                Row(
                  children: [
                    Expanded(
                      child: _scanButton(
                        label: 'Scan Front',
                        side: 'front',
                        image: widget.frontImage,
                        onPressed: widget.onScanFront,
                        color: Colors.cyan,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: _scanButton(
                        label: 'Scan Back',
                        side: 'back',
                        image: widget.backImage,
                        onPressed: widget.onScanBack,
                        color: Colors.purple,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                // Analyze button appears only after at least one scan
                if (widget.frontImage != null || widget.backImage != null)
                  _analyzeButton(),
                const SizedBox(height: 32),
                // Recent scans header
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: const [
                        Icon(Icons.history, size: 16, color: Colors.white54),
                        SizedBox(width: 8),
                        Text(
                          'Recent Scans',
                          style: TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            letterSpacing: 1.0,
                          ),
                        ),
                      ],
                    ),
                    TextButton(
                      onPressed: _loadRecents,
                      child: const Text(
                        'Refresh',
                        style: TextStyle(
                          color: Color(0xFF06B6D4),
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                // Recent scans list
                Column(
                  children: _loading
                      ? [1, 2]
                          .map(
                            (i) => Container(
                              margin: const EdgeInsets.symmetric(vertical: 6),
                              height: 80,
                              decoration: BoxDecoration(
                                color: Colors.white10,
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                          )
                          .toList()
                      : _recentScans.isEmpty
                          ? [
                              Padding(
                                padding:
                                    const EdgeInsets.symmetric(vertical: 12),
                                child: Text(
                                  'No recent scans yet. Submit one to see it here.',
                                  style: TextStyle(
                                    color: Colors.white.withOpacity(0.7),
                                    fontSize: 12,
                                  ),
                                ),
                              ),
                              if (_recentsError != null)
                                Text(
                                  'Recents offline: $_recentsError',
                                  style: const TextStyle(
                                    color: Colors.amberAccent,
                                    fontSize: 11,
                                  ),
                                  textAlign: TextAlign.center,
                                ),
                            ]
                          : _recentScans.map((item) {
                              final imageUrl =
                                  item.thumbnailFront ?? item.thumbnailBack;
                              final grade = item.displayGrade;
                              final isPending = grade.toLowerCase() == 'pending';
                              return Container(
                                margin: const EdgeInsets.symmetric(vertical: 6),
                                padding: const EdgeInsets.all(12),
                                decoration: BoxDecoration(
                                  color: Colors.white10,
                                  borderRadius: BorderRadius.circular(12),
                                  border: Border.all(
                                    color: Colors.white.withOpacity(0.1),
                                  ),
                                ),
                                child: Row(
                                  children: [
                                    ClipRRect(
                                      borderRadius: BorderRadius.circular(8),
                                      child: imageUrl != null && imageUrl.isNotEmpty
                                          ? Image.network(
                                              imageUrl,
                                              width: 48,
                                              height: 48,
                                              fit: BoxFit.cover,
                                            )
                                          : Container(
                                              width: 48,
                                              height: 48,
                                              color: Colors.white.withOpacity(0.05),
                                              child: const Icon(
                                                Icons.photo,
                                                color: Colors.white54,
                                              ),
                                            ),
                                    ),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            item.title,
                                            style: const TextStyle(
                                              color: Colors.white,
                                              fontSize: 14,
                                              fontWeight: FontWeight.w600,
                                            ),
                                          ),
                                          const SizedBox(height: 2),
                                          Text(
                                            _timeAgo(item.completedAt ?? item.submittedAt),
                                            style: const TextStyle(
                                              color: Colors.white54,
                                              fontSize: 12,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    // Badge for grade
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 8, vertical: 4),
                                      decoration: BoxDecoration(
                                        color: isPending
                                            ? Colors.amber.withOpacity(0.1)
                                            : Colors.green.withOpacity(0.1),
                                        border: Border.all(
                                          color: isPending
                                              ? Colors.amber.withOpacity(0.2)
                                              : Colors.green.withOpacity(0.2),
                                        ),
                                        borderRadius: BorderRadius.circular(6),
                                      ),
                                      child: Text(
                                        grade,
                                        style: TextStyle(
                                          color: isPending
                                              ? Colors.amber
                                              : Colors.green,
                                          fontSize: 12,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            }).toList(),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Text(
            'API: ${widget.apiClient.activeBase}',
            style: const TextStyle(
              color: Colors.white38,
              fontSize: 11,
            ),
          ),
        ),
      ],
    );
  }

  /// Glassmorphism helper used for headers and panels.
  Widget _glassSurface({
    required Widget child,
    EdgeInsets? padding,
    BorderRadius? radius,
  }) {
    final borderRadius = radius ?? BorderRadius.circular(24);
    return ClipRRect(
      borderRadius: borderRadius,
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 16, sigmaY: 16),
        child: Container(
          padding: padding,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.04),
            borderRadius: borderRadius,
            border: Border.all(color: Colors.white.withOpacity(0.08)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.3),
                blurRadius: 20,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: child,
        ),
      ),
    );
  }

  /// Builds the scan button for the front and back scans. The button will
  /// display an image preview once the corresponding side has been scanned.
  Widget _scanButton({
    required String label,
    required String side,
    required CapturedImage? image,
    required VoidCallback onPressed,
    required MaterialColor color,
  }) {
    final bool hasImage = image != null;
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        height: 200,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: hasImage
                ? Colors.green.withOpacity(0.4)
                : Colors.white.withOpacity(0.1),
            width: hasImage ? 2 : 1,
          ),
          color: hasImage
              ? Colors.green.withOpacity(0.15)
              : Colors.white.withOpacity(0.05),
        ),
        clipBehavior: Clip.antiAlias,
        child: Stack(
          alignment: Alignment.center,
          children: [
            if (hasImage)
              Positioned.fill(
                child: Image(
                  image: image.provider,
                  fit: BoxFit.cover,
                ),
              ),
            if (hasImage)
              Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [
                      Colors.black.withOpacity(0.4),
                      Colors.transparent,
                    ],
                  ),
                ),
              ),
            if (hasImage)
              Align(
                alignment: Alignment.bottomCenter,
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.6),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: Colors.white.withOpacity(0.08),
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: const [
                        Icon(
                          Icons.check_circle,
                          color: Colors.greenAccent,
                          size: 20,
                        ),
                        SizedBox(width: 6),
                        Text(
                          'Ready',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w600,
                            fontSize: 13,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            if (!hasImage)
              Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                        color: color.withOpacity(0.2),
                      ),
                    ),
                    child: Icon(
                      Icons.photo_camera,
                      color: color,
                      size: 24,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    label,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Required',
                    style: TextStyle(
                      color: Colors.white54,
                      fontSize: 10,
                      letterSpacing: 1.2,
                    ),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }

  /// Builds the analyze button. It becomes enabled only when both the front
  /// and back images have been scanned. The button changes its appearance
  /// accordingly.
  Widget _analyzeButton() {
    final bool enabled = _readyToAnalyze;
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        minimumSize: const Size.fromHeight(56),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        backgroundColor: enabled
            ? Colors.cyan
            : Colors.white.withOpacity(0.05),
        foregroundColor: enabled ? Colors.white : Colors.white54,
        elevation: enabled ? 6 : 0,
      ),
      onPressed: enabled ? widget.onAnalyze : null,
      child: enabled
          ? Row(
              mainAxisSize: MainAxisSize.min,
              children: const [
                Text('Analyze & Grade', style: TextStyle(fontSize: 16)),
                SizedBox(width: 8),
                Icon(Icons.arrow_forward),
              ],
            )
          : const Text(
              'Scan both sides to continue',
              style: TextStyle(fontSize: 14),
            ),
    );
  }
}
