import 'dart:async';

import 'package:flutter/material.dart';

/// Data model for a recent scan entry. Contains a title, date, grade and
/// thumbnail URL. This mirrors the mock API data used in the React version.
class RecentScan {
  final String title;
  final String date;
  final String grade;
  final String imageUrl;

  RecentScan({
    required this.title,
    required this.date,
    required this.grade,
    required this.imageUrl,
  });
}

/// The home screen is the landing page of the app. It shows a greeting,
/// allows the user to scan the front and back of a card, and displays a
/// list of recent scans. Once both sides are captured the "Analyze" button
/// becomes active.
class HomeScreen extends StatefulWidget {
  final String? frontImage;
  final String? backImage;
  final VoidCallback onScanFront;
  final VoidCallback onScanBack;
  final VoidCallback onAnalyze;

  const HomeScreen({
    super.key,
    required this.frontImage,
    required this.backImage,
    required this.onScanFront,
    required this.onScanBack,
    required this.onAnalyze,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _loading = true;
  List<RecentScan> _recentScans = [];

  @override
  void initState() {
    super.initState();
    // Simulate a call to an API to fetch recent scans. We delay for a short
    // period and then populate a list with some sample data. In a real
    // application this would be a network request.
    Timer(const Duration(milliseconds: 800), () {
      setState(() {
        _recentScans = [
          RecentScan(
            title: 'Topps Chrome #23',
            date: '2h ago',
            grade: '9.5',
            imageUrl:
                'https://images.unsplash.com/photo-1534063228518-a6644df1714d?auto=format&fit=crop&w=100&h=100',
          ),
          RecentScan(
            title: 'Panini Prizm #10',
            date: '5h ago',
            grade: '8.0',
            imageUrl:
                'https://images.unsplash.com/photo-1613771404784-3a5686aa2be3?auto=format&fit=crop&w=100&h=100',
          ),
          RecentScan(
            title: 'Upper Deck #45',
            date: '1d ago',
            grade: 'Pending',
            imageUrl:
                'https://images.unsplash.com/photo-1599508704512-2f19efd1e35f?auto=format&fit=crop&w=100&h=100',
          ),
        ];
        _loading = false;
      });
    });
  }

  bool get _readyToAnalyze =>
      widget.frontImage != null && widget.backImage != null;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header section
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
          decoration: BoxDecoration(
            color: Colors.blueGrey.shade900.withOpacity(0.6),
            border: const Border(
              bottom: BorderSide(color: Color(0x33FFFFFF), width: 1),
            ),
          ),
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
                      fontWeight: FontWeight.w500,
                      letterSpacing: 1.2,
                      shadows: [Shadow(blurRadius: 0, color: Colors.transparent)],
                    ),
                  ),
                  SizedBox(height: 4),
                  Text.rich(
                    TextSpan(
                      text: 'TruScore ',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                      children: [
                        TextSpan(
                          text: 'System',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w300,
                            color: Colors.white54,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              // Profile icon button
              IconButton(
                onPressed: () {},
                icon: const Icon(Icons.person_outline),
                color: Colors.white,
                tooltip: 'Profile',
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
                        imageUrl: widget.frontImage,
                        onPressed: widget.onScanFront,
                        color: Colors.cyan,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: _scanButton(
                        label: 'Scan Back',
                        side: 'back',
                        imageUrl: widget.backImage,
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
                      onPressed: () {},
                      child: const Text(
                        'View All',
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
                      : _recentScans.map((item) {
                          return InkWell(
                            onTap: () {},
                            child: Container(
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
                                    child: Image.network(
                                      item.imageUrl,
                                      width: 48,
                                      height: 48,
                                      fit: BoxFit.cover,
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
                                          item.date,
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
                                      color: item.grade == 'Pending'
                                          ? Colors.amber.withOpacity(0.1)
                                          : Colors.green.withOpacity(0.1),
                                      border: Border.all(
                                        color: item.grade == 'Pending'
                                            ? Colors.amber.withOpacity(0.2)
                                            : Colors.green.withOpacity(0.2),
                                      ),
                                      borderRadius: BorderRadius.circular(6),
                                    ),
                                    child: Text(
                                      item.grade,
                                      style: TextStyle(
                                        color: item.grade == 'Pending'
                                            ? Colors.amber
                                            : Colors.green,
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  const Icon(Icons.chevron_right,
                                      color: Colors.white54, size: 20),
                                ],
                              ),
                            ),
                          );
                        }).toList(),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  /// Builds the scan button for the front and back scans. The button will
  /// display an image preview once the corresponding side has been scanned.
  Widget _scanButton({
    required String label,
    required String side,
    required String? imageUrl,
    required VoidCallback onPressed,
    required MaterialColor color,
  }) {
    final bool hasImage = imageUrl != null;
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
            // If we have an image we display it as a background with a
            // translucent gradient overlay. Otherwise leave the background
            // transparent.
            if (hasImage)
              Positioned.fill(
                child: Image.network(
                  imageUrl!,
                  fit: BoxFit.cover,
                  color: Colors.black.withOpacity(0.4),
                  colorBlendMode: BlendMode.darken,
                ),
              ),
            if (hasImage)
              Container(
                alignment: Alignment.center,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: const [
                    Icon(
                      Icons.check_circle,
                      color: Colors.greenAccent,
                      size: 32,
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Ready',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w600,
                        fontSize: 14,
                      ),
                    ),
                  ],
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