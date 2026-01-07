// ignore_for_file: deprecated_member_use

import 'package:flutter/material.dart';

import '../models/captured_image.dart';
import '../models/grading_result.dart';

/// Screen displayed after analysis completes. Shows the returned TruScore,
/// confidence, subgrades, and previews of the captured front/back images.
class ResultsScreen extends StatelessWidget {
  final CapturedImage? frontImage;
  final CapturedImage? backImage;
  final GradingResult? result;
  final String? jobId;
  final VoidCallback onHome;
  final VoidCallback? onViewMarket;
  final VoidCallback? onViewAnalysis;

  const ResultsScreen({
    super.key,
    required this.frontImage,
    required this.backImage,
    required this.result,
    required this.jobId,
    required this.onHome,
    this.onViewMarket,
    this.onViewAnalysis,
  });

  @override
  Widget build(BuildContext context) {
    final grade = result?.front?.formattedGrade ?? 'Pending';
    final confidence = result?.front?.gradeConfidence != null
        ? '${(result!.front!.gradeConfidence! * 100).toStringAsFixed(0)}%'
        : '—';
    final subgrades = _buildSubgrades(result?.front);
    final frontProvider = frontImage?.provider;

    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      body: Column(
        children: [
          // Top section with blurred background and card preview
          SizedBox(
            height: 320,
            width: double.infinity,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Background blurred image
                if (frontProvider != null)
                  Image(
                    image: frontProvider,
                    fit: BoxFit.cover,
                    color: Colors.black.withOpacity(0.72),
                    colorBlendMode: BlendMode.darken,
                  )
                else
                  Container(
                    color: Colors.black,
                  ),
                // Gradient overlay to fade the image into the solid background
                Container(
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [Colors.transparent, Color(0xFF0F172A)],
                      stops: [0.0, 0.8],
                    ),
                  ),
                ),
                // Main card preview
                Align(
                  alignment: Alignment.center,
                  child: Container(
                    width: 170,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(18),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.5),
                          blurRadius: 12,
                          offset: const Offset(0, 4),
                        ),
                      ],
                      border: Border.all(
                        color: Colors.white.withOpacity(0.2),
                      ),
                    ),
                    clipBehavior: Clip.hardEdge,
                    child: frontProvider != null
                        ? Image(
                            image: frontProvider,
                            fit: BoxFit.cover,
                          )
                        : Container(
                            color: Colors.white10,
                            alignment: Alignment.center,
                            child: const Icon(
                              Icons.photo,
                              color: Colors.white38,
                              size: 64,
                            ),
                          ),
                  ),
                ),
                // Back thumbnail if present
                if (backImage != null)
                  Positioned(
                    bottom: 36,
                    right: 36,
                    child: Container(
                      width: 72,
                      height: 72,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.white24),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.6),
                            blurRadius: 10,
                            offset: const Offset(0, 6),
                          )
                        ],
                      ),
                      clipBehavior: Clip.antiAlias,
                      child: Image(image: backImage!.provider, fit: BoxFit.cover),
                    ),
                  ),
                // Share button on top right
                Positioned(
                  top: 16,
                  right: 16,
                  child: IconButton(
                    onPressed: () {},
                    icon: const Icon(Icons.share),
                    color: Colors.white,
                  ),
                ),
              ],
            ),
          ),
          // Content section scrollable
          Expanded(
            child: Container(
              padding: const EdgeInsets.fromLTRB(24, 36, 24, 24),
              decoration: BoxDecoration(
                color: const Color(0xFF0F172A),
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(32),
                  topRight: Radius.circular(32),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.35),
                    offset: const Offset(0, -4),
                    blurRadius: 12,
                  ),
                ],
              ),
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    // Analysis completed badge
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.green.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                          color: Colors.green.withOpacity(0.2),
                        ),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.check_circle,
                              size: 16, color: Colors.greenAccent),
                          const SizedBox(width: 4),
                          Text(
                            result?.front?.success == false
                                ? 'Analysis Completed With Issues'
                                : 'Analysis Complete',
                            style: const TextStyle(
                              color: Colors.greenAccent,
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 1.0,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                    // Title and details
                    const Text(
                      'TruScore Estimate',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      jobId != null ? 'Job • $jobId' : 'Local session',
                      style: const TextStyle(
                        color: Colors.white54,
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 32),
                    // TruScore and Confidence cards
                    Row(
                      children: [
                        Expanded(
                          child: _infoCard(
                            title: 'TruScore',
                            value: grade,
                            subtitle: result?.front?.success == false
                                ? 'Check capture'
                                : 'Estimation',
                            color: Colors.cyan,
                            icon: Icons.star,
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: _infoCard(
                            title: 'Confidence',
                            value: confidence,
                            subtitle: 'Model confidence',
                            color: Colors.green,
                            icon: Icons.shield,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 32),
                    // Subgrades
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Sub-Grades',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.0,
                          ),
                        ),
                        const SizedBox(height: 16),
                        ...subgrades.map((sg) {
                          return Padding(
                            padding: const EdgeInsets.symmetric(vertical: 6),
                            child: Row(
                              children: [
                                SizedBox(
                                  width: 100,
                                  child: Text(
                                    sg.label.toUpperCase(),
                                    style: const TextStyle(
                                      color: Colors.white54,
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                                Expanded(
                                  child: Stack(
                                    children: [
                                      Container(
                                        height: 8,
                                        decoration: BoxDecoration(
                                          color:
                                              Colors.white.withOpacity(0.05),
                                          borderRadius:
                                              BorderRadius.circular(4),
                                        ),
                                      ),
                                      AnimatedContainer(
                                        duration:
                                            const Duration(milliseconds: 600),
                                        height: 8,
                                        width: 200 * (sg.score / 10).clamp(0, 1),
                                        decoration: BoxDecoration(
                                          gradient: LinearGradient(
                                            colors: sg.colors,
                                          ),
                                          borderRadius:
                                              BorderRadius.circular(4),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Text(
                                  sg.score.toStringAsFixed(1),
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 14,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                          );
                        }),
                      ],
                    ),
                    const SizedBox(height: 32),
                    // Action buttons
                    Column(
                      children: [
                        if (onViewAnalysis != null && jobId != null)
                          SizedBox(
                            width: double.infinity,
                            child: ElevatedButton(
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.deepPurple,
                                foregroundColor: Colors.white,
                                padding: const EdgeInsets.symmetric(vertical: 14),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(16),
                                ),
                              ),
                              onPressed: onViewAnalysis,
                              child: const Text(
                                'View Analysis Details',
                                style: TextStyle(fontSize: 15),
                              ),
                            ),
                          ),
                        if (onViewAnalysis != null && jobId != null)
                          const SizedBox(height: 12),
                        if (onViewMarket != null && jobId != null)
                          SizedBox(
                            width: double.infinity,
                            child: ElevatedButton(
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.purple,
                                foregroundColor: Colors.white,
                                padding: const EdgeInsets.symmetric(vertical: 14),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(16),
                                ),
                              ),
                              onPressed: onViewMarket,
                              child: const Text(
                                'View Market Analysis',
                                style: TextStyle(fontSize: 15),
                              ),
                            ),
                          ),
                        if (onViewMarket != null && jobId != null)
                          const SizedBox(height: 12),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.cyan,
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                            onPressed: onHome,
                            child: const Text(
                              'Submit for Official Grading',
                              style: TextStyle(fontSize: 16),
                            ),
                          ),
                        ),
                        const SizedBox(height: 12),
                        SizedBox(
                          width: double.infinity,
                          child: OutlinedButton(
                            style: OutlinedButton.styleFrom(
                              side: BorderSide(
                                  color: Colors.white.withOpacity(0.2)),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                            onPressed: onHome,
                            child: const Text(
                              'Save to Collection',
                              style: TextStyle(fontSize: 16),
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),
                        TextButton(
                          onPressed: onHome,
                          child: const Text(
                            'Return to Dashboard',
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 14,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// Builds an information card used for displaying the TruScore and confidence.
  Widget _infoCard({
    required String title,
    required String value,
    required String subtitle,
    required Color color,
    required IconData icon,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 20, color: color),
              const SizedBox(width: 6),
              Text(
                title.toUpperCase(),
                style: TextStyle(
                  color: color,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.2,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: TextStyle(
              color: color,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  List<_Subgrade> _buildSubgrades(GradingSide? side) {
    if (side == null) {
      return _defaultSubgrades;
    }
    final List<_Subgrade> items = [];
    if (side.centeringScore != null) {
      final centering = side.centeringScore!;
      final normalized = centering > 10 ? centering / 10 : centering;
      items.add(_Subgrade('Centering', normalized.clamp(0, 10),
          const [Colors.cyan, Colors.blue]));
    }
    if (side.cornerScores.isNotEmpty) {
      final numeric = side.cornerScores.values
          .whereType<num>()
          .map((e) => e.toDouble())
          .toList();
      if (numeric.isNotEmpty) {
        final avg = numeric.reduce((a, b) => a + b) / numeric.length;
        final normalized = avg > 10 ? avg / 10 : avg;
        items.add(_Subgrade('Corners', normalized.clamp(0, 10),
            const [Colors.purple, Colors.pink]));
      }
    }
    if (side.surfaceIntegrity != null) {
      final surface = side.surfaceIntegrity!;
      final normalized = surface > 10 ? surface / 10 : surface;
      items.add(_Subgrade('Surface', normalized.clamp(0, 10),
          const [Colors.orange, Colors.red]));
    }
    if (items.isEmpty) {
      return _defaultSubgrades;
    }
    return items;
  }
}

/// Helper class for subgrade entries.
class _Subgrade {
  final String label;
  final double score;
  final List<Color> colors;
  const _Subgrade(this.label, this.score, this.colors);
}

const List<_Subgrade> _defaultSubgrades = [
  _Subgrade('Centering', 9.5, [Colors.cyan, Colors.blue]),
  _Subgrade('Corners', 9.0, [Colors.purple, Colors.pink]),
  _Subgrade('Surface', 9.5, [Colors.orange, Colors.red]),
];
