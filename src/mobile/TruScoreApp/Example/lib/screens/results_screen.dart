import 'package:flutter/material.dart';

/// Screen displayed after analysis completes. Shows a card preview, the final
/// grade, market value, subgrades and some action buttons. This mirrors the
/// ResultsScreen from the React implementation albeit with simplified
/// structure and styling.
class ResultsScreen extends StatelessWidget {
  final String? frontImage;
  final VoidCallback onHome;

  const ResultsScreen({super.key, required this.frontImage, required this.onHome});

  @override
  Widget build(BuildContext context) {
    // Define some subgrade data. In a real implementation these values would
    // come from the analysis back‑end.
    final List<_Subgrade> subgrades = [
      _Subgrade('Centering', 9.5, const [Colors.cyan, Colors.blue]),
      _Subgrade('Corners', 9.0, const [Colors.purple, Colors.pink]),
      _Subgrade('Edges', 10.0, const [Colors.green, Colors.teal]),
      _Subgrade('Surface', 9.5, const [Colors.orange, Colors.red]),
    ];

    return Scaffold(
      backgroundColor: Colors.blueGrey.shade900,
      body: Column(
        children: [
          // Top section with blurred background and card preview
          SizedBox(
            height: 300,
            width: double.infinity,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Background blurred image
                if (frontImage != null)
                  Image.network(
                    frontImage!,
                    fit: BoxFit.cover,
                    color: Colors.black.withOpacity(0.8),
                    colorBlendMode: BlendMode.darken,
                  ),
                // Gradient overlay to fade the image into the solid background
                Container(
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [Colors.transparent, Color(0xFF0F172A), Color(0xFF0F172A)],
                      stops: [0.0, 0.6, 1.0],
                    ),
                  ),
                ),
                // Main card preview
                Align(
                  alignment: Alignment.center,
                  child: Container(
                    width: 160,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(16),
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
                    child: frontImage != null
                        ? Image.network(
                            frontImage!,
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
              padding: const EdgeInsets.fromLTRB(24, 40, 24, 24),
              decoration: BoxDecoration(
                color: Colors.blueGrey.shade900,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(32),
                  topRight: Radius.circular(32),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.3),
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
                        children: const [
                          Icon(Icons.check_circle, size: 16, color: Colors.greenAccent),
                          SizedBox(width: 4),
                          Text(
                            'Analysis Complete',
                            style: TextStyle(
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
                      'Topps Chrome #23',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      '2023 • Baseball • Refractor',
                      style: TextStyle(
                        color: Colors.white54,
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 32),
                    // TruScore and Market Value cards
                    Row(
                      children: [
                        Expanded(
                          child: _infoCard(
                            title: 'TruScore',
                            value: '9.5',
                            subtitle: 'Gem Mint',
                            color: Colors.cyan,
                            icon: Icons.star,
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: _infoCard(
                            title: 'Market Value',
                            value: '\$185',
                            subtitle: '+12%',
                            color: Colors.green,
                            icon: Icons.trending_up,
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
                                  width: 80,
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
                                        duration: const Duration(milliseconds: 600),
                                        height: 8,
                                        width: 200 * (sg.score / 10),
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
                        }).toList(),
                      ],
                    ),
                    const SizedBox(height: 32),
                    // Action buttons
                    Column(
                      children: [
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

  /// Builds an information card used for displaying the TruScore and market
  /// value. Accepts a title, a large value, an optional subtitle and a
  /// colour accent with icon.
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
}

/// Helper class for subgrade entries.
class _Subgrade {
  final String label;
  final double score;
  final List<Color> colors;
  _Subgrade(this.label, this.score, this.colors);
}