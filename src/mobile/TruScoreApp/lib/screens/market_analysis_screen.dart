// ignore_for_file: deprecated_member_use

import 'package:flutter/material.dart';

import '../models/captured_image.dart';
import '../models/grading_result.dart';
import '../models/market_analysis.dart';
import '../services/api_client.dart';

class MarketAnalysisScreen extends StatefulWidget {
  final String jobId;
  final GradingResult? result;
  final CapturedImage? frontImage;
  final CapturedImage? backImage;
  final ApiClient apiClient;
  final VoidCallback onBack;
  final VoidCallback onHome;
  final void Function(String message)? onError;

  const MarketAnalysisScreen({
    super.key,
    required this.jobId,
    required this.result,
    required this.frontImage,
    required this.backImage,
    required this.apiClient,
    required this.onBack,
    required this.onHome,
    this.onError,
  });

  @override
  State<MarketAnalysisScreen> createState() => _MarketAnalysisScreenState();
}

class _MarketAnalysisScreenState extends State<MarketAnalysisScreen> {
  MarketAnalysis? _market;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    if (widget.jobId.isEmpty) {
      setState(() {
        _loading = false;
        _error = 'Missing job id for market analysis.';
      });
      return;
    }
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final data = await widget.apiClient.fetchMarketAnalysis(widget.jobId);
      if (!mounted) return;
      setState(() {
        _market = data;
        _loading = false;
      });
    } catch (e) {
      widget.onError?.call('$e');
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = '$e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final heroProvider = widget.frontImage?.provider ?? widget.backImage?.provider;
    final grade = widget.result?.front?.formattedGrade ?? widget.result?.back?.formattedGrade;

    return Scaffold(
      backgroundColor: const Color(0xFF0B1224),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white),
          onPressed: widget.onBack,
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.home, color: Colors.white70),
            onPressed: widget.onHome,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: _loading
          ? const Center(
              child: CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation(Color(0xFF06B6D4)),
              ),
            )
          : _error != null
              ? _ErrorState(message: _error!, onRetry: _load)
              : RefreshIndicator(
                  color: const Color(0xFF06B6D4),
                  onRefresh: _load,
                  child: ListView(
                    padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
                    physics: const AlwaysScrollableScrollPhysics(),
                    children: [
                      Container(
                        height: 220,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(24),
                          border: Border.all(color: Colors.white.withOpacity(0.08)),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.5),
                              blurRadius: 20,
                              offset: const Offset(0, 10),
                            )
                          ],
                          gradient: const LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              Color(0x3314F9FF),
                              Color(0x3314FFF3),
                              Color(0x660B1224),
                            ],
                          ),
                        ),
                        clipBehavior: Clip.antiAlias,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            if (heroProvider != null)
                              Image(
                                image: heroProvider,
                                fit: BoxFit.cover,
                                color: Colors.black.withOpacity(0.35),
                                colorBlendMode: BlendMode.darken,
                              ),
                            Align(
                              alignment: Alignment.topLeft,
                              child: Padding(
                                padding: const EdgeInsets.all(16),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 12, vertical: 6),
                                      decoration: BoxDecoration(
                                        color: Colors.cyan.withOpacity(0.16),
                                        borderRadius: BorderRadius.circular(14),
                                        border: Border.all(
                                          color: Colors.cyan.withOpacity(0.4),
                                        ),
                                      ),
                                      child: Text(
                                        'JOB ${widget.jobId}',
                                        style: const TextStyle(
                                          color: Colors.cyan,
                                          fontWeight: FontWeight.bold,
                                          letterSpacing: 1.2,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(height: 12),
                                    Text(
                                      'Market Analysis',
                                      style: TextStyle(
                                        fontSize: 26,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white.withOpacity(0.9),
                                      ),
                                    ),
                                    const SizedBox(height: 6),
                                    Text(
                                      grade != null ? 'Grade: $grade' : 'Awaiting grade',
                                      style: const TextStyle(
                                        color: Colors.white70,
                                        fontSize: 14,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 20),
                      _infoCard(
                        title: 'Estimated Value',
                        value: _market?.estimatedValue != null
                            ? '\$${_market!.estimatedValue!.toStringAsFixed(2)}'
                            : 'TBD',
                        subtitle: _market?.marketSummary ??
                            'Placeholder valuation until live market data is wired.',
                        icon: Icons.show_chart,
                        color: Colors.cyan,
                      ),
                      const SizedBox(height: 16),
                      _infoCard(
                        title: 'Population',
                        value: _market?.populationTotal != null
                            ? '${_market!.populationTotal}'
                            : '—',
                        subtitle: 'Based on placeholder population report.',
                        icon: Icons.groups,
                        color: Colors.purpleAccent,
                        child: _populationBars(_market),
                      ),
                      const SizedBox(height: 16),
                      _recentSales(_market),
                      const SizedBox(height: 24),
                      TextButton.icon(
                        onPressed: widget.onHome,
                        icon: const Icon(Icons.check_circle, color: Colors.greenAccent),
                        label: const Text(
                          'Return to Dashboard',
                          style: TextStyle(color: Colors.greenAccent),
                        ),
                      ),
                    ],
                  ),
                ),
    );
  }

  Widget _infoCard({
    required String title,
    required String value,
    required String subtitle,
    required IconData icon,
    required Color color,
    Widget? child,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.04),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 20, color: color),
              const SizedBox(width: 8),
              Text(
                title,
                style: TextStyle(
                  color: color,
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            subtitle,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 13,
            ),
          ),
          if (child != null) ...[
            const SizedBox(height: 12),
            child,
          ],
        ],
      ),
    );
  }

  Widget _recentSales(MarketAnalysis? market) {
    final sales = market?.recentSales ?? const [];
    if (sales.isEmpty) {
      return _infoCard(
        title: 'Recent Sales',
        value: '—',
        subtitle: 'Sales data will appear once market feeds are connected.',
        icon: Icons.receipt_long,
        color: Colors.orangeAccent,
      );
    }
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.04),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: const [
              Icon(Icons.receipt_long, color: Colors.orangeAccent),
              SizedBox(width: 8),
              Text(
                'Recent Sales (Placeholder)',
                style: TextStyle(
                  color: Colors.orangeAccent,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ...sales.map(
            (sale) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      sale.title,
                      style: const TextStyle(color: Colors.white, fontSize: 14),
                    ),
                  ),
                  Text(
                    '\$${sale.price.toStringAsFixed(2)}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    '${sale.daysAgo}d',
                    style: const TextStyle(color: Colors.white54, fontSize: 12),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _populationBars(MarketAnalysis? market) {
    final dist = market?.gradeDistribution ?? const {};
    if (dist.isEmpty) {
      return const SizedBox.shrink();
    }
    final int maxValue = dist.values.fold(0, (prev, val) => val > prev ? val : prev);
    return Column(
      children: dist.entries.map((entry) {
        final ratio = maxValue == 0 ? 0.0 : entry.value / maxValue;
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Row(
            children: [
              SizedBox(
                width: 36,
                child: Text(
                  entry.key,
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ),
              Expanded(
                child: Stack(
                  children: [
                    Container(
                      height: 8,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.08),
                        borderRadius: BorderRadius.circular(6),
                      ),
                    ),
                    FractionallySizedBox(
                      widthFactor: ratio.clamp(0, 1),
                      child: Container(
                        height: 8,
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [Color(0xFF06B6D4), Color(0xFFA855F7)],
                          ),
                          borderRadius: BorderRadius.circular(6),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Text(
                '${entry.value}',
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }
}

class _ErrorState extends StatelessWidget {
  final String message;
  final Future<void> Function() onRetry;

  const _ErrorState({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, color: Colors.redAccent, size: 36),
            const SizedBox(height: 8),
            Text(
              message,
              style: const TextStyle(color: Colors.white70),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.cyan,
                foregroundColor: Colors.white,
              ),
              onPressed: onRetry,
              child: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }
}
