// ignore_for_file: deprecated_member_use

import 'package:flutter/material.dart';

import '../models/grading_result.dart';

class AnalysisDetailsScreen extends StatefulWidget {
  final String jobId;
  final GradingResult result;
  final VoidCallback onBack;

  const AnalysisDetailsScreen({
    super.key,
    required this.jobId,
    required this.result,
    required this.onBack,
  });

  @override
  State<AnalysisDetailsScreen> createState() => _AnalysisDetailsScreenState();
}

class _AnalysisDetailsScreenState extends State<AnalysisDetailsScreen> {
  late String _side;

  @override
  void initState() {
    super.initState();
    _side = widget.result.visualizations.containsKey('front')
        ? 'front'
        : (widget.result.visualizations.keys.isNotEmpty
            ? widget.result.visualizations.keys.first
            : 'front');
  }

  @override
  Widget build(BuildContext context) {
    final bundle = widget.result.visualizations[_side];
    final tabs = _buildStages(bundle);

    return DefaultTabController(
      length: tabs.length,
      child: Scaffold(
        backgroundColor: const Color(0xFF0B1224),
        appBar: AppBar(
          backgroundColor: Colors.transparent,
          elevation: 0,
          leading: IconButton(
            icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white),
            onPressed: widget.onBack,
          ),
          title: Text(
            'Analysis • ${widget.jobId}',
            style: const TextStyle(color: Colors.white, fontSize: 14),
          ),
          actions: _buildSideSwitcher(),
          bottom: TabBar(
            isScrollable: true,
            indicatorColor: const Color(0xFF06B6D4),
            labelColor: Colors.white,
            unselectedLabelColor: Colors.white70,
            tabs: tabs.map((t) => Tab(text: t.title)).toList(),
          ),
        ),
        body: TabBarView(
          children: tabs.map((t) => t.child).toList(),
        ),
      ),
    );
  }

  List<Widget> _buildSideSwitcher() {
    final hasBack = widget.result.visualizations.containsKey('back');
    if (!hasBack) return [];
    return [
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8),
        child: Container(
          padding: const EdgeInsets.all(6),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.06),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: Colors.white.withOpacity(0.08)),
          ),
          child: Row(
            children: [
              _sidePill('Front'),
              const SizedBox(width: 6),
              _sidePill('Back'),
            ],
          ),
        ),
      )
    ];
  }

  Widget _sidePill(String label) {
    final key = label.toLowerCase();
    final active = _side == key;
    return GestureDetector(
      onTap: () => setState(() => _side = key),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: active ? const Color(0xFF06B6D4) : Colors.transparent,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: active ? Colors.black : Colors.white70,
            fontWeight: FontWeight.w700,
            fontSize: 12,
          ),
        ),
      ),
    );
  }

  List<_StageTab> _buildStages(VisualizationBundle? bundle) {
    if (bundle == null) {
      return [
        _StageTab(
          title: 'Summary',
          child: _EmptyState(
            title: 'No visualization data',
            description:
                'This run did not include visualization assets. Try re-scanning.',
          ),
        )
      ];
    }

    final assets = bundle.assets;
    final meta = bundle.meta;
    final allAssets = assets;

    final photometricMeta = meta['photometric'] is Map<String, dynamic>
        ? meta['photometric'] as Map<String, dynamic>
        : null;
    final centeringMeta = meta['centering'] is Map<String, dynamic>
        ? meta['centering'] as Map<String, dynamic>
        : null;
    final cornerMeta = meta['corner_analysis'] is Map<String, dynamic>
        ? meta['corner_analysis'] as Map<String, dynamic>
        : null;
    final borderMeta = meta['border_analysis'] is Map<String, dynamic>
        ? meta['border_analysis'] as Map<String, dynamic>
        : null;
    final insightsMeta = meta['insights'] is Map<String, dynamic>
        ? meta['insights'] as Map<String, dynamic>
        : null;

    VisualizationAsset? firstAssetMatching(String keyword) {
      try {
        return allAssets.firstWhere(
          (a) => a.name.toLowerCase().contains(keyword.toLowerCase()),
        );
      } catch (_) {
        return null;
      }
    }

    return [
      _StageTab(
        title: 'Summary',
        child: _SummaryPane(
          side: _side,
          bundle: bundle,
          grade: _side == 'front'
              ? widget.result.front?.formattedGrade
              : widget.result.back?.formattedGrade,
          insights: insightsMeta,
        ),
      ),
      _StageTab(
        title: 'Centering',
        child: _CenteringPane(
          meta: centeringMeta,
          overlay: firstAssetMatching('centering'),
        ),
      ),
      _StageTab(
        title: 'Photometric',
        child: _AssetGrid(
          assets: [
            if (firstAssetMatching('surface_normals') != null)
              firstAssetMatching('surface_normals')!,
            if (firstAssetMatching('albedo') != null)
              firstAssetMatching('albedo')!,
            if (firstAssetMatching('depth_map') != null)
              firstAssetMatching('depth_map')!,
            if (firstAssetMatching('defect_map') != null)
              firstAssetMatching('defect_map')!,
            if (firstAssetMatching('confidence_map') != null)
              firstAssetMatching('confidence_map')!,
          ],
          meta: photometricMeta,
          emptyText: 'Photometric visualizations will appear here.',
        ),
      ),
      _StageTab(
        title: 'Corners',
        child: _CornerPane(meta: cornerMeta),
      ),
      _StageTab(
        title: 'Borders',
        child: _MetaCard(
          title: 'Border Detection',
          meta: borderMeta,
          emptyText: 'Border detection data not available.',
        ),
      ),
      _StageTab(
        title: 'Defects',
        child: _DefectPane(defects: meta['smart_defects']),
      ),
      _StageTab(
        title: 'Insights',
        child: _MetaCard(
          title: 'Model Insights',
          meta: meta['insights'] as Map<String, dynamic>?,
          emptyText: 'Insights unavailable for this run.',
        ),
      ),
      _StageTab(
        title: 'Assets',
        child: _AssetGrid(
          assets: allAssets,
          meta: meta.isNotEmpty ? meta : null,
          emptyText: 'No assets were produced for this run.',
        ),
      ),
    ];
  }
}

class _StageTab {
  final String title;
  final Widget child;
  _StageTab({required this.title, required this.child});
}

class _SummaryPane extends StatelessWidget {
  final String? grade;
  final String side;
  final VisualizationBundle bundle;
  final Map<String, dynamic>? insights;

  const _SummaryPane({
    required this.side,
    required this.bundle,
    this.grade,
    this.insights,
  });

  @override
  Widget build(BuildContext context) {
    final meta = bundle.meta;
    return _FrostedContainer(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${side[0].toUpperCase()}${side.substring(1)} summary',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              _StatChip(label: 'Grade', value: grade ?? '—'),
              if (insights?['grade_confidence'] != null)
                _StatChip(
                    label: 'Confidence',
                    value:
                        '${(insights!['grade_confidence'] as num).toStringAsFixed(1)}%'),
              if (meta['photometric']?['surface_integrity'] != null)
                _StatChip(
                    label: 'Surface',
                    value:
                        '${(meta['photometric']['surface_integrity'] as num).toStringAsFixed(1)}%'),
              if (meta['centering']?['overall_centering_score'] != null)
                _StatChip(
                    label: 'Centering',
                    value:
                        '${(meta['centering']['overall_centering_score'] as num).toStringAsFixed(1)}%'),
            ],
          ),
          const SizedBox(height: 12),
          if (insights != null) _MetaList(meta: insights!, compact: true),
        ],
      ),
    );
  }
}

class _CenteringPane extends StatelessWidget {
  final Map<String, dynamic>? meta;
  final VisualizationAsset? overlay;

  const _CenteringPane({this.meta, this.overlay});

  @override
  Widget build(BuildContext context) {
    if (meta == null) {
      return const _EmptyState(
        title: 'No centering data',
        description: 'This run did not include centering analysis.',
      );
    }
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (overlay != null)
            _VizCard(
              asset: overlay!,
              dense: true,
              caption: meta?['verdict']?.toString(),
            ),
          const SizedBox(height: 12),
          _MetaList(meta: meta!, compact: true),
        ],
      ),
    );
  }
}

class _CornerPane extends StatelessWidget {
  final Map<String, dynamic>? meta;
  const _CornerPane({this.meta});

  @override
  Widget build(BuildContext context) {
    if (meta == null) {
      return const _EmptyState(
        title: 'No corner data',
        description: 'Corner analysis not available for this run.',
      );
    }
    final rawScores = meta?['scores'];
    final scores = rawScores is Map<String, dynamic>
        ? rawScores.map((k, v) => MapEntry(k.toString(), v))
        : null;
    return _FrostedContainer(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Corner Scores',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 10),
          if (scores != null && scores.isNotEmpty)
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: scores.entries
                  .map((e) => _StatChip(
                        label: e.key,
                        value: (e.value as num).toStringAsFixed(1),
                      ))
                  .toList(),
            )
          else
            const Text(
              'Scores unavailable.',
              style: TextStyle(color: Colors.white70),
            ),
        ],
      ),
    );
  }
}

class _AssetGrid extends StatelessWidget {
  final List<VisualizationAsset> assets;
  final Map<String, dynamic>? meta;
  final String emptyText;

  const _AssetGrid({
    required this.assets,
    this.meta,
    required this.emptyText,
  });

  @override
  Widget build(BuildContext context) {
    if (assets.isEmpty) {
      return _EmptyState(title: 'No assets', description: emptyText);
    }
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: assets.length,
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
              childAspectRatio: 4 / 3,
            ),
            itemBuilder: (context, index) => _VizCard(asset: assets[index]),
          ),
          if (meta != null && meta!.isNotEmpty) ...[
            const SizedBox(height: 16),
            _MetaList(meta: meta!, compact: true),
          ],
        ],
      ),
    );
  }
}

class _VizCard extends StatelessWidget {
  final VisualizationAsset asset;
  final bool dense;
  final String? caption;

  const _VizCard({required this.asset, this.dense = false, this.caption});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => _showFullImage(context, asset),
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.04),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.white.withOpacity(0.08)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              asset.name,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.network(
                  asset.url,
                  fit: BoxFit.contain,
                  loadingBuilder: (context, child, progress) {
                    if (progress == null) return child;
                    return const Center(
                      child: CircularProgressIndicator(
                        valueColor: AlwaysStoppedAnimation(Color(0xFF06B6D4)),
                      ),
                    );
                  },
                  errorBuilder: (context, error, stack) => Container(
                    color: Colors.white.withOpacity(0.05),
                    alignment: Alignment.center,
                    child: Text(
                      'Could not load ${asset.name}',
                      style: const TextStyle(color: Colors.white54),
                    ),
                  ),
                ),
              ),
            ),
            if (caption != null) ...[
              const SizedBox(height: 6),
              Text(
                caption!,
                style: const TextStyle(color: Colors.white70, fontSize: 12),
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),
            ]
          ],
        ),
      ),
    );
  }

  void _showFullImage(BuildContext context, VisualizationAsset asset) {
    showDialog(
      context: context,
      barrierColor: Colors.black.withOpacity(0.9),
      builder: (_) => GestureDetector(
        onTap: () => Navigator.of(context).pop(),
        child: Stack(
          children: [
            Positioned.fill(
              child: InteractiveViewer(
                child: Center(
                  child: Hero(
                    tag: asset.url,
                    child: Image.network(
                      asset.url,
                      fit: BoxFit.contain,
                      errorBuilder: (context, error, stack) => Padding(
                        padding: const EdgeInsets.all(24),
                        child: Text(
                          'Could not load ${asset.name}',
                          style: const TextStyle(color: Colors.white70),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
            Positioned(
              top: 32,
              right: 24,
              child: IconButton(
                icon: const Icon(Icons.close, color: Colors.white),
                onPressed: () => Navigator.of(context).pop(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MetaList extends StatelessWidget {
  final Map<String, dynamic> meta;
  final bool compact;

  const _MetaList({required this.meta, this.compact = false});

  @override
  Widget build(BuildContext context) {
    final entries = meta.entries.toList();
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.03),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: entries
            .map(
              (e) => Padding(
                padding: EdgeInsets.symmetric(vertical: compact ? 2 : 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(
                      width: 130,
                      child: Text(
                        e.key,
                        style: const TextStyle(
                          color: Colors.white70,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    Expanded(
                      child: Text(
                        _formatValue(e.value),
                        style: const TextStyle(
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            )
            .toList(),
      ),
    );
  }

  String _formatValue(dynamic value) {
    if (value == null) return '—';
    if (value is num) return value.toStringAsFixed(value is int ? 0 : 2);
    if (value is List) return value.join(', ');
    if (value is Map) {
      return value.entries.map((e) => '${e.key}: ${_formatValue(e.value)}').join(' • ');
    }
    return value.toString();
  }
}

class _StatChip extends StatelessWidget {
  final String label;
  final String value;
  const _StatChip({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.07),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label.toUpperCase(),
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 11,
              letterSpacing: 0.8,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _DefectPane extends StatelessWidget {
  final dynamic defects;
  const _DefectPane({this.defects});

  @override
  Widget build(BuildContext context) {
    if (defects == null) {
      return const _EmptyState(
        title: 'No defects reported',
        description: 'Smart defect analysis did not return any findings.',
      );
    }
    if (defects is List && defects.isNotEmpty) {
      return ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: defects.length,
        itemBuilder: (context, index) {
          final item = defects[index];
          return _FrostedContainer(
            margin: const EdgeInsets.only(bottom: 12),
            child: Text(
              item.toString(),
              style: const TextStyle(color: Colors.white),
            ),
          );
        },
      );
    }
    return const _EmptyState(
      title: 'No defects detected',
      description: 'Model did not identify meaningful defects.',
    );
  }
}

class _MetaCard extends StatelessWidget {
  final String title;
  final Map<String, dynamic>? meta;
  final String emptyText;
  const _MetaCard({
    required this.title,
    required this.meta,
    required this.emptyText,
  });

  @override
  Widget build(BuildContext context) {
    if (meta == null || meta!.isEmpty) {
      return _EmptyState(title: title, description: emptyText);
    }
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: _FrostedContainer(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 10),
            _MetaList(meta: meta!, compact: true),
          ],
        ),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final String title;
  final String description;
  const _EmptyState({required this.title, required this.description});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              title,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              description,
              style: const TextStyle(color: Colors.white70),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

class _FrostedContainer extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? margin;

  const _FrostedContainer({required this.child, this.margin});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: margin,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.04),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: child,
    );
  }
}
