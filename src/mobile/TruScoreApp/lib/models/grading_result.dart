import 'dart:convert';

/// Structured version of the API response returned by server.py.
class GradingResult {
  final GradingSide? front;
  final GradingSide? back;
  final Map<String, dynamic> metadata;
  final Map<String, VisualizationBundle> visualizations;

  const GradingResult({
    this.front,
    this.back,
    this.metadata = const {},
    this.visualizations = const {},
  });

  /// Return a copy with visualization asset URLs resolved by [resolver].
  GradingResult resolveAssetUrls(String Function(String) resolver) {
    Map<String, VisualizationBundle> remap(Map<String, VisualizationBundle> input) {
      final out = <String, VisualizationBundle>{};
      input.forEach((k, v) {
        out[k] = v.resolve(resolver);
      });
      return out;
    }

    return GradingResult(
      front: front,
      back: back,
      metadata: metadata,
      visualizations: remap(visualizations),
    );
  }

  factory GradingResult.fromJson(Map<String, dynamic> json) {
    Map<String, VisualizationBundle> parseViz(Map<String, dynamic>? data) {
      if (data == null) return const {};
      final map = <String, VisualizationBundle>{};
      data.forEach((key, value) {
        if (value is Map<String, dynamic>) {
          map[key] = VisualizationBundle.fromJson(value);
        }
      });
      return map;
    }

    return GradingResult(
      front: json['front'] != null
          ? GradingSide.fromJson(json['front'] as Map<String, dynamic>)
          : null,
      back: json['back'] != null
          ? GradingSide.fromJson(json['back'] as Map<String, dynamic>)
          : null,
      metadata: (json['metadata'] as Map<String, dynamic>?) ?? const {},
      visualizations:
          parseViz(json['visualizations'] as Map<String, dynamic>?),
    );
  }
}

class GradingSide {
  final bool success;
  final String grade;
  final double? gradeConfidence;
  final double? surfaceIntegrity;
  final double? centeringScore;
  final Map<String, dynamic> cornerScores;
  final int? defectsCount;
  final double? processingTime;
  final String? timestamp;

  const GradingSide({
    required this.success,
    required this.grade,
    this.gradeConfidence,
    this.surfaceIntegrity,
    this.centeringScore,
    this.cornerScores = const {},
    this.defectsCount,
    this.processingTime,
    this.timestamp,
  });

  factory GradingSide.fromJson(Map<String, dynamic> json) {
    double? toDoubleSafe(dynamic value) {
      if (value == null) return null;
      if (value is num) return value.toDouble();
      return double.tryParse(value.toString());
    }

    return GradingSide(
      success: json['success'] == true,
      grade: json['grade']?.toString() ?? 'Unknown',
      gradeConfidence: toDoubleSafe(json['grade_confidence']),
      surfaceIntegrity: toDoubleSafe(json['surface_integrity']),
      centeringScore: toDoubleSafe(json['centering_score']),
      cornerScores:
          (json['corner_scores'] as Map<String, dynamic>?) ?? const {},
      defectsCount: json['defects_count'] as int?,
      processingTime: toDoubleSafe(json['processing_time']),
      timestamp: json['timestamp']?.toString(),
    );
  }

  String get formattedGrade {
    // If it's numeric keep one decimal, otherwise return as-is.
    final numeric = double.tryParse(grade);
    if (numeric != null) return numeric.toStringAsFixed(1);
    return grade;
  }
}

class VisualizationBundle {
  final List<VisualizationAsset> assets;
  final Map<String, dynamic> meta;

  const VisualizationBundle({
    this.assets = const [],
    this.meta = const {},
  });

  VisualizationBundle resolve(String Function(String) resolver) {
    final resolvedAssets = assets
        .map((a) => VisualizationAsset(name: a.name, url: resolver(a.url)))
        .toList();
    return VisualizationBundle(assets: resolvedAssets, meta: meta);
  }

  factory VisualizationBundle.fromJson(Map<String, dynamic> json) {
    final List<dynamic> rawAssets = json['assets'] as List<dynamic>? ?? const [];
    return VisualizationBundle(
      assets: rawAssets
          .whereType<Map<String, dynamic>>()
          .map(VisualizationAsset.fromJson)
          .toList(),
      meta: (json['meta'] as Map<String, dynamic>?) ?? const {},
    );
  }
}

class VisualizationAsset {
  final String name;
  final String url;

  const VisualizationAsset({
    required this.name,
    required this.url,
  });

  factory VisualizationAsset.fromJson(Map<String, dynamic> json) {
    return VisualizationAsset(
      name: json['name']?.toString() ?? 'asset',
      url: json['url']?.toString() ?? '',
    );
  }
}

class JobStatus {
  final String status;
  final GradingResult? result;
  final String? error;

  const JobStatus({
    required this.status,
    this.result,
    this.error,
  });

  JobStatus withResolvedUrls(String Function(String) resolver) {
    return JobStatus(
      status: status,
      result: result?.resolveAssetUrls(resolver),
      error: error,
    );
  }

  bool get isCompleted => status == 'completed';
  bool get isFailed => status == 'failed';
  bool get isRunning => status == 'running' || status == 'queued';

  factory JobStatus.fromJson(Map<String, dynamic> json) {
    return JobStatus(
      status: json['status']?.toString() ?? 'unknown',
      result: json['result'] != null
          ? GradingResult.fromJson(jsonDecode(jsonEncode(json['result'])))
          : null,
      error: json['error']?.toString(),
    );
  }
}
