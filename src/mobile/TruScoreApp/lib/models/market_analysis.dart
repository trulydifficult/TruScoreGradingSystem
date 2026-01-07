/// Placeholder market analysis payload from the API.
class MarketAnalysis {
  final String jobId;
  final String? grade;
  final double? estimatedValue;
  final String? marketSummary;
  final List<MarketSale> recentSales;
  final Map<String, int> gradeDistribution;
  final int? populationTotal;

  const MarketAnalysis({
    required this.jobId,
    this.grade,
    this.estimatedValue,
    this.marketSummary,
    this.recentSales = const [],
    this.gradeDistribution = const {},
    this.populationTotal,
  });

  factory MarketAnalysis.fromJson(Map<String, dynamic> json) {
    double? toDoubleSafe(dynamic value) {
      if (value == null) return null;
      if (value is num) return value.toDouble();
      return double.tryParse(value.toString());
    }

    final List<dynamic> sales = (json['recent_sales'] as List?) ?? const [];
    final Map<String, dynamic> pop = (json['population'] as Map<String, dynamic>?) ?? const {};
    final Map<String, dynamic> dist =
        (pop['grade_distribution'] as Map<String, dynamic>?) ?? const {};

    return MarketAnalysis(
      jobId: json['job_id']?.toString() ?? '',
      grade: json['grade']?.toString(),
      estimatedValue: toDoubleSafe(json['estimated_value']),
      marketSummary: json['market_summary']?.toString(),
      recentSales: sales.map((e) => MarketSale.fromJson(e)).toList(),
      gradeDistribution: dist.map((key, value) => MapEntry(key, int.tryParse(value.toString()) ?? 0)),
      populationTotal: pop['total'] is num ? (pop['total'] as num).toInt() : int.tryParse('${pop['total']}'),
    );
  }
}

class MarketSale {
  final String title;
  final double price;
  final int daysAgo;

  const MarketSale({
    required this.title,
    required this.price,
    required this.daysAgo,
  });

  factory MarketSale.fromJson(Map<String, dynamic> json) {
    double toDoubleSafe(dynamic value) {
      if (value is num) return value.toDouble();
      return double.tryParse(value.toString()) ?? 0.0;
    }

    int toIntSafe(dynamic value) {
      if (value is num) return value.toInt();
      return int.tryParse(value.toString()) ?? 0;
    }

    return MarketSale(
      title: json['title']?.toString() ?? 'Listing',
      price: toDoubleSafe(json['price']),
      daysAgo: toIntSafe(json['days_ago']),
    );
  }
}
