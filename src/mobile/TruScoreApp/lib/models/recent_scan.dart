/// Represents a previous grading submission returned by the API.
class RecentScan {
  final String jobId;
  final String title;
  final String? grade;
  final DateTime? submittedAt;
  final DateTime? completedAt;
  final String? thumbnailFront;
  final String? thumbnailBack;

  const RecentScan({
    required this.jobId,
    required this.title,
    this.grade,
    this.submittedAt,
    this.completedAt,
    this.thumbnailFront,
    this.thumbnailBack,
  });

  factory RecentScan.fromJson(
    Map<String, dynamic> json, {
    String Function(String?)? resolver,
  }) {
    DateTime? parseDate(dynamic value) {
      if (value == null) return null;
      if (value is num) {
        return DateTime.fromMillisecondsSinceEpoch((value * 1000).round());
      }
      return DateTime.tryParse(value.toString());
    }

  String? resolve(String? path) => resolver != null ? resolver(path) : path;

  final gradeValue = json['grade'];
  final grade = gradeValue?.toString();

    return RecentScan(
      jobId: json['job_id']?.toString() ?? '',
      title: json['title']?.toString() ?? 'Submission',
      grade: grade,
      submittedAt: parseDate(json['submitted_at']),
      completedAt: parseDate(json['completed_at']),
      thumbnailFront: resolve(json['thumbnail_front']?.toString()),
      thumbnailBack: resolve(json['thumbnail_back']?.toString()),
    );
  }

  String get displayGrade => grade ?? 'Pending';
}
