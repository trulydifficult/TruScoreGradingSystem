import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../models/grading_result.dart';
import '../models/market_analysis.dart';
import '../models/recent_scan.dart';

const String _defaultBaseUrl =
    String.fromEnvironment('TRUSCORE_API_BASE', defaultValue: '');

class ApiException implements Exception {
  final String message;
  final int? statusCode;

  const ApiException(this.message, [this.statusCode]);

  @override
  String toString() => 'ApiException($statusCode): $message';
}

class ApiClient {
  final http.Client _client;
  String _activeBase = 'http://10.0.2.2:8009'; // Default fallback

  // Static override for runtime updates
  static String? _globalOverride;

  static void updateBaseUrl(String url) {
    _globalOverride = url;
  }

  ApiClient({http.Client? client, String? baseUrl})
  : _client = client ?? http.Client() {
    _initBase(baseUrl);
  }

  void _initBase(String? baseUrl) async {
    if (_globalOverride != null) {
      _activeBase = _globalOverride!;
      return;
    }

    final prefs = await SharedPreferences.getInstance();
    final custom = prefs.getString('custom_api_ip');
    if (custom != null && custom.isNotEmpty) {
      _activeBase = custom;
      // ignore: avoid_print
      print('TruScore API: Using custom IP $_activeBase');
    } else if (baseUrl != null && baseUrl.isNotEmpty) {
      _activeBase = baseUrl;
    }
  }

  String get activeBase => _activeBase;
  // List<String> get baseCandidates => List.unmodifiable(_baseCandidates); // Removed legacy candidate logic

  static String _stripTrailingSlash(String value) =>
  value.endsWith('/') ? value.substring(0, value.length - 1) : value;

  Uri _uri(String base, String path) => Uri.parse('$base$path');

  String _absolute(String? path) {
    if (path == null || path.isEmpty) return '';
    if (path.startsWith('http')) return path;
    // Fix: path is guaranteed not null here, but linter might complain if strict
    if (!path.startsWith('/')) return '$_activeBase/$path';
    return '$_activeBase$path';
  }

  Future<T> _withBaseFallback<T>(Future<T> Function(String base) action) async {
    // Ensure we are using the latest configured IP
    if (_globalOverride != null) {
      _activeBase = _globalOverride!;
    } else {
      final prefs = await SharedPreferences.getInstance();
      final custom = prefs.getString('custom_api_ip');
      if (custom != null && custom.isNotEmpty) {
        _activeBase = custom;
      }
    }

    try {
      return await action(_activeBase);
    } catch (e) {
      throw ApiException('Connection failed to $_activeBase: $e');
    }
  }

  Future<String> submitForGrading({
    required File front,
    File? back,
    Map<String, dynamic>? metadata,
  }) async {
    return _withBaseFallback<String>((base) async {
      final request = http.MultipartRequest('POST', _uri(base, '/api/v1/cards/grade'));
      request.files.add(await http.MultipartFile.fromPath('front', front.path));
      if (back != null) {
        request.files.add(await http.MultipartFile.fromPath('back', back.path));
      }
      if (metadata != null && metadata.isNotEmpty) {
        request.fields['metadata'] = jsonEncode(metadata);
      }

      late http.StreamedResponse streamed;
      try {
        streamed = await request.send().timeout(const Duration(seconds: 45));
      } on Exception catch (e) {
        throw ApiException('Upload failed: $e');
      }

      final response = await http.Response.fromStream(streamed);
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw ApiException('Upload failed (${response.statusCode}) ${response.body}',
            response.statusCode);
      }

      final Map<String, dynamic> json = jsonDecode(response.body);
      final jobId = json['job_id'] ?? json['jobId'];
      if (jobId == null) {
        throw const ApiException('Server did not return a job_id');
      }
      return jobId.toString();
    });
  }

  Future<JobStatus> fetchJob(String jobId) async {
    return _withBaseFallback<JobStatus>((base) async {
      final uri = _uri(base, '/api/v1/cards/$jobId');
      late http.Response response;
      try {
        response = await _client.get(uri).timeout(const Duration(seconds: 20));
      } on Exception catch (e) {
        throw ApiException('Could not reach grading service: $e');
      }

      if (response.statusCode == 404) {
        throw const ApiException('Job not found', 404);
      }
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw ApiException(
          'Server error (${response.statusCode}) ${response.body}',
          response.statusCode,
        );
      }

      final Map<String, dynamic> json = jsonDecode(response.body);
      return JobStatus.fromJson(json).withResolvedUrls(_absolute);
    });
  }

  Future<List<RecentScan>> fetchRecentScans({int limit = 10}) async {
    return _withBaseFallback<List<RecentScan>>((base) async {
      final uri = _uri(base, '/api/v1/cards/recent?limit=$limit');
      late http.Response response;
      try {
        response = await _client.get(uri).timeout(const Duration(seconds: 15));
      } on Exception catch (e) {
        throw ApiException('Could not load recents: $e');
      }

      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw ApiException(
          'Server error (${response.statusCode}) ${response.body}',
          response.statusCode,
        );
      }

      final Map<String, dynamic> json = jsonDecode(response.body);
      final List<dynamic> items = (json['items'] as List?) ?? const [];
      return items
          .map((item) => RecentScan.fromJson(item, resolver: _absolute))
          .toList();
    });
  }

  Future<MarketAnalysis> fetchMarketAnalysis(String jobId) async {
    return _withBaseFallback<MarketAnalysis>((base) async {
      final uri = _uri(base, '/api/v1/cards/$jobId/market');
      late http.Response response;
      try {
        response = await _client.get(uri).timeout(const Duration(seconds: 20));
      } on Exception catch (e) {
        throw ApiException('Could not load market analysis: $e');
      }

      if (response.statusCode == 404) {
        throw const ApiException('Market analysis not available yet', 404);
      }
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw ApiException(
          'Server error (${response.statusCode}) ${response.body}',
          response.statusCode,
        );
      }

      final Map<String, dynamic> json = jsonDecode(response.body);
      return MarketAnalysis.fromJson(json);
    });
  }
}
