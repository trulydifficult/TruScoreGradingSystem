import 'dart:io';

import 'package:flutter/material.dart';

/// Represents a captured or selected image on device storage.
/// We keep only the path plus a flag for analytics/UI on whether the user
/// picked it from the gallery.
class CapturedImage {
  final String path;
  final bool fromGallery;

  const CapturedImage({
    required this.path,
    this.fromGallery = false,
  });

  ImageProvider get provider {
    if (path.startsWith('http')) return NetworkImage(path);
    return FileImage(File(path));
  }
}
