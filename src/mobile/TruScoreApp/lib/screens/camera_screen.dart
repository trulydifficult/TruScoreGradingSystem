import 'package:flutter/material.dart';
import 'package:hl_image_picker/hl_image_picker.dart';
import '../models/captured_image.dart';

class CameraScreen extends StatefulWidget {
  final String side; // 'front' or 'back'
  final ValueChanged<CapturedImage> onCapture;
  final VoidCallback onBack;

  const CameraScreen({
    super.key,
    required this.side,
    required this.onCapture,
    required this.onBack,
  });

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final _picker = HLImagePicker();
  bool _hasLaunched = false;

  @override
  void initState() {
    super.initState();
    // Launch camera immediately after build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_hasLaunched) {
        _hasLaunched = true;
        _openCamera();
      }
    });
  }

  Future<void> _openCamera() async {
    try {
      final image = await _picker.openCamera(
        cropping: true,
        cameraOptions: HLCameraOptions(
          cameraType: CameraType.image,
          isExportThumbnail: true,
          thumbnailCompressFormat: CompressFormat.jpg,
          thumbnailCompressQuality: 0.9,
        ),
        cropOptions: HLCropOptions(
          aspectRatio: const CropAspectRatio(ratioX: 2.5, ratioY: 3.5), // Standard Card Ratio
          compressQuality: 0.9,
          compressFormat: CompressFormat.jpg,
          croppingStyle: CroppingStyle.normal,
        ),
      );

      if (image != null) {
        widget.onCapture(CapturedImage(path: image.path, fromGallery: false));
      } else {
        widget.onBack(); // User cancelled
      }
    } catch (e) {
      debugPrint('Camera error: $e');
      // Fallback or error handling could go here
      widget.onBack();
    }
  }

  @override
  Widget build(BuildContext context) {
    // Show a clean loading state while the picker launches
    return const Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: Colors.cyan),
            SizedBox(height: 16),
            Text("Launching Camera...", style: TextStyle(color: Colors.white54))
          ],
        ),
      ),
    );
  }
}
