import 'package:flutter/material.dart';

/// ISO card formats (copied/adapted from flutter_camera_overlay).
enum OverlayFormat {
  cardID1, // 85.60 × 53.98 mm (credit cards)
  cardID2,
  cardID3,
  simID000,
}

enum OverlayOrientation { landscape, portrait }

class CardOverlay {
  final double ratio;
  final double cornerRadius;
  final OverlayOrientation orientation;

  const CardOverlay({
    required this.ratio,
    required this.cornerRadius,
    required this.orientation,
  });

  factory CardOverlay.byFormat(OverlayFormat format) {
    switch (format) {
      case OverlayFormat.cardID1:
        return const CardOverlay(
            ratio: 1.59,
            cornerRadius: 0.064,
            orientation: OverlayOrientation.landscape);
      case OverlayFormat.cardID2:
        return const CardOverlay(
            ratio: 1.42,
            cornerRadius: 0.067,
            orientation: OverlayOrientation.landscape);
      case OverlayFormat.cardID3:
        return const CardOverlay(
            ratio: 1.42,
            cornerRadius: 0.057,
            orientation: OverlayOrientation.landscape);
      case OverlayFormat.simID000:
        return const CardOverlay(
            ratio: 1.66,
            cornerRadius: 0.073,
            orientation: OverlayOrientation.landscape);
    }
  }
}

/// Draws a translucent overlay with a rounded rectangle cut-out matching the
/// provided [CardOverlay] dimensions.
class CardOverlayShape extends StatelessWidget {
  final CardOverlay model;
  final double? width;
  final double? height;

  const CardOverlayShape({
    super.key,
    required this.model,
    this.width,
    this.height,
  });

  @override
  Widget build(BuildContext context) {
    final media = MediaQuery.of(context);
    final size = media.size;
    final useWidth = width ??
        (media.orientation == Orientation.portrait
            ? size.shortestSide * .9
            : size.longestSide * .5);
    final ratio = model.ratio;
    final useHeight = height ?? useWidth / ratio;
    final radius = model.cornerRadius * useHeight;

    return Stack(
      children: [
        Align(
          alignment: Alignment.center,
          child: Container(
            width: useWidth,
            height: useHeight,
            decoration: ShapeDecoration(
              color: Colors.transparent,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(radius),
                side: const BorderSide(width: 1, color: Colors.white),
              ),
            ),
          ),
        ),
        ColorFiltered(
          colorFilter: const ColorFilter.mode(Colors.black54, BlendMode.srcOut),
              child: Stack(
                children: [
                  Container(
                    decoration: const BoxDecoration(
                      color: Colors.transparent,
                    ),
                    child: Align(
                      alignment: Alignment.center,
                      child: Container(
                        width: useWidth,
                        height: useHeight,
                        decoration: BoxDecoration(
                          color: Colors.black,
                          borderRadius: BorderRadius.circular(radius),
                        ),
                      ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
