// ignore_for_file: deprecated_member_use

import 'dart:ui';

import 'package:flutter/material.dart';

/// A wrapper that simulates a mobile phone frame around its child.
///
/// The original React application was designed for a mobile device. In order to
/// preserve that look and feel in Flutter we wrap our screens in a rounded
/// rectangle with a thick border and a couple of subtle decorative elements
/// (status bar, soft gradient circles and a home indicator). This widget
/// handles all of that decoration and leaves the content area to the child.
class MobileWrapper extends StatelessWidget {
  /// The widget to display inside the mobile frame.
  final Widget child;

  const MobileWrapper({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        width: 390,
        height: 800,
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0F172A),
              Color(0xFF0B1221),
            ],
          ),
          borderRadius: BorderRadius.circular(44),
          border: Border.all(width: 1.5, color: Colors.white10),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.6),
              blurRadius: 30,
              spreadRadius: 4,
              offset: const Offset(0, 18),
            ),
          ],
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(36),
          child: Stack(
            children: [
              BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
                child: Container(
                  color: Colors.white.withOpacity(0.02),
                ),
              ),
              // Soft radial gradient circles in the background to mimic the
              // abstract gradients from the original design. These are kept
              // intentionally subtle so that they don't interfere with the
              // foreground content.
              Positioned(
                top: -100,
                left: -100,
                child: Container(
                  width: 200,
                  height: 200,
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        Color.fromARGB(64, 136, 57, 239),
                        Colors.transparent,
                      ],
                      radius: 0.8,
                    ),
                  ),
                ),
              ),
              Positioned(
                bottom: -120,
                right: -120,
                child: Container(
                  width: 240,
                  height: 240,
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        Color.fromARGB(64, 34, 211, 238),
                        Colors.transparent,
                      ],
                      radius: 0.8,
                    ),
                  ),
                ),
              ),
              Positioned(
                top: 280,
                left: 80,
                child: Container(
                  width: 160,
                  height: 160,
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        Color.fromARGB(32, 59, 130, 246),
                        Colors.transparent,
                      ],
                      radius: 0.8,
                    ),
                  ),
                ),
              ),
              // Foreground layout: content area and home indicator
              Column(
                children: [
                  const SizedBox(height: 12),
                  // Main content area fills the remainder of the frame
                  Expanded(
                    child: child,
                  ),
                  // Home indicator: a rounded bar at the bottom
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Center(
                      child: Container(
                        width: 120,
                        height: 4,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
