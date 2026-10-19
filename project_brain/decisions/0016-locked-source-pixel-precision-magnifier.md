# ADR 0016 — Magnifier is the source of truth for border annotation

Date: 2026-10-19
Status: OPERATOR-ACCEPTED REAL GUI MAGNIFIER PLACEMENT/FREE-FOLLOW; source-render CI PASS; physical GT/model accuracy not established

## Operator requirement

Operator uses the main card only to grab corner/edge handles,
and looks at the magnifier to determine actual physical border
alignment. The zoomed view MUST render the true persisted
image-space polygon against exactly corresponding source-image
pixels. The previous assistant interpretation of a following
mouse viewport as sufficient was rejected: moving a view cannot
excuse stale or misleading geometry. Accuracy is essential for
a 24-station geometry system.

## Source defect

The previous `on_image_mouse_move` invoked `update_magnifier`
BEFORE `current_plugin.handle_drag`, guaranteeing the magnified
line lagged a movement behind the edited quad. On release,
the magnifier was not repainted from final geometry. The old
magnifier repeated its own pixmap-scaled canvas transforms
rather than relying on the Studio's authoritative image
coordinate mapper. A full-screen yellow crosshair at the
cursor could be mistaken for the edited handle after viewport
motion. The viewport traveled with hover after release.

## Accepted design

1. A single pure `MagnifierViewport` in
   `src/modules/annotation_studio/magnifier_geometry.py`
   converts source image-pixel coordinates to 2× or 4× zoomed
   output pixels. Its origin/scale are used both for source
   crop and live annotation points; deterministic round-trip
   with subpixel floats is tested.
2. Render from the exact original/current rotated image pixels
   (same image rotation frame as Studio/main image), with
   nearest-neighbor pixel-preserving enlargement and explicit
   edge padding. Crop originates from shared
   `canvas_to_image_coords` when unlocked, never from an
   independent image-label/pixmap transform.
3. After grabbing a selected corner/side, lock the viewport
   on its actual source-pixel location. The border plugin owns
   `precision_target_point`, which follows its currently
   selected independent vertex or edge midpoint and survives
   mouse release. The Studio owns the locked viewport.
4. Update polygon geometry *before* repainting the magnifier
   during each drag. Keep viewport stationary for precision
   edits, with minimal automatic pan only if a dragged target
   nears its edge. Repaint once after release and maintain
   the same crop through subsequent cursor hover.
5. The crosshair is short and marks the CURRENT live edited
   vertex/midpoint, not an earlier cursor center. An explicit
   `Follow Cursor` control unlocks inspection; clicking
   another handle relocks. Existing 2×/4× controls remain.
6. Reset anchor on image change and invalidate old rotated
   image cache pixels when loading another card. No import/
   export, inference, or training model logic was replaced.

## Verification tiers

- GitHub Annotation Integrity run
  https://github.com/trulydifficult/TruScoreGradingSystem/actions/runs/39227192988
  at 069c87c: **97 PASS**. Tests use real cv2+numpy
  zoomed image pixels and a known original pixel signature,
  actual quad line pixel colors, and source-method press,
  drag, magnifier repaint, release and hover sequencing.
  Also tested affine viewport at 2×/4×, padding and panning.
  Older source contract assertion updated for the unified
  magnifier viewport. These results prove source path
  consistency, NOT real-screen pixel-perfect user experience.
- Still required: operator's PyQt/Wayland workstation test on
  a few safe copied cards with fine image borders and optional
  rotation, including long drags, 2×/4×, freeze on release,
  Follow Cursor unlock, card switch and existing-label resume.
- Model-predicted contour precision, physical card-edge true
  positions and centering station accuracy are NOT inferred
  from correct raster display geometry and remain separate
  ground-truth/benchmark gates.


## 2026-10-19 refinement — operator-verified accuracy, release-to-follow UX

After pulling the prior implementation, operator confirmed the
magnifier correctly displayed the placed lines and did not allow
post-release vertex drift. They requested that the viewport unlock
automatically AFTER release, because their primary verification is
scanning the graphic boundary at 4× without clicking anywhere.
The ADR's previous permanent lock-after-release policy is therefore
**superseded** by this newer approved interaction contract:

1. During press/drag, anchor the viewport to the selected real
   source-image handle. Render edited geometry after drag events.
2. On release, first repaint the finalized line in the anchored
   crop. THEN set `_magnifier_locked_center=None`. Subsequent
   button-free mouse motion follows the cursor via the canonical
   Studio transform; no border coordinates change without a new
   explicit handle press/drag.
3. Cancel lost releases similarly, including the anchor.
4. Magnified quad strokes are 2 OUTPUT pixels thick (previously 1)
   but vertices are NOT moved, rounded earlier or thickened in source
   image/exports. The exact image sampling remains nearest-neighbor.
5. On a 2560×1440 workstation, allow plugin settings to fill
   unused left-column height. The Studio fallback
   `controls_frame` is now vertical-expanding with layout
   stretch 1, rather than an empty stretch below. Retain the
   inner scroll area for small windows and other plugins.

Verified headless: Annotation Integrity run 39230520209,
100 PASS, commit 86c44e2, including direct press→drag→release
→hover and lost-release tests, 2px OpenCV overlays and plugin
layout ownership. The user still needs to accept the resulting
GUI feel and exact frame height on their desktop.
