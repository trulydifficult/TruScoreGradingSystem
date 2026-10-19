# ADR 0015 — Stable pointer-release and responsive Detection controls

Date: 2026-10-19
Status: OPERATOR-ACCEPTED REAL GUI DRAG STABILITY AND CONTROL LAYOUT; full-application acceptance pending
Scope: Existing modular Studio mouse adapter and Border Detection plugin.

## Trigger and observed issue

The operator validated imported-label editing and restart/resume
twice (5 uninterrupted, another 5 with quit/resume after 1).
Screenshot at 1485×1024 revealed the Detection controls extending
into the next Export & Batch group. The operator also reported that
a precisely placed corner/side sometimes continued shifting when
moving the mouse AFTER releasing the left button.

## Actual source defect

- `src/modules/annotation_studio/plugins/border_detection_plugin.py`:
  `on_canvas_click` sets `mouse_pressed=True` with a selected corner,
  side or whole-border drag; its `handle_release` cleared drag mode
  but did **not** set `mouse_pressed=False`. Click hit tolerance
  also allowed the first movement to teleport the handle to the
  pointer rather than preserving its grab offset.
- `src/modules/annotation_studio/modular_annotation_studio.py`:
  mouseMove forwarded events based on a cached `_mouse_pressed`
  state without inspecting the actual `Qt.LeftButton` state, while
  mouseRelease skipped forwarding when the transformed coordinates
  were outside the image.
- The Studio's magnifier follows the cursor on hover by design.
  Viewport panning must be distinguished from edits to underlying
  border-vertex coordinates.

## Fix and ownership decision

1. Studio must own the actual Qt left-button event gate, canvas
   conversion and dispatch. Plugin must own corner/side selection,
   grabbed geometry, pointer grab offset, and final release clearing.
2. On release, whether inside/outside the card, clear plugin
   `mouse_pressed`, dragging_corner/side/border and pointer offset;
   cancel outstanding redraw timer and refresh final geometry.
3. If a release event was lost, next hover event without physical
   left-button pressed cancels stale mouse state and forwards release,
   never drag. A release after no valid press (unloaded/outside card)
   must not invoke plugin.
4. Corner/side grabbed within tolerance retain original vertex
   position until actual pointer movement by preserving cursor-to-
   vertex or cursor-to-side-midpoint delta.
5. The plugin owns its own vertical scroll area and section spacing.
   Detection's four controls are a 2×2 grid inside the Detection
   group, no fixed-height group boundary. Avoid global Studio panel
   resizing or breaking other plugins.
6. Default fresh inference threshold is **0.25** pending a real
   model precision/recall benchmark. Imported-label mode does not
   run YOLO and ignores its confidence control.

## Acceptance evidence and limitations

- Commits: eaef4d4 (scrollable UI / 2×2 grid), c8c3c6d
  (drag state/reset/grab offset), b48e676 (preserve preload
  release contract).
- GitHub Annotation Integrity workflow run 39220200300 at
  b48e676: **84 tests PASSED**. Isolated real methods tested
  with simulated QTimer, corner+side grabs, movement, release,
  subsequent hover, release outside frame, no-image startup and
  original export/request contracts. This verifies source
  behavior, not actual GUI rendering or frame-perfect pointer
  behavior on the operator's COSMIC/Wayland workstation.
- User acceptance still required using a small copied dataset.
  Check 2×2 grid at full and smaller height, no border movement
  after release when hovering, correct magnifier interpretation,
  normal YOLO inference, imported-label Tab/Space/Resume unaffected.
  If defects persist, inspect runtime events/timestamps and Qt
  mouse capture rather than claiming success without evidence.

No model, training code, dataset label geometry or grading
algorithm was modified by this fix.
