# ADR 0006 — Annotation Studio Inference Request Lifecycle

Date: 2026-10-16
Status: Accepted for code-level regression coverage; real GUI runtime acceptance pending
Scope: Active `src/modules/annotation_studio/` implementation

## Context

A production Annotation Studio session with Radeon RX 7900 XTX / ROCm
confirmed the actual `revolutionary_border_detector.pt` delivered outer
class 0 and inner class 1 predictions and four cards exported accurately
under the three established training-format folders.

The full `src/Logs/modular_annotation.log` was inspected:
- 4 image load notifications;
- 6 actual model inference requests (3 for the first card);
- 671 redraws plus 1,348 per-annotation draw logs, which were mistakenly
  perceived as hundreds of detections.

The original automatic timer fired ~500ms after first-card load, but the
first genuine ROCm inference took ~7.4s. Later images returned predictions
quickly. The first card did not fail to schedule; it lacked busy feedback.
There was no evidence justifying a change of model, a CPU fallback, or
loss of the important redraw frequency used for manual box editing.

## Decision

Use **one real detection implementation owned by
`BorderDetectionSettingsWidget.run_detection`** for automatic and
manual requests:

- The Run Detection control dispatches `source="manual"`.
- The plugin's single half-second timer dispatches `source="auto"`.
- Track per-load monotonically increasing tokens, not only image paths,
  to reject delayed callbacks from stale loads or repeated same-path loads.
- Once a manual or automatic inference completes successfully for a
  load token, later automatic callbacks for that token do nothing.
  Explicit manual re-detection stays allowed after completion.
- Hold a busy flag and disable Run Detection while a real inference
  is running. Overlapping requests fail explicitly without launching
  a second model pass.
- Show `Detection queued (automatic)` and `Detecting... (auto|manual)`
  so a long first GPU call is not mistaken for a failure. Process
  UI paint events (excluding user input) just before inference.
- Log source, card path, observed duration, prediction counts and
  failure tracebacks. The fast overlay redraw itself produces no
  per-frame INFO messages.
- Continue to use the actual RGB/BGR image array at its original
  dimensions; Ultralytics owns model preprocessing. Never log invented
  fixed source dimensions.
- Release the busy flag and re-enable the manual button on both success
  and failure.

## Rejected alternatives

- Changing to CPU or a different model as an automatic workaround:
  unnecessary, violates real-model runtime continuity.
- Silencing or throttling overlay redraw: harms precise human correction.
- Rewriting detection around background QThreads without an established
  need: risks Qt thread ownership, current-image changes, and stale results.
- Building a second inference path to handle first-card "warmup":
  adds duplicate ownership and may overwrite user corrections.

## Verification

`tests/annotation/test_detection_request_lifecycle.py` exercises the
active real method bodies with fake model and GUI controls to verify
request-source tagging, busy-state visibility, re-entrant duplicate
rejection, finally-based cleanup on failure, retry, token-based
deduplication after manual inference, and same-path reload safety.
Existing Annotation Studio tests verify model ownership, overlay
colors/handles, three-folder class mapping, and four real exported
card files. GitHub Actions 38983844204 passed **30 Annotation Integrity
tests** after the implementation changes.

**No claim that the new busy indicator or UI event processing has
passed full live PyQt6 + ROCm GUI acceptance yet.**

## Next runtime acceptance

From `run_truscore.py` under the known-good GPU library environment:

1. Choose Border Detection and load a test card.
2. Confirm `Detection queued (automatic)` then visible
   `Detecting... (auto)` during first ROCm initialization.
3. Confirm one outer/one inner editable box after detection.
4. Click Run Detection again deliberately after completion and
   confirm it still re-runs once.
5. Confirm Space saves all three target datasets correctly and the next
   card automatically receives fresh predictions.
6. Inspect source-tagged application logs; there must not be a
   flood of overlay-only INFO lines.

Hardware-neutral model ownership stays intact; library environment
cleanup for ROCm remains a separate startup/environment concern.
