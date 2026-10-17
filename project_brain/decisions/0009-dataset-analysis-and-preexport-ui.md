# ADR 0009 — Project-scoped Dataset Analysis and Pre-export UI

Date: 2026-10-16
Status: Implemented in refactor/phoenix-foundation; source/CI verified, GUI acceptance pending

## User-observed defect

After a live seven-image Dataset Studio session, Verification displayed
7/7 images with visually plausible bounding boxes. Export/Analysis initially
reported 7 images, 216 labels, zero labeled images and zero coverage.
Clicking Refresh Analysis produced seven labeled images and 230 "labels".
The Pre-Export Verification panel showed styled rows with unreadable dark
content; the operator could not determine what it was intended to verify.

## Root cause from current source

- update_analysis_dashboard() always counted files in all three global
  Annotation Studio label directories, then added current imported label
  files or COCO annotations. 216/230 did not denote project annotations.
- Labeled images were inferred by iterating ImageCard UI widgets and
  _image_has_label(), not directly matching the selected image paths.
  UI widgets are not the authoritative project dataset.
- Dashboard initialized during UI construction but was not refreshed by
  on_tab_changed for Export/Analysis (index 4); the manual refresh button
  was the only reliable way to recalculate after imports/conversion.
- update_pre_export_verification() did populate many nested QLabel/QFrame
  rows, but in the user's screenshots their contents were unreadable.
  Its >=10 image threshold and "READY TO EXPORT" mixed quantity guidance
  with training readiness, without true mask or border-accuracy validation.
- Historical Checkplz/claude_fix/data_management.py has a conceptual
  verification_center and quality gates but no functioning current UI
  preflight implementation worth copying.

## Decision and ownership

- Introduce ONE headless, project-scoped summarizer at
  src/shared/dataset_tools/dataset_analysis.py for dashboard metrics and
  pre-export reporting; leave actual COCO conversion/output validation
  owned by the existing shared converter and active exporter.
- Count only selected image paths and their explicitly imported,
  exact-stem-matched YOLO files or corresponding current COCO annotations.
  Prefer source YOLO over its derived COCO copy; never double count.
  Reject ambiguous, empty, malformed or out-of-bounds annotation input.
- Report separate image total, matched YOLO label-file total, actual
  annotation instance count, labeled image total and coverage.
  Display detection COCO export readiness separately from Mask R-CNN
  training readiness; 5-field YOLO boxes do not prove real masks.
- Update Export/Analysis on every tab activation, including following a
  user edit. Retain Refresh Analysis for operator troubleshooting.
- Replace unreadable nested pre-export QLabel rows with a read-only,
  high-contrast QPlainTextEdit report: project type/target, image-label
  coverage, class distribution, missing/invalid pairs, quality-filter
  eligibility, detection export status, and segmentation limitation.
- Restrict high-quality counts to image cards owned by the current
  selected image set. No global caches or guessed training metrics.

## Verification and limitations

- Regression tests exercise the six real card fixtures and exact
  annotation counts, mixed COCO/YOLO deduplication, subset isolation,
  invalid/missing/duplicate labels, Test1-vs-Test10 stem matching,
  COCO-only inputs, export-tab activation and pre-export text.
- GitHub Actions Dataset Integrity validates this headlessly and compiles
  the active Dataset Studio source.
- Actual workstation rendering of QPlainTextEdit, tab transitions,
  readability, and end-to-end dataset export STILL require user GUI
  acceptance. No valid masks, pixel-precision borders, or GPU training
  were inferred by this slice.
- Reporting detection readiness does not replace the exporter's own
  validation at time of write. Generic dataset types beyond current
  border/graphic five-field YOLO detection need separate acceptance.
