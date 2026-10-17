# Seven-card Dataset Studio export integrity audit

Date: 2026-10-16 (operator export); repository verification 2026-10-17 UTC
Branch: refactor/phoenix-foundation
Real export commit: 8eee8ff0d58ced4cf0075b7ee6ab69166eea6c5e
Export source folder: src/modules/dataset_studio/exports/Dataset_20261016_230734/
Evidence: actual seven JPEG Git blobs, actual annotations.json, source
combined_dual_class JPEG/YOLO Git blobs, new headless JPEG/COCO regression test

## Purpose / GUI evidence

The user confirmed that Dataset Studio's updated export/analysis UI was
usable, exported a seven-card detection dataset from the real application,
and committed the actual export for independent verification. This is
the first full committed real Dataset Studio manual COCO export accepted
at the **file-integrity** level after ADR 0008/0009 corrections.

## Actual files

- images/image001F.jpg through image004F.jpg (four files, 1632x2284)
- images/image092F.jpg through image094F.jpg (three files, 1640x2291)
- annotations.json (7 image records and 14 annotation records)
- No dataset_config.json; that is normal for the manual export, NOT
  a directly queue-loadable Phoenix job definition.

Each exported JPEG's Git blob SHA equals the corresponding original
src/shared/annotations/combined_dual_class/images/<stem>.jpg SHA, and
also equals the JPEG in the corresponding graphic_border_model and
outer_border_model source folders. This proves byte identity (no
substitution, resampling or re-encoding).

## Actual COCO evidence

- Top-level keys: info, licenses, images, annotations, categories.
- Category ID 1: outer_border; category ID 2: graphic_border.
- Exactly two annotation rows per image (one of each class).
- Bounding-box format: [x, y, width, height] in source JPEG pixels.
- All 14 bbox coordinate values match the corresponding original
  normalized five-field YOLO rows exactly after expected affine
  scaling. Maximum coordinate difference measured: 0 px.
- Image IDs unique; annotation IDs unique; image/category references
  valid; areas equal width*height; all boxes in image bounds.
- Each graphic border box is inside the outer border box.
- COCO "info.annotation_geometry": axis_aligned_bbox.
- COCO "info.segmentation_ground_truth": false.
- Per-annotation attributes confirm source=yolo_bbox and false mask
  provenance. No segmentation polygons/masks are present.

## Reproducible automated validation

tests/dataset/test_real_seven_card_export.py:
- parses actual committed annotations.json;
- decodes the actual exported JPEGs with Pillow and checks dimensions;
- SHA256-compares each exported JPEG against its corresponding
  combined_dual_class JPEG;
- reads the original YOLO files and validates all 14 COCO bounding
  boxes, classes, areas and containment.

.github/workflows/dataset-integrity.yml now sparse-checks out all
seven exported JPEGs plus seven original source JPEG/YOLO files.
CI run 39002287868: SUCCESS, **43 tests passed in 0.68 s**.
These are real-file validation tests, not synthetic model metrics.

## Remaining limitations / disposition

**Verified:** manual Dataset Studio COCO *detection* export data integrity.
No need to repeat the export unless there are new product requirements.

**Not verified:** independently measured physical border-edge accuracy,
rotation/corner quadrilateral provenance, mask or segmentation ground
truth, native Qt export UI across all paths, and actual training model
accuracy. The previous visual user review suggests plausible boxes;
it is not proof of subpixel physical edge precision.

**Training:** the current Phoenix DETECTRON2 path is Mask R-CNN instance
segmentation and correctly rejects bbox-only data. Seven images can
support a carefully identified integration smoke test for a compatible
bbox detector, but they are not a statistically meaningful precision
training set. Do not silently substitute a box detector or fabricate
masks to make the current Mask R-CNN path accept this export.

**Minor metadata caveat:** the converter still uses a legacy
"YOLO to Mask R-CNN Conversion" text *description* in COCO info despite
explicitly marking annotation_geometry=axis_aligned_bbox and
segmentation_ground_truth=false. Treat the typed geometry/provenance
fields as authoritative; update the legacy description in a future
focused metadata cleanup if desired.

Next priority remains deliberate Phoenix dataset/algorithm routing
and the disconnected Export to Trainer / partly wired Export to Queue
buttons, as audited in DATASET_TO_PHOENIX_HANDOFF_AUDIT.md.
