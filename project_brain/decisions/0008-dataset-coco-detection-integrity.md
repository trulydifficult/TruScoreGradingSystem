# ADR 0008 — Dataset Studio COCO Detection Geometry and Training Truthfulness

Date: 2026-10-16
Status: Implemented on refactor/phoenix-foundation; source-method CI checked; workstation GUI acceptance pending

## Problem

Annotation Studio currently exports five-field YOLO annotations for two classes:
0=outer card border, 1=graphic/inner border. These are **axis-aligned
bounding boxes** on the saved exported image. They do not encode physical
corner quadrilaterals, 24-point centering samples, rotation provenance, or
ground-truth pixel masks.

Before this change Dataset Studio had two distinct conversion paths:

- Manual _export_coco_format forced 640x480 dimensions, misnamed categories
  border/surface, and dropped all segmentation.
- Background ConversionWorker used YOLOToMaskRCNNConverter with synthesized
  48-point polygons derived from boxes, returned a conversion-result wrapper
  (containing coco_data and conversion_stats), and wrote that wrapper instead
  of a top-level COCO JSON object. It also reported zero images by incorrectly
  reading the wrapper's top-level keys.

The preview additionally treated same-stem global annotation-workspace files
as active project labels, preferred potentially stale COCO data over newly
imported source labels, and described filename matches as "verified".
Export UI advertised unimplemented formats and train/validation splitting.

## Decision

1. Preserve the existing Annotation Studio three-target image/YOLO export.
   Do not change its proven card-rotation path here.
2. Use the existing shared YOLOToMaskRCNNConverter, with an explicitly selected
   bbox_only=True mode in both active Dataset Studio conversion routes.
   Preserve true per-image pixel dimensions, bounding-box coordinates,
   class order and original geometry. Do not synthesize segmentation in this
   path. Class IDs are COCO 1=outer_border, 2=graphic_border.
3. Mark box-only geometry in COCO info/attributes. Refuse malformed or
   out-of-bounds normalized boxes, unknown classes, missing labels,
   ambiguous labels and incomplete per-image conversion.
4. Write proper top-level COCO JSON in ConversionWorker and return the
   actual coco_data object to the Verification tab.
5. Make Mask R-CNN format conversion and Phoenix Detectron2Trainer reject
   missing masks and legacy approximate polygons; do not automatically
   route these datasets to a different model family. Genuine segmentation
   must be acquired and verified separately.
6. Prevent unsupported manual export selections and inert split options
   from falsely reporting success; label-only COCO exports are explicitly
   not standalone Phoenix datasets.
7. Verify against only current project labels/COCO data, not shared global
   annotation folders. Invalidate cached conversion/labels when source images
   or imported labels change. Prefer imported source YOLO over stale COCO
   for matching names; flag mismatched COCO dimensions and distinguish a
   filename match from verified border accuracy.

## Scope / limitations

- Existing legacy polygon approximation utilities still exist for code
  archaeology/compatibility; **they are not ground-truth masks**. Any remaining
  call sites that use them require audit before training.
- No 24-point centering, surface quality, or segmentation ground truth was
  invented. No rotated quadrilateral geometry was inferred from boxes.
- File-name/stem matching is still weaker than immutable source-image IDs,
  hashes and annotation-frame provenance; design those before expanding
  precision data generation.
- Mask R-CNN training is explicitly blocked on box-only data; valid bbox
  detection data is not the same as valid segmentation data.
- Production ROCm trainer behavior, real GUI preview, exported ZIP/folder
  workflow, dataset splits, and multi-job Phoenix training remain independent
  acceptance gates. The existing SAM conversion and non-COCO export paths
  require their own audit.

## Verification

New Dataset Integrity CI:
- executes real YOLOToMaskRCNNConverter bbox-only geometry against six committed
  Annotation Studio image/label pairs from fe3f248;
- executes the active ConversionWorker.run and manual _export_coco_format method
  bodies through headless AST isolation with real Pillow JPEGs;
- rejects missing/out-of-bounds labels and synthetic/absent instance masks;
- checks Detectron2 load_dataset early rejection;
- tests that unimported global same-name labels are not counted as verified.

The first full run passed 12 tests after correcting a mistaken test exception
regex (not a production defect). Follow-up CI covers preview invalidation.
Do not equate these source-method checks with workstation Qt GUI acceptance.

## Next tasks

1. Workstation GUI: import a controlled sample of the six known-good saved
   images and their matching combined_dual_class labels. Verify the preview
   against the source YOLO coordinates before conversion and after conversion.
   Confirm actual COCO dimensions, 1/2 categories, and absence of mask claims.
2. Audit all Dataset Studio export variants, validator paths and older scripts
   for remaining silent success/fallback behavior and coordinate-frame drift.
3. Define durable source image/hash, applied rotation, source polygons and
   human-correction provenance. Add actual polygon/mask capture and visual
   acceptance before enabling precision Mask R-CNN training.
4. Continue Phoenix with a real validated dataset plus ROCm training/evaluation
   and unattended multi-job Queue acceptance, not a fabricated 20-image fixture.
