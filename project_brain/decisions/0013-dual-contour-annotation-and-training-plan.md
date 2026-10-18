# ADR 0013 — Model-independent dual-contour labels and staged training delivery

Date: 2026-10-18
Status: DESIGN PROPOSAL / IMPLEMENTATION PENDING. Model selection NOT verified by measured error.
Follows: [ADR 0012](0012-dual-border-precision-contour-contract.md)
Traceability: [PROJECT_REQUIREMENTS.md](../PROJECT_REQUIREMENTS.md) BORDER-01/02/03/04 and FACT-01/02/03/04

## Operator objective and existing value

The operator wants to spend time collecting accurate card annotations while the broader
enterprise refactor continues. Preserve the existing Annotation Studio real YOLO
2-class inference and manual rectangle-correction workflow as a **coarse initialization**,
the existing Dataset Studio JPEG/COCO detection export as a **separate bbox task**,
and Phoenix's real Queue/isolated worker/runtime/provenance. Do not attempt to turn
five-column YOLO bboxes into verified masks by polygon interpolation.

The required final measurement is the *physical card edge* versus the
*printed graphic border*, across 24 full-side stations (5 top, 5 bottom,
7 left, 7 right) including skew. Output accuracy, not a successful
training run, is the definition of success.

## Source-verified gaps (2026-10-18)

- `src/modules/annotation_studio/plugins/border_detection_plugin.py`:
  BorderAnnotation stores x1/y1/x2/y2 and emits a five-value YOLO bbox;
  YOLO inference and hand editing are useful, but not true contour labeling.
- `src/shared/dataset_tools/yolo_to_maskrcnn_converter.py`:
  includes both a genuine polygon *parser* candidate to reuse and a
  bbox-derived `create_refined_polygon` path. The latter is not GT.
- `src/modules/annotation_studio/plugins/prompt_segmentation_plugin.py`:
  has an actual stub-mask generation path; those masks are not GT.
- `src/shared/dataset_tools/phoenix_contract.py`:
  only registers the exact bbox pair `border_detection_2class` +
  `YOLO11s Optimized`. This cannot silently become a contour job.
- `src/modules/phoenix_trainer/trainers/unet_trainer.py`:
  photometric surface-defect-specific, six input channels RGB+normals,
  with binary defect masks. NOT a drop-in 3-channel dual-contour trainer.
- `src/modules/phoenix_trainer/trainers/detectron2_trainer.py`:
  real Mask R-CNN implementation which rejects bbox-only COCO;
  PointRend integration and actual AMD ROCm acceptance not yet verified.
- `src/shared/truscore_system/twentyfour_centering.py`:
  resamples 48 points then chooses a short single segment per side;
  failed ray intersections become 0 and ratio() can return 50:50.
  Requires a full-side geometry repair and explicit invalid results.
- Older `src/revolutionary_border_calibration.py`, converter polygon
  utility methods, `Checkplz/` and Git history remain archaeology sources
  before extracting/replacing any UI or algorithm.
- Full app acceptance goes through `run_truscore.py` per bootstrap.

## Canonical annotation contract — PROPOSED `dual_contour_v1`

A **versioned, editable source record** independent of model/export format,
one per card image, distinct from existing YOLO bbox export:

- Source image SHA-256, exact saved-image dimensions, path or stable ID;
  orientation, applied rotation/affine transform, and declared coordinate
  frame. The authoritative contour vertices are floating-point **saved-
  image pixel coordinates**; store transform back to original capture
  frame when available.
- Two independent semantic components:
  `outer_border` (physical card edge, class 0) and
  `graphic_border` (printed inner border, class 1).
- Each contour is an **ordered closed path** of editable vertices, with
  user-identified semantic corners TL/TR/BR/BL and four ordered *side
  arc ranges*. Do not infer side arcs from extreme short segments of a
  heavily resampled perimeter. Preserve all original vertices.
- Each component records initialization source
  (real YOLO rectangle seed, manual trace, optional segmentation proposal),
  human-reviewed approval and confidence/occlusion/ambiguity flags.
  A generated or bbox-expanded contour must **never** have
  `human_reviewed=true` unless it was actually reviewed/edited and
  accepted. Do not fabricate confidence/precision.
- Explicit status for unavailable/partial/ambiguous graphic borders:
  do not invent closure or interpolate missing evidence as labeled truth.
  Such images may be retained but cannot silently enter supervised
  full-contour training or 24-station quality claims.
- Schema and storage must validate finite, in-bounds coordinates; no
  degenerate/self-intersecting paths; distinct 2-class identity; valid
  corner-side ordering and topology; inner/outer geometric relationship;
  source-image hash and dimensions; and actual approval status.
  Dataset acceptance requires complete independent reviewed contours.
- Keep `dual_contour_v1` in its own shared dataset-tool contract, not
  as changes to existing 5-column YOLO label semantics. Version export
  schemas and tests; allow deliberate schema migration, not silent
  reinterpretation.

## Annotation Studio user workflow

1. Load card and existing genuine two-class YOLO `.pt` inference.
2. Initialize two editable four-corner *proposals* from the boxes,
   visibly marked coarse/unreviewed.
3. Trace and correct the **actual** outer and inner graphic paths
   independently: corner drags, insert/remove/move intermediate
   vertices, per-side zoom/magnifier, overlay visibility and undo.
   Optional real edge snapping / SAM proposals require explicit
   provenance and human review; no stub mask outputs.
4. Save/reopen exact pixel-space paths, transform provenance, two
   reviewed flags, and validation/QA state. Preserve the old bbox
   annotation/export mode; do not disrupt proven 3-target export.
5. Mark only fully reviewed contours as precision training ready.

## Dataset Studio export adapters from ONE source of truth

Canonical dual_contour_v1 → independently validated adapters:

A. **COCO instance segmentation**: two overlapping nested real
   foreground instances, category IDs outer/graphic, complete source
   polygons, derived bboxes, accurate areas and provenance;
   usable for Mask R-CNN / PointRend. Detection-only COCO remains a
   separate purpose and must not claim true segmentation.

B. **High-resolution RGB boundary/region supervision**:
   paired native-resolution outer/graphic binary masks (independent,
   so nested foreground regions may overlap), optional boundary-distance
   targets from the same reviewed original contours, with image-space
   alignment and manifest. Avoid one mutually exclusive 2-class map
   that would discard the fact that both contours enclose nested regions.

C. **YOLO-seg polygon adapter** (optional benchmark): two class-indexed
   normalized polygons from reviewed originals. Never serialize
   detection-format bbox rows and call them masks.

Enforce exact model/dataset pairing in Phoenix handoff registry and
validate mask fidelity with round-trip contour comparisons. Split at
card identity / acquisition-session level to prevent train/held-out
leakage. Keep source-resolution record even if training uses tiles or
resizes; ensure inverse geometric mapping during evaluation.

## Primary trainable model / benchmark sequence

1. First engineering target: **3-channel RGB high-resolution dual-
   contour segmentation baseline**, using PyTorch/ROCm, independent
   boundary/region predictions, native-resolution patch/tiling
   with context and geometrically validated reassembly.
   Implement as a new explicit Phoenix trainer/adaptor; do not
   repurpose the photometric 6-channel U-Net to hide task mismatch.
   U-Net-like high-resolution architecture is a practical initial
   baseline, **not** a predetermined best final model.
2. First boundary-aware comparative model: **Detectron2 Mask R-CNN
   + PointRend**, with genuine COCO masks and explicit dependency/
   AMD/ROCm tests. This is the adopted research priority in ADR 0012,
   not an already-working Phoenix mode.
3. Optional later comparisons: YOLO11-seg and Mask2Former/high-res
   alternatives, using the **same held-out contour dataset**. SAM2/3
   may assist proposals only when operationally supported and reviewed;
   BoxInst/Box2Mask weak supervision cannot substitute verified GT.
4. Select deployment architecture only after fair evaluation on
   native-source-image dimensions, memory/latency and geometric
   evidence. Avoid assuming a candidate wins by name.

## Measurement engine and acceptance

Provide a model-independent geometry interface:
`outer, graphic, source_transform, calibration → 24 measured station
results + centering/skew metrics + validity/reason + overlay evidence`.

- Preserve four explicit semantic side arcs and corner landmarks.
  Distribute 5/5/7/7 stations over each COMPLETE respective outer side,
  not a single segment from the 48-point resample.
- Intersect inward, locally validated normals with the corresponding
  graphic side; never accept backward/incorrect-side crossing without
  a physically validated rule.
- Failed/missing intersections, invalid enclosing geometry or unusable
  calibration yield **invalid/unavailable**, NEVER a zero distance or
  misleading 50:50 centering.
- Distinguish image-pixel offsets and calibrated mm. Confirm printed
  card dimensions/scan metadata before using physical mm outputs.
- Test on deterministic synthetic centered/offset/skewed/rotated cards,
  reversed winding and missing edges; then real independently reviewed
  held-out scans with repeat-acquisition conditions.
- Evaluate boundary displacement (px and calibrated mm), per-station
  distance and ratio error, contour quality, skew-angle error, failure
  coverage and repeatability. Report mean, tails and worst-side
  failures; mask IoU alone is not precision evidence.
- Numerical acceptance thresholds remain TBD until independent
  reference measurements are defined and approved.

## Small-step implementation gates

**Gate 1:** isolated dual_contour_v1 schema, validator and deterministic
geometry unit tests; no UI regression. Compare historical geometry and
converter utilities before writing replacement code.

**Gate 2:** Annotation Studio dual-contour editing and provenance with
real cards; existing bbox workflow remains available.

**Gate 3:** Dataset Studio mask/COCO adapters + visible QA + exact
Phoenix handoff metadata, negative tests for stub and bbox-only masks.

**Gate 4:** Native ROCm Phoenix 3-channel baseline smoke train with
real forward/backward/checkpoints and run provenance; PointRend bench
only after concrete environment and genuine mask validation.

**Gate 5:** Held-out 24-station error comparison and end-to-end
`run_truscore.py` acceptance, then model selection.

No code change, model accuracy or workstation verification is claimed
by recording this proposed implementation plan.

## Relevant external reference evidence

- PointRend boundary refinement: https://ai.meta.com/research/publications/pointrend-image-segmentation-as-rendering/
- COCO/YOLO-seg polygon label format: https://docs.ultralytics.com/datasets/segment
- Boundary IoU vs mask IoU: https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html
- AMD ROCm/Radeon framework support requires version-specific verification:
  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html
