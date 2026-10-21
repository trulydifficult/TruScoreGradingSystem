# ADR 0012 — Dual-Border Precision Contours Are the Primary Training Target

Date: 2026-10-17
Status: REQUIREMENTS CORRECTION ADOPTED; model implementation/selection pending benchmarks
Supersedes: ADR 0011 only where it implies YOLO11s bbox detection is the
operator's primary dual-border/24-point centering model.

## Requirement correction

The operator clarified that the **purpose** of dual-class border
annotation and Detectron2 training is to independently follow the
*physical card edge* and *printed graphic/inner border* accurately
along their actual paths. TruScore must evaluate 24 station distances,
5 per top and bottom edge, 7 per left and right, including horizontal/
vertical centering and visible skew. The original YOLO box labels
were a bootstrap for prediction/manual correction, not sufficient
measurement-grade output.

Source proof: root 24pointside.jpg and 24pointside (1).jpg;
src/shared/truscore_system/twentyfour_centering.py.

## Decision

1. The 24-point task's primary output contract is **two true,
   independent, ordered, native-image-coordinate contours** with
   provenance and validated geometry. Do not claim this is provided
   by COCO bbox detection, OBB rotations or inferred synthetic
   masks. The source saved-image rotation must be preserved.
2. Preserve existing real five-field YOLO → COCO bbox conversion and
   the source/CI-verified seven-image export. These are valid for a
   **coarse detection** task, and are not mask ground truth.
3. ADR 0011's new YOLO11s adapter remains an **optional auxiliary
   coarse locator**, not the designated final dual-border centering
   pipeline. Do not ask the operator to perform a YOLO11s 24-point
   precision acceptance as though it satisfies the requirement.
   No working behavior is discarded before comparing alternatives.
4. The existing Detectron2 Mask R-CNN path will not train
   five-field YOLO→COCO bbox labels as verified segmentation.
   To train the Mask R-CNN mask head, obtain genuine masks/contours.
   A separate explicitly researched box-supervised segmentation
   method (e.g., BoxInst) could use existing boxes, but has to be
   named, implemented, benchmarked and evaluated on real withheld
   contour truth. It is NOT equivalent to converting boxes to masks.
5. Research/benchmark boundary-targeted options, with Detectron2
   PointRend a priority because it directly refines mask edges
   in the existing framework, Mask2Former/high-resolution semantic
   segmentation as alternatives, and SAM2/3 as optional assisted
   labeling (not unreviewed ground truth). Standard YOLO11s
   DETECTION is not the target; YOLO11-SEG is a separate mask
   architecture that may be benchmarked only with genuine masks.
6. Before launching a precision model train, build a truthful
   human-editable two-contour annotation format. Record original
   image hash, rotation/applied affine transform, class/contour
   label semantics, corners and station mapping, source and
   human edits. No placeholder masks or fabricated confidence.
7. Audit and repair twentyfour_centering.py geometry separately:
   currently the 48-point re-sampling feeds a side-selector
   that chooses a single small segment rather than full edges,
   ray failures produce zero distances and may default to 50:50,
   and winding/DPI behavior needs physical reference tests.
   These are **candidate demonstrated source defects**, not
   accepted real-world measurements. Verify with deterministic
   synthetic skew/rotation masks plus actual card examples.
8. Acceptance should be based on held-out real outer+graphic
   contour accuracy at all 24 stations, boundary/corner/skew
   physical-distance error and repeatability. COCO box mAP,
   mask IoU alone, a saved checkpoint or seven-image smoke training
   do not prove the high-accuracy 24-point result.
9. Keep existing Phoenix Queue/Trainer architecture; add a separate
   validated contour training capability contract only when data
   and model-specific ground-truth requirements are met.

## Next gate

- Source geometry contract and 24-station corrected measurement
  algorithm with reference diagrams and automated true-contour tests.
- Design/implement human-verified dual-contour annotations.
- Small manually reviewed mask/contour corpus; bench models against
  actual spatial error on workstation ROCm rather than guessing
  the winner.
- Record model accuracy only after held-out validation.

Full evidence and research URLs:
[DUAL_BORDER_PRECISION_GEOMETRY_AUDIT.md](../DUAL_BORDER_PRECISION_GEOMETRY_AUDIT.md).
