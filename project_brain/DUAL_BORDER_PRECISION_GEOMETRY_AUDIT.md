# Dual-Border Precision Geometry and 24-Point Centering — Audit

Reviewed: 2026-10-17
Status: REQUIREMENTS/RESEARCH AUDIT, not a completed training implementation
Authoritative source: refactor/phoenix-foundation
Decision: ADR 0012 (precision geometry corrects ADR 0011 scope)

## Operator's actual requirement

The dual-border model is not an ordinary object locator. It must
recover **independently identifiable physical outer-card contour and
printed graphic/inner contour** in native source-image coordinates,
with enough geometric accuracy to measure card centering and skew
at 24 positions along their respective full edges.

Measurements: TOP=5, BOTTOM=5, LEFT=7, RIGHT=7. Source image proofs:

- root `24pointside.jpg`: five illustrated vertical measurement
  stations across the card width, drawn through visible printed border;
- root `24pointside (1).jpg`: seven horizontal measurement stations
  from bottom to top, positioned on both left and right edges;
- `src/shared/truscore_system/24point.png` also exists in the repository
  (not independently decoded in this audit).

The exact diagram labels and station numbering are visual-reference
materials; the existing software uses the top/bottom/left/right
grouping above. Do not assume axis-aligned rectangles are suitable
geometric substitutes for real tilted or irregular boundaries.

## Current code and historical state

`src/shared/truscore_system/twentyfour_centering.py` (553 lines)
already ACCEPTS real outer/graphic polygons. Its
`yolo_box_to_polygon` helper explicitly describes rectangles as
temporary input. The current method constructs 48 perimeter sample
vertices, derives 24 outer-to-inner ray intersections and calculates
left-right/top-bottom ratios.

**Critical algorithm-design issues requiring verified repair**:

1. `CenteringAnalyzer.__init__` first calls
   `resample_polygon_perimeter(..., 48)` for each border. Then
   `poly_side_indices` for any polygon with >4 vertices selects
   the *single* edge whose midpoint is most extreme in x/y. Thus
   `sample_points_on_side` distributes its 5 or 7 measurement
   stations across ONE short 48-vertex segment, not across the
   complete top/side/bottom boundary. Geometric logic is
   incompatible with the diagram's full-side station spacing.
2. Image-coordinate winding and `signed_area`/inward-normal
   assumptions need tests with skewed/rotated source polygons.
   Reverse-ray fallback is not a substitute for correct orientation.
3. Missing inner-border intersections currently become **0.0**
   distances; `ratio()` returns 50:50 when both mean gaps are
   zero. This can falsely imply perfect centering on a failed
   measurement. Fail explicitly or mark measurement as invalid.
4. `estimate_dpi_from_geometry` computes card width/height from
   these short selected segments, not full physical border extent
   when passed resampled polygons. This invalidates assumed
   pixel-to-millimeter geometry; scanner metadata alone is not a
   substitute for validated physical-size calibration.
5. The code does not establish a tested connection from these
   24-point results back to a model-generated genuine contour;
   the currently accepted 7-card COCO files have two **axis-aligned
   bboxes**, not masks/contours/physical edge keypoints.

Existing `src/modules/annotation_studio/plugins/border_detection_plugin.py`
produces editable rectangular detection boxes. Annotation Studio
exports five-field YOLO rectangles per class. Dataset Studio
converts those rectangles faithfully to COCO **bbox** fields, but
faithful serialization does NOT magically produce mask supervision.

`src/modules/phoenix_trainer/trainers/detectron2_trainer.py`
is concretely Mask R-CNN and correctly **rejects bbox-only COCO**
with `segmentation_ground_truth=false`; training that architecture
for accurate mask output requires genuine annotation masks or a
deliberately selected and independently validated box-supervised
segmentation algorithm (not standard Mask R-CNN).

`src/modules/annotation_studio/plugins/prompt_segmentation_plugin.py`
already has a segmentation scaffold, but `_run_model_or_stub`
returns a centered fabricated rectangular mask and hardcoded
confidence when real inference is unavailable. It also contains
stub/placeholder inference branches and a subpixel UI flag.
Classification: RESEARCH/UNSAFE FOR TRAINING, not a real
mask-labeling implementation. Never promote its outputs to GT.

Historical `src/revolutionary_border_calibration.py` and
`src/shared/truscore_system/truscore_border_detection.py`
contain additional editable-border/geometry/calibration code.
These are archaeological comparison sources, NOT proof of a
working polygon-capture pipeline. Search relevant historical
implementations and Git history before replacing working logic.

## Candidate precision paths — external research

### A. Detectron2 + PointRend (first comparison baseline)

PointRend performs adaptive point refinement on segmentation
boundaries, specifically aiming to remedy oversmoothing of coarse
masks. It is an extension to the Detectron2 ecosystem rather than
a different category of problem, and was published CVPR 2020.

Source:
https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html
https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend

Requires genuine instance/semantic masks for supervised contour
training, unless an explicitly researched weak-supervision
objective is separately implemented.

### B. Mask2Former or high-resolution semantic segmentation

Mask2Former supports semantic/instance segmentation, published
CVPR 2022. The standalone official project was archived Jan 2025,
so version, dependency and ROCm maintenance concerns are real.
A native high-resolution two-boundary/region segmentation U-Net
or similar semantic model should also be benchmarked before
selecting the winner. Segmenting the entire "card" as one mask
alone is insufficient: the *printed graphic border* has distinct
semantics and may require a dedicated contour class/head.

Sources:
https://github.com/facebookresearch/Mask2Former
https://github.com/facebookresearch/detectron2

### C. SAM 2.1/3 for human-in-the-loop annotation

SAM2 releases promptable image segmentation and fine-tuning
recipes (with optional CUDA extension and installation caveats).
SAM3 also offers text/exemplar prompted segmentation, but the
official model installation documentation requires CUDA 12.6+
and the current project advertises CUDA GPUs; thus RX 7900 XTX
ROCm deployment is UNVERIFIED and must not be promised.
Foundation-model prompts can produce initial polygon proposals,
not authoritative human-reviewed contours. Graphic borders may
not correspond to natural object-mask boundaries. Reject fake
mask fallbacks and ensure the user can correct each edge.

Sources:
https://github.com/facebookresearch/sam2
https://github.com/facebookresearch/sam3

### D. Explicit YOLO11-seg vs YOLO11-detect distinction

Ultralytics separately provides `yolo11s.pt` (bbox detect),
`yolo11s-seg.pt` (instance masks) and `yolo11s-obb.pt`
(rotated rectangular box). The newly implemented Phoenix
YOLO11s adapter is **DETECTION ONLY**, and cannot produce
measurement-grade contours. The existence of YOLO11-seg does
not establish superior boundary accuracy, and OBB quadrilaterals
still cannot capture nonrectilinear printed borders.

Source: https://docs.ultralytics.com/models/yolo11

### E. Box-supervised segmentation (optional bootstrap)

BoxInst and Box2Mask can train a segmentation model with real
bounding-box supervision without inventing hard-coded
ground-truth masks. This is **weak supervision**, not standard
Mask R-CNN and not a certification of edge accuracy. A small,
independently traced polygon ground-truth evaluation set remains
mandatory to establish whether it can reliably localize both
near-parallel borders on this specialized domain.

Sources:
https://github.com/aim-uofa/AdelaiDet
https://github.com/open-mmlab/mmdetection/blob/main/configs/boxinst/README.md
https://pubmed.ncbi.nlm.nih.gov/38319771/

## Recommended architecture to benchmark, not yet accepted

1. Keep proven existing YOLO border bounding-box inference as a
   COARSE locator/annotation initialization only; it is not the
   24-point centering model.
2. New **dual-contour annotation tool**: editable independent
   outer-card and printed-graphic paths (polyline/polygon/edge
   marks), optional SAM-assisted proposals with mandatory human
   correction; preserve source JPEG hash, orientation, source
   pixel coordinate frame, contour vertex ordering, uncertainty,
   user corrections and provenance. No bbox→mask fabrication.
3. Create a reviewed, multi-card/pattern/scanner withheld
   ground-truth corpus with real contours and visible exceptions.
   Use Polygon/COCO segmentation only when faithful to actual
   hand-annotated geometry; maintain richer native
   contour/landmark schema for measurement/provenance.
4. Benchmark (at least) Detectron2 Mask R-CNN baseline,
   Detectron2 PointRend boundary-refined model, and a practical
   high-resolution semantic/edge contour model. Only add others
   when ROCm install and annotation-contract conditions hold.
   Consider weak box supervision only as labeled research.
5. Extract two physically meaningful ordered contours in image
   coordinates, rectify/card-calibrate geometry where appropriate,
   fit/regularize edges with explicit uncertainty, and sample
   the correct 5+5+7+7 full-edge measurement stations. A rotated
   box, coarse mask or its contour alone is not guaranteed to be
   metrology grade.
6. Validate on held-out card families by outer/inner boundary
   pixel MAE/p95/Hausdorff/Boundary IoU, corner/skew error,
   per-station physical-gap error, all 24 ray validity, and
   repeatability vs image rotation/scanner conditions. Boundary
   metrics matter more than ordinary box mAP or region Mask IoU.
   No invented acceptance tolerance; obtain physical precision
   requirements and calibration evidence before declaring PASS.
7. Use Phoenix's current Queue/Trainer framework with a
   separate named **contour/segmentation** dataset/model contract.
   Do NOT redirect existing Detectron2 choice to YOLO11s merely
   because YOLO11s bbox training is already implemented.

Research on why boundary-aware metrics are necessary:
https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html

## Current decision boundary

**Do not run the existing seven-card bbox-only COCO export as
standard Mask R-CNN segmentation ground truth.**
**Do not use the YOLO11s detection pilot as the user's requested
dual-border 24-point precision model.** It remains a separate
optional coarse-box model, not deleted or relabeled as the final
precision solution. No new native geometry model has been selected
or trained. Revisit decision once real contour labels and
benchmark criteria have been agreed.

Next: design true dual-contour annotation/metadata contract,
write deterministic geometry tests, correct 24-point sampling
against root diagrams, then acquire human-corrected ground truth
and benchmark real models on RX 7900 XTX.
