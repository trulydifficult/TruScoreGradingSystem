# Annotation Integrity Gate — Annotation Studio → YOLO → Dataset Studio → COCO → 24-point Centering

Date: 2026-10-15
Status: AUDIT IN PROGRESS — TRAINING BLOCKED

## Authoritative geometry contract

User-confirmed YOLO labels: class 0 = outer physical card boundary; class 1 = inner graphic boundary.
The precision workflow supports rotated boundaries. Standard five-field YOLO
(class, center_x, center_y, width, height) does not preserve corner-specific
rotation or quadrilateral geometry.

The user supplied root images `24pointside.jpg` and `24pointside (1).jpg`
(commit ce8e8b3). The current centering implementation at
`src/shared/truscore_system/twentyfour_centering.py` accepts polygons and
measures 24 rays: top 5, bottom 5, left 7, right 7. These image references
are useful design context; they are not proof of runtime measurement accuracy.

## Verified source findings

- `src/modules/dataset_studio/dataset_studio.py::_export_coco_format`
  hard-codes every image to width=640 and height=480, uses those constants to
  convert normalized YOLO bounding boxes, and emits *no segmentation geometry*.
  This is not a faithful high-precision image geometry export and must not
  be sent to Mask R-CNN segmentation training as if it were annotated masks.
- The same export hard-codes COCO categories 1=border, 2=surface. Actual user
  contract is 1=outer, 2=inner following conversion from YOLO 0/1.
- The separate `_convert_yolo_to_coco` + `YOLOToMaskRCNNConverter` path
  is different. It can create/refine polygons from YOLO boxes, but synthetic
  polygon edges cannot recreate actual rotation that is not stored in the
  original annotation. Do not confuse the existence of converted polygons
  with ground-truth instance masks.
- `_export_coco_format` and `_convert_yolo_to_coco` match image/label
  stems. A label-image mismatch has been suspected but NOT established;
  duplicate stems, incorrect original pairing, and preview discrepancies
  require separate audit.
- `src/modules/annotation_studio/plugins/border_detection_plugin.py`
  exports standard five-value YOLO boxes with `ann.to_yolo_format()`.
  The active implementation includes a historical hard-coded
  `/home/dewster/Projects/Vanguard/src/shared/annotations` export path;
  investigate whether the user workstation uses this path and whether
  rotation is applied consistently to images and annotations.
- The committed `detectron2test.json` has `image_count=0`, `label_count=0`,
  and `data_path=null` despite a separate 20-image/40-annotation export.
  Determine whether it is stale project metadata or a persistence defect.

## Dataset evidence / CI

Original source YOLO files: `20image/yolo/Test001.txt` through
`Test020.txt`; a two-class box format (5 numeric fields per label line).
The 20-image export was committed as
`tests/fixtures/detectron2/coco_20_image_test.zip` (7386ab7).
The ZIP holds `annotations.json` and `images/TestNNN.jpg`. COCO
`images[].file_name` is `TestNNN.jpg`, which is valid *relative to*
the `images/` image root; the first fixture test incorrectly treated the
ZIP root as the image root. That test was corrected.

The later read-only audit compares the fixture with committed YOLO source
labels by filename and bounding-box geometry and verifies image dimensions
with Pillow. It explicitly treats the old export as a *known-defective
negative fixture*, not a successful training acceptance case.
CI outcome for that expanded audit was pending when recorded.

## Acceptance gates before training

1. One-to-one, unambiguous image ↔ annotation association verified for all
   20 image/label pairs, with visual overlays reviewed.
2. Original image dimensions equal COCO metadata; no placeholder dimensions.
3. Labels retain authoritative outer/inner class semantics.
4. Real rotated quadrilateral or polygon geometry preserved through an
   explicit, versioned annotation contract; if only axis-aligned boxes are
   present, fail explicitly for precision-centering/segmentation purposes.
5. Centroid/rotation and 24-point centering results validated against
   measured reference geometry, including non-axis-aligned examples.
6. Dataset Studio exports and Phoenix refuse unsupported or inconsistent
   geometry rather than inventing a training-ready interpretation.

Do NOT report the 20-image COCO ZIP as production-validated until these pass.
