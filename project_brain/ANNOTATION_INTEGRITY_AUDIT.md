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

## Historical cross-check and test result

The older `src/revolutionary_border_calibration.py` preserved human-correction
metadata and a rotation-applied image property in its per-card report, but its
YOLO export still emitted axis-aligned five-field boxes. This material is a
salvage reference for human corrections and original annotation context; it is
not a proven rotated-polygon export pipeline. It remains important to inspect
other archived source before changing annotation ownership.

GitHub Actions 38822223789 and 38822219622 passed 51 unit tests and both
Phoenix Trainer/Queue GUI startup smoke checks. Those tests established that
COCO box/class values in the committed ZIP correspond numerically to the
original same-stem source `20image/yolo/TestNNN.txt` labels. They also
established the negative fixture is missing all 40 segmentation arrays and
contains at least one image whose true dimensions differ from the recorded
640x480 metadata. This is PASS of an integrity *audit*, NOT PASS of the
production dataset acceptance gate. Original YOLO box visual correctness,
per-image annotation provenance and rotated quadrilateral geometry remain
unverified.

## Current active Annotation Studio ownership (source verified)

The current application is `src/modules/annotation_studio/modular_annotation_studio.py`
(`ModularAnnotationStudio`) plus the active
`src/modules/annotation_studio/plugins/border_detection_plugin.py`.
These must be the primary implementation for repairs; the legacy
`src/revolutionary_border_calibration.py` is archaeology, not the active UI.

Actual path for border annotation saving:
`ModularAnnotationStudio.save_and_next_image()` →
`auto_save_current_annotations()` →
`BorderDetectionPlugin.auto_create_training_labels()` →
`BorderDetectionSettingsWidget.auto_create_training_labels()`.
The plugin method delegates to its settings widget, which is the real owner
of the triple YOLO export. It uses `ann.to_yolo_format(w,h)` to serialize
five-field axis-aligned boxes, losing independent rotated corners. The
`BorderAnnotation` dataclass represents x1,y1,x2,y2, not a polygon.

The studio rotation slider redraws a transformed pixel image; its
`save_rotated_image()` calls `cv2.imwrite` on the original path and explicitly
OVERWRITES the image. The plugin `on_image_rotated()` is a no-op. It is not
proven that annotations maintain the same coordinate frame as the overwritten
rotated pixels; that is a high-priority integrity risk. Do not replace this
workflow casually: isolate and test transforms and the current Space-bar save
semantics before any change. The settings widget's actual triple export hardcodes
`/home/dewster/Projects/Vanguard/src/shared/annotations`, which may differ
from the user's current checkout at `/home/dewster/Vanguard`.

Additional concern: `ModularAnnotationStudio.canvas_to_image_coords()`
converts canvas scaling/translation but does not reverse the displayed
rotation. Check with synthetic rotated coordinates and widget tests before
making claims of precise border placement.

## User-confirmed original workflow and refactor invariants

User clarification (2026-10-15): In Annotation Studio, the rotation control
was deliberately implemented to rotate the **card image**, because the outer
and inner bounding boxes were not independently rotatable. Earlier image
exports lost the image rotation, causing boxes to misalign with the image.
The adopted workaround is to **save/export the rotated image together with
the annotations in the same coordinate frame**. This is INTENDED FUNCTIONALITY,
not accidental corruption. Do not "fix" it by reverting exports to an
unrotated image, or by independently rotating boxes without designing and
testing the coordinate-frame transform.

The Space-bar operation is intentionally optimized for repetitive labeling:
(a) save the user's corrected current-card boundaries and appropriately
rotated image; (b) move to the next card; (c) run the user's **single
two-class detector** to produce an approximate outer box (class 0) and inner
box (class 1); (d) allow the user to adjust those boxes with fine control.
Preserve automatic detection and editing, not replace it with two separate
inference models. The three single/dual-class export directory variants are
**outputs**, not evidence of three separate inference models.

Current source confirmation:
- `ModularAnnotationStudio.setup_keyboard_shortcuts` connects Space to
  `save_and_next_image`.
- `save_and_next_image` persists rotated pixels on nonzero rotation,
  exports labels, resets the rotation slider, then navigates.
- `next_image_placeholder` independently invokes annotation auto-save,
  so Space causes **two** auto-save requests. Verify idempotency and
  whether second save can overwrite good data.
- `load_current_image` calls the border plugin's `on_image_changed` and
  the settings widget's `set_current_image`.
- `BorderDetectionPlugin.on_image_changed` schedules the settings
  widget's `run_detection` after 500ms **conditional on the main plugin's
  model_loaded flag**. The settings widget also has its own model state;
  verify single-model ownership and that detector warmup/loaded-state
  updates are synchronized. This is a test target, not a proven regression.
- The main studio's `save_rotated_image()` overwrites the original path,
  while the plugin writes a rotated-image/label pair to export folders.
  Rotated pixels and labels MUST be transformed once, against identical
  dimensions and provenance; verify no double rotation or mixed frames.
- `canvas_to_image_coords()` accounts for zoom/pixmap offset but does not
  invert the displayed rotation. Test mouse edits on nonzero rotation.

Required preservation/acceptance tests:
1. Nonzero-degree image rotation, manual outer/inner box edits, Space save:
   verify exported saved image and both boxes align at known pixel locations.
2. Next card auto-load invokes the existing single two-class model exactly
   once and places two initial editable predictions, preserving user changes
   on the previous card.
3. No save-time re-rotation of already-rotated image or annotations; repeated
   save must be idempotent; original source image retention/replacement
   should be explicit and reversible.
4. Source export pairing tracks image content (not just an ambiguous basename)
   and annotation-frame provenance, including duplicate stems.
5. Test zero rotation, nonzero rotation, zoom and fit mode before changing
   any coordinate transform or current Space-bar behavior.

The present precision objective is not to *automatically* replace the
workflow with rotating boxes. It is to preserve working image-rotation
normalization and add an explicit, verified coordinate-frame contract for
export and future polygon-based 24-point measurement.

## User-confirmed three-target Space-bar export contract (2026-10-16)

The normal Space-bar save must create/refresh **three image/label pairs**
under the ACTUAL current workstation checkout path:
`/home/dewster/Vanguard/src/shared/annotations/`

- `combined_dual_class/images/<stem>.<ext>` and
  `combined_dual_class/labels/<stem>.txt`: both annotations, source
  YOLO class 0=outer physical border, class 1=inner graphic border.
- `graphic_border_model/images/<stem>.<ext>` and
  `graphic_border_model/labels/<stem>.txt`: inner graphic only,
  reindexed to **class 0** for a single-class target.
- `outer_border_model/images/<stem>.<ext>` and
  `outer_border_model/labels/<stem>.txt`: outer physical only,
  class 0 for the single-class target.

All three image files must carry the SAME finalized rotation; their YOLO
boxes must match those exact pixels in the same coordinate frame.
Maintain matching image/label basenames and separate `images/` and
`labels/` subfolders; the triple-export requirement is independent
of the single two-class detection model used for the next card.

Current source `BorderDetectionSettingsWidget.auto_create_training_labels`
already implements the three export targets and per-class filtering, but
still hardcodes the **wrong old path**
`/home/dewster/Projects/Vanguard/src/shared/annotations`.
Fix to a repo-derived path as part of the guarded implementation,
rather than a new workstation-specific hardcoded path. Its saved image
and annotation coordinates still require a rotated-card acceptance test.
The per-format class-ID remapping in the single-class graphic export also
must be verified against `to_yolo_format()`: filtering an annotation of
class 1 does NOT by itself reassign it to single-class 0.

## Guarded source repair — 2026-10-16

Implemented in active `src/` on `refactor/phoenix-foundation`:

- `BorderDetectionSettingsWidget.auto_create_training_labels` computes the
  three-export directory from `Path(__file__).resolve().parents[3] /
  "shared" / "annotations"`. The current checkout location no longer matters;
  old `/home/dewster/Projects/Vanguard` is not used by this export.
- The two single-target training outputs encode class 0, including the
  inner graphic-only target; the combined output still preserves class 0
  (outer) and 1 (inner) without mutating source annotation class IDs.
- `ModularAnnotationStudio.save_and_next_image` now advances with
  `_advance_to_next_image(save_current=False)`, avoiding the second
  annotation/image export after reset_rotation. The original
  `next_image_placeholder()` continues delegating with
  `save_current=True` for ordinary navigation. The Space shortcut and
  model-on-image-change path were not changed.
- Regression file `tests/annotation/test_spacebar_export_contract.py`
  compiles **the active method AST**, executes triple-export behavior
  with synthetic image/annotation stubs, and checks three target paths,
  matching rotated pixel bytes, class remapping, absence of source class
  mutation, Space save exactly once, ordinary Next save, and shortcut/
  detection wiring invariants.
- `.github/workflows/annotation-integrity.yml` isolates these contract
  tests from Phoenix GUI/test dependencies. GitHub Actions run
  `38976427040`: **4 tests passed**, and both active modules compiled.
  This is code-level regression verification only. It does NOT prove
  actual GUI rotation alignment, image provenance, loaded model behavior,
  or annotation-to-pixel geometry under mouse editing.

Still pending: end-to-end run via `run_truscore.py` on workstation,
manual rotation/box overlay checks, a representative two-class detection
on next card, failure propagation/atomic image+label writes, reliable
historical recovery of polygon geometry, and a corrected Dataset Studio
COCO export.

## Two-class detector activation repair — 2026-10-16

User acceptance from the real Annotation Studio reported: loading 3 card
images yielded NO bounding boxes on the first card, and the manual
"Run Detection" button also yielded none. This is a production blocking
failure; previous CI for Space-bar export did not cover model activation.

Root code defects verified in active `src/modules/annotation_studio/`:

- Settings widget and main border plugin hard-coded an obsolete
  `/home/dewster/Projects/Vanguard/src/models/revolutionary_border_detector.pt`
  path. The tracked real checkpoint is at
  `src/models/revolutionary_border_detector.pt` (52 MB); use a path
  derived from `__file__` to be independent of the checkout location.
- `BorderDetectionPlugin.load_model()` previously set
  `self.model = {"path": ..., "loaded": True}` without loading YOLO;
  replaced with delegation to genuine `_load_default_model()`.
  Errors must report real failures, not simulated model availability.
- Settings widget and main plugin each could load the detector
  independently; synchronize and reuse an actually loaded Ultralytics
  model when both target the same checkpoint.
- Previous automatic detection required the main plugin model-loaded
  flag before it would schedule, even though settings-widget startup
  could load later. Now schedule against each new image; the UI
  detection method lazily loads the genuine checkpoint as needed.
  Reject delayed callbacks for a previous image.
- Manual Run Detection also attempts genuine lazy model loading if
  no callable/predictive model is available; use the user-controlled
  confidence threshold (default 0.25), not an unrelated hard-coded 0.5.
  Missing checkpoint or dependencies fail visibly/log explicitly.
- Plugin-panel activation failed to guarantee the settings widget's
  Studio/canvas reference, which could prevent completed detections
  from redrawing. Both plugin activation routes now connect the
  widget to the Studio canvas, and plugin selection with an existing
  image forwards that image through on_image_changed.

The plugin retains ONE two-class YOLO model (outer 0, inner 1), manual
border correction, image-rotation behavior, and Space → triple export.
All modifications were confined to the current active `src/` tree.

Regression tests: `tests/annotation/test_border_detection_model_loading.py`
exercises actual method AST with synthetic Ultralytics detector and
synthetic Qt timer; it checks bundled model resolution, real loader
reuse, missing checkpoint failure, manual two-box production, stale
timer rejection, and plugin selection overlay wiring. The annotation
workflow CI run 38977937318 PASSED after repairs; Phoenix foundation
run 38977896821 PASSED the previous repair set. These are controlled
code-level tests, not an actual RX 7900 XTX/YOLO image inference run.

NEXT USER ACCEPTANCE: pull branch; launch from `run_truscore.py`;
load test cards with Border Detection selected; confirm model status
shows `revolutionary_border_detector.pt`, automatic prediction of
editable outer+inner boxes on each card, button manual prediction, and
correct Space save+next behavior. If predictions are still missing,
read live Annotation Studio logs, model load exceptions, true YOLO
confidence/results and plugin selection messages. Do NOT claim model
weights or GPU inference are verified until those tests pass. Keep
Datasets/COCO training gate blocked pending annotation geometry audit.

## Lost plugin controls after model-path fix — 2026-10-16

User observed a real GUI regression: Border Detection Plugin selected and
reported "Loaded Successfully!", but the left plugin controls region was
empty (screenshot 2026-10-16_17-56.jpg, card counter 0/0). This was a
separate problem from missing bounding boxes; there was not yet an image
loaded in that screenshot.

**Verified active-code cause:**
`BorderDetectionPlugin.create_settings_panel()` (last of two duplicate
definitions within that class) conditionally called
`self.settings_widget.apply_settings(self.settings)` when main-plugin
settings were populated. The existing `BorderDetectionSettingsWidget`
has NO `apply_settings` method. After the real model path repair,
`_load_default_model()` sets `self.settings['model_path']`, turning on
this otherwise dormant AttributeError. Its catch block returned an EMPTY
`QWidget(parent)` instead of the original full controls widget; the
caller considered this success. This explains why fixing checkpoint
loading exposed a blank settings panel.

**Bounded repair:** The active settings widget already owns a `settings`
dictionary, so the panel creation merges the main plugin's config into it
rather than calling the nonexistent method. Panel constructor errors now
log and re-raise, so the Studio reports failed plugin loading rather than
claiming success with a blank widget. No rotation, exports, drawing,
model family, or inference algorithm was changed in this slice.

**Tests:** `tests/annotation/test_border_plugin_panel.py` compiles the
actual effective (last-definition) `create_settings_panel` method
against minimal GUI doubles and asserts the real-model-loaded case
preserves controls, propagates constructor failure, and retains all
existing UI sections. Annotation Integrity GitHub Actions run
`38979071929` passed **15 tests**. This is source-level regression
verification only, not full Qt GUI acceptance.

**NEXT:** Pull latest branch and launch from `run_truscore.py`.
Select Border Detection Plugin; first confirm visible model controls,
Run Detection, outer/graphic selection, confidence, export, and status.
Then load a card and inspect the genuine detection count and rendered boxes.
If still empty, get the actual log traceback rather than adding another
unverified fallback. Current Studio contains duplicate method definitions
in the border plugin; consolidate only after working runtime behavior is
recorded and verified.


## Real ROCm/YOLO image-to-overlay acceptance and first-card lifecycle gap — 2026-10-16

The user verified through the normal `run_truscore.py` UI, launched with
`env -u LD_LIBRARY_PATH -u LD_PRELOAD ./vanguard/bin/python run_truscore.py`,
that the actual `revolutionary_border_detector.pt` predicts an outer
card boundary (YOLO class 0, screenshot confidence ~0.97) and inner
graphics boundary (class 1, screenshot confidence ~0.95), with both
rectangles and adjustment handles clearly displayed. UI reported
`Detection: 1 outer / 1 graphic`. This confirms actual ROCm inference
plus working plugin->Studio overlay rendering for the observed card,
not a simulated test. The inference library-loader failure was caused
by one or both of the original library environment variables; root
variable/collision is not yet isolated.

The user manually corrected and used Save/Next on four cards, reporting
all three exports appear correctly saved. Exported source images and YOLO
pairs have NOT been inspected remotely: no claim of rotation, annotation
pairing, or three-target file-content acceptance yet.

Remaining issue: user had to press `Run Detection` for the first card,
while later workflow operated. Actual active
`ModularAnnotationStudio.load_plugin_from_file()` did not notify a newly
selected plugin if an image had been loaded *before* plugin selection,
unlike alternate `on_plugin_selected()` route. The bounded fix
for `border_detection_plugin` only now detects already-loaded
`original_image`/path after inserting and wiring settings widget;
it calls the existing widget `set_current_image()` and the plugin's
`on_image_changed()` ONCE. The plugin's existing delayed detection,
stale-image guard and single real checkpoint remain authoritative.
When plugin is selected before card loading, normal
`load_current_image()` remains responsible for notification.

Regression tests `tests/annotation/test_first_card_activation.py`
exercise the effective source methods for BOTH startup orders. CI run
38981656677 PASS: 20 Annotation Integrity tests. The real-world
first-card startup behavior has NOT been retested since this fix.
If this persists when the plugin was already selected before card
loading, inspect the exact application log and scheduling messages
rather than blaming the untested activation-order hypothesis.

Next steps: selective sync of only four annotated card images and
matching label files in each of the three output folders; inspect
actual exports visually and verify classes, stems, image dimensions and
rotation. Do not commit the entire virtualenv, archived scripts, or
every older runtime artifact. Keep Mask R-CNN / COCO precision training
blocked pending export validation.
