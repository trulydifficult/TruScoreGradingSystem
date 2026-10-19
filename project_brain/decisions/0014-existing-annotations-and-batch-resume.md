# ADR 0014 — Existing-label Annotation Studio mode and durable batch resume

Date: 2026-10-18
Status: OPERATOR-ACCEPTED REAL GUI IMPORT/EXPORT/QUIT-RESUME SMALL-BATCH WORKFLOW; full 550-card acceptance pending
Scope: Modular Annotation Studio, existing two-class YOLO detection labels and independent four-point border plugin

## Operator-recovered workflow
1. Select **label files first** (many individual `*.txt` files from the combined two-class YOLO dataset).
2. Select individual matching card **image files** (not simply a directory). Only exact-casefolded stem pairs are accepted; duplicate stems and missing labels stop import.
3. For this imported batch, do NOT infer, load, re-detect, or silently substitute YOLO predictions. The selected label pair (class 0 outer, class 1 graphic) creates two editable quad proposals, initially from axis-aligned legacy boxes. Outer is automatically selected; Tab switches to graphic; Space retains three export destinations and saves edited four-point geometry.
4. The independent `Load Cards` action continues the **fresh YOLO** path unchanged.
5. `Resume Interrupted Batch` reopens the last chosen image-and-label list, jumping to the earliest image not confirmed exported. Export success is recorded only after the three-target write has succeeded, **not** from current image navigation alone.

## Ownership
- `src/modules/annotation_studio/modular_annotation_studio.py`: label-first buttons, existing image multi-picker reuse, file pairing, session selection, button/status messaging, progress checkpoint coordination, coordinate transforms, image loading and Space navigation.
- `src/modules/annotation_studio/existing_label_sessions.py`: pure validation of selected paths, exact one-row-per-class five-column YOLO records, durable local checkpoint read/write and first-unsaved resolution.
- `src/modules/annotation_studio/plugins/border_detection_plugin.py`: `load_existing_labels` creates real `BorderAnnotation` quads using the existing `from_yolo` contract, publishes them to Studio, selects outer first, and blocks *all* inference and scheduled model loading in imported mode. False/unknown model confidence is never reported as real precision.
- `src/modules/annotation_studio/plugins/base_plugin.py`: existing StudioContext boundary retained. No duplicated canvas transform or alternate border detector implemented.

## Persistent state and boundaries
- Single active local checkpoint at `$XDG_STATE_HOME/TruScore/annotation_batch_v1.json` (fallback `~/.local/state/TruScore/annotation_batch_v1.json`), **outside Git**.
- Contains absolute source-image path order, selected label paths, and confirmed successful image indices; writes use temp file + fsync + atomic replace.
- Resuming validates original image+label existence and pairing, restores actual saved-count, chooses **first unfinished card**, including out-of-order saves. If a previous batch is incomplete, replacing its checkpoint needs confirmation. Corrupt checkpoints are never silently overwritten.
- Failed exports or failed checkpoint writes cannot advance a card or fabricate completion; the result may need export retry if the files succeeded but the progress checkpoint did not.
- Completed cards may restore the separate quad JSON proposal when its saved-image dimensions/hash match the chosen source. A mismatch fails visibly rather than applying incorrect geometry. Note: source JPEG re-encoding may produce hash differences if the selected source is not the exact exported image; this path requires workstation QA.
- Existing YOLO bbox-only labels are *coarse seeds*, not human-verified quad mask ground truth. A successful import does not imply segmentation training readiness or 24-point physical accuracy.
- Unsaved in-progress mouse edits are NOT checkpointed. Only prior confirmed exports are guaranteed recoverable. The current UI provides one latest batch checkpoint, not multi-workspace session management.

## Archaeology
`src/revolutionary_border_calibration.py` had repeated Dataset Studio JSON prediction-import functions and a `save_session_data`/`load_session` image-index snapshot. They inform the desired workflow but do not provide a verified correct paired-image/label loader or confirmed successful-save resume semantics. Preserve these as REFERENCE, not automatic code restoration. Existing `BorderAnnotation.from_yolo` and current Studio file chooser/event contracts were salvaged.

## Verification and remaining acceptance
Headless Annotation Integrity CI initially PASSED 79 tests at commit `60b735b` (run 39159287317), including simulated source-level import, no auto/manual inference, 550-card / 100-saved resume, existing Tab/Space and all prior YOLO exports. Later guard fixes require the final branch CI check.

**Native workstation acceptance REQUIRED** on copies of a small, previously annotated batch:
- pick combined labels first, individual matching images second;
- confirm immediate two editable borders, outer selected, zero YOLO inference (including delayed callbacks), independent quad dragging, Tab, Space and correct rotated-image export;
- close after a couple successful cards, relaunch and click Resume; confirm first unfinished card opens with correct save count;
- retest normal fresh Load Cards and inference afterwards.
Only then attempt the user's ~550 historical cards.

No Detectron2/PointRend trainer was changed. No simulated training/performance metrics are accepted.

## 2026-10-19 — Live workstation import/resume accepted

Operator verified 5 imported cards continuously and separately
quit/resumed a 5-card batch after one successful card. This upgrades
the current imported-label workflow's basic import/save/resume gate to
**operator workstation accepted**. Precision of saved physical-border
geometry and future full-batch stability remain separate gates.

See ADR 0015 for the subsequent release/overflow repairs and pending
GUI acceptance; the new layout/drag changes have not yet been
operator tested.
