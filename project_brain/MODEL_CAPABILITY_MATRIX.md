# Dataset Studio → Annotation Studio → Phoenix Capability Inventory

Reviewed 2026-10-17 on refactor/phoenix-foundation.
Maintainer contract: ADR 0011. This is an implementation-status inventory
with sources; it is **not** a claim of hardware-level acceptance for
all visible menu choices.

## Interpreting status

- **Implemented source route**: exact contract, exporter and concrete
  trainer code exist; headless tests may pass. End-to-end GPU training
  is still a distinct acceptance gate.
- **Concrete trainer, dataset route missing**: the Phoenix trainer
  implementation exists, but Dataset Studio cannot yet produce the
  required supervised task labels via current plugins.
- **Not implemented**: advertised selection/research, but no truthful
  end-to-end training route. Fails explicitly when published.
- **Historical only**: Checkplz/older script has candidate logic
  to study, but not authorized as current production functionality.

## The 24 dataset type options

| Dataset type key | Required genuine annotation | Current state / next dependency |
|---|---|---|
| border_detection_single | Single class bbox or source polygon | Annotation Studio exports single-class labels, but active Dataset Studio COCO dual-class contract not validated for this task |
| border_detection_2class | Two-class outer+graphic bbox | **Only explicit new route**: YOLO11s Optimized, real COCO 7-card fixture; native GPU/GUI acceptance pending |
| border_ultra_precision | Source corners / rotated polygons, subpixel edge evidence | Not implemented; source 5-field boxes insufficient |
| corner_quality_classification | Human quality class for each corner crop | ViT trainer exists; label-capture/export plugin missing |
| corner_damage_detection | Actual damaged-corner localization | Not implemented; no validated box/mask annotation plugin |
| corner_sharpness_rating | Per-corner quantitative/rating target | Not implemented; regression target/ground truth needed |
| edge_wear_detection | Localized edge wear/defect target | Not implemented; human localization ground truth needed |
| edge_damage_classification | Human damaged-edge classes | Not implemented; classification label workflow missing |
| surface_defect_detection | Verified surface defect boxes/masks | Not implemented; new annotation plugin required |
| surface_quality_rating | Human surface quality/regression score | Not implemented; grading calibration target required |
| surface_damage_classification | Human surface damage classes | Not implemented; class labels needed |
| photometric_surface_normals | Multi-light calibrated normals | Historical photometric implementation; not present as validated Studio→Phoenix source |
| photometric_reflectance | Multi-light reflectance/albedo ground truth | Not implemented; acquisition/albedo labels required |
| photometric_depth | Calibrated depth/multi-view ground truth | Not implemented; real depth supervision required |
| yolo_v10x_precision | Genuine bbox labels and YOLOv10x architecture | Research/selection only; not implemented as Phoenix trainer |
| yolo_v9_accuracy | Genuine bbox labels and YOLOv9/GELAN architecture | Research/selection only; not implemented as Phoenix trainer |
| yolo_11s_advanced | Genuine bbox labels and explicitly selected YOLO11s | Separate dataset type not validated/connected yet; do not equate with 2-class pilot |
| vision_language_fusion | Aligned images, human-reviewed instructions/answers | Future VLM route; Unsloth isolated ROCm evaluation later |
| neural_rendering_hybrid | Calibrated multi-modal geometry/rendering targets | Not implemented |
| tesla_hydra_phoenix | Defined multi-task datasets and verified loss heads | Historical architecture research, not a usable current trainer |
| uncertainty_quantification | Uncertainty/calibration data and benchmark targets | Not implemented |
| classification | Clear classes and source labels | Generic picker option, not connected to a concrete supervised export |
| segmentation | Pixel/instance masks | U-Net/Mask R-CNN code exists; correct mask annotation/export missing |
| truscore_professional | Full independently labeled grading multi-task set | Future integration objective, not implemented |

**Important:** The three annotation folders exported by the current
border plugin are not three independent supervision tasks. They share
source JPEGs; the combined folder contains two-class YOLO boxes,
and the single-class folders remap class IDs.

## The 15 selectable named pipelines

| Picker pipeline label | Phoenix reality |
|---|---|
| Detectron2 (Mask R-CNN + RPN) - Professional | Concrete Detectron2 trainer exists; genuine instance masks required; current border labels are boxes only |
| Sub-Pixel Enhanced Detectron2 | Unimplemented architecture, explicit unsupported |
| YOLOv10x Precision | Unimplemented concrete adapter; explicit unsupported |
| YOLOv9 Gelan-base | Unimplemented concrete adapter; explicit unsupported |
| YOLO11s Optimized | **New explicit bbox detection adapter** for border_detection_2class; headless verified, native GPU pending |
| Feature Pyramid Networks | Architecture component/concept, not an independent working trained model contract |
| Swin Transformer Advanced | Not wired to a verified Phoenix trainer |
| ConvNext Classification | Not wired to a verified Phoenix trainer |
| Custom PS-Net | Historical photometric evidence; current source requires recovery/integration |
| EventPS Real-time (30+ fps) | No verified training implementation |
| Multi-View Networks | No verified training implementation |
| Tesla Hydra Phoenix | Historical/multimodal concept, not accepted trainer |
| Vision-Language Networks | Future isolated VLM/Unsloth candidate, not accepted |
| Bayesian Neural Networks | No verified task-specific current training integration |
| TruScore Professional Pipeline | Proposed composite pipeline, not an accepted single training backend |

**Separate concrete trainer:** a timm Vision Transformer (ViT) was
trained in an actual ROCm smoke job with a purpose-built classification
dataset. It is not a promise that the picker’s Swin, ConvNext or
Feature Pyramid option launches ViT. Current U-Net code exists,
but supervised mask input/export and native runtime acceptance remain
separate.

## Only currently registered Dataset Studio handoff

`dataset_type=border_detection_2class`
+`pipeline=YOLO11s Optimized`
→ current COCO bbox + 2 validated categories
→ atomic Phoenix handoff
→ deterministic source-faithful train/val YOLO labels
→ real Ultralytics YOLO11s native training,
pending workstation acceptance.

**Other 359 theoretical dataset/pipeline combinations are not implied
compatible**: the picker limits displayed combinations, and the
`phoenix_contract` registry accepts only the exact pair above.
The 24×15 combinatorial sweep is a test of non-fallback behavior,
not an invitation to create invalid project combinations.

## Evidence and verification

- Real seven-card JPEG/COCO acceptance:
  DATASET_SEVEN_CARD_EXPORT_AUDIT.md, CI 39002287868 (43 tests).
- Actual Phoenix model routing and historical code:
  PHOENIX_TRAINER_AUDIT.md and CODE_ARCHAEOLOGY.md.
- New handoff source and pending acceptance:
  ADR 0011, tests/phoenix/test_yolo11_dataset_handoff.py,
  .github/workflows/phoenix-dataset-handoff.yml.
- Unsloth AMD ROCm / VLM research:
  PHOENIX_TRAINER_AUDIT.md. Deferred, not installed in Phoenix.

## Required next action

After dedicated source/CI tests pass, perform an actual one-dataset
UI acceptance with an explicitly selected YOLO11s pipeline, then
a **real** short ROCm train that creates best.pt/last.pt and real
Ultralytics output. Document commands, hardware, logs, file hashes
and genuine evaluation results. A seven-image smoke job cannot
establish scientific accuracy or generalization.

When new Annotation Studio plugins are added, update this inventory
and the central capability registry in the same reviewed change.
