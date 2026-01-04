# Action Items from Research Docs (Jan 2025)

## High-Value Tech Themes (from NewTech, State-of-the-Art, Next-Gen Architectures, LLM Meta-Learning, Advanced Training)
- Vision-transformer + SAM/SAM2 + Detectron2 ensembles; YOLOv10x/YOLOv9 dual-stack for precision/speed with sub-pixel edge refinement (Steger/PGI).
- Photometric stereo as first-class input (PS-FCN/EventPS, PSBDDS); 24-point centering with 1/1000 mm precision; hybrid multi-view + photometric depth and reflectance.
- Vision-language fusion (FILM/FusionSAM) for promptable segmentation and explanation; LLM meta-learner with Bayesian uncertainty + episodic memory + natural-language justifications.
- Active learning/uncertainty pipelines (MC Dropout/Deep Ensembles, BADGE) with continuous feedback loops; automated drift detection and requeue for low-confidence items.
- Multi-model orchestration (YOLO11-seg + Mask R-CNN, ViT/DeiT/CaiT/BeiT, Swin/ConvNeXt, U-Net/PS-Net) with distributed training, TensorRT export, and ensemble fusion.
- Blockchain/NFC digital twin for provenance; mobile scanning with phone-grade photometric/UV/IR capture; real-time market/pricing APIs (Ximilar/TraderMade/Marketstack/TCG/eBay).

## Annotation Studio: Gaps to Add
- Promptable segmentation: add FusionSAM/FILM-backed “prompt” tool (natural language + points) with live masks; enable text prompts like “highlight edge wear on bottom-right”.
- Sub-pixel/precision modes: optional Steger-based edge snap for borders/corners; measurement overlay showing μm deltas to feed 24-point centering.
- Multi-modal layers: support importing photometric normals/depth/reflectance layers and toggling them in canvas; allow annotating directly on surface-normal view.
- Active-learning hooks: “flag low-confidence” button writes to `active_learning_queue` with model logits/entropy for trainer pickup.
- Export coverage: ensure YOLOv10/11, Detectron2, SAM masks, ultra-precision JSON (sub-pixel coords), and vision-language prompt annotations (text+mask pairing) are available.
- QA workflows: drift/consistency checks (schema + resolution/DPI warnings) before export; template presets for each dataset_type in `dataset_types.txt`.

## Dataset Studio: Gaps to Add
- Full dataset types coverage: add templates for experimental sets (vision_language_fusion, neural_rendering_hybrid, uncertainty_quantification, tesla_hydra_phoenix, photometric_depth/reflectance).
- Multi-modal packaging: bundle RGB + normals + depth + reflectance + UV/IR + text prompts; validate presence and alignments.
- Synthetic/augmentation: hooks for synthetic data generation and photometric augmentation; photometric stereo export already present—extend to normals/depth/point clouds.
- Active learning + drift: integrate BADGE/uncertainty samplers to auto-build “label_next” sets; monitor PSI/KS for drift and push alerts.
- Queue metadata: export to training queue with pipeline hints (YOLOv10x precision, ViT ensemble, FusionSAM, PS-Net) and hardware needs (GPU/VRAM).

## Trainer / Phoenix Trainer: Gaps to Add
- Multi-model orchestration: presets for YOLOv10x+YOLOv9 dual, YOLO11-seg + Mask R-CNN fusion, ViT/DeiT ensembles, SAM2 fine-tune, PS-FCN/EventPS, Swin/ConvNeXt surface classifiers, U-Net 6-channel (RGB+normals).
- Vision-language & LLM track: training path for FusionSAM/FILM promptable segmentation and the Revolutionary LLM Meta-Learner (multi-modal encoders + Bayesian heads + explanation generator).
- Uncertainty + active learning: bake MC Dropout/Deep Ensemble eval; emit entropy/variance into training_status.json; auto-export low-confidence samples back to annotation queue.
- Distributed/MLOps: DDP/ZeRO presets, TensorRT export, RT-DETR/YOLO-Former option for real-time, A/B champion-challenger with rollback; metrics to Prometheus/Grafana hook.
- Queue enhancements: priority + hardware-aware scheduling; allow manual dataset load already present—add batch training and dependency ordering (e.g., train border before centering fusion).

## Photometric / Scoring System
- EventPS real-time mode; hybrid multi-view + photometric depth/reflectance; sub-pixel centering and edge sharpness metrics surfaced in UI.
- Confidence-layer fusion: combine RGB detections + photometric surface cues + uncertainty thresholds before final grade; queue low-confidence for human review/active learning.

## Platform/Business Integrations
- Blockchain/NFC digital twin export (Polygon Supernets-style) with audit log; attach grading reports to token IDs.
- Mobile capture profile: presets for phone capture (multi-light burst, UV/IR toggles), edge deployment of YOLO11s/Florence-2 for ID and pre-grade.
- Market intelligence: optional real-time price feeds (Ximilar/TraderMade/Marketstack/eBay/TCGPlayer) into reports; API keys configurable per environment.

## Next Steps (suggested order)
1) Add Annotation Studio promptable segmentation + sub-pixel snap modes; wire low-confidence export.
2) Expand Dataset Studio templates/exports for experimental and photometric/vision-language sets with alignment validation.
3) Add trainer presets for dual-YOLO + SAM2 + PS-FCN + ViT ensemble + uncertainty reporting; connect active-learning loop.
4) Surface photometric depth/reflectance + confidence fusion in grading UI; emit blockchain/NFC-ready report hook.
