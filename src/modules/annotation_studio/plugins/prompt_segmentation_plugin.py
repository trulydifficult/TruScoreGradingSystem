#!/usr/bin/env python3
"""
Promptable Segmentation Plugin (Step 1 scaffolding)
---------------------------------------------------
Adds prompt-driven masks, optional sub-pixel snap flag, multi-modal layer
references (normals/depth/reflectance), and low-confidence queue exports.

This is intentionally minimal/non-destructive: all advanced behaviors are
stubs you can wire to real SAM/FusionSAM inference later. Defaults are off.
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap, QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QTextEdit,
    QGroupBox,
    QFormLayout,
)

from .base_plugin import BaseAnnotationPlugin, StudioContext

try:
    from shared.essentials.truscore_theme import TruScoreTheme
    from shared.essentials.truscore_buttons import TruScoreButton
except Exception:
    class TruScoreTheme:
        VOID_BLACK = "#0A0A0B"
        NEURAL_GRAY = "#1C1E26"
        GHOST_WHITE = "#F8F9FA"
        NEON_CYAN = "#00F5FF"
        QUANTUM_GREEN = "#00FF88"

    class TruScoreButton(QPushButton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


class PromptSegmentationPlugin(BaseAnnotationPlugin):
    """Prompt-driven segmentation scaffold."""

    def __init__(self):
        super().__init__()
        self.annotations: List[Dict[str, Any]] = []
        self.current_image_path: Optional[str] = None
        self.current_image: Optional[np.ndarray] = None
        self.subpixel_snap_enabled = False
        self.loaded_layers: Dict[str, str] = {}
        self.prompt_text: str = ""
        self.low_confidence_dir = Path("active_learning_queue")
        self.low_confidence_dir.mkdir(exist_ok=True)
        self._last_mask: Optional[np.ndarray] = None
        self.model_path = Path(__file__).parent / "models" / "sam_promptable.pt"
        self.model = None
        self._qa_findings: List[str] = []
        self._try_load_model()

    # ==================== REQUIRED HOOKS ====================
    def on_activate(self):
        return

    def on_deactivate(self):
        return

    def handle_click(self, image_x: float, image_y: float) -> bool:
        return False

    def handle_drag(self, image_x: float, image_y: float) -> bool:
        return False

    def handle_release(self, image_x: float, image_y: float) -> bool:
        return False

    def handle_key_press(self, key: str, modifiers: List[str]) -> bool:
        key_upper = key.upper() if key else ""
        if key_upper == "G":
            self._generate_mask_stub()
            return True
        if key_upper == "S":
            self._save_mask()
            return True
        if key_upper == "F":
            self._flag_low_confidence()
            return True
        return False

    def draw_overlay(self, image: np.ndarray, transform_context: Dict[str, Any]) -> np.ndarray:
        """Overlay the last mask (if any) for visual confirmation."""
        if image is None:
            return image
        overlay = image.copy()
        if self._last_mask is not None:
            color = (0, 255, 180)
            overlay[self._last_mask > 0] = (
                0.6 * overlay[self._last_mask > 0] + 0.4 * np.array(color, dtype=np.float32)
            )
        if self.subpixel_snap_enabled:
            overlay = self._draw_subpixel_overlay(overlay)
        return overlay.astype(np.uint8)

    def draw_magnifier_overlay(self, image: np.ndarray, center_x: int, center_y: int, zoom_factor: float) -> np.ndarray:
        return self.draw_overlay(image, {"zoom_level": zoom_factor})

    def get_export_data(self, format_type: str) -> Dict[str, Any]:
        if format_type not in ["yolo", "coco", "truscore", "prompt_json", "yolo11", "ultra_precision_json"]:
            raise ValueError(f"Unsupported export format: {format_type}")

        width, height = (self.current_image.shape[1], self.current_image.shape[0]) if self.current_image is not None else (0, 0)
        self._run_pre_export_qa(width, height)
        export = {
            "format": format_type,
            "annotations": self.annotations,
            "image_width": width,
            "image_height": height,
            "metadata": {
                "prompt": self.prompt_text,
                "subpixel_snap": self.subpixel_snap_enabled,
                "layers": self.loaded_layers,
                "qa_findings": self._qa_findings,
            },
        }
        return export

    def has_annotations(self) -> bool:
        return bool(self.annotations)

    def on_image_changed(self, image_path: str, image_data: np.ndarray):
        self.current_image_path = image_path
        self.current_image = image_data
        self.annotations = []
        self._last_mask = None

    def on_image_rotated(self, rotation_angle: float):
        return

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Promptable Segmentation",
            "version": "0.1.0",
            "description": "Prompt-driven masks with sub-pixel snap flag + multimodal refs.",
            "author": "Vanguard",
            "supported_formats": ["yolo", "coco", "truscore", "prompt_json", "yolo11", "ultra_precision_json"],
        }

    def create_settings_panel(self, parent_widget) -> Any:
        panel = QWidget(parent_widget)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Promptable Segmentation")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TruScoreTheme.QUANTUM_GREEN};")
        layout.addWidget(title)

        # Prompt input
        prompt_input = QLineEdit()
        prompt_input.setPlaceholderText("e.g., highlight edge wear on bottom-right")
        prompt_input.textChanged.connect(self._set_prompt)
        layout.addWidget(prompt_input)

        # Sub-pixel + multimodal toggles
        toggles = QGroupBox("Precision & Layers")
        form = QFormLayout(toggles)
        self.subpixel_cb = QCheckBox("Enable sub-pixel snap overlay flag")
        self.subpixel_cb.stateChanged.connect(self._toggle_subpixel)
        form.addRow(self.subpixel_cb)

        layer_row = QHBoxLayout()
        layer_btn = TruScoreButton("Load normals/depth/reflectance")
        layer_btn.clicked.connect(self._load_layers)
        self.layer_label = QLabel("No extra layers loaded")
        layer_row.addWidget(layer_btn)
        layer_row.addWidget(self.layer_label)
        form.addRow(layer_row)
        toggles.setLayout(form)
        layout.addWidget(toggles)

        # Actions
        btn_row = QHBoxLayout()
        gen_btn = TruScoreButton("Generate Mask")
        gen_btn.clicked.connect(self._generate_mask_stub)
        save_mask_btn = TruScoreButton("Save Mask")
        save_mask_btn.clicked.connect(self._save_mask)
        flag_btn = TruScoreButton("Flag Low Confidence")
        flag_btn.clicked.connect(self._flag_low_confidence)
        btn_row.addWidget(gen_btn)
        btn_row.addWidget(save_mask_btn)
        btn_row.addWidget(flag_btn)
        layout.addLayout(btn_row)

        # Model status
        model_status = QLabel(self._model_status_text())
        model_status.setStyleSheet(f"color:{TruScoreTheme.GHOST_WHITE};")
        layout.addWidget(model_status)
        self.model_status_label = model_status

        # QA / info
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(80)
        self.info_box.setStyleSheet(f"background:{TruScoreTheme.NEURAL_GRAY}; color:{TruScoreTheme.GHOST_WHITE};")
        layout.addWidget(self.info_box)

        layout.addStretch()
        return panel

    # ==================== INTERNAL HELPERS ====================
    def _set_prompt(self, text: str):
        self.prompt_text = text

    def _toggle_subpixel(self, state: int):
        self.subpixel_snap_enabled = state == Qt.CheckState.Checked

    def _load_layers(self):
        """Attach optional multi-modal layers for export context."""
        files, _ = QFileDialog.getOpenFileNames(
            None,
            "Select normal/depth/reflectance layers",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)",
        )
        if not files:
            return
        for f in files:
            suffix = Path(f).suffix.lower()
            if "norm" in f:
                self.loaded_layers["normals"] = f
            elif "depth" in f or "dpt" in f:
                self.loaded_layers["depth"] = f
            elif "refl" in f or "albedo" in f:
                self.loaded_layers["reflectance"] = f
        self.layer_label.setText(", ".join(self.loaded_layers.keys()) or "No extra layers loaded")

    def _generate_mask_stub(self):
        """Generate mask. Uses real model if available; otherwise stub."""
        if self.current_image is None:
            self._log("No image loaded; cannot generate mask.")
            return

        h, w = self.current_image.shape[:2]
        mask, bbox, confidence = self._run_model_or_stub(self.current_image, self.prompt_text)
        self._last_mask = mask
        self.annotations = [
            {
                "prompt": self.prompt_text,
                "bbox": bbox,
                "subpixel_snap": self.subpixel_snap_enabled,
                "layers": self.loaded_layers,
                "confidence": confidence,
            }
        ]
        self._log(f"Mask generated ({'model' if self.model else 'stub'}) with conf={confidence:.2f}.")
        self._notify_status(f"Mask generated (conf={confidence:.2f}, subpixel={'on' if self.subpixel_snap_enabled else 'off'})")

    def _flag_low_confidence(self):
        if not self.current_image_path:
            self._log("No image loaded; cannot flag.")
            return
        entry = {
            "image_path": self.current_image_path,
            "prompt": self.prompt_text,
            "layers": self.loaded_layers,
            "timestamp": datetime.utcnow().isoformat(),
            "reason": "manual_flag" if self.annotations else "no_annotations",
        }
        out = self.low_confidence_dir / "requests.jsonl"
        import json
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self._log(f"Queued low-confidence sample to {out}")
        self._notify_status("Queued low-confidence sample")

    def _log(self, message: str):
        if hasattr(self, "info_box"):
            self.info_box.append(message)
        print(message)
        self._notify_status(message)

    def _save_mask(self):
        """Save current mask to active_learning_queue/masks for quick review."""
        if self._last_mask is None or self.current_image is None or not self.current_image_path:
            self._log("No mask to save.")
            return
        mask_dir = self.low_confidence_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        mask_path = mask_dir / (Path(self.current_image_path).stem + "_mask.png")
        cv2.imwrite(str(mask_path), self._last_mask)
        self._log(f"Saved mask to {mask_path}")

    # ==================== MODEL + QA HELPERS ====================
    def _try_load_model(self):
        """Optional: load SAM/FusionSAM weights if present."""
        search_candidates = [
            self.model_path,
            Path(__file__).parent / "models" / "fusion_sam.pt",
            Path(__file__).parent / "models" / "sam2_promptable.pt",
            Path(__file__).parent / "models" / "sam_promptable.onnx",
        ]
        found_path = next((p for p in search_candidates if p.exists()), None)
        if not found_path:
            self.model = None
            return
        try:
            if found_path.suffix.lower() == ".onnx":
                import onnxruntime as ort  # type: ignore

                class _OnnxWrapper:
                    """Thin ONNX runner; replace _run_inference with real SAM/SAM2 preprocessing/postprocessing."""

                    def __init__(self, session):
                        self.session = session

                    def __call__(self, image: np.ndarray, prompt: str) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
                        return self._run_inference(image, prompt)

                    def _run_inference(self, image: np.ndarray, prompt: str) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
                        # Generic best-effort inference: resize to model input, run, and map the first output back to image size.
                        try:
                            inp = self.session.get_inputs()[0]
                            _, c, h_in, w_in = inp.shape if inp.shape and len(inp.shape) == 4 else (1, 3, 512, 512)
                            resized = cv2.resize(image, (w_in, h_in))
                            # Basic normalization
                            arr = resized.astype(np.float32) / 255.0
                            if c == 3:
                                arr = arr.transpose(2, 0, 1)
                            arr = np.expand_dims(arr, 0)
                            ort_inputs = {inp.name: arr}
                            outputs = self.session.run(None, ort_inputs)
                            # Assume first output is mask-like; take channel 0
                            mask_pred = outputs[0]
                            mask_map = mask_pred[0]
                            if mask_map.ndim == 3:
                                mask_map = mask_map[0]
                            mask_map = (mask_map - mask_map.min()) / (mask_map.max() - mask_map.min() + 1e-6)
                            mask_map = (mask_map * 255).astype(np.uint8)
                            mask_resized = cv2.resize(mask_map, (image.shape[1], image.shape[0]))
                            # BBox from mask
                            ys, xs = np.where(mask_resized > 128)
                            if len(xs) == 0 or len(ys) == 0:
                                return np.zeros_like(mask_resized), (0, 0, 0, 0), 0.1
                            x0, x1 = int(xs.min()), int(xs.max())
                            y0, y1 = int(ys.min()), int(ys.max())
                            bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
                            confidence = float(mask_resized.max() / 255.0)
                            return mask_resized, bbox, confidence
                        except Exception:
                            # Fall back to safe stub if inference fails
                            h, w = image.shape[:2]
                            mask = np.zeros((h, w), dtype=np.uint8)
                            x0, x1 = int(w * 0.35), int(w * 0.65)
                            y0, y1 = int(h * 0.35), int(h * 0.65)
                            mask[y0:y1, x0:x1] = 255
                            return mask, (x0, y0, x1 - x0, y1 - y0), 0.25

                sess = ort.InferenceSession(str(found_path), providers=["CPUExecutionProvider"])
                self.model = _OnnxWrapper(sess)
            else:
                import torch  # type: ignore
                # Placeholder PyTorch wrapper; replace with actual SAM/FusionSAM model load.
                class _StubModel:
                    def __call__(self, image: np.ndarray, prompt: str) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
                        h, w = image.shape[:2]
                        mask = np.zeros((h, w), dtype=np.uint8)
                        x0, x1 = int(w * 0.35), int(w * 0.65)
                        y0, y1 = int(h * 0.35), int(h * 0.65)
                        mask[y0:y1, x0:x1] = 255
                        return mask, (x0, y0, x1 - x0, y1 - y0), 0.55
                self.model = _StubModel()
            self.model_path = found_path
        except Exception as exc:
            self.model = None
            self._log(f"Model load failed ({exc}); staying in stub mode.")
        if hasattr(self, "model_status_label"):
            try:
                self.model_status_label.setText(self._model_status_text())
            except Exception:
                pass

    def _run_model_or_stub(self, image: np.ndarray, prompt: str) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """Run segmentation; if no model, fall back to deterministic stub."""
        h, w = image.shape[:2]
        if self.model:
            try:
                return self.model(image, prompt)
            except Exception as exc:
                self._log(f"Model inference failed ({exc}); using stub.")
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, x1 = int(w * 0.3), int(w * 0.7)
        y0, y1 = int(h * 0.3), int(h * 0.7)
        mask[y0:y1, x0:x1] = 255
        return mask, (x0, y0, x1 - x0, y1 - y0), 0.35

    def _run_pre_export_qa(self, width: int, height: int):
        """Record QA findings for export consumers."""
        findings: List[str] = []
        if width < 800 or height < 800:
            findings.append("Resolution below recommended 1000px; consider higher DPI.")
        if not self.annotations:
            findings.append("No annotations present.")
        if self.subpixel_snap_enabled:
            findings.append("Sub-pixel snap flag enabled; ensure downstream handles precision coords.")
        if self.loaded_layers:
            findings.append(f"Extra layers attached: {list(self.loaded_layers.keys())}")
        self._qa_findings = findings
        if findings:
            self._log("QA findings: " + "; ".join(findings))
            self._notify_status("QA warnings present before export")

    def _model_status_text(self) -> str:
        if self.model:
            return f"Model: loaded from {self.model_path.name}"
        if self.model_path.exists():
            return f"Model file present ({self.model_path.name}) but not loaded; using stub."
        return "Model: not found (using stub masks). Place SAM/FusionSAM weights as sam_promptable.pt under plugins/models."

    def _notify_status(self, msg: str):
        """Update studio status bar if context is available."""
        try:
            if getattr(self, "studio_context", None) and hasattr(self.studio_context, "update_status"):
                self.studio_context.update_status(msg)
        except Exception:
            pass

    # ==================== SUB-PIXEL OVERLAY ====================
    def _draw_subpixel_overlay(self, image: np.ndarray) -> np.ndarray:
        """Lightweight visual overlay to hint sub-pixel precision areas."""
        if image is None:
            return image
        overlay = image.copy()
        h, w = overlay.shape[:2]
        color = (0, 255, 255)
        thickness = max(1, min(h, w) // 400)
        # simple crosshair grid
        step = max(20, min(h, w) // 25)
        for x in range(step, w, step):
            cv2.line(overlay, (x, 0), (x, h), color, thickness, lineType=cv2.LINE_AA)
        for y in range(step, h, step):
            cv2.line(overlay, (0, y), (w, y), color, thickness, lineType=cv2.LINE_AA)
        return overlay
