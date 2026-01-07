"""
TruScore Enterprise Mobile API Bridge
=====================================

Professional-grade FastAPI bridge for the TruScore ecosystem.

Architecture:
- API Layer (FastAPI): Handles HTTP requests, auth (future), and file uploads.
- Job Manager: Manages state, persistence, and indexing.
- Worker Layer: Wraps the heavy PyQt/AI pipeline in a thread-safe executor.
- Discovery: Auto-detects LAN IP for easy local development.

Scalability Note:
For production (1000+ users), the 'GradingWorker' class should be replaced
with a Celery task queue (Redis/RabbitMQ) to distribute load across multiple
GPU servers.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from PyQt6.QtWidgets import QApplication

from shared.essentials.truscore_logging import (
    log_component_status,
    setup_truscore_logging,
)

from modules.continuous_learning.guru_dispatcher import get_global_guru, GuruEvent

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
class ServerConfig:
    HOST = os.getenv("TRUSCORE_HOST", "0.0.0.0")
    PORT = int(os.getenv("TRUSCORE_PORT", "8009"))
    MAX_WORKERS = int(os.getenv("TRUSCORE_WORKERS", "1"))

    # Environment setup
    REPO_ROOT = Path(__file__).resolve().parents[2]
    EXPORT_ROOT = REPO_ROOT / "exports" / "mobile_jobs"
    INDEX_FILE = EXPORT_ROOT / "jobs_index.jsonl"

    # Headless configuration
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")

    @staticmethod
    def setup_path():
        if str(ServerConfig.REPO_ROOT) not in sys.path:
            sys.path.append(str(ServerConfig.REPO_ROOT))

ServerConfig.setup_path()
ServerConfig.EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

logger = setup_truscore_logging(__name__, "truscore_mobile_api.log")

SERVER_STARTED_AT = datetime.now().isoformat()


# -----------------------------------------------------------------------------
# CORE LOGIC: JOB MANAGER
# -----------------------------------------------------------------------------
class JobManager:
    """
    Manages the lifecycle, persistence, and retrieval of grading jobs.
    Abstracts away the file-system details to allow future DB migration.
    """
    def __init__(self, root_dir: Path, index_file: Path):
        self.root_dir = root_dir
        self.index_file = index_file
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.load_history()

    def create_job(self, metadata: Dict[str, Any], front_path: str, back_path: Optional[str], job_id: Optional[str] = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "submitted_at": time.time(),
                "front_path": front_path,
                "back_path": back_path,
                "metadata": metadata,
            }
        self.persist_job(job_id, force=True)
        return job_id

    def update_status(self, job_id: str, status: str, result: Optional[Dict] = None, error: Optional[str] = None):
        with self._lock:
            if job_id not in self._jobs:
                return
            job = self._jobs[job_id]
            job["status"] = status

            if status == "running":
                job["started_at"] = time.time()
            elif status in ["completed", "failed"]:
                job["completed_at"] = time.time()

            if result:
                job["result"] = result
            if error:
                job["error"] = error

        self.persist_job(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_recents(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            items = [
                (jid, job)
                for jid, job in self._jobs.items()
                if job.get("status") in {"completed", "running", "queued"}
            ]

        # Prefer surfacing recents that actually have images available.
        items_with_media = []
        for jid, job in items:
            if self._resolve_media_path(jid, "front", job.get("front_path")) is not None:
                items_with_media.append((jid, job))
                continue
            if self._resolve_media_path(jid, "back", job.get("back_path")) is not None:
                items_with_media.append((jid, job))
        if items_with_media:
            items = items_with_media

        # Sort by completion time (newest first)
        items.sort(
            key=lambda pair: pair[1].get("completed_at") or pair[1].get("submitted_at") or 0,
            reverse=True,
        )

        # Transform to lightweight summary
        results = []
        for jid, job in items[:max(1, limit)]:
            results.append(self._format_recent(jid, job))
        return results

    def _format_recent(self, job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
        result = job.get("result") or {}
        front = result.get("front") or {}
        grade = front.get("grade")
        if isinstance(grade, float):
            grade = f"{grade:.1f}"

        front_exists = self._resolve_media_path(job_id, "front", job.get("front_path")) is not None
        back_exists = self._resolve_media_path(job_id, "back", job.get("back_path")) is not None

        return {
            "job_id": job_id,
            "title": (job.get("metadata") or {}).get("title") or f"Submission {job_id[:6]}",
            "submitted_at": job.get("submitted_at"),
            "completed_at": job.get("completed_at"),
            "status": job.get("status"),
            "grade": grade,
            "thumbnail_front": f"/api/v1/cards/{job_id}/front" if front_exists else None,
            "thumbnail_back": f"/api/v1/cards/{job_id}/back" if back_exists else None,
        }

    def _resolve_media_path(self, job_id: str, kind: str, stored_path: Optional[str]) -> Optional[Path]:
        if stored_path:
            try:
                p = Path(stored_path)
                if p.exists() and p.is_file():
                    return p
            except Exception:
                pass

            try:
                marker = "exports/mobile_jobs/"
                if marker in stored_path:
                    tail = stored_path.split(marker, 1)[1]
                    candidate = self.root_dir / tail
                    if candidate.exists() and candidate.is_file():
                        return candidate
            except Exception:
                pass

        job_dir = self.root_dir / job_id
        if not job_dir.exists():
            return None

        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
            candidate = job_dir / f"{kind}{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate

        matches = sorted(job_dir.glob(f"{kind}.*"))
        for m in matches:
            if m.is_file():
                return m
        return None

    def repair_media_paths(self, job_id: str) -> Optional[Dict[str, Optional[str]]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

        front = self._resolve_media_path(job_id, "front", job.get("front_path"))
        back = self._resolve_media_path(job_id, "back", job.get("back_path"))

        # If legacy data wrote images into a different job folder, copy them into the
        # canonical folder for this job_id so future requests are stable.
        job_dir = self.root_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        def _canonicalize(kind: str, src: Optional[Path]) -> Optional[Path]:
            if not src or not src.exists() or not src.is_file():
                return None
            try:
                src_rel = src.resolve().relative_to(self.root_dir.resolve())
            except Exception:
                return src

            # If already in the right folder, keep it.
            if len(src_rel.parts) >= 2 and src_rel.parts[0] == job_id:
                return src

            dest = job_dir / f"{kind}{src.suffix or '.jpg'}"
            if dest.exists() and dest.is_file():
                return dest

            try:
                shutil.copy2(src, dest)
                return dest
            except Exception as exc:
                logger.error(f"Failed to canonicalize {kind} image for job {job_id}: {exc}")
                return src

        front = _canonicalize("front", front)
        back = _canonicalize("back", back)

        changed = False
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if front and job.get("front_path") != str(front):
                job["front_path"] = str(front)
                changed = True
            if back and job.get("back_path") != str(back):
                job["back_path"] = str(back)
                changed = True

        if changed:
            self.persist_job(job_id, force=True)

        return {
            "front_path": str(front) if front else None,
            "back_path": str(back) if back else None,
        }

    def persist_job(self, job_id: str, force: bool = False):
        """Write job state to disk (Simulates DB commit)"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            # Deep copy to avoid threading issues during write
            record = job.copy()

        record["job_id"] = job_id

        if not force and "result" not in record and record.get("status") not in {"completed", "failed"}:
            return

        try:
            job_dir = self.root_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            # 1. Write Summary
            (job_dir / "summary.json").write_text(json.dumps(record, indent=2))

            # 2. Write Details (if heavy)
            if record.get("result") and "details" in record["result"]:
                (job_dir / "details.json").write_text(json.dumps(record["result"]["details"], indent=2))

            # 3. Append to Index
            with self.index_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        except Exception as exc:
            logger.error(f"Failed to persist job {job_id}: {exc}")

    def load_history(self):
        """Rehydrate memory from disk on startup"""
        try:
            count = 0
            for summary_file in self.root_dir.glob("*/summary.json"):
                try:
                    data = json.loads(summary_file.read_text())
                    jid = data.get("job_id") or summary_file.parent.name
                    if jid:
                        with self._lock:
                            self._jobs[jid] = data
                        count += 1
                except Exception:
                    continue
            if count > 0:
                logger.info(f"Rehydrated {count} jobs from disk storage.")
        except Exception as exc:
            logger.error(f"Failed to load history: {exc}")


# -----------------------------------------------------------------------------
# CORE LOGIC: PIPELINE WORKER
# -----------------------------------------------------------------------------
class GradingWorker:
    """
    Wraps the PyQt-based pipeline.
    In production, this class would be replaced by a Celery Worker interface.
    """
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.executor = ThreadPoolExecutor(max_workers=ServerConfig.MAX_WORKERS)
        self._qt_app = None
        self._pipeline = None
        self._lock = threading.Lock()

        # Ensure headless Qt for image processing
        self._ensure_qt()

    def _ensure_qt(self):
        if QApplication.instance() is None:
            self._qt_app = QApplication([])

    def _get_pipeline(self):
        """Lazy load the massive AI pipeline"""
        with self._lock:
            if self._pipeline is None:
                logger.info("Initializing TruScore Master Pipeline (Lazy Load)...")
                try:
                    from modules.truscore_grading.truscore_master_pipeline import TruScoreMasterPipeline
                    self._pipeline = TruScoreMasterPipeline()
                    logger.info("Pipeline Ready.")
                except Exception as exc:
                    logger.critical(f"Pipeline init failed: {exc}")
                    raise
            return self._pipeline

    def submit_job(self, job_id: str):
        job = self.job_manager.get_job(job_id)
        if not job:
            return

        repaired = self.job_manager.repair_media_paths(job_id) or {}
        front_resolved = repaired.get("front_path") or job.get("front_path")
        if not front_resolved:
            self.job_manager.update_status(job_id, "failed", error="Missing front image for job")
            return

        front_path = Path(front_resolved)
        back_resolved = repaired.get("back_path") or job.get("back_path")
        back_path = Path(back_resolved) if back_resolved else None
        metadata = job["metadata"]

        self.executor.submit(self._run_job, job_id, front_path, back_path, metadata)

    def _run_job(self, job_id: str, front_path: Path, back_path: Optional[Path], metadata: Dict):
        """The actual heavy lifting"""
        logger.info(f"WORKER: Starting job {job_id}")
        self.job_manager.update_status(job_id, "running")

        try:
            pipeline = self._get_pipeline()

            # Analyze Front
            front_result = pipeline.analyze_card_master_pipeline(str(front_path))

            # Analyze Back (Optional)
            back_result = None
            if back_path and back_path.exists():
                back_result = pipeline.analyze_card_master_pipeline(str(back_path))

            summary = {
                "front": _summarize_result(front_result),
                "back": _summarize_result(back_result) if back_result else None,
                "metadata": metadata or {},
                "details": {
                    "front": _detailed_snapshot(front_result),
                    "back": _detailed_snapshot(back_result) if back_result else None,
                },
                "visualizations": {},
            }

            # Serialize visualizations (saves images to disk)
            summary["visualizations"]["front"] = _serialize_visualizations(front_result, job_id, "front", front_path)
            if back_result:
                summary["visualizations"]["back"] = _serialize_visualizations(back_result, job_id, "back", back_path)

            self.job_manager.update_status(job_id, "completed", result=summary)
            logger.info(f"WORKER: Job {job_id} Finished successfully.")

            try:
                guru = get_global_guru()

                learning_payload = {
                    "job_id": job_id,
                    "final_grade": (summary.get("front") or {}).get("grade"),
                    "front_scores": ((summary.get("details") or {}).get("front") or {}).get("scores", {}),
                    "centering_data": ((summary.get("details") or {}).get("front") or {}).get("centering", {}),
                    "defects_count": (summary.get("front") or {}).get("defects_count"),
                    "surface_integrity": (summary.get("front") or {}).get("surface_integrity"),
                    "visualization_assets": ((summary.get("visualizations") or {}).get("front") or {}).get("assets", []),
                }
                if back_result:
                    learning_payload["back_scores"] = ((summary.get("details") or {}).get("back") or {}).get("scores", {})

                event = GuruEvent(
                    event_type="mobile_grading_completed",
                    source_system="mobile_api",
                    data_payload=learning_payload,
                    metadata={
                        "front_path": str(front_path),
                        "back_path": str(back_path) if back_path else None,
                        "user_metadata": metadata,
                        "pipeline_version": "master_v2",
                    },
                    timestamp=datetime.now().isoformat(),
                    quality_score=None,
                )

                absorbed = guru.absorb_event(event)
                if absorbed:
                    logger.info(f"WORKER: Job {job_id} absorbed by Continuous Learning Model")
            except Exception as exc:
                logger.error(f"WORKER: Continuous Learning absorption failed for {job_id}: {exc}")

        except Exception as exc:
            logger.error(f"WORKER: Job {job_id} Failed: {exc}")
            self.job_manager.update_status(job_id, "failed", error=str(exc))


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (Preserved from original for compatibility)
# -----------------------------------------------------------------------------
# Re-implementing essential helpers here for the Worker to use
# (In a real refactor, these would move to 'utils.py')

def _sanitize_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "x") and hasattr(value, "y") and callable(value.x) and callable(value.y):
        return [float(value.x()), float(value.y())]
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    if isinstance(value, dict):
        clean_dict = {}
        for k, v in value.items():
            if k in {"pixmap", "image_data", "buffer"}:
                continue
            if "Pixmap" in str(type(v)):
                continue
            clean_dict[k] = _sanitize_value(v)
        return clean_dict
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    try:
        return float(value)
    except Exception:
        return str(value)

def _summarize_result(result):
    if result is None:
        return {}
    if hasattr(result, "scores"):
        scores = getattr(result, "scores", None)
        surface = getattr(result, "surface_data", None)
        centering = getattr(result, "centering_data", {}) or {}
        corners = getattr(result, "corner_data", {}) or {}
        corner_scores = corners.get("scores", {}) if isinstance(corners, dict) else {}

        grade_val = None
        if scores:
            try:
                grade_val = round(float(scores.final_grade), 1)
            except Exception:
                grade_val = getattr(scores, "final_grade", None)

        return {
            "success": bool(getattr(result, "success", False)),
            "grade": grade_val if grade_val is not None else "Unknown",
            "grade_confidence": None,
            "surface_integrity": getattr(surface, "surface_integrity", None) if surface else None,
            "defects_count": getattr(surface, "defect_count", None) if surface else None,
            "centering_score": centering.get("overall_centering_score") if isinstance(centering, dict) else None,
            "corner_scores": _sanitize_value(corner_scores),
            "processing_time": getattr(result, "processing_time", None),
            "timestamp": getattr(result, "timestamp", None),
        }

    return {"success": True, "grade": "Unknown"}

def _detailed_snapshot(result):
    if result is None:
        return {}

    details: Dict[str, Any] = {}
    scores = getattr(result, "scores", None)
    if scores:
        details["scores"] = {
            "final_grade": getattr(scores, "final_grade", None),
            "corners": getattr(scores, "corners", None),
            "centering": getattr(scores, "centering", None),
            "surface": getattr(scores, "surface", None),
            "edges": getattr(scores, "edges", None),
            "total": getattr(scores, "total", None),
        }

    centering = getattr(result, "centering_data", None)
    if isinstance(centering, dict):
        details["centering_analysis"] = _sanitize_value(centering)
        details["centering"] = details["centering_analysis"]

    corners = getattr(result, "corner_data", None)
    if isinstance(corners, dict):
        corners_clean = corners.copy()
        if "crops" in corners_clean:
            del corners_clean["crops"]
        details["corner_analysis"] = _sanitize_value(corners_clean)
        details["corners"] = details["corner_analysis"]

    surface = getattr(result, "surface_data", None)
    if surface:
        details["photometric_analysis"] = {
            "surface_integrity": getattr(surface, "surface_integrity", None),
            "defect_count": getattr(surface, "defect_count", None),
            "surface_roughness": getattr(surface, "surface_roughness", None),
            "surface_quality": getattr(surface, "surface_quality", None),
        }
        details["surface_analysis"] = details["photometric_analysis"]

    border = getattr(result, "border_data", None)
    if isinstance(border, dict):
        details["border_analysis"] = _sanitize_value(border)

    viz_data = getattr(result, "visualization_data", {})
    if isinstance(viz_data, dict):
        if "insights" in viz_data:
            details["insights"] = _sanitize_value(viz_data["insights"])
        if "smart_defects" in viz_data:
            details["smart_defects"] = _sanitize_value(viz_data["smart_defects"])

    return details

# NOTE: The _serialize_visualizations logic is complex and file-IO heavy.
# It is assumed to be available or we can keep the original implementation's logic.
# For this "Server.py" replacement, I will ensure we keep the necessary imports or methods.
# To keep this response clean, I will assume the original helper methods are present or I will
# include the critical ones below.

def _normalize_to_uint8(arr, colorize=False, colormap=cv2.COLORMAP_VIRIDIS):
    try:
        arr = np.nan_to_num(arr)
        if arr.ndim == 2:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.applyColorMap(arr, colormap) if colorize else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return arr
    except Exception:
        pass
    return None

def _render_centering_overlay(image_path: Optional[Path], centering: Dict[str, Any], dest: Path) -> Optional[str]:
    if not image_path or not image_path.exists():
        return None
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        def _to_pts(poly):
            return np.array([[int(x), int(y)] for x, y in poly], dtype=np.int32)

        outer = centering.get("outer")
        inner = centering.get("inner")
        rays = centering.get("rays") or []

        if (outer is None or inner is None) and centering.get("_border_box_fallback"):
            bb = centering["_border_box_fallback"]
            ob = bb.get("outer_border")
            ib = bb.get("inner_border")
            if ob and len(ob) == 4:
                x1, y1, x2, y2 = ob
                outer = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            if ib and len(ib) == 4:
                x1, y1, x2, y2 = ib
                inner = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        if outer:
            cv2.polylines(img, [_to_pts(outer)], isClosed=True, color=(255, 0, 255), thickness=3)
        if inner:
            cv2.polylines(img, [_to_pts(inner)], isClosed=True, color=(0, 200, 0), thickness=3)
        for idx, ray in enumerate(rays, start=1):
            if not isinstance(ray, (list, tuple)) or len(ray) != 2:
                continue
            (sx, sy), (ex, ey) = ray
            cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(
                img,
                str(idx),
                (int(sx) + 2, int(sy) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), img)
        return str(dest)
    except Exception as exc:
        logger.error(f"Failed to render centering overlay: {exc}")
        return None

def _write_image(arr, dest):
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), arr)
        return str(dest)
    except: return None

def _serialize_visualizations(result, job_id, side, path):
    out = {"assets": [], "meta": {}}
    if not result or not hasattr(result, "visualization_data"): return out
    viz = result.visualization_data
    if not isinstance(viz, dict): return out

    viz_dir = ServerConfig.EXPORT_ROOT / job_id / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    photo = viz.get("photometric_analysis")
    if photo:
        if hasattr(photo, "surface_normals"):
            dest = viz_dir / f"{side}_normals.png"
            arr = (np.clip(photo.surface_normals, -1, 1) + 1.0) * 127.5
            norm = _normalize_to_uint8(arr)
            if norm is not None and _write_image(norm, dest):
                out["assets"].append({"name": "surface_normals", "label": "Surface Normals (3D)", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})
        if hasattr(photo, "depth_map"):
            dest = viz_dir / f"{side}_depth.png"
            norm = _normalize_to_uint8(np.array(photo.depth_map), colorize=True, colormap=cv2.COLORMAP_PLASMA)
            if norm is not None and _write_image(norm, dest):
                out["assets"].append({"name": "depth_map", "label": "Depth Map / Texture", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})
        if hasattr(photo, "albedo_map"):
            dest = viz_dir / f"{side}_albedo.png"
            norm = _normalize_to_uint8(np.array(photo.albedo_map))
            if norm is not None and _write_image(norm, dest):
                out["assets"].append({"name": "albedo_map", "label": "Surface Albedo", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})

        if hasattr(photo, "defect_map"):
            dest = viz_dir / f"{side}_defect_map.png"
            norm = _normalize_to_uint8(np.array(photo.defect_map), colorize=True, colormap=cv2.COLORMAP_MAGMA)
            if norm is not None and _write_image(norm, dest):
                out["assets"].append({"name": "defect_map", "label": "Defect Heatmap", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})

        if hasattr(photo, "confidence_map"):
            dest = viz_dir / f"{side}_confidence_map.png"
            norm = _normalize_to_uint8(np.array(photo.confidence_map), colorize=True, colormap=cv2.COLORMAP_TURBO)
            if norm is not None and _write_image(norm, dest):
                out["assets"].append({"name": "confidence_map", "label": "Confidence Map", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})

    centering = viz.get("centering_analysis")
    if isinstance(centering, dict):
        border_box = viz.get("border_analysis") if isinstance(viz.get("border_analysis"), dict) else None
        centering_for_overlay = dict(centering)
        if border_box:
            centering_for_overlay["_border_box_fallback"] = border_box

        clean_centering = {k: v for k, v in centering.items() if k not in {"visualization_data", "pixmap"}}
        out["meta"]["centering"] = _sanitize_value(clean_centering)

        overlay_path = viz_dir / f"{side}_centering_overlay.png"
        rendered = _render_centering_overlay(path, centering_for_overlay, overlay_path)
        if rendered:
            out["assets"].append({"name": "centering_overlay", "label": "24-Point Centering", "url": f"/api/v1/cards/{job_id}/viz/{overlay_path.name}"})

    corners_data = viz.get("corner_analysis")
    if isinstance(corners_data, dict):
        crops = corners_data.get("crops")
        if isinstance(crops, dict):
            for corner_key, crop_img in crops.items():
                if isinstance(crop_img, np.ndarray) and crop_img.size > 0:
                    fname = f"{side}_{corner_key}.png"
                    dest = viz_dir / fname
                    if _write_image(crop_img, dest):
                        readable_label = corner_key.replace("_", " ").title().replace("Tl", "Top Left").replace("Tr", "Top Right").replace("Bl", "Bottom Left").replace("Br", "Bottom Right")
                        out["assets"].append({"name": corner_key, "label": readable_label, "url": f"/api/v1/cards/{job_id}/viz/{fname}"})
        clean_corners = corners_data.copy()
        if "crops" in clean_corners:
            del clean_corners["crops"]
        out["meta"]["corner_analysis"] = _sanitize_value(clean_corners)

    for key in ("border_analysis", "smart_defects", "insights"):
        if key in viz:
            out["meta"][key] = _sanitize_value(viz.get(key))

    for key, val in viz.items():
        if key in {"photometric_analysis", "centering_analysis", "corner_analysis", "border_analysis", "smart_defects", "insights"}:
            continue
        if val is None:
            continue
        if isinstance(val, (str, Path)) and Path(val).exists():
            src = Path(val)
            fname = f"{side}_{src.name}"
            dest = viz_dir / fname
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)
                out["assets"].append({"name": key, "label": key.replace("_", " ").title(), "url": f"/api/v1/cards/{job_id}/viz/{fname}"})
            except Exception as exc:
                logger.error(f"Failed to copy viz asset {src}: {exc}")
            continue
        if hasattr(val, "shape"):
            fname = f"{side}_{key}.png"
            dest = viz_dir / fname
            saved = _write_image(val, dest)
            if saved:
                out["assets"].append({"name": key, "label": key.replace("_", " ").title(), "url": f"/api/v1/cards/{job_id}/viz/{fname}"})
            continue
        if isinstance(val, list) and val and hasattr(val[0], "shape"):
            for i, arr in enumerate(val[:2]):
                fname = f"{side}_{key}_{i}.png"
                dest = viz_dir / fname
                saved = _write_image(arr, dest)
                if saved:
                    out["assets"].append({"name": f"{key}_{i}", "label": f"{key.replace('_', ' ').title()} {i+1}", "url": f"/api/v1/cards/{job_id}/viz/{fname}"})
            continue
        out["meta"][key] = _sanitize_value(val)

    return out


# -----------------------------------------------------------------------------
# API SETUP
# -----------------------------------------------------------------------------
job_manager = JobManager(ServerConfig.EXPORT_ROOT, ServerConfig.INDEX_FILE)
worker = GradingWorker(job_manager)

app = FastAPI(title="TruScore Enterprise API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "server_started_at": SERVER_STARTED_AT,
        "export_root": str(ServerConfig.EXPORT_ROOT),
    }

@app.get("/")
def root():
    return HTMLResponse(f"""
    <html><body style="background:#111;color:#eee;font-family:sans-serif;padding:2rem;">
        <h1 style="color:#38bdf8">TruScore Enterprise API</h1>
        <p>Status: <span style="color:#4ade80">ONLINE</span></p>
        <p><b>Server IP:</b> {get_lan_ip()}</p>
        <p>Use this IP in your mobile app settings.</p>
    </body></html>
    """)

@app.post("/api/v1/cards/grade")
async def grade_card(
    front: UploadFile = File(...),
    back: UploadFile = File(None),
    metadata: Optional[str] = Form(None)
):
    try:
        meta_obj = json.loads(metadata) if metadata else {}
    except:
        meta_obj = {}

    job_id = uuid.uuid4().hex
    job_dir = ServerConfig.EXPORT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    front_suffix = Path(front.filename or "front.jpg").suffix or ".jpg"
    front_path = job_dir / f"front{front_suffix}"
    with front_path.open("wb") as f:
        shutil.copyfileobj(front.file, f)

    back_path_str = None
    if back:
        back_suffix = Path(back.filename or "back.jpg").suffix or ".jpg"
        back_path = job_dir / f"back{back_suffix}"
        with back_path.open("wb") as f:
            shutil.copyfileobj(back.file, f)
        back_path_str = str(back_path)

    final_id = job_manager.create_job(meta_obj, str(front_path), back_path_str, job_id=job_id)
    worker.submit_job(final_id)

    return {"job_id": final_id, "status": "queued"}

@app.get("/api/v1/cards/recent")
def get_recents(limit: int = 10):
    items = job_manager.get_recents(limit)
    return {"items": items, "count": len(items)}

@app.get("/api/v1/cards/{job_id}")
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job: raise HTTPException(404, "Job not found")
    return job

@app.get("/api/v1/cards/{job_id}/visualizations")
def get_visualizations(job_id: str):
    job = job_manager.get_job(job_id)
    if not job: raise HTTPException(404, "Job not found")
    result = (job.get("result") or {}).get("visualizations") or {}
    return result

@app.get("/api/v1/cards/{job_id}/viz/{filename}")
def get_viz_file(job_id: str, filename: str):
    safe_name = Path(filename).name
    path = ServerConfig.EXPORT_ROOT / job_id / "viz" / safe_name
    if path.exists(): return FileResponse(path)
    raise HTTPException(404)

@app.get("/api/v1/cards/{job_id}/front")
def get_front(job_id: str):
    job = job_manager.get_job(job_id)
    if job:
        path = job_manager._resolve_media_path(job_id, "front", job.get("front_path"))
        if path:
            media_type, _ = mimetypes.guess_type(path)
            return FileResponse(path, media_type=media_type or "image/jpeg")
        # Job exists but image is missing; return a tiny placeholder to avoid client crashes.
        return Response(
            content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\n\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82",
            media_type="image/png",
        )
    raise HTTPException(404)

@app.get("/api/v1/cards/{job_id}/back")
def get_back(job_id: str):
    job = job_manager.get_job(job_id)
    if job:
        path = job_manager._resolve_media_path(job_id, "back", job.get("back_path"))
        if path:
            media_type, _ = mimetypes.guess_type(path)
            return FileResponse(path, media_type=media_type or "image/jpeg")
        return Response(
            content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\n\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82",
            media_type="image/png",
        )
    raise HTTPException(404)

@app.get("/api/v1/cards/{job_id}/market")
def get_market(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    result = (job.get("result") or {}).get("front") or {}

    grade = result.get("grade")
    try:
        grade_num = float(grade)
    except Exception:
        grade_num = None

    base_value = 120.0 + (grade_num or 5) * 30
    return {
        "job_id": job_id,
        "grade": grade,
        "estimated_value": round(base_value, 2),
        "market_summary": "Market analysis placeholder. Live comps & population data coming soon.",
        "recent_sales": [
            {"title": "eBay Comp #1", "price": round(base_value * 0.95, 2), "days_ago": 2},
            {"title": "eBay Comp #2", "price": round(base_value * 1.05, 2), "days_ago": 5},
            {"title": "Auction House", "price": round(base_value * 1.2, 2), "days_ago": 9},
        ],
        "population": {
            "total": 1200,
            "grade_distribution": {
                "10": 120,
                "9": 380,
                "8": 420,
                "7": 210,
                "other": 70,
            },
        },
    }


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def get_lan_ip():
    """Auto-detects the machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


if __name__ == "__main__":
    lan_ip = get_lan_ip()
    print("\n" + "="*60)
    print(f" TRUSCORE ENTERPRISE API SERVER")
    print(f" Running on: http://{lan_ip}:{ServerConfig.PORT}")
    print(f" Configure your mobile app to point to this IP.")
    print("="*60 + "\n")

    uvicorn.run(app, host=ServerConfig.HOST, port=ServerConfig.PORT, log_level="info")
