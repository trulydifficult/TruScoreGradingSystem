"""
Local Mobile API bridge for TruScore.

Purpose:
- Accept front/back card images from the Flutter mobile app
- Queue grading jobs and return a job_id for async status polling
- Wrap the existing TruScore photometric pipeline in a headless adapter

Notes:
- Designed for local use (loopback) during development. CORS is limited to localhost.
- The current pipeline is PyQt6-based; we load it lazily and keep one worker to
  avoid concurrent Qt model access.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional
import sys
import numpy as np
import cv2

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from PyQt6.QtWidgets import QApplication

from shared.essentials.truscore_logging import (
    log_component_status,
    setup_truscore_logging,
)

# Force headless Qt/matplotlib for server context
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

logger = setup_truscore_logging(__name__, "truscore_mobile_api.log")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = REPO_ROOT / "exports" / "mobile_jobs"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
INDEX_FILE = EXPORT_ROOT / "jobs_index.jsonl"

# Ensure repository modules (e.g., shared, modules) are importable even when
# running as `python server.py` from this directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Single-worker executor to keep the current pipeline thread-safe
executor = ThreadPoolExecutor(max_workers=1)

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()

_PIPELINE = None
_pipeline_lock = threading.Lock()
_QT_APP = None


def _ensure_qt_app():
    """Create a headless QApplication so QPixmap/QImage operations succeed."""
    global _QT_APP
    if QApplication.instance() is None:
        _QT_APP = QApplication([])  # Offscreen due to QT_QPA_PLATFORM=offscreen

# Initialize Qt app on import (main thread) to avoid warnings when worker threads use QPixmap.
_ensure_qt_app()


def job_dir_for(job_id: str) -> Path:
    return EXPORT_ROOT / job_id


def _persist_job(job_id: str, force: bool = False):
    """Write the latest job summary to disk and append to index for auditing/learning."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        record = {
            "job_id": job_id,
            "status": job.get("status"),
            "submitted_at": job.get("submitted_at"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "front_path": job.get("front_path"),
            "back_path": job.get("back_path"),
            "metadata": job.get("metadata") or {},
            "result": job.get("result"),
        }
        if not force and "result" not in job:
            return

    try:
        job_dir = job_dir_for(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        summary_path = job_dir / "summary.json"
        summary_path.write_text(json.dumps(record, indent=2))

        # Save details separately for consumers.
        if record.get("result") and "details" in (record["result"] or {}):
            details_path = job_dir / "details.json"
            details_path.write_text(json.dumps(record["result"]["details"], indent=2))

        # Append to rolling JSONL index for downstream learning pipelines.
        with INDEX_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # pragma: no cover - defensive persistence
        logger.error(f"Failed to persist job {job_id}: {exc}")


def _load_existing_jobs():
    """Load any previously completed jobs from disk into memory so recents persist."""
    try:
        for summary_file in EXPORT_ROOT.glob("*/summary.json"):
            try:
                data = json.loads(summary_file.read_text())
                jid = data.get("job_id") or summary_file.parent.name
                if not jid:
                    continue
                with jobs_lock:
                    jobs[jid] = {
                        "status": data.get("status", "completed"),
                        "submitted_at": data.get("submitted_at"),
                        "started_at": data.get("started_at"),
                        "completed_at": data.get("completed_at"),
                        "front_path": data.get("front_path"),
                        "back_path": data.get("back_path"),
                        "metadata": data.get("metadata") or {},
                        "result": data.get("result"),
                    }
            except Exception:
                continue
        if jobs:
            logger.info(f"Loaded {len(jobs)} historical mobile jobs from disk")
    except Exception as exc:
        logger.error(f"Failed to load historical jobs: {exc}")


_load_existing_jobs()


def _get_pipeline():
    """Lazy-load the TruScore Master Pipeline (same as Card Manager "Grade this card")."""
    global _PIPELINE
    with _pipeline_lock:
        if _PIPELINE is None:
            _ensure_qt_app()
            try:
                from modules.truscore_grading.truscore_master_pipeline import (
                    TruScoreMasterPipeline,
                )
            except Exception as exc:  # pragma: no cover - import guard
                logger.error(f"Failed to import TruScore Master Pipeline: {exc}")
                raise

            _PIPELINE = TruScoreMasterPipeline()
            logger.info("TruScore Master Pipeline initialized for Mobile API")
    return _PIPELINE


def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _job_to_recent(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a stored job into a lightweight recent scan entry."""
    result = job.get("result") or {}
    front = result.get("front") or {}

    thumb_front = f"/api/v1/cards/{job_id}/front" if job.get("front_path") else None
    thumb_back = f"/api/v1/cards/{job_id}/back" if job.get("back_path") else None

    grade = front.get("grade")
    if isinstance(grade, float):
        grade = f"{grade:.1f}"

    return {
        "job_id": job_id,
        "title": (job.get("metadata") or {}).get("title") or f"Submission {job_id[:6]}",
        "submitted_at": job.get("submitted_at"),
        "completed_at": job.get("completed_at"),
        "status": job.get("status"),
        "grade": grade,
        "thumbnail_front": thumb_front,
        "thumbnail_back": thumb_back,
    }


def _summarize_result(result: Any) -> Dict[str, Any]:
    """
    Reduce the rich pipeline output to API-friendly fields.

    Supports the master pipeline result object (TruScoreResults) as well as
    legacy dict-shaped results.
    """
    if result is None:
        return {}

    # Master pipeline object support
    if hasattr(result, "scores"):
        scores = getattr(result, "scores", None)
        surface = getattr(result, "surface_data", None)
        centering = getattr(result, "centering_data", {}) or {}
        corners = getattr(result, "corner_data", {}) or {}
        corner_scores = corners.get("scores", {}) if isinstance(corners, dict) else {}

        surface_integrity = getattr(surface, "surface_integrity", None)
        defects_count = getattr(surface, "defect_count", None)
        centering_score = centering.get("overall_centering_score") if isinstance(centering, dict) else None

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
            "surface_integrity": surface_integrity,
            "centering_score": centering_score,
            "corner_scores": corner_scores,
            "defects_count": defects_count,
            "processing_time": getattr(result, "processing_time", None),
            "timestamp": getattr(result, "timestamp", None),
        }

    # Legacy dict fallback
    insights = result.get("insights", {}) or {}
    photometric = result.get("photometric_analysis")
    centering = result.get("centering_analysis", {}) or {}
    corners = result.get("corner_analysis", {}) or {}

    surface_integrity = getattr(photometric, "surface_integrity", None) if photometric else None
    corner_scores = corners.get("scores", {}) if isinstance(corners, dict) else {}
    centering_score = centering.get("overall_centering_score")

    try:
        defects_count = len(result.get("smart_defects") or [])
    except Exception:
        defects_count = None

    return {
        "success": bool(result.get("success", False)),
        "grade": insights.get("overall_grade_estimate", "Unknown"),
        "grade_confidence": insights.get("grade_confidence"),
        "surface_integrity": surface_integrity,
        "centering_score": centering_score,
        "corner_scores": corner_scores,
        "defects_count": defects_count,
        "processing_time": result.get("processing_time"),
        "timestamp": result.get("timestamp"),
    }


def _sanitize_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    try:
        return float(value)
    except Exception:
        return None


def _detailed_snapshot(result: Any) -> Dict[str, Any]:
    """Extract a JSON-friendly snapshot for UI/learning without heavy arrays."""
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
        details["centering"] = _sanitize_value(centering)

    corners = getattr(result, "corner_data", None)
    if isinstance(corners, dict):
        details["corners"] = _sanitize_value(corners)

    surface = getattr(result, "surface_data", None)
    if surface:
        details["surface"] = {
            "surface_integrity": getattr(surface, "surface_integrity", None),
            "defect_count": getattr(surface, "defect_count", None),
            "surface_roughness": getattr(surface, "surface_roughness", None),
        }

    border = getattr(result, "border_data", None)
    if isinstance(border, dict):
        details["border"] = _sanitize_value(border)

    quality = getattr(result, "quality_statements", None)
    if isinstance(quality, dict):
        details["quality_statements"] = {
            k: v if isinstance(v, list) else [v] for k, v in quality.items() if k != "error"
        }

    return details


def _write_image(array: Any, dest: Path) -> Optional[str]:
    try:
        arr = np.array(array)
        if arr.size == 0:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), arr)
        return str(dest)
    except Exception as exc:
        logger.error(f"Failed to write visualization image {dest.name}: {exc}")
        return None


def _normalize_to_uint8(arr: np.ndarray, colorize: bool = False, colormap: int = cv2.COLORMAP_VIRIDIS) -> Optional[np.ndarray]:
    """Normalize an array to uint8 BGR for writing to disk."""
    try:
        arr = np.nan_to_num(arr)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            # Assume already color-ish; clamp to 0-255
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
            arr = arr.astype(np.uint8)
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            return arr
        if arr.ndim == 2:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if colorize:
                arr = cv2.applyColorMap(arr, colormap)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return arr
    except Exception as exc:
        logger.error(f"Failed to normalize array: {exc}")
    return None


def _render_centering_overlay(image_path: Optional[Path], centering: Dict[str, Any], dest: Path) -> Optional[str]:
    """Render a centering overlay (outer/inner polygons + rays) onto the captured image."""
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

        # Fallback: build polygons from border detector boxes if provided
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


def _serialize_visualizations(result: Any, job_id: str, side: str, image_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Extract visualization data into disk assets + lightweight metadata.
    Returns {assets: [{name, url}], meta: {...}}
    """
    viz = getattr(result, "visualization_data", None)
    out: Dict[str, Any] = {"assets": [], "meta": {}}
    if not isinstance(viz, dict):
        return out

    viz_dir = job_dir_for(job_id) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Photometric stereo maps (surface normals, depth, albedo, defects, confidence)
    photometric = viz.get("photometric_analysis")
    if photometric is not None:
        normals = getattr(photometric, "surface_normals", None)
        if normals is not None:
            arr = (np.clip(normals, -1, 1) + 1.0) * 127.5
            norm_img = _normalize_to_uint8(arr)
            if norm_img is not None:
                dest = viz_dir / f"{side}_surface_normals.png"
                cv2.imwrite(str(dest), norm_img)
                out["assets"].append({"name": "surface_normals", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})

        depth = getattr(photometric, "depth_map", None)
        if depth is not None:
            arr = _normalize_to_uint8(np.array(depth), colorize=True, colormap=cv2.COLORMAP_PLASMA)
            if arr is not None:
                dest = viz_dir / f"{side}_depth_map.png"
                cv2.imwrite(str(dest), arr)
                out["assets"].append({"name": "depth_map", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})

        albedo = getattr(photometric, "albedo_map", None)
        if albedo is not None:
            arr = _normalize_to_uint8(np.array(albedo))
            if arr is not None:
                dest = viz_dir / f"{side}_albedo_map.png"
                cv2.imwrite(str(dest), arr)
                out["assets"].append({"name": "albedo_map", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})

        defects = getattr(photometric, "defect_map", None)
        if defects is not None:
            arr = _normalize_to_uint8(np.array(defects), colorize=True, colormap=cv2.COLORMAP_MAGMA)
            if arr is not None:
                dest = viz_dir / f"{side}_defect_map.png"
                cv2.imwrite(str(dest), arr)
                out["assets"].append({"name": "defect_map", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})

        confidence = getattr(photometric, "confidence_map", None)
        if confidence is not None:
            arr = _normalize_to_uint8(np.array(confidence), colorize=True, colormap=cv2.COLORMAP_TURBO)
            if arr is not None:
                dest = viz_dir / f"{side}_confidence_map.png"
                cv2.imwrite(str(dest), arr)
                out["assets"].append({"name": "confidence_map", "url": f"/api/v1/cards/{job_id}/viz/{dest.name}"})

        out["meta"]["photometric"] = {
            "surface_integrity": getattr(photometric, "surface_integrity", None),
            "defect_count": getattr(photometric, "defect_count", None),
            "surface_roughness": getattr(photometric, "surface_roughness", None),
            "resolution": getattr(photometric, "resolution", None),
        }

    # Centering overlay (rendered even if no pixmap was preserved)
    centering = viz.get("centering_analysis")
    if isinstance(centering, dict):
        # Inject border box fallback so overlay can render even if polygons missing
        border_box = viz.get("border_analysis") if isinstance(viz.get("border_analysis"), dict) else None
        centering_for_overlay = dict(centering)
        if border_box:
            centering_for_overlay["_border_box_fallback"] = border_box

        out["meta"]["centering"] = _sanitize_value({k: v for k, v in centering.items() if k != "visualization_data"})
        overlay_path = viz_dir / f"{side}_centering_overlay.png"
        rendered = _render_centering_overlay(image_path, centering_for_overlay, overlay_path)
        if rendered:
            out["assets"].append({"name": "centering_overlay", "url": f"/api/v1/cards/{job_id}/viz/{overlay_path.name}"})

    # Corners + borders + insights
    for key in ("corner_analysis", "border_analysis", "smart_defects", "insights"):
        if key in viz:
            out["meta"][key] = _sanitize_value(viz.get(key))

    # Generic catch-all for any other visualization payloads
    for key, val in viz.items():
        if key in {"photometric_analysis", "centering_analysis", "corner_analysis", "border_analysis", "smart_defects", "insights"}:
            continue
        if val is None:
            continue
        # If val is a path-like string, copy it
        if isinstance(val, (str, Path)) and Path(val).exists():
            src = Path(val)
            fname = f"{side}_{src.name}"
            dest = viz_dir / fname
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)
                out["assets"].append({"name": key, "url": f"/api/v1/cards/{job_id}/viz/{fname}"})
            except Exception as exc:
                logger.error(f"Failed to copy viz asset {src}: {exc}")
            continue
        # If val looks like an array, save it as PNG
        if hasattr(val, "shape"):
            fname = f"{side}_{key}.png"
            dest = viz_dir / fname
            saved = _write_image(val, dest)
            if saved:
                out["assets"].append({"name": key, "url": f"/api/v1/cards/{job_id}/viz/{fname}"})
            continue
        # If it's a list of arrays, persist the first one or two
        if isinstance(val, list) and val and hasattr(val[0], "shape"):
            for i, arr in enumerate(val[:2]):
                fname = f"{side}_{key}_{i}.png"
                dest = viz_dir / fname
                saved = _write_image(arr, dest)
                if saved:
                    out["assets"].append({"name": f"{key}_{i}", "url": f"/api/v1/cards/{job_id}/viz/{fname}"})
            continue
        # Otherwise, sanitize into meta
        out["meta"][key] = _sanitize_value(val)
    return out


def _process_job(job_id: str, front_path: Path, back_path: Optional[Path], metadata: Dict[str, Any]):
    """Worker that runs the TruScore pipeline and records the result."""
    logger.info(f"Processing job {job_id} (front={front_path.name}, back={back_path.name if back_path else 'none'})")
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = time.time()

    try:
        pipeline = _get_pipeline()

        front_result = pipeline.analyze_card_master_pipeline(str(front_path))
        back_result = None
        if back_path and back_path.exists():
            back_result = pipeline.analyze_card_master_pipeline(str(back_path))

        # Run grading
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

        summary["visualizations"]["front"] = _serialize_visualizations(front_result, job_id, "front", front_path)
        if back_result:
            summary["visualizations"]["back"] = _serialize_visualizations(back_result, job_id, "back", back_path)

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_at"] = time.time()
            jobs[job_id]["result"] = summary
            jobs[job_id]["result_path"] = str(job_dir_for(job_id) / "summary.json")

        logger.info(f"Job {job_id} completed")
        _persist_job(job_id)
    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}")
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)
            jobs[job_id]["completed_at"] = time.time()
        _persist_job(job_id)


app = FastAPI(
    title="TruScore Mobile API Bridge",
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# Local-only CORS for dev
# Dev-friendly CORS (local/LAN). Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>TruScore Mobile API</title>
      <style>
        body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }
        .card { max-width: 720px; margin: 0 auto; background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
        h1 { color: #38bdf8; margin-bottom: 8px; }
        h2 { color: #c084fc; margin-top: 20px; }
        code, pre { background: #1f2937; color: #e5e7eb; padding: 2px 6px; border-radius: 6px; }
        a { color: #38bdf8; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .pill { display: inline-block; padding: 6px 10px; border-radius: 999px; background: #22c55e22; color: #22c55e; font-weight: 600; }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>TruScore Mobile API</h1>
        <div class="pill">Local Dev</div>
        <p>This endpoint is running. Use it with the mobile app or cURL.</p>

        <h2>Health Check</h2>
        <pre><code>GET /health</code></pre>

        <h2>Submit for Grading (multipart)</h2>
        <pre><code>POST /api/v1/cards/grade
front: required image file
back: optional image file
metadata: optional JSON string</code></pre>

        <h2>Check Job</h2>
        <pre><code>GET /api/v1/cards/&lt;job_id&gt;</code></pre>

        <h2>Docs</h2>
        <p><a href="/docs">OpenAPI / Swagger UI</a></p>
      </div>
    </body>
    </html>
    """


@app.post("/api/v1/cards/grade")
async def grade_card(
    front: UploadFile = File(..., description="Front image (300+ DPI recommended)"),
    back: UploadFile = File(None, description="Back image (optional but recommended)"),
    metadata: Optional[str] = Form(None, description="JSON string with metadata"),
):
    """Submit a card for grading. Returns a job_id for async polling."""
    try:
        meta_obj = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}") from exc

    job_id = uuid.uuid4().hex
    job_dir = EXPORT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Preserve original extensions when possible
    front_suffix = Path(front.filename or "front.jpg").suffix or ".jpg"
    back_suffix = Path(back.filename or "back.jpg").suffix or ".jpg" if back else ".jpg"

    front_path = job_dir / f"front{front_suffix}"
    back_path = job_dir / f"back{back_suffix}" if back else None

    _save_upload(front, front_path)
    if back and back_path:
        _save_upload(back, back_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "submitted_at": time.time(),
            "front_path": str(front_path),
            "back_path": str(back_path) if back_path else None,
            "metadata": meta_obj,
        }

    # Persist initial submission metadata/files for learning/archive.
    _persist_job(job_id, force=True)

    executor.submit(_process_job, job_id, front_path, back_path, meta_obj)
    logger.info(f"Job {job_id} queued")

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/v1/cards/recent")
def get_recent(limit: int = 10):
    """Return a list of recent completed/queued jobs for populating the home screen."""
    with jobs_lock:
        items = [
            (jid, job)
            for jid, job in jobs.items()
            if job.get("status") in {"completed", "running", "queued"}
        ]
    # Sort by completion time (fallback to submission time)
    items.sort(
        key=lambda pair: pair[1].get("completed_at") or pair[1].get("submitted_at") or 0,
        reverse=True,
    )
    recents = [_job_to_recent(jid, job) for jid, job in items[: max(1, limit)]]
    return {"items": recents, "count": len(recents)}


@app.get("/api/v1/cards/{job_id}")
def get_job(job_id: str):
    """Fetch job status/result."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/v1/cards/{job_id}/front")
def get_front_image(job_id: str):
    """Return the raw front image for a job (for thumbnails/previews)."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or not job.get("front_path"):
            raise HTTPException(status_code=404, detail="Not found")
        path = Path(job["front_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.get("/api/v1/cards/{job_id}/back")
def get_back_image(job_id: str):
    """Return the raw back image for a job (for thumbnails/previews)."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or not job.get("back_path"):
            raise HTTPException(status_code=404, detail="Not found")
        path = Path(job["back_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.get("/api/v1/cards/recent")
def get_recent(limit: int = 10):
    """Return a list of recent completed/queued jobs for populating the home screen."""
    with jobs_lock:
        items = [
            (jid, job)
            for jid, job in jobs.items()
            if job.get("status") in {"completed", "running", "queued"}
        ]
    # Sort by completion time (fallback to submission time)
    items.sort(
        key=lambda pair: pair[1].get("completed_at") or pair[1].get("submitted_at") or 0,
        reverse=True,
    )
    recents = [_job_to_recent(jid, job) for jid, job in items[: max(1, limit)]]
    return {"items": recents, "count": len(recents)}


@app.get("/api/v1/cards/{job_id}/market")
def get_market(job_id: str):
    """Placeholder market analysis response for a graded job."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        result = (job.get("result") or {}).get("front") or {}

    grade = result.get("grade")
    try:
        grade_num = float(grade)
    except Exception:
        grade_num = None

    # Lightweight placeholder data until real market pipeline is wired
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


@app.get("/api/v1/cards/{job_id}/visualizations")
def get_visualizations(job_id: str):
    """Return visualization metadata for front/back (assets + meta)."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        result = (job.get("result") or {}).get("visualizations") or {}
    return result


@app.get("/api/v1/cards/{job_id}/viz/{filename}")
def get_visualization_file(job_id: str, filename: str):
    """Serve a visualization asset (png, etc.) for the given job."""
    safe_name = Path(filename).name  # prevent path traversal
    path = job_dir_for(job_id) / "viz" / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


if __name__ == "__main__":
    # Allow direct execution: python -m mobile_api.server
    try:
        import uvicorn

        log_component_status("Mobile API Bridge", True)
        # Log available routes for easier diagnostics
        try:
            from fastapi.routing import APIRoute
            route_paths = [route.path for route in app.routes if isinstance(route, APIRoute)]
            logger.info(f"Mobile API routes: {route_paths}")
        except Exception:
            pass

        uvicorn.run(app, host="0.0.0.0", port=8009, reload=False)
    except Exception as exc:
        log_component_status("Mobile API Bridge", False, str(exc))
        raise
