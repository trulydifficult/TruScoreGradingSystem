#!/usr/bin/env python3
"""
SQLite cache/index for Dataset Studio
- Stores image metadata, thumbnail locations, and quality metrics
- One database per project under <project_root>/cache/dataset.sqlite
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import time

SCHEMA = {
    "images": (
        "CREATE TABLE IF NOT EXISTS images (\n"
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        "  path TEXT UNIQUE NOT NULL,\n"
        "  file_size INTEGER DEFAULT 0,\n"
        "  width INTEGER DEFAULT 0,\n"
        "  height INTEGER DEFAULT 0,\n"
        "  mtime INTEGER DEFAULT 0,\n"
        "  hash TEXT,\n"
        "  added_at INTEGER\n"
        ")"
    ),
    "thumbs": (
        "CREATE TABLE IF NOT EXISTS thumbs (\n"
        "  image_id INTEGER UNIQUE NOT NULL,\n"
        "  thumb_path TEXT NOT NULL,\n"
        "  thumb_w INTEGER DEFAULT 0,\n"
        "  thumb_h INTEGER DEFAULT 0,\n"
        "  updated_at INTEGER,\n"
        "  FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE\n"
        ")"
    ),
    "quality": (
        "CREATE TABLE IF NOT EXISTS quality (\n"
        "  image_id INTEGER UNIQUE NOT NULL,\n"
        "  lap_var REAL,\n"
        "  sobel_var REAL,\n"
        "  hist_var REAL,\n"
        "  score REAL,\n"
        "  computed_at INTEGER,\n"
        "  FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE\n"
        ")"
    ),
}

class CacheDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def _ensure_schema(self):
        cur = self.conn.cursor()
        for sql in SCHEMA.values():
            cur.execute(sql)
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # Images
    def upsert_image(self, path: Path, file_size: int, width: int, height: int, mtime: int, hashval: Optional[str] = None) -> int:
        cur = self.conn.cursor()
        now = int(time.time())
        cur.execute(
            "INSERT INTO images(path, file_size, width, height, mtime, hash, added_at)\n"
            "VALUES(?,?,?,?,?,?,?)\n"
            "ON CONFLICT(path) DO UPDATE SET file_size=excluded.file_size, width=excluded.width, height=excluded.height, mtime=excluded.mtime, hash=excluded.hash",
            (str(path), file_size, width, height, mtime, hashval, now),
        )
        self.conn.commit()
        cur.execute("SELECT id FROM images WHERE path=?", (str(path),))
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def get_image_id(self, path: Path) -> Optional[int]:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM images WHERE path=?", (str(path),))
        row = cur.fetchone()
        return int(row[0]) if row else None

    # Thumbnails
    def upsert_thumb(self, image_id: int, thumb_path: Path, w: int, h: int):
        cur = self.conn.cursor()
        now = int(time.time())
        cur.execute(
            "INSERT INTO thumbs(image_id, thumb_path, thumb_w, thumb_h, updated_at)\n"
            "VALUES(?,?,?,?,?)\n"
            "ON CONFLICT(image_id) DO UPDATE SET thumb_path=excluded.thumb_path, thumb_w=excluded.thumb_w, thumb_h=excluded.thumb_h, updated_at=excluded.updated_at",
            (image_id, str(thumb_path), w, h, now),
        )
        self.conn.commit()

    def get_thumb(self, image_id: int) -> Optional[Tuple[Path, int, int]]:
        cur = self.conn.cursor()
        cur.execute("SELECT thumb_path, thumb_w, thumb_h FROM thumbs WHERE image_id=?", (image_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Path(row[0]), int(row[1]), int(row[2])

    # Quality
    def upsert_quality(self, image_id: int, lap_var: float, sobel_var: float, hist_var: float, score: float):
        cur = self.conn.cursor()
        now = int(time.time())
        cur.execute(
            "INSERT INTO quality(image_id, lap_var, sobel_var, hist_var, score, computed_at)\n"
            "VALUES(?,?,?,?,?,?)\n"
            "ON CONFLICT(image_id) DO UPDATE SET lap_var=excluded.lap_var, sobel_var=excluded.sobel_var, hist_var=excluded.hist_var, score=excluded.score, computed_at=excluded.computed_at",
            (image_id, float(lap_var), float(sobel_var), float(hist_var), float(score), now),
        )
        self.conn.commit()

    def get_quality(self, image_id: int) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT lap_var, sobel_var, hist_var, score, computed_at FROM quality WHERE image_id=?", (image_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "lap_var": float(row[0]),
            "sobel_var": float(row[1]),
            "hist_var": float(row[2]),
            "score": float(row[3]),
            "computed_at": int(row[4]),
        }

__all__ = ["CacheDB"]
