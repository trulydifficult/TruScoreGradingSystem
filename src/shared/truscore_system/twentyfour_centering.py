from __future__ import annotations

"""
Centering Analyzer
------------------
Drop-in module for your PyQt6 grading popup.

Features
- Accepts two borders as polygons (outer & graphic) OR temporary YOLO boxes.
- Measures 24 distances (mm) between borders at fixed locations:
  * Top 5 (points 1–5)
  * Bottom 5 (6–10)
  * Left 7 (11–17)
  * Right 7 (18–24)
- Computes centering ratios (top/bottom and left/right) and a final analysis string.
- Optional visualization: draws borders, measurement rays, and labels.

Usage (example)
---------------
from centering_analyzer import CenteringAnalyzer, yolo_box_to_polygon

outer = yolo_box_to_polygon((x1, y1, x2, y2))            # temporary YOLO box
inner = yolo_box_to_polygon((gx1, gy1, gx2, gy2))         # or a real polygon [(x,y),...]

an = CenteringAnalyzer(image_path, outer, inner)
results = an.run_analysis(show_visual=True)

# results dict contains:
# - measurements_mm: list[float] (24 values, order described above)
# - groups: dict with top/bottom/left/right arrays and means
# - ratios: dict with 'top_bottom' and 'left_right' as (pctA, pctB)
# - verdict: str
# - pixmap: Optional[QPixmap] (only if show_visual=True)

Integration
-----------
Embed results["pixmap"] in your existing tab (QLabel.setPixmap or a QGraphicsView),
show the table/ratios/verdict in your UI controls.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import math

from PyQt6.QtGui import QImage, QPainter, QPen, QBrush, QColor, QPixmap, QFont
from PyQt6.QtCore import QPointF, QRectF, Qt

Point = Tuple[float, float]
Polygon = List[Point]

CARD_WIDTH_MM = 63.5
CARD_HEIGHT_MM = 88.9
DEFAULT_DPI = 600.0
MM_PER_INCH = 25.4

# ----------------------------
# Utility geometry functions
# ----------------------------

def yolo_box_to_polygon(box: Tuple[float, float, float, float]) -> Polygon:
    """Convert YOLO box (x1,y1,x2,y2) to rectangle polygon in clockwise order.
    """
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def edge_vectors(poly: Polygon) -> List[Tuple[Point, Point, Point]]:
    """Return list of edges with (A, B, N) where A->B is the edge, N is inward unit normal.
    Assumes polygon is in clockwise order (common for image coords).
    """
    n = len(poly)
    edges = []
    for i in range(n):
        A = poly[i]
        B = poly[(i + 1) % n]
        # edge vector
        ex, ey = (B[0] - A[0], B[1] - A[1])
        length = math.hypot(ex, ey)
        if length == 0:
            continue
        ux, uy = ex / length, ey / length
        # outward normal for CW is (uy, -ux); inward is (-uy, ux)
        nx, ny = (-uy, ux)
        edges.append((A, B, (nx, ny)))
    return edges


def lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def line_intersection(p: Point, d: Point, a: Point, b: Point) -> Optional[Point]:
    """Intersect ray (p + t*d, t>=0) with segment ab. Return point or None if no forward hit.
    Using 2D cross product method.
    """
    px, py = p
    dx, dy = d
    ax, ay = a
    bx, by = b

    rx, ry = (dx, dy)
    sx, sy = (bx - ax, by - ay)
    rxs = rx * sy - ry * sx
    if abs(rxs) < 1e-9:
        return None  # parallel or colinear – ignore

    qpx, qpy = ax - px, ay - py
    t = (qpx * sy - qpy * sx) / rxs
    u = (qpx * ry - qpy * rx) / rxs
    if t >= 0 and 0 <= u <= 1:
        return (px + t * rx, py + t * ry)
    return None


def polygon_ray_intersections(origin: Point, direction: Point, poly: Polygon) -> List[Point]:
    hits = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ip = line_intersection(origin, direction, a, b)
        if ip is not None:
            hits.append(ip)
    # sort by distance from origin
    hits.sort(key=lambda q: math.hypot(q[0] - origin[0], q[1] - origin[1]))
    return hits


def poly_side_indices(poly: Polygon) -> Dict[str, Tuple[int, int]]:
    """Return indices (start,end) for logical sides Top, Right, Bottom, Left for a convex
    quadrilateral polygon in CW order. We choose the shortest path between vertices.
    If the polygon has 4 points, assume order: TL, TR, BR, BL for CW.
    For >4 points, we approximate by picking extreme points and ordering.
    """
    if len(poly) == 4:
        # Assume CW: (TL, TR, BR, BL)
        return {
            'top': (0, 1),
            'right': (1, 2),
            'bottom': (2, 3),
            'left': (3, 0),
        }

    # Fallback for arbitrary convex polygon: use bounding box extremes to pick edges
    # Find closest edges to top/bottom/left/right by average y/x.
    edges = edge_vectors(poly)
    def edge_mid(e):
        (a, b, _) = e
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    mids = [(i, edge_mid(e)) for i, e in enumerate(edges)]
    top_i = min(mids, key=lambda m: m[1][1])[0]
    bottom_i = max(mids, key=lambda m: m[1][1])[0]
    left_i = min(mids, key=lambda m: m[1][0])[0]
    right_i = max(mids, key=lambda m: m[1][0])[0]

    return {
        'top': (top_i, (top_i + 1) % len(poly)),
        'right': (right_i, (right_i + 1) % len(poly)),
        'bottom': (bottom_i, (bottom_i + 1) % len(poly)),
        'left': (left_i, (left_i + 1) % len(poly)),
    }


def sample_points_on_side(poly: Polygon, side: Tuple[int, int], count: int) -> List[Point]:
    i, j = side
    a, b = poly[i], poly[j]
    # Place samples evenly excluding vertices: t in (1/(n+1), ..., n/(n+1))
    pts = []
    for k in range(1, count + 1):
        t = k / (count + 1)
        pts.append(lerp(a, b, t))
    return pts


def inward_normal(a: Point, b: Point, cw: bool = True) -> Point:
    # Edge vector a->b
    ex, ey = (b[0] - a[0], b[1] - a[1])
    length = math.hypot(ex, ey)
    if length == 0:
        return (0.0, 0.0)
    ux, uy = ex / length, ey / length
    # For image coordinates (y down), CW polygon inward normal is (-uy, ux)
    nx, ny = (-uy, ux) if cw else (uy, -ux)
    return (nx, ny)


def distance(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# ----------------------------
# DPI & scaling helpers
# ----------------------------

def qimage_dpi(image: QImage) -> Optional[float]:
    # QImage stores dots-per-meter; convert if present
    dpmx = image.dotsPerMeterX()
    dpmy = image.dotsPerMeterY()
    if dpmx > 0 and dpmy > 0:
        # average to reduce anisotropy
        dpm = 0.5 * (dpmx + dpmy)
        dpi = dpm * 25.4 / 1000.0  # 1 meter = 1000 mm
        return dpi
    return None


def estimate_dpi_from_geometry(outer: Polygon) -> float:
    """As a fallback, estimate DPI from outer polygon and known card size.
    We assume the outer polygon is a (near) rectangle and use top and left edge lengths.
    """
    sides = poly_side_indices(outer)
    top_a, top_b = outer[sides['top'][0]], outer[sides['top'][1]]
    left_a, left_b = outer[sides['left'][0]], outer[sides['left'][1]]
    width_px = distance(top_a, top_b)
    height_px = distance(left_a, left_b)
    # Compute dpi from both dimensions (px / inches)
    dpi_w = (width_px / (CARD_WIDTH_MM / MM_PER_INCH)) if CARD_WIDTH_MM > 0 else DEFAULT_DPI
    dpi_h = (height_px / (CARD_HEIGHT_MM / MM_PER_INCH)) if CARD_HEIGHT_MM > 0 else DEFAULT_DPI
    # Geometric mean is robust
    dpi_est = math.sqrt(max(dpi_w, 1e-6) * max(dpi_h, 1e-6))
    return float(dpi_est)


# ----------------------------
# Core Analyzer
# ----------------------------
@dataclass
class AnalysisResults:
    measurements_mm: List[float]
    groups: Dict[str, Dict[str, List[float] | float]]
    ratios: Dict[str, Tuple[float, float]]  # e.g., { 'top_bottom': (55.0, 45.0), ... }
    verdict: str
    pixmap: Optional[QPixmap]
    rays: List[Tuple[Point, Point]]
    outer: Polygon
    inner: Polygon


class CenteringAnalyzer:
    def __init__(self,
                 image_path: str,
                 outer_border: Polygon | Tuple[float, float, float, float],
                 graphic_border: Polygon | Tuple[float, float, float, float],
                 force_clockwise: bool = True,
                 default_dpi: float = DEFAULT_DPI):
        self.image_path = image_path
        self.force_cw = force_clockwise  # Set this BEFORE calling _normalize_border
        self.default_dpi = default_dpi
        self.outer = self._normalize_border(outer_border)
        self.inner = self._normalize_border(graphic_border)

        self.image = QImage(self.image_path)
        if self.image.isNull():
            raise ValueError(f"Failed to load image: {self.image_path}")
        self.width = self.image.width()
        self.height = self.image.height()

        # Determine DPI
        dpi = qimage_dpi(self.image)
        if dpi is None or dpi <= 0:
            dpi = estimate_dpi_from_geometry(self.outer)
            if not math.isfinite(dpi) or dpi <= 50:  # sanity floor
                dpi = self.default_dpi
        self.dpi = dpi
        self.px_to_mm = MM_PER_INCH / self.dpi

    # ------------------------
    # Public API
    # ------------------------
    def run_analysis(self, show_visual: bool = False) -> AnalysisResults:
        measurements_px, viz, rays = self._measure_all(show_visual=show_visual)
        measurements_mm = [m * self.px_to_mm for m in measurements_px]

        # Grouping: top(1-5), bottom(6-10), left(11-17), right(18-24)
        top = measurements_mm[0:5]
        bottom = measurements_mm[5:10]
        left = measurements_mm[10:17]
        right = measurements_mm[17:24]

        def avg(arr: List[float]) -> float:
            return sum(arr) / len(arr) if arr else 0.0

        means = {
            'top': avg(top), 'bottom': avg(bottom), 'left': avg(left), 'right': avg(right)
        }

        # Ratios as percentages (A/B means A% vs B%)
        def ratio(a: float, b: float) -> Tuple[float, float]:
            s = a + b
            if s <= 0:
                return (50.0, 50.0)
            return (round(100.0 * a / s, 2), round(100.0 * b / s, 2))

        top_bottom_pct = ratio(means['top'], means['bottom'])
        left_right_pct = ratio(means['left'], means['right'])

        verdict = self._compose_verdict(top_bottom_pct, left_right_pct)

        results = AnalysisResults(
            measurements_mm=[round(v, 3) for v in measurements_mm],
            groups={
                'top': {'values': [round(v, 3) for v in top], 'avg': round(means['top'], 3)},
                'bottom': {'values': [round(v, 3) for v in bottom], 'avg': round(means['bottom'], 3)},
                'left': {'values': [round(v, 3) for v in left], 'avg': round(means['left'], 3)},
                'right': {'values': [round(v, 3) for v in right], 'avg': round(means['right'], 3)},
            },
            ratios={
                'top_bottom': top_bottom_pct,
                'left_right': left_right_pct,
            },
            verdict=verdict,
            pixmap=viz,
            rays=rays,
            outer=self.outer,
            inner=self.inner
        )
        return results

    # ------------------------
    # Internals
    # ------------------------
    def _normalize_border(self, border: Polygon | Tuple[float, float, float, float]) -> Polygon:
        if isinstance(border, (tuple, list)) and len(border) == 4 and all(isinstance(v, (int, float)) for v in border):
            poly = yolo_box_to_polygon(tuple(border))
        else:
            poly = [(float(x), float(y)) for x, y in border]  # type: ignore
        # Ensure clockwise order if desired (shoelace area < 0 means CW for image coords)
        if self.force_cw:
            if signed_area(poly) > 0:
                poly = list(reversed(poly))
        return poly

    def _measure_all(self, show_visual: bool) -> Tuple[List[float], Optional[QPixmap], List[Tuple[Point, Point]]]:
        outer = self.outer
        inner = self.inner

        # Determine sides of outer polygon
        sides_outer = poly_side_indices(outer)

        counts = {'top': 5, 'bottom': 5, 'left': 7, 'right': 7}
        order = ['top', 'bottom', 'left', 'right']

        measurements_px: List[float] = []
        ray_points: List[Tuple[Point, Point]] = []  # (start, end) for visualization

        for side_name in order:
            idx = sides_outer[side_name]
            a, b = outer[idx[0]], outer[idx[1]]
            samples = sample_points_on_side(outer, idx, counts[side_name])
            nvec = inward_normal(a, b, cw=True)

            for sp in samples:
                hits = polygon_ray_intersections(sp, nvec, inner)
                if not hits:
                    # If inward normal failed (e.g., polygons overlap oddly), try reverse
                    hits = polygon_ray_intersections(sp, (-nvec[0], -nvec[1]), inner)
                if hits:
                    ip = hits[0]
                    measurements_px.append(distance(sp, ip))
                    ray_points.append((sp, ip))
                else:
                    measurements_px.append(0.0)
                    ray_points.append((sp, sp))

        pixmap = self._render_overlay(outer, inner, ray_points) if show_visual else None
        return measurements_px, pixmap, ray_points

    def _render_overlay(self, outer: Polygon, inner: Polygon,
                         rays: List[Tuple[Point, Point]]) -> QPixmap:
        img = QPixmap.fromImage(self.image)
        pm = QPixmap(img.size())
        pm.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Draw original image first onto a new pixmap for compositing
        canvas = QPixmap(img)
        painter_img = QPainter(canvas)

        # Borders
        pen_outer = QPen(QColor(255, 0, 255), 3)  # pink
        pen_inner = QPen(QColor(0, 200, 0), 3)    # green
        painter_img.setPen(pen_outer)
        draw_polygon(painter_img, outer)
        painter_img.setPen(pen_inner)
        draw_polygon(painter_img, inner)

        # Rays + labels
        painter_img.setPen(QPen(QColor(255, 215, 0), 2))  # gold lines
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter_img.setFont(font)

        for i, (s, e) in enumerate(rays, start=1):
            painter_img.drawLine(QPointF(*s), QPointF(*e))
            # place label slightly outside the outer point
            lx, ly = s[0] - 6, s[1] - 6
            painter_img.drawText(QPointF(lx, ly), str(i))

        painter_img.end()
        painter.drawPixmap(0, 0, canvas)
        painter.end()
        return canvas

    def _compose_verdict(self, tb: Tuple[float, float], lr: Tuple[float, float]) -> str:
        top_pct, bot_pct = tb
        left_pct, right_pct = lr

        vertical_shift = "Centered"
        if abs(top_pct - 50.0) >= 0.5:
            vertical_shift = "Up" if top_pct > 50.0 else "Down"

        horizontal_shift = "Centered"
        if abs(left_pct - 50.0) >= 0.5:
            horizontal_shift = "Left" if left_pct > 50.0 else "Right"

        return (
            f"Card is {('slightly ' if (abs(top_pct-50)<3 and abs(left_pct-50)<3) else '')}"
            f"shifted {vertical_shift.lower()} and {horizontal_shift.lower()} with ratios "
            f"Top/Bottom = {top_pct:.2f}/{bot_pct:.2f}, "
            f"Left/Right = {left_pct:.2f}/{right_pct:.2f}."
        )


# ----------------------------
# Drawing helpers & geometry
# ----------------------------

def draw_polygon(p: QPainter, poly: Polygon):
    if not poly:
        return
    pts = [QPointF(x, y) for x, y in poly]
    for i in range(len(pts)):
        p.drawLine(pts[i], pts[(i + 1) % len(pts)])


def signed_area(poly: Polygon) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


# ----------------------------
# Convenience: pretty-print results as lines of text
# ----------------------------

def format_results_text(results: AnalysisResults) -> str:
    mm = results.measurements_mm
    g = results.groups
    tb = results.ratios['top_bottom']
    lr = results.ratios['left_right']

    lines = []
    lines.append("24-Point Measurements (mm):")
    lines.append(f"Top (1–5):    {g['top']['values']}  → Avg = {g['top']['avg']} mm")
    lines.append(f"Bottom (6–10): {g['bottom']['values']}  → Avg = {g['bottom']['avg']} mm")
    lines.append(f"Left (11–17): {g['left']['values']}  → Avg = {g['left']['avg']} mm")
    lines.append(f"Right (18–24): {g['right']['values']}  → Avg = {g['right']['avg']} mm")
    lines.append("")
    lines.append(f"Top/Bottom Centering: {tb[0]:.2f}% / {tb[1]:.2f}%")
    lines.append(f"Left/Right Centering: {lr[0]:.2f}% / {lr[1]:.2f}%")
    lines.append("")
    lines.append(results.verdict)
    return "\n".join(lines)


# ----------------------------
# Minimal demo hook (optional)
# ----------------------------
if __name__ == "__main__":
    # This block is intentionally light so the module stays drop-in friendly.
    # Example quick test (requires you to pass polygons programmatically).
    import sys
    from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QTextEdit, QSizePolicy

    # Dummy example: two nested rectangles
    outer = [(50, 50), (1530, 50), (1530, 2120), (50, 2120)]
    inner = [(130, 130), (1450, 130), (1450, 2050), (130, 2050)]

    image_path = sys.argv[1] if len(sys.argv) > 1 else 'sample.jpg'

    app = QApplication(sys.argv)
    try:
        analyzer = CenteringAnalyzer(image_path, outer, inner)
        res = analyzer.run_analysis(show_visual=True)
    except Exception as e:
        raise SystemExit(f"Error: {e}")

    w = QWidget()
    w.setWindowTitle("Centering Analyzer Demo")
    layout = QVBoxLayout(w)

    if res.pixmap:
        label = QLabel()
        label.setPixmap(res.pixmap)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    text = QTextEdit()
    text.setReadOnly(True)
    text.setText(format_results_text(res))
    layout.addWidget(text)

    w.resize(1000, 800)
    w.show()
    sys.exit(app.exec())
