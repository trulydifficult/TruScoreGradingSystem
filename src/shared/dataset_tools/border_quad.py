"""Pure image-pixel quadrilateral geometry for annotation borders.

Vertex order is TL, TR, BR, BL (clockwise visually in image coordinates).
This module knows nothing about Qt, YOLO, inference, or image rotation.
"""
from __future__ import annotations

from math import hypot, isfinite
from typing import Iterable

CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
SIDE_NAMES = ("top", "right", "bottom", "left")
Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


def validate_quad(vertices: Iterable[Iterable[float]]) -> Quad:
    """Require four finite, strictly convex, nondegenerate ordered vertices."""
    points = tuple(tuple(map(float, point)) for point in vertices)
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise ValueError("A border quad requires four (x, y) vertices")
    if any(not isfinite(value) for point in points for value in point):
        raise ValueError("Border quad contains non-finite coordinates")
    crosses = []
    for index in range(4):
        a, b, c = (points[(index + offset) % 4] for offset in (0, 1, 2))
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        crosses.append(cross)
    if min(crosses) <= 1e-6:
        raise ValueError("Border vertices must remain clockwise, convex, and nondegenerate")
    return points  # type: ignore[return-value]


def rect_to_quad(x1: float, y1: float, x2: float, y2: float) -> Quad:
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    return validate_quad(((left, top), (right, top), (right, bottom), (left, bottom)))


def quad_bounds(vertices: Quad) -> tuple[float, float, float, float]:
    return (min(p[0] for p in vertices), min(p[1] for p in vertices),
            max(p[0] for p in vertices), max(p[1] for p in vertices))


def side_midpoint(vertices: Quad, name: str) -> Point:
    index = SIDE_NAMES.index(name)
    a, b = vertices[index], vertices[(index + 1) % 4]
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def nearest_corner(vertices: Quad, x: float, y: float, tolerance: float) -> str | None:
    for index, (px, py) in enumerate(vertices):
        if hypot(float(x) - px, float(y) - py) <= tolerance:
            return CORNER_NAMES[index]
    return None


def nearest_side_handle(vertices: Quad, x: float, y: float, tolerance: float) -> str | None:
    for name in SIDE_NAMES:
        px, py = side_midpoint(vertices, name)
        if hypot(float(x) - px, float(y) - py) <= tolerance:
            return name
    return None


def quad_contains(vertices: Quad, x: float, y: float, margin: float = 0.0) -> bool:
    """Convex signed-halfplane hit test in image coordinates."""
    point = (float(x), float(y))
    for index in range(4):
        a, b = vertices[index], vertices[(index + 1) % 4]
        cross = ((b[0] - a[0]) * (point[1] - a[1])
                 - (b[1] - a[1]) * (point[0] - a[0]))
        if cross < -max(0.0, float(margin)) * hypot(b[0] - a[0], b[1] - a[1]):
            return False
    return True


def moved_corner(vertices: Quad, name: str, x: float, y: float) -> Quad:
    result = list(vertices)
    result[CORNER_NAMES.index(name)] = (float(x), float(y))
    return validate_quad(result)


def moved_side(vertices: Quad, name: str, x: float, y: float) -> Quad:
    """Move just two vertices by the requested midpoint displacement."""
    index = SIDE_NAMES.index(name)
    previous = side_midpoint(vertices, name)
    dx, dy = float(x) - previous[0], float(y) - previous[1]
    result = list(vertices)
    for vertex_index in (index, (index + 1) % 4):
        a, b = result[vertex_index]
        result[vertex_index] = (a + dx, b + dy)
    return validate_quad(result)


def moved_quad(vertices: Quad, dx: float, dy: float) -> Quad:
    return validate_quad(((x + float(dx), y + float(dy)) for x, y in vertices))


def quad_inside_image(vertices: Quad, width: int, height: int) -> bool:
    return all(0 <= x <= width and 0 <= y <= height for x, y in vertices)
