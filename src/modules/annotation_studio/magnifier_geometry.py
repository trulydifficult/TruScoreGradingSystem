"""Authoritative image-pixel to magnifier-pixel affine viewport.

The displayed photo crop and polygon edges MUST use this same viewport.
No separate canvas transform, mouse state, or plugin rendering pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class MagnifierViewport:
    """The source image's cropped square mapped to a fixed output square."""

    left: int
    top: int
    side: int
    scale: int
    output: int = 400

    @classmethod
    def centered(cls, image_x: float, image_y: float,
                 magnification: int, output: int = 400) -> "MagnifierViewport":
        if magnification not in (2, 4):
            raise ValueError("Only the Studio's 2x and 4x levels are supported")
        if output <= 0 or output % magnification:
            raise ValueError("Magnifier dimensions must be divisible by magnification")
        if not all(isfinite(v) for v in (image_x, image_y)):
            raise ValueError("Magnifier position must be finite")
        source_side = output // magnification
        return cls(
            left=round(image_x) - source_side // 2,
            top=round(image_y) - source_side // 2,
            side=source_side,
            scale=magnification,
            output=output,
        )

    def project(self, x: float, y: float) -> tuple[float, float]:
        """Image source point -> magnifier point (before any raster rounding)."""
        return ((x - self.left) * self.scale, (y - self.top) * self.scale)

    def unproject(self, x: float, y: float) -> tuple[float, float]:
        """Magnifier point -> source image point, preserving subpixel fractions."""
        return (self.left + x / self.scale, self.top + y / self.scale)
