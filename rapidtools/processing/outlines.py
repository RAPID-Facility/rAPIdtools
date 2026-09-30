# Copyright (c) 2025 The University of Washington
#
# This file is part of rapidtools.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# rapidtools. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 09-29-2026

"""
Drawing asset outlines onto image crops.

Shared by the aerial and street-level imagery extractors so that the
``outline_shape`` / ``outline_buffer`` / ``outline_width`` / ``outline_color``
options mean the same thing everywhere. All geometry is handled in the crop's
pixel space with shapely; real-world buffer distances are converted with a
metres-per-pixel figure supplied by the caller.

Example:
    >>> from PIL import Image
    >>> from rapidtools.processing.outlines import draw_outline
    >>> img = Image.new('RGB', (100, 100), 'black')
    >>> draw_outline(
    ...     img, [(30, 30), (70, 30), (70, 70), (30, 70)], is_closed=True,
    ...     shape='corners', buffer='4 px', width=3, color='red',
    ... )
    >>> img.getpixel((26, 26))
    (255, 0, 0)
"""

from __future__ import annotations

import math
import re

from PIL import Image, ImageColor, ImageDraw
from shapely.geometry import LineString, Point, Polygon

from rapidtools.constants import METERS_CONVERSION_FACTORS, UNIT_ALIASES

# Accepted values for ``outline_shape``:
OUTLINE_SHAPES = ('geometry', 'bbox', 'rotated_bbox', 'convex_hull', 'corners')

# Radius of the dot drawn for point assets without a buffer:
POINT_RADIUS_PX = 5


def parse_outline_buffer(buffer: float | str) -> tuple[float, str]:
    """
    Parse an ``outline_buffer`` specification into a magnitude and a kind.

    Args:
        buffer (float | str):
            A bare number or ``'12 px'`` (pixels), a percentage such as
            ``'10%'`` of the asset's longest pixel extent, or a real-world
            distance such as ``'2 m'`` or ``'5 ft'``.

    Returns:
        tuple[float, str]:
            The non-negative magnitude and its kind: ``'px'``, ``'%'`` or
            ``'m'`` (the magnitude is then in metres).

    Raises:
        ValueError: If the value cannot be parsed, uses an unsupported unit,
            or is negative.

    Example:
        >>> value, kind = parse_outline_buffer('5 ft')
        >>> round(value, 3), kind
        (1.524, 'm')
        >>> parse_outline_buffer('10%')
        (10.0, '%')
    """
    if isinstance(buffer, bool):
        raise ValueError(f'Could not parse outline buffer: {buffer!r}')
    try:
        if isinstance(buffer, (int, float)):
            value, kind = float(buffer), 'px'
        else:
            text = str(buffer).strip().lower()
            if text.endswith('%'):
                value, kind = float(text[:-1]), '%'
            elif match := re.match(r'^([\d.]+)\s*([a-z]+)$', text):
                value, unit = float(match.group(1)), match.group(2)
                standard_unit = UNIT_ALIASES.get(unit)
                if standard_unit == 'pixels':
                    kind = 'px'
                elif standard_unit in METERS_CONVERSION_FACTORS:
                    value /= METERS_CONVERSION_FACTORS[standard_unit]
                    kind = 'm'
                else:
                    raise ValueError(f"Unsupported outline buffer unit '{unit}'.")
            elif re.match(r'^[\d.]+$', text):
                value, kind = float(text), 'px'
            else:
                raise ValueError(f'Could not parse outline buffer: {buffer!r}')
    except ValueError as exc:
        if 'outline buffer' in str(exc):
            raise
        raise ValueError(f'Could not parse outline buffer: {buffer!r}') from exc
    if value < 0:
        raise ValueError('outline_buffer must not be negative.')
    return value, kind


def parse_outline_width(width: int | float | str) -> tuple[float, str]:
    """
    Parse an ``outline_width`` specification into a magnitude and a kind.

    Args:
        width (int | float | str):
            Pixels (``6``, ``'6'``, ``'6 px'``) or a percentage of the shorter
            image side (``'1.5%'``).

    Returns:
        tuple[float, str]: The positive magnitude and ``'px'`` or ``'%'``.

    Raises:
        ValueError: If the value cannot be parsed, is not positive, or uses a
            real-world unit.

    Example:
        >>> parse_outline_width('1.5%')
        (1.5, '%')
    """
    try:
        value, kind = parse_outline_buffer(width)
    except ValueError as exc:
        raise ValueError(
            f'Could not parse outline width: {width!r} (use pixels or a '
            'percentage of the shorter image side).'
        ) from exc
    if kind == 'm':
        raise ValueError(
            'outline_width must be in pixels or a percentage, not a real-world '
            'distance.'
        )
    if value <= 0:
        raise ValueError('outline_width must be positive.')
    return value, kind


def validate_outline_color(color: str | tuple[int, int, int]) -> None:
    """
    Raise ``ValueError`` for a colour PIL cannot interpret.

    Example:
        >>> validate_outline_color('#00ffff')
        >>> validate_outline_color('not-a-colour')
        Traceback (most recent call last):
        ...
        ValueError: unknown color specifier: 'not-a-colour'
    """
    if isinstance(color, str):
        ImageColor.getrgb(color)


def resolve_buffer_px(
    spec: tuple[float, str], extent_px: float, meters_per_pixel: float
) -> float:
    """
    Resolve a parsed buffer to pixels for one crop.

    Args:
        spec: Output of :func:`parse_outline_buffer`.
        extent_px: Longest side of the outline shape's bounds, in pixels (the
            reference for percentage buffers).
        meters_per_pixel: Ground sampling distance of the crop; ``0`` disables
            real-world buffers.

    Example:
        >>> resolve_buffer_px((2.0, 'm'), 50.0, 0.5)
        4.0
    """
    value, kind = spec
    if kind == '%':
        return extent_px * value / 100.0
    if kind == 'm':
        return value / meters_per_pixel if meters_per_pixel > 0 else 0.0
    return value


def resolve_width_px(spec: tuple[float, str], image_size: tuple[int, int]) -> int:
    """
    Resolve a parsed stroke width to whole pixels for one crop.

    Example:
        >>> resolve_width_px((10.0, '%'), (200, 100))
        10
        >>> resolve_width_px((0.2, 'px'), (200, 100))
        1
    """
    value, kind = spec
    if kind == '%':
        value = min(image_size) * value / 100.0
    return max(1, int(round(value)))


def outline_geometry(
    pixel_coords: list[tuple[float, float]], is_closed: bool, shape: str
):
    """
    Build the pixel-space shapely geometry that the outline traces.

    Args:
        pixel_coords: The asset's vertices in crop pixel coordinates.
        is_closed: Whether the vertices describe a polygon ring.
        shape: One of :data:`OUTLINE_SHAPES` (``'corners'`` uses the rotated
            rectangle).

    Returns:
        shapely.geometry.base.BaseGeometry: A ``Point``, ``LineString`` or
        ``Polygon`` in pixel coordinates (multi-part for self-crossing rings
        that had to be repaired).

    Example:
        >>> ring = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        >>> outline_geometry(ring, True, 'bbox').bounds
        (0.0, 0.0, 10.0, 10.0)
    """
    if len(pixel_coords) == 1:
        base = Point(pixel_coords[0])
    elif is_closed and len(pixel_coords) >= 3:
        base = Polygon(pixel_coords)
        if not base.is_valid:
            repaired = base.buffer(0)
            base = repaired if not repaired.is_empty else base.convex_hull
    else:
        base = LineString(pixel_coords)

    if shape == 'bbox':
        return base.envelope
    if shape in ('rotated_bbox', 'corners'):
        return base.minimum_rotated_rectangle
    if shape == 'convex_hull':
        return base.convex_hull
    return base


def draw_outline(
    image: Image.Image,
    pixel_coords: list[tuple[float, float]],
    is_closed: bool,
    *,
    shape: str = 'geometry',
    buffer: float | str | tuple[float, str] = 0,
    width: int | float | str | tuple[float, str] = 6,
    color: str | tuple[int, int, int] = 'red',
    meters_per_pixel: float = 0.0,
) -> None:
    """
    Draw an asset outline onto a crop, in place.

    Args:
        image: The crop to annotate.
        pixel_coords: The asset's vertices in crop pixel coordinates.
        is_closed: Whether the vertices describe a polygon ring.
        shape: One of :data:`OUTLINE_SHAPES`.
        buffer: Expansion of the shape, as accepted by
            :func:`parse_outline_buffer` or already parsed.
        width: Stroke width, as accepted by :func:`parse_outline_width` or
            already parsed.
        color: PIL colour of the stroke.
        meters_per_pixel: Ground sampling distance for real-world buffers.
    """
    if not pixel_coords:
        return
    if shape not in OUTLINE_SHAPES:
        raise ValueError(
            f'outline shape must be one of {OUTLINE_SHAPES}, got {shape!r}.'
        )
    buffer_spec = buffer if isinstance(buffer, tuple) else parse_outline_buffer(buffer)
    width_spec = width if isinstance(width, tuple) else parse_outline_width(width)
    geom = outline_geometry(pixel_coords, is_closed, shape)
    minx, miny, maxx, maxy = geom.bounds
    buffer_px = resolve_buffer_px(
        buffer_spec, max(maxx - minx, maxy - miny), meters_per_pixel
    )
    if buffer_px > 0:
        # Mitred joins keep rectangles rectangular; points become rings.
        geom = geom.buffer(buffer_px, join_style='mitre')
    draw_geometry(
        ImageDraw.Draw(image),
        geom,
        width=resolve_width_px(width_spec, image.size),
        color=color,
        brackets=shape == 'corners',
    )


def draw_geometry(draw, geom, width: int, color, brackets: bool) -> None:
    """
    Trace a pixel-space geometry with PIL.

    Args:
        draw: A :class:`PIL.ImageDraw.ImageDraw` bound to the crop.
        geom: The geometry to trace; multi-part geometries are drawn part by
            part.
        width: Stroke width in pixels.
        color: PIL colour of the stroke.
        brackets: Draw four-cornered polygons as corner brackets instead of a
            closed ring.
    """
    if geom.is_empty:
        return
    if geom.geom_type.startswith('Multi') or geom.geom_type == 'GeometryCollection':
        for part in geom.geoms:
            draw_geometry(draw, part, width, color, brackets)
    elif geom.geom_type == 'Point':
        x, y, r = geom.x, geom.y, POINT_RADIUS_PX
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    elif geom.geom_type == 'LineString':
        draw.line(list(geom.coords), fill=color, width=width)
    elif geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
        if brackets and len(coords) == 5:
            draw_corner_brackets(draw, coords[:4], width, color)
        else:
            draw.line(coords, fill=color, width=width)


def draw_corner_brackets(draw, corners, width: int, color) -> None:
    """
    Draw an L-shaped bracket at each corner of a quadrilateral.

    Each arm runs a quarter of the shorter side (at least twice the stroke
    width) towards the neighbouring corner, leaving the asset's edges
    uncovered.

    Args:
        draw: A :class:`PIL.ImageDraw.ImageDraw` bound to the crop.
        corners: The four corner points in ring order.
        width: Stroke width in pixels.
        color: PIL colour of the stroke.
    """
    n = len(corners)
    sides = [math.dist(corners[i], corners[(i + 1) % n]) for i in range(n)]
    arm = max(2.0 * width, 0.25 * min(sides))
    for i, (cx, cy) in enumerate(corners):
        ends = []
        for nx, ny in (corners[i - 1], corners[(i + 1) % n]):
            length = math.hypot(nx - cx, ny - cy)
            if length == 0:
                continue
            t = min(arm, length / 2.0) / length
            ends.append((cx + (nx - cx) * t, cy + (ny - cy) * t))
        if len(ends) == 2:
            draw.line([ends[0], (cx, cy), ends[1]], fill=color, width=width)
            # Fill the notch a thick polyline leaves at its corner:
            r = width / 2.0
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
