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
Observation records linking a physical asset to the street-level image it was
detected in.

Street-level discovery (for example
:class:`~rapidtools.processing.MapillaryFeatureExtractor`) sees the same object
from several camera positions. Each sighting is an :class:`Observation`: which
image, where the camera stood and looked, the object's outline in normalised
image coordinates, and the ground position estimated from that single view.
The observations of one asset are stored as plain dictionaries under
``asset.attributes['observations']`` so they survive GeoJSON round trips, and
the imagery extractors read them back to decide which images to download and
where to crop.

Example:
    >>> from rapidtools.core import Observation
    >>> obs = Observation(
    ...     image_id='123', label='object--vehicle--car',
    ...     polygon=[(0.40, 0.55), (0.46, 0.55), (0.46, 0.62), (0.40, 0.62)],
    ...     camera_lon=-117.41, camera_lat=47.66, compass_angle=90.0,
    ... )
    >>> obs.bbox
    (0.4, 0.55, 0.46, 0.62)
    >>> Observation.from_dict(obs.to_dict()) == obs
    True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Observation:
    """
    One sighting of an object in a street-level image.

    Attributes:
        image_id (str):
            Provider image identifier (Mapillary image key).
        label (str):
            Class label of the detection, in the provider's vocabulary for
            Mapillary detections (``'object--vehicle--car'``) or the text
            prompt for model-based detections.
        polygon (list[tuple[float, float]]):
            Exterior ring of the detection in normalised image coordinates:
            ``x`` grows to the right and ``y`` downwards, both in ``[0, 1]``.
        camera_lon (float):
            Longitude of the camera in WGS84 degrees.
        camera_lat (float):
            Latitude of the camera in WGS84 degrees.
        compass_angle (float):
            Heading of the image centre column in degrees clockwise from
            north.
        is_pano (bool):
            ``True`` for equirectangular panoramas (360 degrees across the
            width, 180 down the height).
        sequence_id (str | None):
            Capture sequence the image belongs to; observations of one
            sequence share a camera rig.
        captured_at (str | None):
            Capture date or timestamp as an ISO string.
        confidence (float | None):
            Detector confidence in ``[0, 1]`` when the provider reports one.
        source (str):
            Where the detection came from: ``'mapillary'`` or ``'sam3'``.
        bearing (float | None):
            Absolute bearing from the camera to the object, degrees clockwise
            from north. Filled in by the localisation step.
        elevation (float | None):
            Elevation angle of the object's ground contact (its lowest pixel)
            in degrees; negative below the horizon.
        range_m (float | None):
            Estimated horizontal distance from camera to object in metres.
        lon (float | None):
            Single-view estimate of the object's longitude.
        lat (float | None):
            Single-view estimate of the object's latitude.
        image_width (int | None):
            Pixel width of the source image, when known.
        image_height (int | None):
            Pixel height of the source image, when known.
        extra (dict[str, Any]):
            Free-form provider metadata.
    """

    image_id: str
    label: str
    polygon: list[tuple[float, float]]
    camera_lon: float
    camera_lat: float
    compass_angle: float
    is_pano: bool = True
    sequence_id: str | None = None
    captured_at: str | None = None
    confidence: float | None = None
    source: str = 'mapillary'
    bearing: float | None = None
    elevation: float | None = None
    range_m: float | None = None
    lon: float | None = None
    lat: float | None = None
    image_width: int | None = None
    image_height: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise the polygon to a list of float tuples."""
        self.polygon = [(float(x), float(y)) for x, y in self.polygon]
        if len(self.polygon) < 3:
            raise ValueError('An observation polygon needs at least three vertices.')

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """
        Axis-aligned bounds of the polygon as ``(min_x, min_y, max_x, max_y)``.

        Example:
            >>> Observation('i', 'car', [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4)],
            ...             0.0, 0.0, 0.0).bbox
            (0.1, 0.2, 0.3, 0.4)
        """
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def area(self) -> float:
        """
        Polygon area as a fraction of the image (shoelace formula).

        Example:
            >>> Observation('i', 'car', [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
            ...             0.0, 0.0, 0.0).area
            0.25
        """
        pts = self.polygon
        total = 0.0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1], strict=True):
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    @property
    def has_location(self) -> bool:
        """``True`` once a single-view ground position has been estimated."""
        return self.lon is not None and self.lat is not None

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable dictionary of the observation.

        Example:
            >>> obs = Observation('i', 'car', [(0, 0), (1, 0), (1, 1)], 1.0, 2.0, 3.0)
            >>> d = obs.to_dict()
            >>> d['image_id'], d['polygon'][0]
            ('i', [0.0, 0.0])
        """
        data = asdict(self)
        data['polygon'] = [[x, y] for x, y in self.polygon]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """
        Rebuild an observation from :meth:`to_dict` output.

        Unknown keys are kept under ``extra`` so records written by a newer
        version still load.

        Example:
            >>> Observation.from_dict({
            ...     'image_id': 'i', 'label': 'car',
            ...     'polygon': [[0, 0], [1, 0], [1, 1]],
            ...     'camera_lon': 1.0, 'camera_lat': 2.0, 'compass_angle': 3.0,
            ... }).label
            'car'
        """
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = dict(kwargs.pop('extra', {}) or {})
        extra.update({k: v for k, v in data.items() if k not in known})
        kwargs['polygon'] = [tuple(p) for p in kwargs['polygon']]
        return cls(extra=extra, **kwargs)
