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
# Barbaros Cetiner
#
# Last updated:
# 09-22-2026

"""
Geometry helpers for matching street-level panoramas to physical assets.

This module contains the spatial machinery used to pick the best 360-degree
panorama for each face of a building footprint and to crop that panorama
down to the part of the image that actually shows the asset:

    * Spatial indexing of footprints (:func:`build_footprint_index`) and
      panorama locations (:func:`index_panos`).
    * A local equirectangular projection (:func:`get_local_projection_func`)
      and principal-axis estimation (:func:`get_principal_axes`) used to
      rotate candidate cameras into a footprint-aligned frame.
    * Ray casting and line-of-sight occlusion checks
      (:func:`find_best_panos`, :func:`_check_occlusion`).
    * Equirectangular cropping that handles the 180/-180 degree seam and
      optional mask-driven vertical trimming (:func:`crop_panorama_to_asset`).

Example:
    >>> from rapidtools.processing.pano_utils import (
    ...     build_footprint_index, index_panos, find_best_panos
    ... )
    >>>
    >>> tree, geoms = build_footprint_index(buildings)  # doctest: +SKIP
    >>> pano_tree, coords, headings = index_panos(panos)  # doctest: +SKIP
    >>> best = find_best_panos(
    ...     buildings[0], panos, tree, geoms, pano_tree, coords
    ... )  # doctest: +SKIP
    >>> sorted(best)  # doctest: +SKIP
    ['major_neg', 'major_pos', 'minor_neg', 'minor_pos']
"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
from PIL.ImageChops import offset
from scipy.spatial import cKDTree
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree

from rapidtools.constants import EARTH_RADIUS_KM, LATITUDE_SPACING_KM
from rapidtools.core import (
    ImageAsset,
    ImageCollection,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.data_sources.mapillary_client import SegmentationLabels

logger = logging.getLogger(__name__)

# Define global variables:
PLOT_FOOTPRINT = False
DEBUG_PLOTS = False

SEARCH_RADIUS = 60
MAX_CANDIDATES = 100
RAY_TOLERANCE_METERS = 10.0
OFF_AXIS_INTERVAL_DEG = 90


def _transform_polygon_to_local_rotated(
    wgs84_poly: Polygon, projector_func, cos_t: float, sin_t: float
) -> Polygon:
    """
    Project and rotate a WGS84 polygon into the local analysis frame.

    Only the exterior ring is transformed; holes are ignored because the
    result is used for visualization and occlusion sketches only.

    Args:
        wgs84_poly (Polygon):
            Polygon in longitude/latitude coordinates.
        projector_func (Callable):
            Function ``(lon, lat) -> (x, y)`` returning local planar
            coordinates in meters, e.g. from
            :func:`get_local_projection_func`.
        cos_t (float):
            Cosine of the rotation angle applied after projection.
        sin_t (float):
            Sine of the rotation angle applied after projection.

    Returns:
        Polygon:
            The projected and rotated polygon in local meters.

    Example:
        >>> from shapely.geometry import box
        >>>
        >>> project = get_local_projection_func(0.0, 0.0)
        >>> poly = _transform_polygon_to_local_rotated(
        ...     box(0, 0, 0.001, 0.001), project, 1.0, 0.0
        ... )
        >>> round(poly.area)
        12364
    """
    coords = list(wgs84_poly.exterior.coords)

    local_coords = []
    for lon, lat in coords:
        # Project to meters (relative to target centroid):
        x, y = projector_func(lon, lat)

        # Rotate to align with target axes:
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        local_coords.append((x_rot, y_rot))

    return Polygon(local_coords)


def build_footprint_index(
    asset_collection: PhysicalAssetCollection,
) -> tuple[STRtree, list]:
    """
    Build a spatial index (R-tree) for building footprints.

    Args:
        asset_collection (PhysicalAssetCollection):
            Iterable of ``PhysicalAsset`` objects. Assets whose ``geometry``
            is ``None`` are skipped.

    Returns:
        tuple[STRtree, list]:
            ``(tree, geoms)`` where ``tree`` is the Shapely ``STRtree`` and
            ``geoms`` is the list of geometries in tree order. Because
            Shapely 2 queries return integer indices, ``geoms`` is needed to
            look up the actual polygon.

    Example:
        >>> from shapely.geometry import box
        >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
        >>>
        >>> assets = PhysicalAssetCollection([
        ...     PhysicalAsset(id='a', geometry=box(0, 0, 1, 1)),
        ...     PhysicalAsset(id='b', geometry=box(5, 5, 6, 6)),
        ... ])
        >>> tree, geoms = build_footprint_index(assets)
        >>> len(geoms)
        2
    """
    geoms = [asset.geometry for asset in asset_collection if asset.geometry is not None]
    tree = STRtree(geoms)
    return tree, geoms


def get_nearby_buildings(
    target_poly: Polygon, tree: STRtree, geom_list: list, radius_meters: float
) -> list[Polygon]:
    """
    Find footprint polygons within a radius of a target using the R-tree.

    The metric radius is converted into an approximate degree envelope
    around the target's bounds, the tree is queried with that envelope, and
    the target polygon itself is excluded from the result.

    Args:
        target_poly (Polygon):
            The polygon (WGS84) around which to search.
        tree (STRtree):
            Spatial index built by :func:`build_footprint_index`.
        geom_list (list):
            Geometries in the same order used to build ``tree``.
        radius_meters (float):
            Search radius in meters.

    Returns:
        list[Polygon]:
            Candidate polygons whose bounding boxes fall within the search
            envelope, excluding any geometry equal to ``target_poly``.

    Example:
        >>> from shapely.geometry import box
        >>> from shapely.strtree import STRtree
        >>>
        >>> geoms = [box(0, 0, 0.0001, 0.0001), box(0.0002, 0, 0.0003, 0.0001)]
        >>> tree = STRtree(geoms)
        >>> len(get_nearby_buildings(geoms[0], tree, geoms, radius_meters=50))
        1
    """
    # Convert radius (meters) to degrees (approximate). Latitude: 1 deg ~=
    # 111 km. Longitude: 1 deg ~= 111 km * cos(lat).
    lat = target_poly.centroid.y
    deg_per_meter_lat = 1.0 / 111111.0
    deg_per_meter_lon = 1.0 / (111111.0 * math.cos(math.radians(lat)))

    dx = radius_meters * deg_per_meter_lon
    dy = radius_meters * deg_per_meter_lat

    # Create a search bounding box (envelope) around the target:
    minx, miny, maxx, maxy = target_poly.bounds
    search_envelope = box(minx - dx, miny - dy, maxx + dx, maxy + dy)

    # Query the tree. In Shapely 2.0+ tree.query returns geometry indices:
    indices = tree.query(search_envelope)

    # Retrieve actual polygons, excluding the target itself:
    neighbors = []
    for i in indices:
        candidate = geom_list[i]
        if not candidate.equals(target_poly):
            neighbors.append(candidate)

    return neighbors


def index_panos(
    image_collection: ImageCollection,
) -> tuple[cKDTree | None, np.ndarray, np.ndarray]:
    """
    Build a spatial index over pano images for fast nearest-neighbor search.

    Reads ``longitude``, ``latitude`` and ``compass_angle`` from each image's
    ``properties`` (defaulting to ``0.0`` when missing).

    Args:
        image_collection (ImageCollection):
            Collection of ``ImageAsset`` objects with geographic properties.

    Returns:
        tuple[cKDTree | None, np.ndarray, np.ndarray]:
            ``(tree, coords_deg, headings)`` where ``tree`` is a
            ``scipy.spatial.cKDTree`` over ``(lon, lat)`` pairs,
            ``coords_deg`` is an ``(N, 2)`` float32 array of those pairs and
            ``headings`` is an ``(N,)`` array of compass angles. If the
            collection is empty, ``(None, [], [])`` is returned.

    Example:
        >>> from rapidtools.core import ImageAsset, ImageCollection
        >>>
        >>> panos = ImageCollection([
        ...     ImageAsset(
        ...         path='p1.jpg', id='p1', allow_missing_file=True,
        ...         properties={'longitude': -118.1, 'latitude': 34.2,
        ...                     'compass_angle': 90.0},
        ...     )
        ... ])
        >>> tree, coords, headings = index_panos(panos)
        >>> coords.shape, float(headings[0])
        ((1, 2), 90.0)
    """
    # Iterate over the collection once to grab the properties:
    data = [
        (
            img.properties.get('longitude', 0.0),
            img.properties.get('latitude', 0.0),
            img.properties.get('compass_angle', 0.0),
        )
        for img in image_collection
    ]

    if not data:
        return None, np.array([]), np.array([])

    # Bulk conversion to NumPy: create one array (N, 3) and then slice it:
    data_arr = np.array(data, dtype=np.float32)

    # Get pano coordinates: All rows, columns 0 and 1 (lon, lat):
    camera_coords_deg = data_arr[:, 0:2]

    # Get pano headings: All rows, column 2:
    camera_headings = data_arr[:, 2]

    # Build the KDTree:
    pano_tree = cKDTree(camera_coords_deg)

    return pano_tree, camera_coords_deg, camera_headings


def get_local_projection_func(centroid_lon: float, centroid_lat: float):
    """
    Return a function projecting geographic to local planar coordinates.

    The returned projector applies an equirectangular approximation centred
    on ``(centroid_lon, centroid_lat)``: X grows eastward and Y grows
    northward, both in meters. Longitude deltas are wrapped to
    ``[-180, 180)`` degrees before scaling.

    Args:
        centroid_lon (float):
            Longitude of the local origin in degrees.
        centroid_lat (float):
            Latitude of the local origin in degrees.

    Returns:
        Callable[[float | np.ndarray, float | np.ndarray], tuple]:
            ``project(lon, lat) -> (x, y)`` accepting scalars or NumPy
            arrays and returning meters relative to the origin.

    Example:
        >>> project = get_local_projection_func(0.0, 0.0)
        >>> x, y = project(0.0, 0.001)
        >>> round(float(x)), round(float(y))
        (0, 111)
    """
    # Calculate function constants:
    r_earth = EARTH_RADIUS_KM * 1000
    centroid_lon_rad = np.radians(centroid_lon)
    centroid_lat_rad = np.radians(centroid_lat)
    cos_centroid_lat = np.cos(centroid_lat_rad)

    def project(lon, lat):
        """Project ``(lon, lat)`` degrees to local ``(x, y)`` meters."""
        # Convert longitude and latitude values to radians:
        lon_rad = np.radians(lon, dtype=np.float32)
        lat_rad = np.radians(lat, dtype=np.float32)

        # Calculate deltas between point location and centroid:
        dlon = lon_rad - centroid_lon_rad
        dlon = (dlon + np.pi) % (2 * np.pi) - np.pi

        # Equirectangular projection:
        x = dlon * cos_centroid_lat * r_earth
        y = (lat_rad - centroid_lat_rad) * r_earth

        return x, y

    return project


def get_principal_axes(polygon_local) -> tuple[float, float]:
    """
    Compute the orientations of the major and minor axes of a polygon.

    The polygon's minimum rotated rectangle is used to determine the two
    principal directions. Angles follow the compass convention:
    0 degrees = +Y (north), 90 degrees = +X (east), clockwise positive.

    Args:
        polygon_local (Polygon):
            Polygon in a planar coordinate system (e.g., local meters).

    Returns:
        tuple[float, float]:
            ``(major_angle_deg, minor_angle_deg)`` in ``[0, 360)``.

    Example:
        >>> from shapely.geometry import box
        >>>
        >>> major, minor = get_principal_axes(box(0, 0, 10, 2))
        >>> round(major % 180), round(minor % 180)
        (90, 0)
    """
    # Convert the building outline to a rotated rectangle to identify principal
    # directions:
    mbr = polygon_local.minimum_rotated_rectangle
    x, y = mbr.exterior.coords.xy

    # Get the first three unique corners of the rotated rectangle:
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[1], y[1]])
    p2 = np.array([x[2], y[2]])

    # Compute edge vectors for the two adjacent sides of the rectangle
    # and determine their lengths (short side vs. long side):
    edge1 = p1 - p0
    edge2 = p2 - p1
    edge1_len = np.linalg.norm(edge1)
    edge2_len = np.linalg.norm(edge2)

    # Determine the direction of the longer (major) edge and minor (shorter)
    # edge:
    if edge1_len >= edge2_len:
        major_vec = edge1
        minor_vec = edge2
    else:
        major_vec = edge2
        minor_vec = edge1

    # Convert Cartesian vector to angle with convention
    # 0 = +Y, 90 = +X, clockwise positive:
    return _vec_to_angle(major_vec), _vec_to_angle(minor_vec)


def _vec_to_angle(v) -> float:
    """
    Convert a 2D vector into a compass orientation angle in degrees.

    Args:
        v (Sequence[float]):
            A ``(dx, dy)`` vector.

    Returns:
        float:
            Angle in ``[0, 360)`` where 0 = +Y, 90 = +X (clockwise positive).

    Example:
        >>> float(_vec_to_angle((1.0, 0.0)))
        90.0
        >>> float(_vec_to_angle((0.0, -1.0)))
        180.0
    """
    dx, dy = v[0], v[1]
    rads = np.arctan2(dx, dy)
    return np.degrees(rads) % 360


def _check_occlusion(
    camera_pt_wgs84: tuple[float, float],
    target_poly_wgs84: Polygon,
    neighboring_polys_wgs84: list[Polygon],
) -> bool:
    """
    Check if the line of sight from camera to target is blocked by neighbors.

    A line is drawn from the camera to the centroid of the target polygon.
    If that line passes through the interior of any neighboring polygon
    (intersection length > 0), the view is considered occluded. Merely
    touching a corner does not count.

    Args:
        camera_pt_wgs84 (tuple[float, float]):
            ``(lon, lat)`` of the camera.
        target_poly_wgs84 (Polygon):
            The Shapely polygon of the building we want to see.
        neighboring_polys_wgs84 (list[Polygon]):
            Other building polygons to check against. Any polygon equal to
            the target is skipped.

    Returns:
        bool:
            ``True`` if the view is occluded (blocked), ``False`` if clear.
            Also ``False`` when there are no neighbors or when the camera
            lies inside the target itself.

    Example:
        >>> from shapely.geometry import box
        >>>
        >>> target = box(0, 0, 1, 1)
        >>> blocker = box(2, 0, 3, 1)
        >>> _check_occlusion((5.0, 0.5), target, [blocker])
        True
        >>> _check_occlusion((0.5, 5.0), target, [blocker])
        False
    """
    if not neighboring_polys_wgs84:
        return False

    cam_point = Point(camera_pt_wgs84)

    # If the camera is inside the target building, strictly speaking it's not
    # occluded by *neighbors*, though it might be invalid for other reasons.
    if target_poly_wgs84.contains(cam_point):
        return False

    # Construct a line of sight (LOS) from the camera to the target centroid:
    target_centroid = target_poly_wgs84.centroid
    los_line = LineString([cam_point, target_centroid])

    # Check intersection with neighbors:
    for poly in neighboring_polys_wgs84:
        # Skip if the neighbor is actually the target itself:
        if poly.equals(target_poly_wgs84):
            continue

        if poly.intersects(los_line):
            # If the intersection has length > 0 (i.e. it passes through the
            # building interior or along an edge), it is an occlusion.
            # Touching a corner (Point intersection) doesn't count.
            intersection = poly.intersection(los_line)
            if not intersection.is_empty and intersection.length > 1e-9:
                return True

    return False


def find_best_panos(
    target_asset_wgs84: PhysicalAsset,
    pano_collection: ImageCollection,
    building_tree: STRtree,
    building_geoms: list,
    tree: cKDTree,
    coords_deg: np.ndarray,
    print_results: bool = False,
    search_radius_meters: float = SEARCH_RADIUS,
    ray_tolerance_meters: float = RAY_TOLERANCE_METERS,
    interval_deg: float = OFF_AXIS_INTERVAL_DEG,
    max_candidates: int = MAX_CANDIDATES,
    cast_corner_rays: bool = False,
) -> dict:
    """
    Find the single best panorama for each viewing direction of a footprint.

    The algorithm:

        1. Extracts the (largest) footprint polygon of the target asset.
        2. Collects nearby footprints as potential occluders.
        3. Queries the pano KD-tree for candidates within the search radius.
        4. Projects candidates to local meters and rotates them so the
           footprint's major axis aligns with +X.
        5. Casts rays from the footprint centroid (the four principal axes
           plus either corner rays or evenly spaced off-axis rays) and
           collects the candidates aligned with each ray.
        6. Returns, per ray, the best-aligned candidate whose line of sight
           is not blocked by a neighboring footprint.

    Args:
        target_asset_wgs84 (PhysicalAsset):
            Asset with a ``Polygon`` or ``MultiPolygon`` geometry in WGS84.
        pano_collection (ImageCollection):
            The collection of panorama assets that ``tree`` indexes.
        building_tree (STRtree):
            Footprint index from :func:`build_footprint_index`.
        building_geoms (list):
            Geometries backing ``building_tree``.
        tree (cKDTree):
            Pano location index from :func:`index_panos`.
        coords_deg (np.ndarray):
            ``(N, 2)`` array of pano ``(lon, lat)`` from :func:`index_panos`.
        print_results (bool):
            If ``True``, prints the selected image ID per ray. Defaults to
            ``False``.
        search_radius_meters (float):
            Maximum camera distance from the footprint centroid in meters.
        ray_tolerance_meters (float):
            Maximum perpendicular distance from a ray for a camera to count
            as aligned with it.
        interval_deg (float):
            Angular spacing of off-axis rays when ``cast_corner_rays`` is
            ``False``.
        max_candidates (int):
            Maximum number of KD-tree neighbours to consider.
        cast_corner_rays (bool):
            If ``True``, casts rays toward the four corners of the minimum
            rotated rectangle instead of evenly spaced off-axis rays.

    Returns:
        dict[str, ImageAsset | None]:
            Mapping from ray name (``'major_pos'``, ``'minor_pos'``,
            ``'major_neg'``, ``'minor_neg'``, ``'corner_N'`` or
            ``'ray_N'``) to the chosen ``ImageAsset``, or ``None`` when no
            unobstructed candidate exists for that ray. An empty dict is
            returned if the geometry is not polygonal or no candidates fall
            within the search radius.

    Example:
        >>> tree, geoms = build_footprint_index(buildings)  # doctest: +SKIP
        >>> pano_tree, coords, _ = index_panos(panos)  # doctest: +SKIP
        >>> best = find_best_panos(
        ...     buildings['bldg_1'], panos, tree, geoms, pano_tree, coords,
        ...     search_radius_meters=40,
        ... )  # doctest: +SKIP
        >>> best['major_pos'].id  # doctest: +SKIP
        '1234567890'
    """
    # 1. Geometry validation & setup:
    raw_geom = target_asset_wgs84.geometry

    if raw_geom.geom_type not in ['Polygon', 'MultiPolygon']:
        return {}

    # Extract the building footprint:
    if raw_geom.geom_type == 'MultiPolygon':
        footprint_wgs84 = max(raw_geom.geoms, key=lambda a: a.area)
    else:
        footprint_wgs84 = raw_geom

    # Extract the centroid:
    centroid = footprint_wgs84.centroid
    centroid_lon, centroid_lat = centroid.x, centroid.y

    # 2. Efficient neighbor detection (occlusion). Use an occlusion radius
    # slightly larger than the search radius to catch buildings that might
    # block the view:
    occlusion_radius = search_radius_meters * 1.5

    potential_occluders = get_nearby_buildings(
        footprint_wgs84, building_tree, building_geoms, radius_meters=occlusion_radius
    )

    # get_nearby_buildings only excludes geometries equal to the extracted
    # footprint. When the target is a MultiPolygon, its full geometry is still
    # in the index and would otherwise occlude itself:
    potential_occluders = [p for p in potential_occluders if not p.equals(raw_geom)]

    # 3. Pano search (broad phase):
    radius_deg = (search_radius_meters * 2) / (LATITUDE_SPACING_KM * 1000)
    dists_deg, indices = tree.query(
        [centroid_lon, centroid_lat], k=max_candidates, distance_upper_bound=radius_deg
    )

    dists_deg = np.atleast_1d(dists_deg)
    indices = np.atleast_1d(indices)
    valid_mask = np.isfinite(dists_deg)
    indices = indices[valid_mask]

    if len(indices) == 0:
        return {}

    # 4. Projection to local meters:
    projector = get_local_projection_func(centroid_lon, centroid_lat)

    cand_lons_filter1 = coords_deg[indices, 0]
    cand_lats_filter1 = coords_deg[indices, 1]
    cand_x, cand_y = projector(cand_lons_filter1, cand_lats_filter1)

    # Filter by exact radius in meters:
    dists_meters = np.sqrt(cand_x**2 + cand_y**2)
    dist_filter = dists_meters <= search_radius_meters

    if not np.any(dist_filter):
        return {}

    # Keep only valid candidates (WGS84 coords retained for occlusion check):
    cand_x = cand_x[dist_filter]
    cand_y = cand_y[dist_filter]
    cand_lons_filter1 = cand_lons_filter1[dist_filter]
    cand_lats_filter1 = cand_lats_filter1[dist_filter]
    indices = indices[dist_filter]

    # 5. Rotation alignment. Get local building coords:
    local_poly_coords = [
        projector(lon, lat) for lon, lat in footprint_wgs84.exterior.coords
    ]
    local_poly = Polygon(local_poly_coords)
    building_angle_deg, _ = get_principal_axes(local_poly)

    # Calculate rotation:
    beta = (90.0 - building_angle_deg) % 360.0
    theta_rad = -np.radians(beta)

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Rotate candidates:
    x_rot = cand_x * cos_t - cand_y * sin_t
    y_rot = cand_x * sin_t + cand_y * cos_t

    # 6. Ray generation. Force the 4 principal axes (faces) to always be
    # included:
    target_angles = {0.0, 90.0, 180.0, 270.0}
    corner_labels = {}

    if cast_corner_rays:
        mbr = local_poly.minimum_rotated_rectangle
        mbr_x, mbr_y = mbr.exterior.xy
        mbr_x = np.array(mbr_x)
        mbr_y = np.array(mbr_y)

        # Rotate MBR to aligned space:
        mbr_x_rot = mbr_x * cos_t - mbr_y * sin_t
        mbr_y_rot = mbr_x * sin_t + mbr_y * cos_t

        unique_x = mbr_x_rot[:4]
        unique_y = mbr_y_rot[:4]

        base_angles = np.degrees(np.arctan2(unique_y, unique_x)) % 360
        sorted_indices = np.argsort(base_angles)

        # Exact angle to the 4 corners, no offsets:
        for i, idx in enumerate(sorted_indices):
            base_angle = base_angles[idx]
            target_angles.add(base_angle)
            corner_labels[base_angle] = f'corner_{i + 1}'
    else:
        for ang in np.arange(0, 360, interval_deg):
            target_angles.add(float(ang))

    sorted_angles = sorted(target_angles)

    # 7. Ray matching & occlusion check:
    results_map = {}
    raw_results_map = {}  # Geometric bests, ignoring occlusion
    axis_map = {
        0.0: 'major_pos',
        90.0: 'minor_pos',
        180.0: 'major_neg',
        270.0: 'minor_neg',
    }

    for angle in sorted_angles:
        name = _name_ray(angle, axis_map, corner_labels)

        # Find all matches aligned with ray:
        candidate_indices_local = _find_all_matches_for_ray(
            angle, x_rot, y_rot, tolerance_meters=ray_tolerance_meters
        )

        # Capture the "raw" best (first index = best geometric match):
        if len(candidate_indices_local) > 0:
            raw_local_idx = candidate_indices_local[0]
            raw_global_idx = indices[raw_local_idx]
            raw_results_map[name] = pano_collection[raw_global_idx]

        # Iterate to find best non-occluded match:
        found_match = False
        for local_idx in candidate_indices_local:
            global_idx = indices[local_idx]
            cand_lon = cand_lons_filter1[local_idx]
            cand_lat = cand_lats_filter1[local_idx]

            is_blocked = _check_occlusion(
                (cand_lon, cand_lat), footprint_wgs84, potential_occluders
            )

            if not is_blocked:
                results_map[name] = pano_collection[global_idx]
                found_match = True
                break

        if not found_match:
            results_map[name] = None

    # 8. Printing & plotting:
    if print_results:
        sorted_keys = sorted(results_map.keys(), key=_sort_key)
        for ray_name in sorted_keys:
            img = results_map[ray_name]
            status = f'ID: {img.id}' if img else 'No match (or Occluded)'
            print(f'{ray_name.ljust(12)}: {status}')

    if DEBUG_PLOTS:

        def get_plot_coords(img_map):
            """Map ray names to rotated local coordinates of chosen images."""
            pts = {}
            for key, img in img_map.items():
                if img is None:
                    continue
                target_idx = _result_img_index_lookup(img, indices, pano_collection)
                loc_idx_arr = np.where(indices == target_idx)[0]
                if loc_idx_arr.size > 0:
                    l_idx = loc_idx_arr[0]
                    pts[key] = (x_rot[l_idx], y_rot[l_idx])
            return pts

        plot_points_final = get_plot_coords(results_map)
        plot_points_raw = get_plot_coords(raw_results_map)

        rot_neighbors = []
        for poly in potential_occluders:
            try:
                if poly.geom_type == 'MultiPolygon':
                    p = max(poly.geoms, key=lambda a: a.area)
                else:
                    p = poly
                rot_neighbors.append(
                    _transform_polygon_to_local_rotated(p, projector, cos_t, sin_t)
                )
            except Exception as e:
                logger.debug(f'Skipping neighbor in debug plot: {e}')

        _plot_dynamic_results(
            local_poly,
            x_rot,
            y_rot,
            theta_rad,
            ray_tolerance_meters,
            sorted_angles,
            best_points=plot_points_final,
            raw_points=plot_points_raw,
            neighbor_polys=rot_neighbors,
        )

    return results_map


def _name_ray(angle: float, axis_map: dict, corner_labels: dict) -> str:
    """
    Resolve a human-readable name for a ray angle.

    Args:
        angle (float):
            Ray angle in degrees within the footprint-aligned frame.
        axis_map (dict[float, str]):
            Mapping of the four principal-axis angles to their names.
        corner_labels (dict[float, str]):
            Mapping of corner angles to ``'corner_N'`` labels.

    Returns:
        str:
            An axis name, a corner label, or ``'ray_<int(angle)>'``.

    Example:
        >>> axes = {0.0: 'major_pos', 90.0: 'minor_pos'}
        >>> _name_ray(90.0, axes, {})
        'minor_pos'
        >>> _name_ray(45.0, axes, {45.0: 'corner_1'})
        'corner_1'
        >>> _name_ray(30.0, axes, {})
        'ray_30'
    """
    for ax_angle, ax_name in axis_map.items():
        if np.isclose(angle, ax_angle) or np.isclose(angle, ax_angle + 360.0):
            return ax_name
    for c_angle, c_label in corner_labels.items():
        if np.isclose(angle, c_angle):
            return c_label
    return f'ray_{int(angle)}'


def _find_all_matches_for_ray(
    angle_deg: float, x_rot: np.ndarray, y_rot: np.ndarray, tolerance_meters: float
) -> np.ndarray:
    """
    Return indices of all candidates aligned with a ray, best fit first.

    A candidate matches when its perpendicular distance to the ray is within
    ``tolerance_meters`` and it lies in the ray's forward half-space. Matches
    are ordered by increasing perpendicular distance.

    Args:
        angle_deg (float):
            Ray direction in degrees (0 = +X, counter-clockwise positive).
        x_rot (np.ndarray):
            Candidate X coordinates in the aligned local frame.
        y_rot (np.ndarray):
            Candidate Y coordinates in the aligned local frame.
        tolerance_meters (float):
            Maximum perpendicular distance from the ray.

    Returns:
        np.ndarray:
            Integer indices into ``x_rot``/``y_rot`` sorted by alignment
            quality; empty if no candidate matches.

    Example:
        >>> import numpy as np
        >>>
        >>> xs = np.array([10.0, 20.0, -10.0])
        >>> ys = np.array([0.5, 4.0, 0.0])
        >>> _find_all_matches_for_ray(0.0, xs, ys, tolerance_meters=5.0).tolist()
        [0, 1]
    """
    rad = np.radians(angle_deg)
    ray_vec_x = np.cos(rad)
    ray_vec_y = np.sin(rad)

    perp_dist = np.abs(x_rot * -ray_vec_y + y_rot * ray_vec_x)
    dot_prod = x_rot * ray_vec_x + y_rot * ray_vec_y

    mask = (perp_dist <= tolerance_meters) & (dot_prod > 0)

    if not np.any(mask):
        return np.array([], dtype=int)

    # Get indices where mask is True:
    valid_local_indices = np.where(mask)[0]
    valid_dists = perp_dist[mask]

    # Sort by smallest perpendicular distance (best alignment to ray):
    sorted_order = np.argsort(valid_dists)

    return valid_local_indices[sorted_order]


def _sort_key(k: str) -> tuple:
    """
    Sort key ordering ray names: principal axes, corners, then other rays.

    Args:
        k (str):
            A ray name such as ``'major_pos'``, ``'corner_2'`` or
            ``'ray_45'``.

    Returns:
        tuple[int, int]:
            A ``(group, order)`` pair suitable for ``sorted(key=...)``.

    Example:
        >>> sorted(['ray_45', 'corner_2', 'minor_neg', 'major_pos'], key=_sort_key)
        ['major_pos', 'minor_neg', 'corner_2', 'ray_45']
    """
    if 'major_pos' in k:
        return (0, 0)
    if 'minor_pos' in k:
        return (1, 0)
    if 'major_neg' in k:
        return (2, 0)
    if 'minor_neg' in k:
        return (3, 0)

    if 'corner' in k:
        try:
            payload = k.split('_')[1]
            digits = ''.join(filter(str.isdigit, payload))
            letters = ''.join(filter(str.isalpha, payload))
            num = int(digits) if digits else 0
            suffix_val = ord(letters[0]) if letters else 0
            return (4, num * 100 + suffix_val)
        except (IndexError, ValueError):
            return (4, 9999)

    try:
        val = int(k.split('_')[1])
    except (IndexError, ValueError):
        val = 0
    return (5, val)


def _plot_candidates(
    footprint: Polygon,
    lons_red: np.ndarray,
    lats_red: np.ndarray,
    lons_green: np.ndarray,
    lats_green: np.ndarray,
) -> None:
    """
    Plot candidate pano locations resulting from radial filtering.

    Args:
        footprint (Polygon):
            Shapely Polygon in lon/lat (WGS84) coordinates.
        lons_red (np.ndarray):
            Longitudes for the first (coarse) round of radial filtering.
        lats_red (np.ndarray):
            Latitudes for the first (coarse) round of radial filtering.
        lons_green (np.ndarray):
            Longitudes surviving the second, accurate distance filter.
        lats_green (np.ndarray):
            Latitudes surviving the second, accurate distance filter.

    Returns:
        None

    Example:
        >>> import numpy as np
        >>> from shapely.geometry import box
        >>>
        >>> _plot_candidates(
        ...     box(0, 0, 1, 1),
        ...     np.array([2.0]), np.array([2.0]),
        ...     np.array([]), np.array([]),
        ... )  # doctest: +SKIP
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the building footprint (blue):
    x_poly, y_poly = footprint.exterior.xy
    ax.fill(
        x_poly, y_poly, alpha=0.4, fc='lightblue', ec='blue', label='Building footprint'
    )

    # Plot the centroid (black X):
    centroid = footprint.centroid
    ax.scatter(
        centroid.x,
        centroid.y,
        color='black',
        marker='x',
        s=100,
        zorder=10,
        label='Centroid',
    )

    # Plot the first set of pano locations resulting from approximate radial
    # filtering (red):
    if lons_red.size and lats_red.size:
        ax.scatter(
            lons_red,
            lats_red,
            color='red',
            s=30,
            alpha=0.6,
            label='Coarse radial search (Red)',
        )

    # Plot the second set of pano locations based on accurate distance
    # calculations filtering (green):
    if lons_green.size and lats_green.size:
        ax.scatter(
            lons_green,
            lats_green,
            color='green',
            s=40,
            alpha=0.9,
            zorder=5,
            label='Accurate distance filter (Green)',
        )

    # Clean up the figure:
    ax.set_title('Search Results Comparison')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    # Only add legend if there is something to show:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper left')

    plt.show()


def _result_img_index_lookup(target_img, valid_indices, collection) -> int:
    """
    Find the collection index of an image among a set of candidate indices.

    Args:
        target_img (ImageAsset):
            The image to locate.
        valid_indices (Iterable[int]):
            Candidate indices into ``collection``.
        collection (ImageCollection):
            The collection to search.

    Returns:
        int:
            The first index in ``valid_indices`` whose asset equals
            ``target_img``, or ``-1`` if not found.

    Example:
        >>> from rapidtools.core import ImageAsset, ImageCollection
        >>>
        >>> a = ImageAsset(path='a.jpg', id='a', allow_missing_file=True)
        >>> b = ImageAsset(path='b.jpg', id='b', allow_missing_file=True)
        >>> col = ImageCollection([a, b])
        >>> _result_img_index_lookup(b, [0, 1], col)
        1
        >>> _result_img_index_lookup(b, [0], col)
        -1
    """
    for idx in valid_indices:
        if collection[idx] == target_img:
            return idx
    return -1


def _plot_dynamic_results(
    local_poly: Polygon,
    x_rot: np.ndarray,
    y_rot: np.ndarray,
    theta_rad: float,
    tolerance: float,
    angles: list,
    best_points: dict,
    raw_points: dict | None = None,
    neighbor_polys: list[Polygon] | None = None,
) -> None:
    """
    Visualize target, neighbors, rays, raw candidates and final selections.

    Raw (geometric best) matches are drawn as hollow circles and final
    (occlusion-checked) matches as solid stars.

    Args:
        local_poly (Polygon):
            Target footprint in local meters (unrotated).
        x_rot (np.ndarray):
            Candidate X coordinates in the aligned frame.
        y_rot (np.ndarray):
            Candidate Y coordinates in the aligned frame.
        theta_rad (float):
            Rotation applied to candidates, in radians.
        tolerance (float):
            Ray tolerance in meters (informational only).
        angles (list[float]):
            Ray angles in degrees to draw.
        best_points (dict[str, tuple[float, float]]):
            Final selections keyed by ray name.
        raw_points (dict[str, tuple[float, float]] | None):
            Geometric-best selections keyed by ray name.
        neighbor_polys (list[Polygon] | None):
            Occluder footprints already transformed to the aligned frame.

    Returns:
        None

    Example:
        >>> import numpy as np
        >>> from shapely.geometry import box
        >>>
        >>> _plot_dynamic_results(
        ...     box(-5, -2, 5, 2), np.array([20.0]), np.array([0.0]), 0.0,
        ...     10.0, [0.0, 90.0], best_points={'major_pos': (20.0, 0.0)},
        ... )  # doctest: +SKIP
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. Plot neighbors (the occluders):
    if neighbor_polys:
        for npoly in neighbor_polys:
            nx, ny = npoly.exterior.xy
            ax.fill(nx, ny, alpha=0.3, fc='salmon', ec='red', label='_nolegend_')
        ax.plot(
            [],
            [],
            color='salmon',
            alpha=0.6,
            linewidth=5,
            label='Neighboring Buildings (Occluders)',
        )

    # 2. Plot target building (gray):
    theta_deg = float(np.degrees(theta_rad))
    rot_poly = shapely_rotate(local_poly, theta_deg, origin=(0.0, 0.0))
    px, py = rot_poly.exterior.xy
    ax.fill(px, py, alpha=0.5, fc='gray', ec='black', label='Target Building')

    # 3. Plot all candidate dots background (light gray):
    if x_rot.size and y_rot.size:
        ax.scatter(x_rot, y_rot, color='lightgray', s=15, alpha=0.5, zorder=1)

    # 4. Plot rays:
    lim = 80
    for angle in angles:
        rad = np.radians(angle)
        if np.isclose(angle % 180, 0):
            color = 'blue'
        elif np.isclose(angle % 180, 90):
            color = 'purple'
        else:
            color = 'green'
        ex = lim * np.cos(rad)
        ey = lim * np.sin(rad)
        ax.plot([0, ex], [0, ey], color=color, linestyle='--', lw=0.8, alpha=0.7)

    # 5. Plot raw matches (the geometrically closest) as hollow circles:
    if raw_points:
        for key, (rx, ry) in raw_points.items():
            ax.scatter(
                rx,
                ry,
                facecolors='none',
                edgecolors=_ray_color(key),
                marker='o',
                s=150,
                linewidth=2,
                zorder=9,
                label='_nolegend_',
            )

    # 6. Plot final matches (the visible ones) as solid stars:
    for key, (cx, cy) in best_points.items():
        ax.scatter(
            cx,
            cy,
            c=_ray_color(key),
            marker='*',
            s=200,
            edgecolors='black',
            zorder=10,
            label='_nolegend_',
        )

    ax.set_title(
        'Occlusion Analysis\nCircles=Best Geometry (Raw) | Stars=Best Visible (Final)'
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Custom legend:
    legend_elements = [
        Line2D([0], [0], color='gray', lw=4, label='Target'),
        Line2D([0], [0], color='salmon', lw=4, label='Occluder'),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markeredgecolor='black',
            markerfacecolor='none',
            markersize=10,
            label='Raw Match (Ignored Occlusion)',
        ),
        Line2D(
            [0],
            [0],
            marker='*',
            color='w',
            markerfacecolor='black',
            markersize=15,
            label='Final Match (Visible)',
        ),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()


def _ray_color(key: str) -> str:
    """
    Pick a plot colour for a ray name.

    Args:
        key (str):
            Ray name such as ``'major_pos'`` or ``'corner_1'``.

    Returns:
        str:
            ``'blue'`` for major-axis rays, ``'purple'`` for minor-axis rays
            and ``'green'`` otherwise.

    Example:
        >>> _ray_color('major_neg'), _ray_color('minor_pos'), _ray_color('ray_45')
        ('blue', 'purple', 'green')
    """
    if 'major' in key:
        return 'blue'
    if 'minor' in key:
        return 'purple'
    return 'green'


def crop_panorama_to_asset(
    target_asset: PhysicalAsset,
    pano_image: ImageAsset,
    horizontal_padding_deg: float = 20.0,
    vertical_padding_percent: float = 1.0,
    vertical_crop_mode: str = 'smart',
    base_sky_crop_ratio: float = 0.25,
) -> Image.Image:
    """
    Crop an equirectangular panorama to the part that shows an asset.

    Horizontally, the bearings from the camera to every footprint vertex are
    computed relative to the camera heading and the smallest angular window
    covering all of them (plus padding) is cut out, correctly handling
    assets that straddle the 180/-180 degree seam at the image edges.
    Vertically, the top ``base_sky_crop_ratio`` of the image is always
    removed; in ``'smart'`` mode the semantic mask is additionally used to
    trim remaining sky above the first non-sky row and to cut off the survey
    vehicle at the bottom.

    Args:
        target_asset (PhysicalAsset):
            Asset with a ``Polygon``, ``MultiPolygon`` (largest part is
            used) or other geometry (convex hull is used).
        pano_image (ImageAsset):
            Panorama whose ``_pil_image`` is loaded and whose ``properties``
            contain ``latitude``, ``longitude`` and ``compass_angle``. In
            ``'smart'`` mode the semantic mask is loaded via
            ``load_mask('semantic')`` and ``semantic_map`` is consulted for
            the sky and ego-vehicle IDs.
        horizontal_padding_deg (float):
            Extra angular margin added on both sides of the asset. Defaults
            to ``20.0``.
        vertical_padding_percent (float):
            Vertical padding (percentage of image height) applied to the
            smart crop lines. Defaults to ``1.0``.
        vertical_crop_mode (str):
            ``'smart'`` to use the semantic mask; any other value applies
            only the base sky crop. Defaults to ``'smart'``.
        base_sky_crop_ratio (float):
            Fraction of the image height removed from the top
            unconditionally. Defaults to ``0.25``.

    Returns:
        PIL.Image.Image:
            The cropped image strip.

    Raises:
        TypeError:
            If ``pano_image`` does not hold a loaded PIL image.

    Example:
        >>> pano.load_image_from_disk()  # doctest: +SKIP
        >>> crop = crop_panorama_to_asset(
        ...     building, pano, vertical_crop_mode='full'
        ... )  # doctest: +SKIP
        >>> crop.size  # doctest: +SKIP
        (1365, 3072)
    """
    # Extract the PIL image:
    pil_image = pano_image._pil_image
    if not isinstance(pil_image, Image.Image):
        raise TypeError('pano_image does not contain a loaded PIL image.')
    img_w, img_h = pil_image.size

    # Extract the segmentation mask:
    mask_data = None
    mask_strip = None
    if vertical_crop_mode == 'smart':
        mask_data = pano_image.load_mask('semantic')

    # Get the geometry for the asset and process it into a polygon of
    # appropriate type:
    geom = target_asset.geometry
    if geom.geom_type == 'MultiPolygon':
        poly = max(geom.geoms, key=lambda a: a.area)
    elif geom.geom_type == 'Polygon':
        poly = geom
    else:
        poly = geom.convex_hull
    coords = list(poly.exterior.coords)

    # Calculate bearing angles of asset vertices relative to North
    # (-180 deg to 180 deg):
    cam_lat = pano_image.properties['latitude']
    cam_lon = pano_image.properties['longitude']
    bearings = [_get_bearing(cam_lat, cam_lon, lat, lon) for lon, lat in coords]

    # Calculate bearing angles of asset vertices relative to the image center:
    rel_bearings = []
    for b in bearings:
        diff = b - pano_image.properties['compass_angle']
        # Normalize to -180 to 180:
        diff = (diff + 180) % 360 - 180
        rel_bearings.append(diff)

    # Sort bearings to find the largest gap. This identifies if the asset spans
    # across the 180/-180 seam:
    rel_bearings.sort()

    # Calculate gaps between adjacent sorted angles:
    max_gap = 0
    start_index = 0

    for i in range(len(rel_bearings) - 1):
        gap = rel_bearings[i + 1] - rel_bearings[i]
        if gap > max_gap:
            max_gap = gap
            start_index = i + 1

    # Check the gap between the last and first point (wrapping around 360):
    wrap_gap = (rel_bearings[0] + 360) - rel_bearings[-1]

    if wrap_gap > max_gap:
        # If the largest gap is the wrap-around gap, the asset DOES NOT cross
        # the seam. Normal logic applies:
        min_angle = rel_bearings[0] - horizontal_padding_deg
        max_angle = rel_bearings[-1] + horizontal_padding_deg
        crosses_seam = False
    else:
        # If the largest gap is somewhere in the middle of the array, the asset
        # CROSSES the seam (the "back" of the image). All the points, excluding
        # the largest gap, are retained. Start from the index after the gap and
        # wrap around to the index before the gap:
        min_angle = rel_bearings[start_index] - horizontal_padding_deg
        max_angle = rel_bearings[start_index - 1] + horizontal_padding_deg
        crosses_seam = True

    # Calculate the pixel count per degree and the location of the image
    # center in pixel coordinates:
    pixels_per_deg = img_w / 360.0
    center_x = img_w / 2.0

    # Calculate crop coordinates:
    if not crosses_seam:
        # The asset is fully visible within the continuous image frame:
        start_x = max(0, int(center_x + (min_angle * pixels_per_deg)))
        end_x = min(img_w, int(center_x + (max_angle * pixels_per_deg)))

        # Crop image in horizontal direction:
        image_strip = pil_image.crop((start_x, 0, end_x, img_h))

        # Crop segmentation mask in horizontal direction:
        if mask_data is not None:
            mask_strip = mask_data[:, start_x:end_x]
    else:
        # Because the crop area crosses the seam, min_angle is positive (e.g.,
        # 170) and max_angle is negative (e.g., -170). Shift the image so the
        # seam is in the middle to make cropping easy.

        # Normalize angles to 0-360 for width calculation:
        norm_min = (min_angle + 360) % 360
        norm_max = (max_angle + 360) % 360

        # Calculate width in degrees (handling wrap) and pixels:
        width_deg = (norm_max - norm_min + 360) % 360
        width_px = int(width_deg * pixels_per_deg)

        # Calculate where the "left" edge (min_angle) starts in pixels:
        start_px_rel = int(center_x + (min_angle * pixels_per_deg)) % img_w

        # Offset the image so 'start_px_rel' becomes x=0:
        image_strip = offset(pil_image, -start_px_rel, 0).crop((0, 0, width_px, img_h))

        # Crop segmentation mask in horizontal direction using the same
        # approach:
        if mask_data is not None:
            rolled_mask = np.roll(mask_data, -start_px_rel, axis=1)
            mask_strip = rolled_mask[:, :width_px]

    # Apply the base sky crop immediately (cuts off the top zenith sky):
    final_top, final_bottom = int(img_h * base_sky_crop_ratio), img_h

    # Perform a vertical "smart" crop:
    if vertical_crop_mode == 'smart' and mask_strip is not None:
        try:
            # Resolve semantic IDs:
            sky_id, vehicle_id = None, None
            if pano_image.semantic_map:
                name_to_id = {k.lower(): v for v, k in pano_image.semantic_map.items()}
                sky_id = name_to_id.get(SegmentationLabels.SKY)
                vehicle_id = name_to_id.get(SegmentationLabels.SURVEY_VEHICLE)

            # Save mask IDs to ignore:
            ids_to_ignore = [0]
            if sky_id is not None:
                ids_to_ignore.append(sky_id)
            if vehicle_id is not None:
                ids_to_ignore.append(vehicle_id)

            # Create the content mask (everything but sky and survey vehicle):
            valid_content_mask = ~np.isin(mask_strip, ids_to_ignore)

            # Check which rows contain at least one content pixel:
            rows_with_content = np.any(valid_content_mask, axis=1)
            content_indices = np.flatnonzero(rows_with_content)

            # Calculate vertical padding relative to original image height:
            padding_px = int(img_h * vertical_padding_percent / 100)

            if content_indices.size > 0:
                # Find the very first row with content below the sky line:
                smart_top = max(0, content_indices[0] - padding_px)

                # Take whichever cut removes MORE sky (base crop or smart crop):
                final_top = max(final_top, smart_top)

                if vehicle_id is not None:
                    # Check rows where the vehicle appears:
                    vehicle_indices = np.flatnonzero(
                        np.any(mask_strip == vehicle_id, axis=1)
                    )

                    if vehicle_indices.size > 0:
                        # The crop line is the highest point (min row) of the
                        # vehicle:
                        final_bottom = max(0, vehicle_indices[0] - padding_px)
            else:
                logger.info(
                    'Smart crop could not be performed: Mask strip contains '
                    'only background/sky/vehicle.'
                )

        except Exception as e:
            logger.warning(f'Smart crop failed: {e}')

    # Safety check: make sure the crop is valid (i.e., top is above bottom):
    if final_top >= final_bottom:
        logger.warning(
            f'Invalid vertical crop calculated {final_top} to {final_bottom}. '
            'Resetting.'
        )
        final_top, final_bottom = int(img_h * base_sky_crop_ratio), img_h

    # Final vertical crop on the already horizontally-cropped strip:
    return image_strip.crop((0, final_top, image_strip.width, final_bottom))


def _get_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial compass bearing (forward azimuth) between two points.

    This function uses spherical trigonometry to determine the direction
    one must initially travel to get from point 1 to point 2. The result
    is normalized to a compass circle (0 deg = North, 90 deg = East).

    Args:
        lat1 (float): Latitude of the starting point in decimal degrees.
        lon1 (float): Longitude of the starting point in decimal degrees.
        lat2 (float): Latitude of the destination point in decimal degrees.
        lon2 (float): Longitude of the destination point in decimal degrees.

    Returns:
        float:
            The bearing in degrees (0.0 to 360.0).

    Example:
        Bearing from (0, 0) towards the east (0, 10):

        >>> _get_bearing(0.0, 0.0, 0.0, 10.0)
        90.0

        Bearing from (0, 0) towards the north (10, 0):

        >>> _get_bearing(0.0, 0.0, 10.0, 0.0)
        0.0
    """
    d_lon = math.radians(lon2 - lon1)
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    y = math.sin(d_lon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(
        lat2_r
    ) * math.cos(d_lon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360
