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
# 05-25-2026

import math
import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ImageChops import offset
from scipy.spatial import cKDTree
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon, LineString, Point
from shapely.strtree import STRtree
from shapely.geometry import box

from rapidtools.constants import EARTH_RADIUS_KM, LATITUDE_SPACING_KM
from rapidtools.core import (
    ImageCollection,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.data_sources.mapillary_client import SegmentationLabels

# Define global varibles:
PLOT_FOOTPRINT = False
DEBUG_PLOTS = False

SEARCH_RADIUS = 60
MAX_CANDIDATES = 100
RAY_TOLERANCE_METERS = 10.0
OFF_AXIS_INTERVAL_DEG = 90

def _transform_polygon_to_local_rotated(
    wgs84_poly: Polygon, 
    projector_func, 
    cos_t: float, 
    sin_t: float
) -> Polygon:
    """
    Helper to project and rotate a WGS84 polygon into the local analysis frame.
    """
    # 1. Extract exterior coordinates
    # (Simplified: ignoring holes for visualization purposes)
    coords = list(wgs84_poly.exterior.coords)
    
    local_coords = []
    for lon, lat in coords:
        # Project to meters (relative to target centroid)
        x, y = projector_func(lon, lat)
        
        # Rotate to align with target axes
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        local_coords.append((x_rot, y_rot))
        
    return Polygon(local_coords)

def build_footprint_index(asset_collection: PhysicalAssetCollection) -> tuple[STRtree, list]:
    """
    Builds a spatial index (R-tree) for the building footprints.
    
    Args:
        asset_collection: The list/collection of infrastructure assets.
        
    Returns:
        tuple: (STRtree instance, List of geometry objects corresponding to the tree)
    """
    # Extract just the geometries. The STRtree stores these.
    # We keep a reference to the list because STRtree (in newer Shapely versions)
    # returns indices, which we use to look up the actual polygon in this list.
    geoms = [asset.geometry for asset in asset_collection if asset.geometry is not None]
    tree = STRtree(geoms)
    return tree, geoms

def get_nearby_buildings(
    target_poly: Polygon, 
    tree: STRtree, 
    geom_list: list, 
    radius_meters: float
) -> list[Polygon]:
    """
    Efficiently finds polygons within a radius using the R-tree.
    """
    # 1. Convert radius (meters) to degrees (approximate)
    # Latitude: 1 deg ~= 111km. Longitude: 1 deg ~= 111km * cos(lat)
    # We use a safe approximation.
    lat = target_poly.centroid.y
    deg_per_meter_lat = 1.0 / 111111.0
    deg_per_meter_lon = 1.0 / (111111.0 * math.cos(math.radians(lat)))
    
    dx = radius_meters * deg_per_meter_lon
    dy = radius_meters * deg_per_meter_lat
    
    # 2. Create a search bounding box (envelope) around the target
    minx, miny, maxx, maxy = target_poly.bounds
    search_envelope = box(minx - dx, miny - dy, maxx + dx, maxy + dy)
    
    # 3. Query the tree
    # usage for Shapely 2.0+: tree.query returns indices of geometries
    indices = tree.query(search_envelope)
    
    # 4. Retrieve actual polygons
    neighbors = []
    for i in indices:
        candidate = geom_list[i]
        # Exclude the target itself (simple equality check)
        if not candidate.equals(target_poly):
             neighbors.append(candidate)
             
    return neighbors

def index_panos(
        image_collection: ImageCollection
    ) -> tuple[cKDTree | None, np.ndarray, np.ndarray]:
    """
    Build a spatial index over pano images for fast nearest-neighbor search.
    """
    # Iterate over the collection once to grab the properties:
    data = [
        (
            img.properties.get('longitude', 0.0),
            img.properties.get('latitude', 0.0),
            img.properties.get('compass_angle', 0.0)
        )
        for img in image_collection
    ]
    
    if not data:
        return None, np.array([]), np.array([])
    
    # Bulk Conversion to NumPy Create one array (N, 3) and then slice it:    
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
    Return a function projecting from geographic to local planar coordinates.
    """
    # Calculate function constants:
    r_earth = EARTH_RADIUS_KM*1000
    centroid_lon_rad = np.radians(centroid_lon)
    centroid_lat_rad = np.radians(centroid_lat)
    cos_centroid_lat = np.cos(centroid_lat_rad)
    
    def project(lon, lat):
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
    Convert a 2D vector into an orientation angle in degrees.
    """
    dx, dy = v[0], v[1]
    rads = np.arctan2(dx, dy)
    return np.degrees(rads) % 360

def _check_occlusion(
    camera_pt_wgs84: tuple[float, float],
    target_poly_wgs84: Polygon,
    neighboring_polys_wgs84: list[Polygon]
) -> bool:
    """
    Check if the line of sight from camera to target is blocked by neighbors.
    
    Args:
        camera_pt_wgs84: Tuple (lon, lat) of the camera.
        target_poly_wgs84: The Shapely Polygon of the building we want to see.
        neighboring_polys_wgs84: List of other building polygons to check against.
        
    Returns:
        True if the view is occluded (blocked), False if clear.
    """
    if not neighboring_polys_wgs84:
        return False

    cam_point = Point(camera_pt_wgs84)
    
    # 1. Construct a Line of Sight (LOS)
    # We draw a line from the camera to the *centroid* of the target.
    # Note: A stricter check might cast rays to multiple corners, but centroid
    # is usually sufficient for general occlusion culling.
    target_centroid = target_poly_wgs84.centroid
    
    # If the camera is inside the target building, strictly speaking it's not occluded 
    # by *neighbors*, though it might be invalid for other reasons.
    if target_poly_wgs84.contains(cam_point):
        return False
        
    los_line = LineString([cam_point, target_centroid])
    
    # 2. Check intersection with neighbors
    for poly in neighboring_polys_wgs84:
        # Skip if the neighbor is actually the target itself (based on equality or ID if avail)
        if poly.equals(target_poly_wgs84):
            continue
            
        # Optimization: prepared geometry checks are faster
        # (Though constructing `prep` for every single check might be overkill if list is huge.
        # Ideally, `neighboring_polys_wgs84` is a pre-filtered list of nearby buildings).
        
        # If the line intersects a neighbor polygon...
        if poly.intersects(los_line):
            # We must ensure the intersection isn't just "touching" the exterior 
            # or overlapping the target itself.
            
            # The LOS naturally intersects the target. We care if it intersects *others*.
            # Calculate the intersection geometry.
            intersection = poly.intersection(los_line)
            
            # If the intersection has length > 0 (i.e. it passes through the building interior
            # or along an edge), it is an occlusion.
            # Touching a corner (Point intersection) usually doesn't count as full occlusion.
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
        cast_corner_rays: bool = False
    ) -> dict:
    """
    Finds the SINGLE best panorama for multiple angles, filtering out occluded views
    using an STRtree for efficient neighbor lookup.
    """
    # -----------------------------------------------------------
    # 1. GEOMETRY VALIDATION & SETUP
    # -----------------------------------------------------------
    raw_geom = target_asset_wgs84.geometry
    
    if raw_geom.geom_type not in ['Polygon', 'MultiPolygon']:
        return {}
        
    # Extract the building footprint
    footprint_wgs84 = max(raw_geom.geoms, key=lambda a: a.area) if raw_geom.geom_type == 'MultiPolygon' else raw_geom 
    
    # Extract the centroid
    centroid = footprint_wgs84.centroid
    centroid_lon, centroid_lat = centroid.x, centroid.y
    
    # ---------------------------------------------------------
    # 2. EFFICIENT NEIGHBOR DETECTION (OCCLUSION)
    # ---------------------------------------------------------
    # We define an "Occlusion Radius" slightly larger than search radius
    # to catch buildings that might block the view.
    occlusion_radius = search_radius_meters * 1.5
    
    # Use the STRtree to find only relevant neighbors
    potential_occluders = get_nearby_buildings(
        footprint_wgs84,
        building_tree,
        building_geoms,
        radius_meters=occlusion_radius
    )

    # ---------------------------------------------------------
    # 3. PANO SEARCH (BROAD PHASE)
    # ---------------------------------------------------------
    radius_deg = (search_radius_meters*2) / (LATITUDE_SPACING_KM*1000)
    dists_deg, indices = tree.query(
        [centroid_lon, centroid_lat],
        k=max_candidates,
        distance_upper_bound=radius_deg
    )
    
    valid_mask = np.isfinite(dists_deg)
    indices = indices[valid_mask]
    
    if len(indices) == 0:
        return {}
    
    # ---------------------------------------------------------
    # 4. PROJECTION TO LOCAL METERS
    # ---------------------------------------------------------
    projector = get_local_projection_func(centroid_lon, centroid_lat)
    
    cand_lons_filter1 = coords_deg[indices, 0]
    cand_lats_filter1 = coords_deg[indices, 1]
    cand_x, cand_y = projector(cand_lons_filter1, cand_lats_filter1)
    
    # Filter by exact radius in meters
    dists_meters = np.sqrt(cand_x**2 + cand_y**2)
    dist_filter = dists_meters <= search_radius_meters
    
    if not np.any(dist_filter):
        return {}
    
    # Keep only valid candidates
    cand_x = cand_x[dist_filter]
    cand_y = cand_y[dist_filter]
    cand_lons_filter1 = cand_lons_filter1[dist_filter] # Keep wgs84 coords for occlusion check
    cand_lats_filter1 = cand_lats_filter1[dist_filter]
    indices = indices[dist_filter]

    # ---------------------------------------------------------
    # 5. ROTATION ALIGNMENT
    # ---------------------------------------------------------
    # Get local building coords
    local_poly_coords = [projector(lon, lat) for lon, lat in footprint_wgs84.exterior.coords]
    local_poly = Polygon(local_poly_coords)
    building_angle_deg, _ = get_principal_axes(local_poly)
    
    # Calculate Rotation
    beta = (90.0 - building_angle_deg) % 360.0
    theta_rad = -np.radians(beta)
    
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Rotate candidates
    x_rot = cand_x * cos_t - cand_y * sin_t
    y_rot = cand_x * sin_t + cand_y * cos_t
    
    # -----------------------------------------------------------
    # 6. RAY GENERATION
    # -----------------------------------------------------------
    target_angles = set()            
    corner_labels = {} 
    
    # Force the 4 principal axes (faces) to always be included
    target_angles.update({0.0, 90.0, 180.0, 270.0})
    
    if cast_corner_rays:
        mbr = local_poly.minimum_rotated_rectangle
        mbr_x, mbr_y = mbr.exterior.xy
        mbr_x = np.array(mbr_x)
        mbr_y = np.array(mbr_y)
        
        # Rotate MBR to aligned space
        mbr_x_rot = mbr_x * cos_t - mbr_y * sin_t
        mbr_y_rot = mbr_x * sin_t + mbr_y * cos_t
        
        unique_x = mbr_x_rot[:4]
        unique_y = mbr_y_rot[:4]
        
        base_angles = np.degrees(np.arctan2(unique_y, unique_x)) % 360
        sorted_indices = np.argsort(base_angles)
        
        # Exact angle to the 4 corners, no offsets
        for i, idx in enumerate(sorted_indices):
            base_angle = base_angles[idx]
            target_angles.add(base_angle)
            corner_labels[base_angle] = f"corner_{i+1}"
    else:
        for ang in np.arange(0, 360, interval_deg):
            target_angles.add(float(ang))
    
    sorted_angles = sorted(list(target_angles))
    
    # -----------------------------------------------------------
    # 7. RAY MATCHING & OCCLUSION CHECK
    # -----------------------------------------------------------
    results_map = {}
    raw_results_map = {} # <--- NEW: To store geometric bests
    axis_map = {0.0: 'major_pos', 90.0: 'minor_pos', 180.0: 'major_neg', 270.0: 'minor_neg'}

    for angle in sorted_angles:
        # Naming Logic
        name = None
        for ax_angle, ax_name in axis_map.items():
            if np.isclose(angle, ax_angle) or np.isclose(angle, ax_angle + 360.0): name = ax_name; break
        if name is None:
            for c_angle, c_label in corner_labels.items():
                if np.isclose(angle, c_angle): name = c_label; break
        if name is None: name = f"ray_{int(angle)}"
        
        # 1. Find all matches aligned with ray
        candidate_indices_local = _find_all_matches_for_ray(
            angle, x_rot, y_rot, tolerance_meters=RAY_TOLERANCE_METERS
        )
        
        # 2. Capture the "Raw" best (First index = best geometric match)
        if len(candidate_indices_local) > 0:
            raw_local_idx = candidate_indices_local[0]
            raw_global_idx = indices[raw_local_idx]
            raw_results_map[name] = pano_collection[raw_global_idx]
        
        # 3. Iterate to find best non-occluded match
        found_match = False
        for local_idx in candidate_indices_local:
            global_idx = indices[local_idx]
            cand_lon = cand_lons_filter1[local_idx]
            cand_lat = cand_lats_filter1[local_idx]
            
            is_blocked = _check_occlusion((cand_lon, cand_lat), footprint_wgs84, potential_occluders)
            
            if not is_blocked:
                results_map[name] = pano_collection[global_idx]
                found_match = True
                break 
        
        if not found_match:
            results_map[name] = None

    # -----------------------------------------------------------
    # 8. PRINTING & PLOTTING
    # -----------------------------------------------------------
    if print_results:
        sorted_keys = sorted(results_map.keys(), key=_sort_key)
        for ray_name in sorted_keys:
            img = results_map[ray_name]
            status = f"ID: {img.id}" if img else "No match (or Occluded)"
            print(f"{ray_name.ljust(12)}: {status}")

    if DEBUG_PLOTS:
        # Helper to get plot coords from an image map
        def get_plot_coords(img_map):
            pts = {}
            for key, img in img_map.items():
                if img is None: continue
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
                p = max(poly.geoms, key=lambda a: a.area) if poly.geom_type == 'MultiPolygon' else poly
                rot_neighbors.append(_transform_polygon_to_local_rotated(p, projector, cos_t, sin_t))
            except: pass

        _plot_dynamic_results(
            local_poly, x_rot, y_rot, 
            theta_rad, ray_tolerance_meters, sorted_angles,
            best_points=plot_points_final,
            raw_points=plot_points_raw, 
            neighbor_polys=rot_neighbors
        )

    return results_map

def _find_all_matches_for_ray(
    angle_deg: float,
    x_rot: np.ndarray,
    y_rot: np.ndarray,
    tolerance_meters: float
) -> np.ndarray:
    """
    Returns indices of ALL candidates matching the ray, sorted by 'best' fit.
    This replaces _find_match_idx_for_ray which only returned the single best.
    """
    rad = np.radians(angle_deg)
    ray_vec_x = np.cos(rad)
    ray_vec_y = np.sin(rad)
    
    perp_dist = np.abs(x_rot * -ray_vec_y + y_rot * ray_vec_x)
    dot_prod = x_rot * ray_vec_x + y_rot * ray_vec_y
    
    mask = (perp_dist <= tolerance_meters) & (dot_prod > 0)
    
    if not np.any(mask):
        return np.array([], dtype=int)
    
    # Get indices where mask is True
    valid_local_indices = np.where(mask)[0]
    valid_dists = perp_dist[mask]
    
    # Sort by smallest perpendicular distance (best alignment to ray)
    # You could also sort by `dot_prod` (closeness to building) if preferred,
    # but usually ray alignment is the priority.
    sorted_order = np.argsort(valid_dists)
    
    return valid_local_indices[sorted_order]

def _sort_key(k: str) -> tuple:
    """Sort key helper."""
    if "major_pos" in k: return (0, 0)
    if "minor_pos" in k: return (1, 0)
    if "major_neg" in k: return (2, 0)
    if "minor_neg" in k: return (3, 0)
    
    if "corner" in k: 
        try:
            payload = k.split('_')[1] 
            digits = "".join(filter(str.isdigit, payload))
            letters = "".join(filter(str.isalpha, payload))
            num = int(digits) if digits else 0
            suffix_val = ord(letters[0]) if letters else 0
            return (4, num * 100 + suffix_val)
        except:
            return (4, 9999)

    try:
        val = int(k.split("_")[1])
    except:
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
        footprint:
            Shapely Polygon in lon/lat (WGS84) coordinates.
        lons_red, lats_red:
            NumPy arrays of longitudes and latitudes for the first round of
            radial filtering.
        lons_green, lats_green:
            NumPy arrays of longitudes and latitudes resulting from the
            second round of radial filtering based on more accurate
            distance calculations.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the Building Footprint (Blue):
    x_poly, y_poly = footprint.exterior.xy
    ax.fill(
        x_poly,
        y_poly,
        alpha=0.4,
        fc="lightblue",
        ec="blue",
        label="Building footprint",
    )

    # Plot the Centroid (Black X):
    centroid = footprint.centroid
    ax.scatter(
        centroid.x,
        centroid.y,
        color="black",
        marker="x",
        s=100,
        zorder=10,
        label="Centroid",
    )

    # Plot the first set of pano locations resulting from approximate radial
    # filtering (Red):
    if lons_red.size and lats_red.size:
        ax.scatter(
            lons_red,
            lats_red,
            color="red",
            s=30,
            alpha=0.6,
            label="Coarse radial search (Red)",
        )

    # Plot the second set of pano locations based on accurate distance 
    # calculations filtering (Green):
    if lons_green.size and lats_green.size:
        ax.scatter(
            lons_green,
            lats_green,
            color="green",
            s=40,
            alpha=0.9,
            zorder=5,
            label="Accurate distance filter (Green)",
        )

    # Clean up the figure
    ax.set_title("Search Results Comparison")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal")

    # Only add legend if there is something to show
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left")

    plt.show()

def _result_img_index_lookup(target_img, valid_indices, collection):
    """Lookup helper."""
    for idx in valid_indices:
        if collection[idx] == target_img:
            return idx
    return -1
"""
def _plot_dynamic_results(local_poly, x_rot, y_rot, theta_rad, tolerance, angles, best_points):
    fig, ax = plt.subplots(figsize=(12, 12))
    theta_deg = float(np.degrees(theta_rad))
    rot_poly = shapely_rotate(local_poly, theta_deg, origin=(0.0, 0.0))
    px, py = rot_poly.exterior.xy
    ax.fill(px, py, alpha=0.3, fc="gray", ec="black", label="Rotated Building")
    if x_rot.size and y_rot.size:
        ax.scatter(x_rot, y_rot, color="lightgray", s=15, alpha=0.5, zorder=1)
    lim = 70
    for angle in angles:
        rad = np.radians(angle)
        color = "blue" if np.isclose(angle % 180, 0) else ("purple" if np.isclose(angle % 180, 90) else "green")
        ex, ey = lim * np.cos(rad), lim * np.sin(rad)
        ax.plot([0, ex], [0, ey], color=color, linestyle="--", lw=0.8, alpha=0.7)
    for key, (cx, cy) in best_points.items():
        c = "blue" if "major" in key else ("purple" if "minor" in key else "green")
        ax.scatter(cx, cy, c=c, marker="*", s=150, edgecolors="black", zorder=10)
    ax.set_aspect("equal")
    plt.show()
"""
def _plot_dynamic_results(
    local_poly: Polygon,
    x_rot: np.ndarray,
    y_rot: np.ndarray,
    theta_rad: float,
    tolerance: float,
    angles: list,
    best_points: dict,        # Final (Occlusion Checked) results
    raw_points: dict = None,  # <--- NEW: Raw (Geometric Best) results
    neighbor_polys: list[Polygon] = None
) -> None:
    """
    Visualizes Target, Neighbors, Rays, Raw Candidates (Circles), and Final Selections (Stars).
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. Plot Neighbors (The Occluders)
    if neighbor_polys:
        for npoly in neighbor_polys:
            nx, ny = npoly.exterior.xy
            ax.fill(
                nx, ny, 
                alpha=0.3, 
                fc="salmon", 
                ec="red", 
                label="_nolegend_"
            )
        ax.plot([], [], color="salmon", alpha=0.6, linewidth=5, label="Neighboring Buildings (Occluders)")

    # 2. Plot Target Building (Gray)
    theta_deg = float(np.degrees(theta_rad))
    rot_poly = shapely_rotate(local_poly, theta_deg, origin=(0.0, 0.0))
    px, py = rot_poly.exterior.xy
    ax.fill(px, py, alpha=0.5, fc="gray", ec="black", label="Target Building")

    # 3. Plot all candidate dots background (Light Gray)
    if x_rot.size and y_rot.size:
        ax.scatter(x_rot, y_rot, color="lightgray", s=15, alpha=0.5, zorder=1)

    # 4. Plot Rays
    lim = 80 
    for angle in angles:
        rad = np.radians(angle)
        if np.isclose(angle % 180, 0): color = "blue"
        elif np.isclose(angle % 180, 90): color = "purple"
        else: color = "green"
        ex = lim * np.cos(rad)
        ey = lim * np.sin(rad)
        ax.plot([0, ex], [0, ey], color=color, linestyle="--", lw=0.8, alpha=0.7)

    # 5. Plot RAW Matches (The Geometrically Closest) - HOLLOW CIRCLES
    if raw_points:
        for key, (rx, ry) in raw_points.items():
            if "major" in key: c = "blue"
            elif "minor" in key: c = "purple"
            else: c = "green"
            
            # Plot hollow circle
            ax.scatter(
                rx, ry, 
                facecolors='none', 
                edgecolors=c, 
                marker='o', 
                s=150, 
                linewidth=2,
                zorder=9,
                label="_nolegend_"
            )

    # 6. Plot FINAL Matches (The Visible Ones) - SOLID STARS
    for key, (cx, cy) in best_points.items():
        if "major" in key: c = "blue"
        elif "minor" in key: c = "purple"
        else: c = "green"

        ax.scatter(
            cx, cy, 
            c=c, 
            marker="*", 
            s=200, 
            edgecolors="black", 
            zorder=10, 
            label="_nolegend_"
        )

    ax.set_title("Occlusion Analysis\nCircles=Best Geometry (Raw) | Stars=Best Visible (Final)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=4, label='Target'),
        Line2D([0], [0], color='salmon', lw=4, label='Occluder'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='none', markersize=10, label='Raw Match (Ignored Occlusion)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=15, label='Final Match (Visible)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()

def crop_panorama_to_asset(
        target_asset, 
        pano_image, 
        horizontal_padding_deg=5.0, 
        vertical_padding_percent=1.0, 
        vertical_crop_mode='full'
    ):
    """
    Crop the panorama based on the extent of the asset.
    """
    
    # Extract the PIL image:
    pil_image = pano_image._pil_image
    img_w, img_h = pil_image.size
    
    # Extract the segmentation mask:
    if vertical_crop_mode == 'smart':
        mask_data = pano_image.load_mask('semantic')
    
    if not isinstance(pil_image, Image.Image): 
        raise ImportError('Invalid PIL Image')

    # Get the geometry for the asset and process it into a polygon of 
    # appropriate type:
    geom = target_asset.geometry
    poly = max(geom.geoms, key=lambda a: a.area) \
        if geom.geom_type == "MultiPolygon" else \
            (geom if geom.geom_type == "Polygon" else geom.convex_hull)
    coords = list(poly.exterior.coords)

    # Calculate bearing angles of asset vertices relative to North:
    # (-180 deg to 180 deg):
    bearings = [_get_bearing(pano_image.properties['latitude'], pano_image.properties['longitude'], lat, lon) for lon, lat in coords]
    
    # Calculate bearings angles of asset vertices relative to the image center:
    rel_bearings = []
    for b in bearings:
        diff = b - pano_image.properties['compass_angle']
        # Normalize to -180 to 180:
        diff = (diff + 180) % 360 - 180
        rel_bearings.append(diff)

    # Sort bearings to find the largest gap
    # This identifies if the asset spans across the 180/-180 seam
    rel_bearings.sort()
    
    # Calculate gaps between adjacent sorted angles:
    max_gap = 0
    start_index = 0
    
    # Check gaps between consecutive points:
    for i in range(len(rel_bearings) - 1):
        gap = rel_bearings[i+1] - rel_bearings[i]
        if gap > max_gap:
            max_gap = gap
            start_index = i + 1

    # Check the gap between the last and first point (wrapping around 360)
    wrap_gap = (rel_bearings[0] + 360) - rel_bearings[-1]
    
    if wrap_gap > max_gap:
        # If the largest gap is the wrap-around gap, this means the asset
        # DOES NOT cross the seam. Normal logic applies:
        min_angle = rel_bearings[0] - horizontal_padding_deg
        max_angle = rel_bearings[-1] + horizontal_padding_deg
        crosses_seam = False
    else:
        # If the largest gap is somewhere in the middle of the array, this 
        # means the asset CROSSES the seam (the "back" of the image).
        # In this care all the points, excluding the largest gap, are retained.
        # Logic: Start from the index after the gap, wrap around to the index 
        # before the gap:
        
        # Identify the boundary angles roughly
        # The "left" edge is the value after the gap
        # The "right" edge is the value before the gap
        min_angle = rel_bearings[start_index] - horizontal_padding_deg
        max_angle = rel_bearings[start_index - 1] + horizontal_padding_deg 
        crosses_seam = True

    # Calculate the pixels count per number of degrees and the location of the
    # image center in pixel coordinates:
    pixels_per_deg = img_w / 360.0
    center_x = img_w / 2.0

    # Calculate Crop Coordinates:
    if not crosses_seam:
        # The asset is fully visible within the continuous image frame, i.e.
        # it does NOT split across the left/right edges (the -180/180 degree
        # seam):
        
        # Convert angles to pixel:
        start_x = int(center_x + (min_angle * pixels_per_deg))
        end_x = int(center_x + (max_angle * pixels_per_deg))
        
        # In case the cropping area is pushed off-canvas, clamp it to image
        # bounds:
        start_x = max(0, start_x)
        end_x = min(img_w, end_x)
        
        # Crop image in horizontal direction:
        image_strip = pil_image.crop((start_x, 0, end_x, img_h))
        
        # Crop segmentation mask in horizontal direction: 
        if mask_data is not None:
            mask_strip = mask_data[:, start_x:end_x]
    
    else:
        # Because the crop area crosses the seam, min_angle is positive 
        # (e.g., 170) and max_angle is negative (e.g., -170) and they need to 
        # be normalized. Here the image is shifted so the seam is in the middle
        # to make cropping easy.
        
        # Calculate width of the asset Normalize angles to 0-360 for width
        # calculation:
        norm_min = (min_angle + 360) % 360
        norm_max = (max_angle + 360) % 360
        
        # Calculate width in degrees (handling wrap):
        width_deg = (norm_max - norm_min + 360) % 360
        width_px = int(width_deg * pixels_per_deg)

        # Calculate where the "left" edge (min_angle) starts in pixel 
        # coordinates:
        start_px_rel = int(center_x + (min_angle * pixels_per_deg))
        
        # If start_px_rel is > img_w, modulo it:
        start_px_rel = start_px_rel % img_w

        # Offset the image so 'start_px_rel' becomes x=0
        # If start is at 3000px, we shift by -3000
        image_strip = offset(pil_image, -start_px_rel, 0).crop((0, 0, width_px, img_h))

        # Crop segmentation mask in horizontal direction using the same 
        # approach: 
        if mask_data is not None:
            rolled_mask = np.roll(mask_data, -start_px_rel, axis=1)
            mask_strip = rolled_mask[:, :width_px]
            
            
    # Perform a vertical crop:
    final_top = 0
    final_bottom = img_h

    if vertical_crop_mode == 'smart' and mask_strip is not None:
        try:
            # Resolve semantic IDs:
            sky_id, vehicle_id = None, None
    
            if pano_image.semantic_map:
                # Create a lowercase map for case-insensitive lookup:
                name_to_id = {
                    k.lower(): v for v, k in pano_image.semantic_map.items()
                }
                sky_id = name_to_id.get(SegmentationLabels.SKY)
                vehicle_id = name_to_id.get(SegmentationLabels.SURVEY_VEHICLE)
                
            # Save mask IDs to ignore:
            ids_to_ignore = [0]
            if sky_id is not None:
                ids_to_ignore.append(sky_id)
            if vehicle_id is not None:
                ids_to_ignore.append(vehicle_id)
        
            # Create the content mask (i.e., everything but the sky and
            # collection vehicle):
            valid_content_mask = ~np.isin(mask_strip, ids_to_ignore)
            
            # Check which rows contain at least one content pixel:
            rows_with_content = np.any(valid_content_mask, axis=1)
            
            # Get the indices of rows with content:
            content_indices = np.flatnonzero(rows_with_content)
            
            # Calculate vertical padding relative to original image height:
            padding_px = int(img_h * vertical_padding_percent/100)
            
            if content_indices.size > 0:
                # Find the very first row with content below the sky line:
                min_row = content_indices[0]
                
                # Calculate the top crop:
                final_top = max(0, min_row - padding_px)
            
                if vehicle_id is not None:
                    # Check rows where the vehicle appears:
                    vehicle_rows = np.any(mask_strip == vehicle_id, axis=1)
                    vehicle_indices = np.flatnonzero(vehicle_rows)
                    
                    if vehicle_indices.size > 0:
                        # The non-padded crop line is the highest point 
                        # (min row) of the vehicle:
                        vehicle_top_row = vehicle_indices[0]
                        
                        # Calculate the bottom crop:
                        final_bottom = max(0, vehicle_top_row - padding_px)

            else:
                logging.info(
                    'Smart crop could not be performed: Mask strip '
                    'contains only background/sky/data collection vehicle.'
                )
                
        except Exception as e:
            logging.warning(
                f'Smart crop failed, returning full height strip: {e}'
            )

    elif vertical_crop_mode == 'full':
        pass

    else:
        logging.warning(
            f"Unsupported vertical crop mode '{vertical_crop_mode}'. "
            "Returning a full height image."
        )
        
    # Safety Check: Make sure the crop is valid (i.e., top is above bottom):
    if final_top >= final_bottom:
        logging.warning(
            f'Invalid vertical crop calculated {final_top} to {final_bottom}. ' 
            'Resetting to full.'
        )
        final_top = 0
        final_bottom = img_h

    # Final Vertical Crop on the already horizontally-cropped strip
    # Note: strip width is already correct, we just adjust Y:
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
        >>> # Bearing from Equator/Prime Meridian (0,0) towards East (0, 10)
        >>> get_bearing(0.0, 0.0, 0.0, 10.0)
        90.0
        
        >>> # Bearing from Equator/Prime Meridian (0,0) towards North (10, 0)
        >>> get_bearing(0.0, 0.0, 10.0, 0.0)
        0.0
    """
    d_lon = math.radians(lon2 - lon1)
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    y = math.sin(d_lon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) \
        * math.cos(lat2_r) * math.cos(d_lon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360