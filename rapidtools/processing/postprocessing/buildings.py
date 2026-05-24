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
# 05-24-2026

import math
from shapely.geometry import LineString
from shapely.ops import split

import logging
import uuid
from typing import Any

import networkx as nx
import numpy as np
import pyproj
import rasterio.features
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform, unary_union

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
from rapidtools.processing.image_segmenters import SAM3ImageSegmenter

logger = logging.getLogger(__name__)

# Pipeline configuration:
DEFAULT_PROMPT = 'building'
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_RETRY_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_RETRY_MASK_THRESHOLD = 0.3
DEFAULT_SNAP_TOLERANCE_FT = 0.5


class BuildingRegularizer:
    """
    Pipeline component to regularize and refine preliminary building polygons.

    This regularizer takes assets that have preliminary building geometries and
    associated cropped imagery, runs a batched SAM 3 segmentation to find the
    exact physical boundaries of the buildings, and maps the resulting pixel
    masks back to real-world WGS84 coordinates. 
    
    It filters out AI hallucinations and giant blobs using IoU, splits merged 
    preliminary shapes using Recall, and runs a second low-threshold pass 
    on dropped assets to rescue shadowed buildings.
    """

    def __init__(
        self,
        prompt: str | list[str] = DEFAULT_PROMPT,
        batch_size: int = 4,
        load_in_4bit: bool = True,
        min_overlap_ratio: float = 0.15,
        max_gap_bridge_ft: float = 15.0,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        retry_dropped: bool = True,
        retry_confidence_threshold: float = DEFAULT_RETRY_CONFIDENCE_THRESHOLD,
        retry_mask_threshold: float = DEFAULT_RETRY_MASK_THRESHOLD,
        snap_tolerance_ft: float = DEFAULT_SNAP_TOLERANCE_FT,
    ) -> None:
        """
        Initializes the BuildingRegularizer with SAM 3 and pipeline thresholds.

        Args:
            prompt (str | list[str]): 
                The text prompt(s) used for SAM 3 segmentation (e.g., 'building roof').
            batch_size (int): 
                Batch size for SAM 3 inference.
            load_in_4bit (bool): 
                Whether to load the SAM 3 model in 4-bit precision to save memory.
            min_overlap_ratio (float): 
                Minimum Intersection over Union (IoU) required to keep a mask.
            max_gap_bridge_ft (float): 
                Maximum distance in feet to dynamically bridge fragmented polygons.
            confidence_threshold (float): 
                Primary confidence threshold for SAM 3 mask generation.
            mask_threshold (float): 
                Primary threshold for SAM 3 mask binarization.
            retry_dropped (bool): 
                Whether to run a second pass with lower thresholds on dropped assets.
            retry_confidence_threshold (float): 
                Lower confidence threshold for the retry pass.
            retry_mask_threshold (float): 
                Lower mask threshold for the retry pass.
            snap_tolerance_ft (float): 
                Tolerance in feet for snapping polygon vertices.
        """
        self.min_overlap_ratio = min_overlap_ratio
        self.max_gap_bridge_ft = max_gap_bridge_ft
        self.snap_tolerance_ft = snap_tolerance_ft
        self.retry_dropped = retry_dropped
        self.retry_confidence_threshold = retry_confidence_threshold
        self.retry_mask_threshold = retry_mask_threshold

        # Safely handle lists by converting them to the period-separated string
        # format:
        if isinstance(prompt, list):
            safe_prompt = '. '.join(prompt)
        else:
            safe_prompt = prompt

        self.segmenter = SAM3ImageSegmenter(
            prompt=safe_prompt,
            batch_size=batch_size,
            load_in_4bit=load_in_4bit,
            threshold=confidence_threshold,
            mask_threshold=mask_threshold,
        )

    def __call__(
        self, input_assets: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Allows the regularizer instance to be called as a function.

        Args:
            input_assets (PhysicalAssetCollection): 
                The preliminary building assets to regularize.

        Returns:
            PhysicalAssetCollection: 
                The fully processed and refined building assets.
        """
        return self.process(input_assets)

    def process(
        self, input_assets: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Executes the full building regularization pipeline.

        This pipeline consists of:
        1. Running batched SAM 3 segmentation on the image crops.
        2. Georeferencing masks and applying geometric/spatial filters.
        3. (Optional) Retrying dropped assets with lower thresholds.
        4. Dissolving overlapping duplicates.
        5. Truncating neighboring intrusions to form clean property lines.

        Args:
            input_assets (PhysicalAssetCollection): 
                Collection of preliminary building assets.

        Returns:
            PhysicalAssetCollection: 
                Collection of refined, georeferenced, and non-overlapping building assets.
        """        
        logger.info('Starting Building Regularization Pipeline...')

        if not input_assets:
            logger.warning('No assets provided to BuildingRegularizer. Exiting.')
            return PhysicalAssetCollection()

        # Pass 1:
        logger.info('Step 1/4: Running SAM 3 segmentation on image crops...')
        segmented_assets = self.segmenter(input_assets)

        logger.info('Step 2/4: Georeferencing masks and dynamically bridging gaps...')
        refined_collection = self._georeference_and_filter(segmented_assets)

        # Pass 2: Retry dropped assets with lower thresholds:
        if self.retry_dropped:
            # --- BUG FIX 1: Strip the suffix so the IDs match input assets ---
            successful_ids = {asset.id.split('_split_')[0] for asset in refined_collection}
            dropped_assets = PhysicalAssetCollection()
            
            for asset in input_assets:
                if asset.id not in successful_ids:
                    dropped_assets.add(asset)

            if len(dropped_assets) > 0:
                logger.info(
                    f'Step 3/4: Retrying {len(dropped_assets)} dropped assets '
                    f'with lower thresholds...'
                )
                
                # Temporarily overwrite segmenter thresholds:
                orig_conf = self.segmenter.threshold
                orig_mask = self.segmenter.mask_threshold
                self.segmenter.threshold = self.retry_confidence_threshold
                self.segmenter.mask_threshold = self.retry_mask_threshold
                
                # Run the retry pass:
                retry_segmented = self.segmenter(dropped_assets)
                retry_refined = self._georeference_and_filter(retry_segmented)
                
                # Restore original thresholds:
                self.segmenter.threshold = orig_conf
                self.segmenter.mask_threshold = orig_mask
                
                logger.info(
                    f'Pass 2 recovered {len(retry_refined)} previously dropped assets.'
                )
                
                # Merge recovered assets into the main collection:
                for asset in retry_refined:
                    refined_collection.add(asset)
            else:
                logger.info('Step 3/4: No dropped assets to retry.')
        else:
            logger.info('Step 3/4: Retry step skipped (retry_dropped=False).')

        # Dissolve:
        logger.info('Step 4/4: Dissolving overlapping and adjoining polygons...')
        merged_collection = self._dissolve_overlapping_assets(refined_collection)
        
        # Clip minor overlaps
        logger.info('Step 5: Truncating intrusions between neighboring polygons...')
        final_collection = self._truncate_intrusions(merged_collection)

        logger.info('Building regularization complete.')
        return final_collection

    def _fill_holes(self, geometry: BaseGeometry) -> BaseGeometry:
        """
        Removes interior rings (holes) from polygons to ensure solid footprints.
        """
        if geometry.geom_type == 'Polygon':
            return Polygon(geometry.exterior)
        elif geometry.geom_type in ('MultiPolygon', 'GeometryCollection'):
            polygons = [
                Polygon(p.exterior) 
                for p in getattr(geometry, 'geoms', []) 
                if p.geom_type == 'Polygon'
            ]
            return MultiPolygon(polygons) if polygons else geometry
            
        return geometry

    def _truncate_intrusions(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Slices off intrusions by extending the penetrated wall of the 
        larger building into a straight cutting line, guaranteeing clean cuts 
        without L-shaped bites or notches.
        """
        # Sort by area descending so large, primary buildings act as the unyielding 'walls'
        assets = sorted(list(asset_collection), key=lambda a: a.geometry.area, reverse=True)
        
        resolved_collection = PhysicalAssetCollection()
        seen_geoms = [] 

        for asset in assets:
            geom = asset.geometry.buffer(0)
            if geom.is_empty:
                continue
                
            # Check for overlaps against all primary buildings we've already locked in
            for larger_geom in seen_geoms:
                if not geom.intersects(larger_geom):
                    continue
                    
                overlap = geom.intersection(larger_geom)
                if overlap.is_empty or overlap.area < 1e-5:
                    continue
                    
                # 1. Extract all exterior wall segments of the primary building
                segments = []
                if larger_geom.geom_type == 'Polygon':
                    ext_coords = list(larger_geom.exterior.coords)
                    segments = [
                        LineString([ext_coords[k], ext_coords[k+1]]) 
                        for k in range(len(ext_coords)-1)
                    ]
                
                # 2. Find the walls that the smaller building is touching/crossing
                intersecting_walls = [
                    seg for seg in segments 
                    if seg.intersects(overlap) or seg.distance(overlap) < 1e-3
                ]
                
                if intersecting_walls:
                    # 3. Pick the longest wall segment involved in the collision
                    main_wall = max(intersecting_walls, key=lambda s: s.length)
                    
                    # 4. Extend this wall into an infinite cutting line
                    coords = list(main_wall.coords)
                    p1, p2 = coords[0], coords[1]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = math.hypot(dx, dy)
                    
                    if length > 0:
                        nx, ny = dx / length, dy / length
                        ext = 5000  # Make the line long enough to slice the whole neighbor
                        ext_p1 = (p1[0] - nx * ext, p1[1] - ny * ext)
                        ext_p2 = (p2[0] + nx * ext, p2[1] + ny * ext)
                        cut_line = LineString([ext_p1, ext_p2])
                        
                        # 5. Slice the invading building cleanly along the extended wall!
                        try:
                            parts = split(geom, cut_line)
                            valid_polys = [
                                p for p in parts.geoms 
                                if p.geom_type == 'Polygon' and not p.is_empty
                            ]
                            
                            if len(valid_polys) > 1:
                                # Keep the piece that safely sits OUTSIDE the main building
                                geom = min(valid_polys, key=lambda p: p.intersection(larger_geom).area)
                            else:
                                geom = geom.difference(larger_geom) # Fallback
                        except Exception:
                            geom = geom.difference(larger_geom) # Fallback
                    else:
                        geom = geom.difference(larger_geom)
                else:
                    geom = geom.difference(larger_geom) # Fallback
                    
            # 6. Clean up the final shape (drop tiny splinters if any)
            if geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
                polys = [g for g in getattr(geom, 'geoms', []) if g.geom_type == 'Polygon']
                if polys:
                    geom = max(polys, key=lambda p: p.area)
                else:
                    continue
                    
            asset.geometry = geom
            seen_geoms.append(geom)
            resolved_collection.add(asset)
            
        return resolved_collection

    def _bridge_fragments(self, geometry: BaseGeometry) -> BaseGeometry:
        """
        Dynamically calculates the distance needed to connect disconnected 
        polygon parts and applies Morphological Closing to fuse them.

        Useful for reconnecting building components split by visual obstructions 
        like tree branches or shadows.
        """
        if geometry.geom_type not in ('MultiPolygon', 'GeometryCollection'):
            return geometry

        proj_wgs84 = pyproj.CRS('EPSG:4326')
        proj_local = pyproj.CRS(
            f'+proj=aeqd +lat_0={geometry.centroid.y} '
            f'+lon_0={geometry.centroid.x} +datum=WGS84 +units=us-ft'
        )
        
        to_local = pyproj.Transformer.from_crs(
            proj_wgs84, proj_local, always_xy=True
        ).transform
        to_wgs84 = pyproj.Transformer.from_crs(
            proj_local, proj_wgs84, always_xy=True
        ).transform

        local_geom = shapely_transform(to_local, geometry)
        polygons = [
            g for g in getattr(local_geom, 'geoms', []) 
            if g.geom_type == 'Polygon'
        ]
        
        if len(polygons) <= 1:
            return geometry

        G = nx.Graph()
        for i, p1 in enumerate(polygons):
            G.add_node(i)
            for j, p2 in enumerate(polygons):
                if i < j:
                    G.add_edge(i, j, weight=p1.distance(p2))

        mst = nx.minimum_spanning_tree(G)
        max_gap = max(data['weight'] for _, _, data in mst.edges(data=True))
        dynamic_buffer = (max_gap / 2.0) * 1.05
        dynamic_buffer = min(dynamic_buffer, self.max_gap_bridge_ft)
        
        bridged_local = local_geom.buffer(dynamic_buffer, join_style=2).buffer(
            -dynamic_buffer, join_style=2
        )
        
        bridged_wgs84 = shapely_transform(to_wgs84, bridged_local)
        
        if bridged_wgs84.geom_type in ('MultiPolygon', 'GeometryCollection'):
            polygons = [
                g for g in getattr(bridged_wgs84, 'geoms', []) 
                if g.geom_type == 'Polygon'
            ]
            if polygons:
                return max(polygons, key=lambda p: p.area)
                
        return bridged_wgs84

    def _georeference_and_filter(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        refined_collection = PhysicalAssetCollection()
        """
        Georeferences pixel masks to WGS84 and applies spatial quality filters.

        Evaluates SAM 3 masks against their preliminary seed geometry using 
        Intersection over Union (IoU), Recall, and Precision. Pieces that pass 
        are combined (to reconstruct complex hip-roofs) and converted to WGS84 
        polygons.
        """

        for asset in asset_collection:
            prelim_geom = asset.geometry
            if not prelim_geom or prelim_geom.is_empty:
                continue


            # If coordinates are outside valid WGS84 bounds (-180 to 180),
            # dynamically reproject prelim_geom to WGS84 so they can intersect:
            minx, miny, maxx, maxy = prelim_geom.bounds
            if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
                transformer = pyproj.Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
                prelim_geom = shapely_transform(transformer.transform, prelim_geom)

            sam3_masks_dict: dict[str, Any] = asset.attributes.get('sam3_masks', {})
            if not sam3_masks_dict:
                continue

            valid_instances = []

            for img_asset in asset.image_assets:
                masks = sam3_masks_dict.get(img_asset.id)
                if masks is None or len(masks) == 0:
                    continue

                wgs84_bounds = img_asset.properties.get('wgs84_bounds')
                if not wgs84_bounds:
                    continue

                min_lon, min_lat, max_lon, max_lat = wgs84_bounds

                if masks.ndim == 2:
                    masks = masks[np.newaxis, ...]

                height, width = masks.shape[-2:]
                transform = from_bounds(
                    min_lon, min_lat, max_lon, max_lat, width, height
                )

                for instance_mask in masks:
                    binary_mask = np.squeeze(instance_mask > 0.5).astype(np.uint8)
                    instance_parts = []

                    for geom_dict, val in rasterio.features.shapes(
                        binary_mask, transform=transform
                    ):
                        if val == 1:
                            poly = shape(geom_dict)
                            if poly.is_valid and not poly.is_empty:
                                instance_parts.append(poly)

                    if not instance_parts:
                        continue

                    instance_geom = unary_union(instance_parts)

                    if instance_geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
                        instance_geom = self._bridge_fragments(instance_geom)

                    # Calculate areas safely:
                    prelim_area = prelim_geom.area
                    if prelim_area == 0 or instance_geom.area == 0:
                        continue

                    intersection_area = instance_geom.intersection(prelim_geom).area
                    
                    # RECALL: Did the new mask cover the preliminary seed:
                    recall = intersection_area / prelim_geom.area
                    
                    # IoU: The ultimate measure of spatial similarity:
                    union_area = prelim_geom.union(instance_geom).area
                    iou = intersection_area / union_area if union_area > 0 else 0.0

                    # Accept the mask if it is a solid 1-to-1 match (IoU) OR if
                    # it is a cleanly extracted building that covers the 
                    # preliminary seed (recall):
                    if iou >= self.min_overlap_ratio or recall >= 0.50:
                        valid_instances.append(instance_geom)

            # Process all valid instances we collected:
            if valid_instances:
                for idx, valid_geom in enumerate(valid_instances):
                    # HEAL: Fix any self-intersections instantly:
                    healed_geom = valid_geom.buffer(0)
                    
                    if healed_geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
                        valid_polys = [
                            g for g in getattr(healed_geom, 'geoms', []) 
                            if g.geom_type == 'Polygon' and not g.is_empty
                        ]
                    elif healed_geom.geom_type == 'Polygon' and not healed_geom.is_empty:
                        valid_polys = [healed_geom]
                    else:
                        continue 

                    # Add each piece as a brand new asset to the collection:
                    for p_idx, final_poly in enumerate(valid_polys):
                        new_attrs = asset.attributes.copy() if asset.attributes else {}
                        new_attrs['preliminary_geometry_wkt'] = prelim_geom.wkt
                        new_attrs['sam3_refined'] = True
                        
                        new_asset = PhysicalAsset(
                            id=f"{asset.id}_split_{idx}_{p_idx}",
                            geometry=final_poly,
                            attributes=new_attrs
                        )
                        
                        if hasattr(asset, 'image_assets'):
                            for img in asset.image_assets:
                                new_asset.add_image_assets(img)
                                
                        refined_collection.add(new_asset)
            else:
                logger.debug(
                    f"Asset '{asset.id}' dropped. No SAM 3 instances passed filters."
                )

        return refined_collection

    def _dissolve_overlapping_assets(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Takes all refined geometries and dissolves ONLY highly overlapping ones 
        (duplicates). Keeps adjacent but distinct neighbor buildings separate.

        Builds a network graph of polygons that overlap by > 50% and merges 
        connected components into single distinct physical structures.
        """
        geometries = [
            asset.geometry
            for asset in asset_collection
            if asset.geometry and not asset.geometry.is_empty
        ]

        if not geometries:
            return PhysicalAssetCollection()

        # Build a network graph to group duplicates:
        G = nx.Graph()
        for i, geom in enumerate(geometries):
            G.add_node(i, geom=geom)

        # Check overlap between every pair of polygons:
        for i in range(len(geometries)):
            for j in range(i + 1, len(geometries)):
                g1 = geometries[i]
                g2 = geometries[j]

                if not (
                    g1.bounds[0] > g2.bounds[2] or 
                    g1.bounds[2] < g2.bounds[0] or 
                    g1.bounds[1] > g2.bounds[3] or 
                    g1.bounds[3] < g2.bounds[1]
                ):
                    if g1.intersects(g2):
                        inter_area = g1.intersection(g2).area
                        min_area = min(g1.area, g2.area)
                        
                        # If they overlap by more than 50% of the smaller 
                        # polygon's area, they are duplicates targeting the same
                        # physical building:
                        if min_area > 0 and (inter_area / min_area) > 0.5:
                            G.add_edge(i, j)

        final_collection = PhysicalAssetCollection()

        # Merge grouped duplicates and create the final assets:
        for comp in nx.connected_components(G):
            comp_geoms = [G.nodes[n]['geom'] for n in comp]
            
            merged_geom = unary_union(comp_geoms).buffer(0)
            
            clean_parts = (
                [merged_geom] 
                if merged_geom.geom_type == 'Polygon' 
                else list(getattr(merged_geom, 'geoms', []))
            )
            
            for poly in clean_parts:
                if poly.is_empty or poly.geom_type != 'Polygon':
                    continue
                    
                # Fill any holes caused by HVACs, shadows, or trees:
                solid_poly = self._fill_holes(poly)
    
                new_asset = PhysicalAsset(
                    id=f'bldg_merged_{uuid.uuid4().hex[:8]}',
                    geometry=solid_poly,
                    attributes={'asset_type': 'building_footprint'},
                )
                final_collection.add(new_asset)

        return final_collection