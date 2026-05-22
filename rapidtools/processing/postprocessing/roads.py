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
# 05-21-2026

import logging
import math
import statistics
import uuid
from typing import Any

import networkx as nx
import pyproj
from shapely.geometry import LineString, MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, transform, unary_union, voronoi_diagram
from tqdm import tqdm

# Import your core inventory classes (adjust the import path as needed)
from rapidtools.core import PhysicalAsset, PhysicalAssetCollection

logger = logging.getLogger(__name__)


# ==============================================================================
# PIPELINE CONFIGURATION CONSTANTS
# ==============================================================================
# These constants define the heuristic thresholds and multipliers used by the
# geometry processing algorithms. They can be modified to tune the regularization
# behavior for different scales of assets (e.g., highways vs. alleyways).

# 1. Loading & Initial Processing
MIN_POLYGON_AREA_SQFT = 1000.0  # Ignore stray artifacts smaller than this
CLEANUP_BUFFER_FT = 0.1  # Buffer used to fuse slightly disconnected polygons
MIN_APPROX_WIDTH_FT = 10.0  # Absolute baseline width fallback for roads

# 2. Skeletonization Parameters (Multipliers of calculated approx_width)
SKELETON_SIMPLIFY_FACTOR = 0.1  # Initial smoothing tolerance
SKELETON_BUFFER_FACTOR = 0.2  # Outward/Inward buffer to remove jagged edges
SKELETON_SEGMENTIZE_FACTOR = 0.5  # Distance between points injected along boundary
SKELETON_CLIP_FACTOR = 0.05  # Negative buffer to clip Voronoi lines inside polygon

# 3. Graph Healing & Pruning
COORD_ROUNDING_DECIMALS = 1  # Precision used when identifying graph nodes
DEAD_END_PRUNING_FACTOR = 2.5  # Multiplier of approx_width to identify stub dead-ends

# 4. Collinear Segment Merging
COLLINEAR_SIMPLIFY_FACTOR = 0.75  # Tolerance for straightening wobbly centerlines
MIN_SEGMENT_LENGTH_FT = 1.0  # Drop segments shorter than this after simplification
MAX_MERGE_ANGLE_DEG = 25.0  # Maximum angle difference to merge two lines into one
FINAL_LINE_SIMPLIFY_TOLERANCE = 2.0  # Final smoothing tolerance applied after merge

# 5. Statistical Width Sampling
MIN_SAMPLES_PER_SEGMENT = 3  # Min width measurements per road segment
SAMPLE_SPACING_FACTOR = 0.5  # Spacing between samples (multiplier of approx_width)
DERIVATIVE_DELTA_FT = 1.0  # Delta offset used to calculate the normal vector (heading)
MEASURING_TAPE_FACTOR = 5.0  # Multiplier of approx_width defining max sampling ray cast

# 6. Output Formatting
OUTPUT_ROUNDING_DECIMALS = 2  # Decimals used when storing properties (width, azimuth)


class RoadwayRegularizer:
    """
    Pipeline component to regularize jagged raster-to-vector road polygons into
    clean, statistically sized, continuous straight segments.

    This regularizer processes raw input geometries, extracts a Voronoi-based
    centerline skeleton, heals the graph, merges collinear segments, and
    statistically samples widths. It outputs in-memory `PhysicalAssetCollection`
    objects for both the centerlines and the reconstructed polygons.

    Args:
        min_width_ft (float, optional):
            The minimum enforced width for any regularized road segment in
            feet. Defaults to 22.0.
        min_network_length_ft (float, optional):
            The minimum cumulative length a polygon collection must possess to be
            considered a valid road network. Any disconnected stub networks
            shorter than this value will be filtered out. Defaults to 100.0.
    """

    def __init__(
        self,
        min_width_ft: float = 22.0,
        min_network_length_ft: float = 100.0,
    ):
        self.min_width_ft = min_width_ft
        self.min_network_length_ft = min_network_length_ft

    def __call__(
        self, input_assets: PhysicalAssetCollection
    ) -> tuple[PhysicalAssetCollection, PhysicalAssetCollection]:
        """
        Allows the instance to be called directly like a function.

        Args:
            input_assets (PhysicalAssetCollection): The raw input road polygons.

        Returns:
            tuple[PhysicalAssetCollection, PhysicalAssetCollection]: A tuple containing
                the centerlines collection and the reconstructed polygons collection.
        """
        return self.process(input_assets)

    def process(
        self, input_assets: PhysicalAssetCollection
    ) -> tuple[PhysicalAssetCollection, PhysicalAssetCollection]:
        """
        Executes the regularization logic and returns asset collections.

        Args:
            input_assets (PhysicalAssetCollection): The raw input road polygons.

        Returns:
            tuple[PhysicalAssetCollection, PhysicalAssetCollection]: A tuple containing:
                - `centerlines_collection`: Assets representing the road centerlines.
                - `polygons_collection`: Assets representing the regularized polygons.
        """
        logger.info('Starting Roadway Regularization Pipeline...')

        # 1. Extract Geometries & Project
        raw_geometries = [asset.geometry for asset in input_assets]
        
        if not raw_geometries:
            logger.warning('No valid geometries found in input collection. Exiting.')
            return PhysicalAssetCollection(), PhysicalAssetCollection()

        union_all = unary_union(raw_geometries)
        project_to_feet, project_to_wgs84 = self._get_transformers(union_all)

        valid_polys = [
            transform(project_to_feet, g).buffer(0)
            for g in raw_geometries
            if transform(project_to_feet, g).area > MIN_POLYGON_AREA_SQFT
        ]

        all_networks = unary_union(
            [
                p.buffer(CLEANUP_BUFFER_FT, join_style=2).buffer(
                    -CLEANUP_BUFFER_FT, join_style=2
                )
                for p in valid_polys
            ]
        )

        largest_poly = self._get_largest_polygon(all_networks)
        approx_width = max(
            MIN_APPROX_WIDTH_FT, 2.0 * (largest_poly.area / largest_poly.length)
        )

        # 2. Skeletonization
        merged_lines = self._generate_voronoi_skeleton(all_networks, approx_width)

        # 3. Graph Healing & Pruning
        graph = self._build_and_heal_graph(merged_lines, approx_width)

        # 4. Filter Stub Networks
        graph = self._filter_stub_networks(graph, approx_width)

        # 5. Merge Collinear Segments
        straight_segments = self._merge_collinear_segments(graph, approx_width)

        # 6. Sample Widths & Azimuth
        analyzed_segments = self._calculate_widths_and_azimuths(
            straight_segments, all_networks, approx_width
        )

        # 7. Build Centerlines Collection
        centerlines_collection = self._build_centerlines_collection(
            analyzed_segments, project_to_wgs84
        )

        # 8. Reconstruct Polygons & Build Collection
        polygons_collection = self._build_polygons_collection(
            analyzed_segments, project_to_wgs84
        )

        logger.info('Roadway regularization complete.')
        return centerlines_collection, polygons_collection

    # --- Private Helper Methods ---

    def _get_geoms(self, geometry: BaseGeometry) -> list[BaseGeometry]:
        """Extracts a list of sub-geometries from a multi-part Shapely geometry."""
        if hasattr(geometry, 'geoms'):
            return list(geometry.geoms)
        return [geometry]

    def _get_lines(self, geom: BaseGeometry) -> list[LineString]:
        """Recursively extracts LineStrings from a geometry or geometry collection."""
        lines = []
        if geom.geom_type == 'LineString':
            lines.append(geom)
        elif hasattr(geom, 'geoms'):
            for part in geom.geoms:
                lines.extend(self._get_lines(part))
        return lines

    def _get_transformers(self, base_geom: BaseGeometry) -> tuple[Any, Any]:
        """Configures pyproj transformers to convert between metric and WGS84."""
        proj_wgs84 = pyproj.CRS('EPSG:4326')
        proj_feet = pyproj.CRS(
            f'+proj=aeqd +lat_0={base_geom.centroid.y} '
            f'+lon_0={base_geom.centroid.x} +datum=WGS84 +units=us-ft'
        )
        project_to_feet = pyproj.Transformer.from_crs(
            proj_wgs84, proj_feet, always_xy=True
        ).transform
        project_to_wgs84 = pyproj.Transformer.from_crs(
            proj_feet, proj_wgs84, always_xy=True
        ).transform
        return project_to_feet, project_to_wgs84

    def _get_largest_polygon(self, geom: BaseGeometry) -> BaseGeometry:
        """Finds the largest single polygon within a multi-polygon."""
        if geom.geom_type in ['MultiPolygon', 'GeometryCollection']:
            return max(
                [g for g in self._get_geoms(geom) if g.geom_type == 'Polygon'],
                key=lambda p: p.area,
            )
        return geom

    def _extract_all_coords(self, geom: BaseGeometry) -> list[tuple[float, float]]:
        """Recursively extracts coordinate tuples from a geometry."""
        if hasattr(geom, 'geoms'):
            pts = []
            for g in geom.geoms:
                pts.extend(self._extract_all_coords(g))
            return pts
        elif hasattr(geom, 'coords'):
            return list(geom.coords)
        return []

    def _generate_voronoi_skeleton(
        self, network_geom: BaseGeometry, approx_width: float
    ) -> BaseGeometry:
        """Generates a skeleton geometry from raw polygons using a Voronoi diagram."""
        logger.info('Generating Voronoi skeleton...')
        smoothed_poly = (
            network_geom.simplify(approx_width * SKELETON_SIMPLIFY_FACTOR)
            .buffer(approx_width * SKELETON_BUFFER_FACTOR, join_style=2)
            .buffer(-approx_width * SKELETON_BUFFER_FACTOR, join_style=2)
        )

        boundary_pts = smoothed_poly.boundary.segmentize(
            approx_width * SKELETON_SEGMENTIZE_FACTOR
        )
        coords = self._extract_all_coords(boundary_pts)

        voronoi_edges = voronoi_diagram(MultiPoint(coords), edges=True)
        clipped_lines = voronoi_edges.intersection(
            smoothed_poly.buffer(-approx_width * SKELETON_CLIP_FACTOR)
        )

        return linemerge(unary_union(clipped_lines))

    def _build_and_heal_graph(
        self, merged_lines: BaseGeometry, approx_width: float
    ) -> nx.Graph:
        """Builds a NetworkX graph from lines and prunes or heals segments."""
        logger.info('Building and healing graph...')
        G = nx.Graph()

        # Build initial graph
        for line in self._get_geoms(merged_lines):
            if line.geom_type != 'LineString':
                continue
            start = (
                round(line.coords[0][0], COORD_ROUNDING_DECIMALS),
                round(line.coords[0][1], COORD_ROUNDING_DECIMALS),
            )
            end = (
                round(line.coords[-1][0], COORD_ROUNDING_DECIMALS),
                round(line.coords[-1][1], COORD_ROUNDING_DECIMALS),
            )
            if start != end:
                G.add_edge(start, end, geom=line, length=line.length)

        # Heal and Prune
        pruning = True
        while pruning:
            pruning = False
            healing = True

            while healing:
                healing = False
                deg2_nodes = [n for n in G.nodes if G.degree(n) == 2]
                for n in deg2_nodes:
                    if G.degree(n) == 2:
                        u, v = list(G.neighbors(n))
                        if u != v and not G.has_edge(u, v):
                            lines_to_merge = self._get_lines(
                                G[u][n]['geom']
                            ) + self._get_lines(G[n][v]['geom'])
                            merged_geom = linemerge(lines_to_merge)
                            new_length = G[u][n]['length'] + G[n][v]['length']
                            G.add_edge(u, v, geom=merged_geom, length=new_length)
                            G.remove_node(n)
                            healing = True

            dead_ends = [n for n in G.nodes if G.degree(n) == 1]
            for node in dead_ends:
                if G.has_node(node) and G.degree(node) == 1:
                    neighbor = list(G.neighbors(node))[0]
                    if (
                        G[node][neighbor]['length']
                        < approx_width * DEAD_END_PRUNING_FACTOR
                    ):
                        G.remove_node(node)
                        pruning = True
        return G

    def _filter_stub_networks(self, G: nx.Graph, approx_width: float) -> nx.Graph:
        """Filters out structurally insignificant stub networks based on path length."""
        logger.info('Structurally filtering stub networks...')
        valid_edges = []
        for comp in list(nx.connected_components(G)):
            subgraph = G.subgraph(comp)
            total_length = sum(
                data['length'] for _, _, data in subgraph.edges(data=True)
            )
            if total_length > self.min_network_length_ft:
                valid_edges.extend(subgraph.edges(data=True))

        filtered_G = nx.Graph()
        filtered_G.add_edges_from(valid_edges)
        return filtered_G

    def _merge_collinear_segments(
        self, G: nx.Graph, approx_width: float
    ) -> list[LineString]:
        """Simplifies and merges collinear sub-segments into continuous lines."""
        logger.info('Merging collinear segments into continuous roads...')
        straight_segments = []
        for _, _, data in G.edges(data=True):
            straightened = data['geom'].simplify(
                approx_width * COLLINEAR_SIMPLIFY_FACTOR, preserve_topology=False
            )
            for part in self._get_lines(straightened):
                if part.length > MIN_SEGMENT_LENGTH_FT:
                    straight_segments.append(part)

        merged_something = True
        while merged_something:
            merged_something = False
            for i in range(len(straight_segments)):
                for j in range(i + 1, len(straight_segments)):
                    l1, l2 = straight_segments[i], straight_segments[j]
                    c1, c2 = list(l1.coords), list(l2.coords)

                    shared = None
                    if c1[-1] == c2[0]:
                        shared, other1, other2 = c1[-1], c1[-2], c2[1]
                        new_coords = c1 + c2[1:]
                    elif c1[0] == c2[-1]:
                        shared, other1, other2 = c1[0], c1[1], c2[-2]
                        new_coords = c2 + c1[1:]
                    elif c1[-1] == c2[-1]:
                        shared, other1, other2 = c1[-1], c1[-2], c2[-2]
                        new_coords = c1 + c2[::-1][1:]
                    elif c1[0] == c2[0]:
                        shared, other1, other2 = c1[0], c1[1], c2[1]
                        new_coords = c1[::-1] + c2[1:]

                    if shared:
                        a1 = math.atan2(shared[1] - other1[1], shared[0] - other1[0])
                        a2 = math.atan2(other2[1] - shared[1], other2[0] - shared[0])
                        diff = math.degrees(
                            abs((a1 - a2 + math.pi) % (2 * math.pi) - math.pi)
                        )

                        if diff < MAX_MERGE_ANGLE_DEG:
                            straight_segments.pop(j)
                            straight_segments.pop(i)
                            merged_line = LineString(new_coords).simplify(
                                FINAL_LINE_SIMPLIFY_TOLERANCE, preserve_topology=False
                            )
                            straight_segments.append(merged_line)
                            merged_something = True
                            break
                if merged_something:
                    break
        return straight_segments

    def _calculate_widths_and_azimuths(
        self, segments: list[LineString], network: BaseGeometry, approx_width: float
    ) -> list[dict[str, Any]]:
        """Samples the physical width and heading of segments against the network."""
        logger.info('Statistically sampling road widths & calculating azimuth...')
        analyzed = []
        for seg in tqdm(segments, desc='   -> Sampling'):
            length = seg.length
            num_samples = max(
                MIN_SAMPLES_PER_SEGMENT,
                int(length / (approx_width * SAMPLE_SPACING_FACTOR)),
            )
            sampled_widths = []

            for i in range(1, num_samples):
                dist = i * length / num_samples
                pt = seg.interpolate(dist)

                pt1 = seg.interpolate(max(0, dist - DERIVATIVE_DELTA_FT))
                pt2 = seg.interpolate(min(length, dist + DERIVATIVE_DELTA_FT))
                dx, dy = pt2.x - pt1.x, pt2.y - pt1.y
                norm = math.hypot(dx, dy)
                if norm == 0:
                    continue
                nx_vec, ny_vec = -dy / norm, dx / norm

                measuring_tape = LineString(
                    [
                        (
                            pt.x + nx_vec * approx_width * MEASURING_TAPE_FACTOR,
                            pt.y + ny_vec * approx_width * MEASURING_TAPE_FACTOR,
                        ),
                        (
                            pt.x - nx_vec * approx_width * MEASURING_TAPE_FACTOR,
                            pt.y - ny_vec * approx_width * MEASURING_TAPE_FACTOR,
                        ),
                    ]
                )
                sampled_widths.append(measuring_tape.intersection(network).length)

            final_width = max(
                self.min_width_ft,
                statistics.median(sampled_widths)
                if sampled_widths
                else self.min_width_ft,
            )

            coords = list(seg.coords)
            seg_dx, seg_dy = coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1]
            azimuth = (math.degrees(math.atan2(seg_dx, seg_dy)) + 360) % 360

            analyzed.append({'geometry': seg, 'width': final_width, 'azimuth': azimuth})
        return analyzed

    def _build_centerlines_collection(
        self, analyzed_segments: list[dict[str, Any]], proj_transform: Any
    ) -> PhysicalAssetCollection:
        """
        Converts analyzed LineStrings into a PhysicalAssetCollection.

        Args:
            analyzed_segments (list[dict[str, Any]]): Data generated by the analyzer
                containing geometry, width, and azimuth.
            proj_transform (Any): A pyproj coordinate transformer.

        Returns:
            PhysicalAssetCollection: The resulting centerline assets.
        """
        logger.info('Building centerlines PhysicalAssetCollection...')
        collection = PhysicalAssetCollection()

        for item in analyzed_segments:
            seg_wgs84 = transform(proj_transform, item['geometry'])
            for part in self._get_geoms(seg_wgs84):
                if part.geom_type == 'LineString' and not part.is_empty:
                    asset = PhysicalAsset(
                        id=f'cl_{uuid.uuid4().hex[:8]}',
                        geometry=part,
                        attributes={
                            'asset_type': 'road_centerline',
                            'width_ft': round(item['width'], OUTPUT_ROUNDING_DECIMALS),
                            'azimuth_deg': round(
                                item['azimuth'], OUTPUT_ROUNDING_DECIMALS
                            ),
                        },
                    )
                    collection.add(asset)

        return collection

    def _build_polygons_collection(
        self, analyzed_segments: list[dict[str, Any]], proj_transform: Any
    ) -> PhysicalAssetCollection:
        """
        Reconstructs linear polygons and packages them into a PhysicalAssetCollection.

        Args:
            analyzed_segments (list[dict[str, Any]]): Data generated by the analyzer
                containing geometry, width, and azimuth.
            proj_transform (Any): A pyproj coordinate transformer.

        Returns:
            PhysicalAssetCollection: The resulting polygon assets.
        """
        logger.info(
            'Reconstructing linear polygons and building PhysicalAssetCollection...'
        )
        analyzed_segments.sort(key=lambda x: x['width'], reverse=True)
        collection = PhysicalAssetCollection()
        placed_polys = []

        for item in analyzed_segments:
            poly = item['geometry'].buffer(item['width'] / 2.0, cap_style=2)

            for p, p_bounds in placed_polys:
                poly_bounds = poly.bounds
                if not (
                    poly_bounds[2] < p_bounds[0]
                    or poly_bounds[0] > p_bounds[2]
                    or poly_bounds[3] < p_bounds[1]
                    or poly_bounds[1] > p_bounds[3]
                ):
                    if poly.intersects(p):
                        poly = poly.difference(p)
                        if poly.is_empty:
                            break

            if not poly.is_empty:
                placed_polys.append((poly, poly.bounds))
                for part in self._get_geoms(transform(proj_transform, poly)):
                    if part.geom_type == 'Polygon' and not part.is_empty:
                        asset = PhysicalAsset(
                            id=f'poly_{uuid.uuid4().hex[:8]}',
                            geometry=part,
                            attributes={
                                'asset_type': 'road_polygon',
                                'width_ft': round(
                                    item['width'], OUTPUT_ROUNDING_DECIMALS
                                ),
                                'azimuth_deg': round(
                                    item['azimuth'], OUTPUT_ROUNDING_DECIMALS
                                ),
                            },
                        )
                        collection.add(asset)

        return collection