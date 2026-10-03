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
Regularization of raster-derived road polygons into clean centerlines.

This module provides :class:`RoadwayRegularizer`, a pure-geometry pipeline
component that converts the jagged blobs produced by raster-to-vector road
segmentation (e.g., from
:class:`~rapidtools.processing.SAM3OrthoFeatureExtractor` with the prompt
``'road'``) into straight centerline segments with statistically sampled
widths, and rebuilds clean, non-overlapping road polygons from them.

All computation happens in a local azimuthal-equidistant projection in US
survey feet; inputs and outputs are WGS84.

Example:
    >>> from rapidtools.core import PhysicalAssetCollection
    >>> from rapidtools.processing import RoadwayRegularizer
    >>> raw_roads = PhysicalAssetCollection.from_geojson('roads_raw.geojson')
    >>> centerlines, polygons = RoadwayRegularizer(min_width_ft=20)(raw_roads)
    >>> centerlines.to_geojson('centerlines.geojson')
    >>> polygons.to_geojson('roads_clean.geojson')
"""

import logging
import math
import statistics
import uuid
from typing import Any, Generic, Literal, TypeVar, cast, overload

import networkx as nx
import pyproj
import shapely
from shapely.geometry import LineString, MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, transform, unary_union, voronoi_diagram
from tqdm import tqdm

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
from rapidtools.processing.step import Stage

logger = logging.getLogger(__name__)

# What ``RoadwayRegularizer.__call__`` returns: a ``(centerlines, polygons)``
# tuple for ``output='both'`` or a single collection otherwise.
_OutT = TypeVar('_OutT')
_Both = tuple[PhysicalAssetCollection, PhysicalAssetCollection]

ASSET_TYPE = 'road'

# Pipeline configuration constants:
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
SKELETON_SNAP_GRID_FT = 0.001  # Grid used to snap Voronoi edge endpoints before merging

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


class RoadwayRegularizer(Generic[_OutT]):
    """
    Pipeline component to regularize jagged raster-to-vector road polygons.

    This regularizer processes raw input geometries, extracts a Voronoi-based
    centerline skeleton, heals the graph, merges collinear segments, and
    statistically samples widths. It outputs in-memory
    ``PhysicalAssetCollection`` objects for both the centerlines and the
    reconstructed polygons.

    Args:
        min_width_ft (float, optional):
            The minimum enforced width for any regularized road segment in
            feet. Defaults to 22.0.
        min_network_length_ft (float, optional):
            The minimum cumulative length a connected skeleton component must
            possess to be considered a valid road network. Any disconnected
            stub networks shorter than this value will be filtered out.
            Defaults to 100.0.

    Example:
        >>> from rapidtools.core import PhysicalAssetCollection
        >>> from rapidtools.processing import RoadwayRegularizer
        >>>
        >>> raw = PhysicalAssetCollection.from_geojson('roads_raw.geojson')
        >>> regularizer = RoadwayRegularizer(
        ...     min_width_ft=22.0, min_network_length_ft=150.0
        ... )
        >>> centerlines, polygons = regularizer(raw)
        >>> centerlines[0].attributes
        {'asset_type': 'road_centerline', 'width_ft': 24.5, 'azimuth_deg': 91.2}
    """

    stage = Stage.REGULARIZE

    @overload
    def __init__(
        self: 'RoadwayRegularizer[_Both]',
        min_width_ft: float = ...,
        min_network_length_ft: float = ...,
        output: Literal['both'] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: 'RoadwayRegularizer[PhysicalAssetCollection]',
        min_width_ft: float = ...,
        min_network_length_ft: float = ...,
        *,
        output: Literal['polygons', 'centerlines'],
    ) -> None: ...

    @overload
    def __init__(
        self: 'RoadwayRegularizer[Any]',
        min_width_ft: float = ...,
        min_network_length_ft: float = ...,
        output: str = ...,
    ) -> None: ...

    def __init__(
        self,
        min_width_ft: float = 22.0,
        min_network_length_ft: float = 100.0,
        output: str = 'both',
    ):
        """
        Initialize the regularizer thresholds.

        Args:
            min_width_ft (float, optional):
                Minimum enforced road width in feet.
            min_network_length_ft (float, optional):
                Minimum cumulative length of a skeleton component in feet.
        """
        self.min_width_ft = min_width_ft
        self.min_network_length_ft = min_network_length_ft
        if output not in ('both', 'polygons', 'centerlines'):
            raise ValueError(
                f"output must be 'both', 'polygons' or 'centerlines', got {output!r}."
            )
        # 'both' returns (centerlines, polygons); the single-collection modes make
        # the regularizer usable as a Pipeline step.
        self.output = output

    def __call__(self, input_assets: PhysicalAssetCollection) -> _OutT:
        """
        Allow the instance to be called directly like a function.

        Args:
            input_assets (PhysicalAssetCollection):
                The raw input road polygons (WGS84).

        Returns:
            With ``output='both'`` (the default), a tuple containing the
            centerlines collection and the reconstructed polygons collection;
            with ``output='polygons'`` or ``output='centerlines'``, just that
            collection.

        Example:
            >>> centerlines, polygons = RoadwayRegularizer()(raw_roads)
        """
        centerlines, polygons = self.process(input_assets)
        # The casts are backed by the ``__init__`` overloads, which tie
        # ``_OutT`` to the ``output`` mode:
        if self.output == 'polygons':
            return cast(_OutT, polygons)
        if self.output == 'centerlines':
            return cast(_OutT, centerlines)
        return cast(_OutT, (centerlines, polygons))

    def process(
        self, input_assets: PhysicalAssetCollection
    ) -> tuple[PhysicalAssetCollection, PhysicalAssetCollection]:
        """
        Execute the regularization logic and return asset collections.

        The steps are: project to a local feet-based CRS, drop polygons
        smaller than ``MIN_POLYGON_AREA_SQFT``, fuse near-touching polygons,
        skeletonize with a Voronoi diagram, heal/prune the skeleton graph,
        remove short stub networks, merge collinear segments, sample widths
        and azimuths, and finally rebuild WGS84 centerline and polygon
        assets.

        Args:
            input_assets (PhysicalAssetCollection):
                The raw input road polygons (WGS84).

        Returns:
            tuple[PhysicalAssetCollection, PhysicalAssetCollection]:
                A tuple containing:

                - ``centerlines_collection``: Assets representing the road
                  centerlines (``LineString`` geometries) with the attributes
                  ``'asset_type': 'road_centerline'``, ``'width_ft'`` and
                  ``'azimuth_deg'``.
                - ``polygons_collection``: Assets representing the regularized
                  polygons with ``'asset_type': 'road_polygon'`` and the same
                  width/azimuth attributes.

                Both collections are empty when the input is empty or when no
                polygon exceeds the minimum area.

        Example:
            >>> centerlines, polygons = RoadwayRegularizer().process(raw_roads)
            >>> polygons[0].attributes['asset_type']
            'road_polygon'
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

        if not valid_polys:
            logger.warning(
                f'No input polygon exceeds {MIN_POLYGON_AREA_SQFT} sq ft. Exiting.'
            )
            return PhysicalAssetCollection(), PhysicalAssetCollection()

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

        # 9. Default undefined asset types to 'road'
        centerlines_collection.set_asset_type(ASSET_TYPE, overwrite=False)
        polygons_collection.set_asset_type(ASSET_TYPE, overwrite=False)

        logger.info('Roadway regularization complete.')
        return centerlines_collection, polygons_collection

    # --- Private Helper Methods ---

    def _get_geoms(self, geometry: BaseGeometry) -> list[BaseGeometry]:
        """
        Extract the list of sub-geometries from a (multi-part) geometry.

        Args:
            geometry (BaseGeometry):
                Any Shapely geometry.

        Returns:
            list[BaseGeometry]:
                The parts of a multi-part geometry, or ``[geometry]`` for a
                single-part geometry.
        """
        if hasattr(geometry, 'geoms'):
            return list(geometry.geoms)
        return [geometry]

    def _get_lines(self, geom: BaseGeometry) -> list[LineString]:
        """
        Recursively extract ``LineString`` parts from a geometry.

        Args:
            geom (BaseGeometry):
                A ``LineString``, ``MultiLineString`` or nested
                ``GeometryCollection``.

        Returns:
            list[LineString]:
                All ``LineString`` parts found (empty for point/polygon input).
        """
        lines = []
        if geom.geom_type == 'LineString':
            lines.append(geom)
        elif hasattr(geom, 'geoms'):
            for part in geom.geoms:
                lines.extend(self._get_lines(part))
        return lines

    def _get_transformers(self, base_geom: BaseGeometry) -> tuple[Any, Any]:
        """
        Build pyproj transformers between WGS84 and a local feet-based CRS.

        The local CRS is an azimuthal equidistant projection centred on the
        centroid of ``base_geom`` with units of US survey feet.

        Args:
            base_geom (BaseGeometry):
                WGS84 geometry whose centroid defines the projection origin.

        Returns:
            tuple[Callable, Callable]:
                ``(project_to_feet, project_to_wgs84)`` transform functions
                suitable for :func:`shapely.ops.transform`.
        """
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
        """
        Find the largest single polygon within a multi-part geometry.

        Args:
            geom (BaseGeometry):
                A ``Polygon``, ``MultiPolygon`` or ``GeometryCollection``.

        Returns:
            BaseGeometry:
                The polygon part with the largest area, or ``geom`` itself
                when it is not multi-part.
        """
        if geom.geom_type in ['MultiPolygon', 'GeometryCollection']:
            return max(
                [g for g in self._get_geoms(geom) if g.geom_type == 'Polygon'],
                key=lambda p: p.area,
            )
        return geom

    def _extract_all_coords(self, geom: BaseGeometry) -> list[tuple[float, float]]:
        """
        Recursively extract coordinate tuples from a geometry.

        Args:
            geom (BaseGeometry):
                Any geometry; multi-part geometries are flattened.

        Returns:
            list[tuple[float, float]]:
                All coordinates found, in order. Objects with neither
                ``geoms`` nor ``coords`` yield an empty list.
        """
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
        """
        Generate a centerline skeleton from road polygons using a Voronoi diagram.

        The polygon is smoothed, its boundary densified, and the Voronoi edges
        of the boundary points are clipped to the polygon interior. Edges that
        remain form the medial axis approximation. Edge endpoints are snapped
        to ``SKELETON_SNAP_GRID_FT`` so that adjoining edges merge cleanly.

        Args:
            network_geom (BaseGeometry):
                The fused road polygons in the local feet CRS.
            approx_width (float):
                Estimated road width in feet; scales all tolerances.

        Returns:
            BaseGeometry:
                A ``LineString`` or ``MultiLineString`` of merged skeleton
                edges.
        """
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

        # Voronoi edge endpoints computed from different cells can differ by
        # floating-point noise; snap them to a fine grid so that linemerge
        # recognises them as shared vertices and produces continuous lines.
        snapped_lines = shapely.set_precision(
            unary_union(clipped_lines), SKELETON_SNAP_GRID_FT
        )
        return linemerge(snapped_lines)

    def _build_and_heal_graph(
        self, merged_lines: BaseGeometry, approx_width: float
    ) -> nx.Graph:
        """
        Build a NetworkX graph from skeleton lines and prune or heal segments.

        Nodes are line endpoints rounded to ``COORD_ROUNDING_DECIMALS``; the
        line vertices are snapped onto those node coordinates so edges join
        exactly. Edges carry ``geom`` and ``length`` attributes. Degree-2 nodes
        are collapsed by merging their two edges,
        and dead-end edges shorter than ``approx_width *
        DEAD_END_PRUNING_FACTOR`` are removed. The two operations alternate
        until the graph stops changing.

        Args:
            merged_lines (BaseGeometry):
                Skeleton lines from :meth:`_generate_voronoi_skeleton`.
            approx_width (float):
                Estimated road width in feet.

        Returns:
            nx.Graph:
                The healed and pruned skeleton graph.
        """
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
                # Snap the line's end vertices onto the rounded node keys so
                # that every edge meeting at a node shares exactly the same
                # coordinate; otherwise linemerge cannot fuse them when the
                # node is healed below.
                coords = list(line.coords)
                coords[0] = start
                coords[-1] = end
                line = LineString(coords)
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
        """
        Remove connected components whose total length is too short.

        Args:
            G (nx.Graph):
                Skeleton graph with ``length`` edge attributes.
            approx_width (float):
                Estimated road width in feet (currently unused; kept for
                signature symmetry with the other graph helpers).

        Returns:
            nx.Graph:
                A new graph containing only the edges of components longer
                than ``min_network_length_ft``.
        """
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
        """
        Simplify graph edges and merge collinear sub-segments into straight lines.

        Each edge geometry is simplified with a tolerance proportional to
        ``approx_width`` and split into straight parts. Pairs of parts that
        share an endpoint and deviate by less than ``MAX_MERGE_ANGLE_DEG`` are
        merged repeatedly until no further merge is possible.

        Args:
            G (nx.Graph):
                Healed skeleton graph with ``geom`` edge attributes.
            approx_width (float):
                Estimated road width in feet.

        Returns:
            list[LineString]:
                Straight, merged centerline segments.
        """
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
        """
        Sample the physical width and heading of each segment against the network.

        At evenly spaced stations along a segment, a "measuring tape" line
        perpendicular to the local heading is intersected with the road
        polygon; the median intersection length (floored at
        ``min_width_ft``) is the segment width. The azimuth is the compass
        bearing (degrees clockwise from north) from the first to the last
        vertex.

        Args:
            segments (list[LineString]):
                Straight centerline segments in the local feet CRS.
            network (BaseGeometry):
                The fused road polygons in the same CRS.
            approx_width (float):
                Estimated road width in feet; sets sample spacing and tape
                length.

        Returns:
            list[dict[str, Any]]:
                One dict per segment with keys ``'geometry'``, ``'width'``
                (feet) and ``'azimuth'`` (degrees).
        """
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
        Convert analyzed LineStrings into a WGS84 ``PhysicalAssetCollection``.

        Args:
            analyzed_segments (list[dict[str, Any]]):
                Data generated by :meth:`_calculate_widths_and_azimuths`
                containing geometry, width, and azimuth.
            proj_transform (Any):
                A pyproj transform function from the local CRS to WGS84.

        Returns:
            PhysicalAssetCollection:
                The resulting centerline assets (IDs ``'cl_<hash>'``).
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
        Reconstruct non-overlapping road polygons from analyzed centerlines.

        Segments are buffered by half their width (flat caps) from widest to
        narrowest; each new polygon has all previously placed polygons
        subtracted so that intersections are owned by the wider road.

        Args:
            analyzed_segments (list[dict[str, Any]]):
                Data generated by :meth:`_calculate_widths_and_azimuths`
                containing geometry, width, and azimuth. Sorted in place by
                width (descending).
            proj_transform (Any):
                A pyproj transform function from the local CRS to WGS84.

        Returns:
            PhysicalAssetCollection:
                The resulting polygon assets (IDs ``'poly_<hash>'``).
        """
        logger.info(
            'Reconstructing linear polygons and building PhysicalAssetCollection...'
        )
        analyzed_segments.sort(key=lambda x: x['width'], reverse=True)
        collection = PhysicalAssetCollection()
        placed_polys: list[tuple[BaseGeometry, tuple[float, float, float, float]]] = []

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
