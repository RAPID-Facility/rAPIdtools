Feature detection and regularization
====================================

When you have imagery but no inventory, rAPIdtools can discover assets
itself. SAM 3 scans a raster for a text concept and returns polygons, and the
regularizers turn those raw masks into clean building footprints or road
centrelines.

Finding assets in a raster
--------------------------

:class:`~rapidtools.processing.SAM3OrthoFeatureExtractor` tiles a raster into
square patches of a real-world size, runs SAM 3 on each batch with your
prompt, and maps the resulting masks back to WGS84 polygons.

.. code-block:: python

   from rapidtools import SAM3OrthoFeatureExtractor

   extractor = SAM3OrthoFeatureExtractor(
       prompt='building roof',
       patch_size=50, unit='meters',   # tile size on the ground
       overlap_ratio=0.25,             # overlap between tiles to avoid edge cuts
       batch_size=4,
       threshold=0.5,                  # detection confidence
       mask_threshold=0.4,             # mask binarization
   )
   buildings = extractor('eaton_bing.tiff')
   buildings.to_geojson('buildings_preliminary.geojson')

Smaller patches keep texture visible for small objects; larger ones are
faster and suit roads and vegetation. Because neighbouring tiles overlap,
the same object is often detected more than once. By default touching
detections dissolve into one polygon, which is right for roofs, roads and
vegetation. For countable objects such as vehicles, pass
``merge_overlaps=False`` to keep one asset per instance with its SAM 3
``confidence``, and only remove duplicates seen from overlapping tiles.

Segmenting attached images
--------------------------

:class:`~rapidtools.processing.SAM3ImageSegmenter` runs the same model on the
images already attached to assets, batching across the whole collection, and
stores the masks and boxes in each asset's attributes under ``sam3_masks``
and ``sam3_bounding_boxes``.

Regularizing building footprints
--------------------------------

Raw masks from a basemap are jagged, sometimes merged with a neighbour, and
occasionally hallucinated. :class:`~rapidtools.processing.BuildingRegularizer`
crops high-resolution imagery around each preliminary footprint, runs SAM 3
again at the higher resolution, and keeps a mask only when it agrees with
its seed (by intersection-over-union or recall). Merged seeds are split,
fragments closer than ``max_gap_bridge_ft`` are bridged, and dropped assets
get a second, lower-threshold pass to rescue buildings in shadow.

.. code-block:: python

   from rapidtools import AerialImageryExtractor, BuildingRegularizer

   with_crops = AerialImageryExtractor(
       'eaton_bing.tiff', save_directory='output/crops',
       buffer_asset='20 m', force_square_image=True,
   )(buildings)
   final = BuildingRegularizer(prompt='building roof', batch_size=8)(with_crops)

Every image the regularizer reads must carry georeferencing in its
properties, which :class:`~rapidtools.processing.AerialImageryExtractor`
writes automatically: ``native_georef`` (the crop's affine transform and
CRS, used to vectorize masks in the raster's own grid before reprojecting to
WGS84) and ``wgs84_bounds``. Crops from older runs that only have
``wgs84_bounds`` still work, but their polygons are mapped through the
lon/lat envelope, which is exact only for north-up WGS84 rasters.

Regularizing roads
------------------

:class:`~rapidtools.processing.RoadwayRegularizer` extracts a Voronoi
skeleton from raw road polygons, heals the graph, merges collinear segments,
samples widths statistically and drops disconnected stubs shorter than
``min_network_length_ft``. It returns both centrelines and rebuilt polygons.

.. code-block:: python

   from rapidtools import RoadwayRegularizer, SAM3OrthoFeatureExtractor

   raw = SAM3OrthoFeatureExtractor(
       prompt='paved road, street', patch_size=100, unit='meters',
       threshold=0.25, mask_threshold=0.25,
   )('eaton_patch_20250214.tiff')
   centerlines, polygons = RoadwayRegularizer(min_width_ft=22)(raw)
   centerlines[0].attributes
   # {'asset_type': 'road_centerline', 'width_ft': 24.5, 'azimuth_deg': 91.2}
