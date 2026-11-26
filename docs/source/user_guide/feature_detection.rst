Feature detection and regularization
====================================

If you have imagery but no inventory, rAPIdtools can find the assets for you.
SAM 3 searches a raster for whatever a short text prompt describes and
returns polygons. The regularizers then turn those rough masks into clean
building footprints or road centerlines.

Finding assets in a raster
--------------------------

:class:`~rapidtools.processing.SAM3OrthoFeatureExtractor` cuts a raster into
square tiles of a fixed size on the ground, runs SAM 3 on them in batches
with your prompt, and converts the masks back into WGS84 polygons.

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

Small tiles keep the texture of small objects visible. Large tiles are faster
and suit roads and vegetation, which need more context. Because neighboring
tiles overlap, the same object is often detected more than once, so by
default detections that touch are dissolved into one polygon. That is what
you want for roofs, roads and vegetation. For objects you want to count,
such as vehicles, pass ``merge_overlaps=False``. Each instance then stays a
separate asset with its SAM 3 ``confidence``, and only the duplicates from
overlapping tiles are removed.

Segmenting attached images
--------------------------

:class:`~rapidtools.processing.SAM3ImageSegmenter` runs the same model on the
images already attached to your assets, batching across the whole
collection. The masks and boxes are stored in each asset's attributes as
``sam3_masks`` and ``sam3_bounding_boxes``.

Regularizing building footprints
--------------------------------

Masks from a basemap are jagged, sometimes merged with a neighbor, and now
and then pure invention. :class:`~rapidtools.processing.BuildingRegularizer`
crops high-resolution imagery around each preliminary footprint, runs SAM 3
again at that resolution, and keeps the new mask only if it agrees with the
original one (by intersection over union or by recall). Footprints that
covered two buildings are split, pieces closer than ``max_gap_bridge_ft``
are joined, and buildings that were dropped get a second pass at a lower
threshold, which rescues many that sit in shadow.

.. code-block:: python

   from rapidtools import AerialImageryExtractor, BuildingRegularizer

   with_crops = AerialImageryExtractor(
       'eaton_bing.tiff', save_directory='output/crops',
       buffer_asset='20 m', force_square_image=True,
   )(buildings)
   final = BuildingRegularizer(prompt='building roof', batch_size=8)(with_crops)

The regularizer needs to know where each image sits on the ground.
:class:`~rapidtools.processing.AerialImageryExtractor` records this in two
properties of every crop: ``native_georef``, the crop's affine transform and
CRS, which lets masks be traced in the raster's own grid before they are
reprojected to WGS84; and ``wgs84_bounds``. Crops from older runs that only
have ``wgs84_bounds`` still work, but their polygons are placed using the
longitude and latitude box, which is exact only for north-up WGS84 rasters.

Regularizing roads
------------------

:class:`~rapidtools.processing.RoadwayRegularizer` reduces raw road polygons
to a Voronoi skeleton, repairs gaps in the resulting network, merges
segments that line up, and samples the width along each one. Disconnected
stubs shorter than ``min_network_length_ft`` are dropped. You get back both
the centerlines and road polygons rebuilt from them.

.. code-block:: python

   from rapidtools import RoadwayRegularizer, SAM3OrthoFeatureExtractor

   raw = SAM3OrthoFeatureExtractor(
       prompt='paved road, street', patch_size=100, unit='meters',
       threshold=0.25, mask_threshold=0.25,
   )('eaton_patch_20250214.tiff')
   centerlines, polygons = RoadwayRegularizer(min_width_ft=22)(raw)
   centerlines[0].attributes
   # {'asset_type': 'road_centerline', 'width_ft': 24.5, 'azimuth_deg': 91.2}
