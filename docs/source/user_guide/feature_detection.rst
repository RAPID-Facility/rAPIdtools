Feature detection and regularization
====================================

If you have imagery but no inventory, rAPIdtools can find the assets for you.
SAM 3 searches a raster for whatever a short text prompt describes and
returns polygons. The regularizers then turn those rough masks into clean
building footprints or road centerlines. This works on Bing and Google
basemaps (stitched with :class:`~rapidtools.processing.BingOrthomosaicExtractor`
and :class:`~rapidtools.processing.GoogleOrthomosaicExtractor`) as well as on
drone orthomosaics.

Finding assets in a raster
--------------------------

:class:`~rapidtools.processing.SAM3OrthoFeatureExtractor` cuts a raster into
square tiles of a fixed size on the ground, runs SAM 3 on them in batches
with your prompt, and converts the masks back into WGS84 polygons.

.. code-block:: python

   from rapidtools import SAM3OrthoFeatureExtractor

   extractor = SAM3OrthoFeatureExtractor(
       prompt='tree canopy',
       patch_size=50, unit='meters',   # tile size on the ground
       overlap_ratio=0.25,             # overlap between tiles to avoid edge cuts
       batch_size=4,
       threshold=0.5,                  # detection confidence
       mask_threshold=0.4,             # mask binarization
   )
   canopy = extractor('eaton_bing.tiff')
   canopy.to_geojson('canopy.geojson')

Small tiles keep the texture of small objects visible. Large tiles are faster
and suit roads and vegetation, which need more context. Because neighboring
tiles overlap, the same object is often detected more than once, so by
default detections that touch are dissolved into one polygon. That is what
you want for roads, vegetation and water, but it melts neighboring objects
together: two houses whose masks meet, a house and its garage, cars parked
side by side.

For anything you want one polygon per object, such as vehicles, pass
``merge_overlaps=False``. Every object is then fused from all the tiles
that saw it: two detections from different tiles count as the same object
only when they agree inside the area both tiles covered (so a building cut
by one tile's edge still matches the whole building in the next tile), two
instances from the same tile are never merged, and the outline is the one
most tiles agree on. Each asset carries the mean SAM 3 ``confidence`` and
the number of tiles that detected it (``views``).

Tiles larger than 1,024 pixels (a 50 m tile of a 2 cm drone orthomosaic is
2,600 pixels across) are downsampled before they reach SAM 3, which works
at 1,008 pixels anyway, so centimeter-resolution rasters scan quickly.

Finding buildings
-----------------

A prompt that names buildings (``'building'``, ``'house'``, ``'structure'``,
also inside longer words such as ``'warehouse'``) needs no other settings:
the scan keeps one footprint per building and gives every footprint
straight, square walls.

.. code-block:: python

   buildings = SAM3OrthoFeatureExtractor(
       prompt='building', patch_size=50, unit='meters',
   )('eaton_bing.tiff')

This is the same as spelling out the building defaults:

.. code-block:: python

   buildings = SAM3OrthoFeatureExtractor(
       prompt='building',
       patch_size=50, unit='meters',
       overlap_ratio=0.5,          # every building seen whole by several tiles
       merge_overlaps=False,       # one footprint per building
       neck_width_m=1.0,           # cut masks that bleed across a gap between roofs
       min_detection_rate=0.3,     # drop buildings most tiles that saw them missed
       regularize=True,            # straight, square walls (FootprintRegularizer)
   )('eaton_bing.tiff')

Any of them can be set to something else; ``regularize=False`` returns the
raw mask outlines. ``extractor.scan_settings()`` shows what the next scan
will use. Settings left out follow the prompt, so an extractor that is
reused for another prompt (to keep the loaded model) switches back to the
plain defaults for, say, ``'tree'``. Tiles of about 50 m suit buildings.

For the best outlines, refine the scan on close-up crops with
:class:`~rapidtools.processing.BuildingRegularizer` (see below). The GUI
and the ``detect.aerial`` job step do this by default.

Segmenting attached images
--------------------------

:class:`~rapidtools.processing.SAM3ImageSegmenter` runs the same model on the
images already attached to your assets, batching across the whole
collection. The masks and boxes are stored in each asset's attributes as
``sam3_masks`` and ``sam3_bounding_boxes``.

Regularizing building footprints
--------------------------------

Outlines traced from masks follow the pixels: they are stair-stepped, their
corners are rounded, and a tree or a shadow along a wall leaves a bulge or a
dent. :class:`~rapidtools.processing.FootprintRegularizer` redraws every
footprint with straight walls that meet at right angles. It needs no model
or imagery and runs in seconds. Building scans already run it; use it
directly on footprints from elsewhere (another detector, a scan with
``regularize=False``, an old inventory) or with other settings:

.. code-block:: python

   from rapidtools import FootprintRegularizer

   clean = FootprintRegularizer(min_area_m2=10)(raw_footprints)
   clean.to_geojson('buildings.geojson')

Each footprint is rasterized on a grid turned to its main wall direction
(the direction of its own walls, of its minimum rotated rectangle, or the
one its neighbors share, whichever fits best), cleaned of bulges and dents
narrower than about a meter, and traced back; steps shorter than about a
meter are straightened and a footprint that is nearly a rectangle becomes
one. Round buildings are only simplified, and a footprint the square walls
fit poorly keeps a simplified outline. Neighbors are then made
non-overlapping. The functions behind it,
:func:`~rapidtools.processing.postprocessing.footprints.regularize_footprint`
and :func:`~rapidtools.processing.postprocessing.footprints.regularize_footprints`,
work on shapely polygons in meters.

:class:`~rapidtools.processing.BuildingRegularizer` goes further: it crops
high-resolution imagery around each preliminary footprint (from a SAM 3
scan, OpenStreetMap or a parcel layer), segments the crops again, and
returns one regularized footprint per building. A building usually shows up
in its own crop and in its neighbors' crops; these views are fused the same
way as the tiles of a scan, so neighbors that touch stay apart, masks that
bleed across the gap between two roofs are cut (``neck_width_m``), and a
preliminary footprint that covered two buildings comes back as two.
Buildings that were dropped get a second pass at a lower threshold, which
rescues many that sit in shadow, and because every building is seen up
close, roofs the scan outlined only in part (a roof of two materials, for
example) come back whole.

.. code-block:: python

   from rapidtools import AerialImageryExtractor, BuildingRegularizer

   with_crops = AerialImageryExtractor(
       'eaton_bing.tiff', save_directory='output/crops',
       buffer_asset='20 m', force_square_image=True,
   )(buildings)
   final = BuildingRegularizer(prompt='building', batch_size=8)(with_crops)

Every output footprint has the attributes ``confidence``, ``views`` (the
crops that detected it) and ``source_ids`` (the preliminary footprints it
matched). Masks are kept at no more than 1,024 pixels a side
(``max_mask_side``) and released as soon as they are traced, so crops of
drone orthomosaics do not exhaust memory.

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
