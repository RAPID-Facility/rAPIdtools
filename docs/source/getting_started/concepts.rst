Concepts
========

The library is built around a few ideas. *Assets* and *collections* describe
the things you are studying, *image assets* hold the pictures of them,
*pipeline steps* do the work, and *models* are interchangeable engines
behind the steps that analyze images.

Assets and collections
----------------------

A :class:`~rapidtools.core.PhysicalAsset` stands for one real-world object,
such as a building, a road segment or a utility pole. It has a Shapely
geometry, a dictionary of ``attributes`` and an
:class:`~rapidtools.core.ImageCollection` with the pictures taken of it.

.. code-block:: python

   from shapely.geometry import Point
   from rapidtools import PhysicalAsset

   pole = PhysicalAsset(
       id='pole_123',
       geometry=Point(-118.14, 34.19),
       attributes={'asset_type': 'utility_pole', 'material': 'wood'},
   )

A :class:`~rapidtools.core.PhysicalAssetCollection` holds assets in order,
with unique IDs and fast lookup by ID. It reads and writes GeoJSON and
Shapefiles, converts to a pandas DataFrame, filters by geometry or by
attribute, and merges with other collections.

.. code-block:: python

   from rapidtools import BoundingBox, PhysicalAssetCollection

   buildings = PhysicalAssetCollection.from_geojson('buildings.geojson')
   subset = buildings.filter_by_geometry(BoundingBox(-118.15, 34.18, -118.13, 34.20))
   damaged = subset.filter_by_attribute('gemini_chs_level', '4')
   damaged.to_shapefile('damaged.shp', ignore_properties=['image_assets'])

   buildings['bldg_01']          # lookup by ID
   buildings[:10]                # slice into a new collection
   len(buildings), 'bldg_01' in buildings

Regions
-------

:class:`~rapidtools.core.BoundingBox` and
:class:`~rapidtools.core.PolygonRegion` describe areas of interest in WGS84.
You can build a bounding box from a raster's extent, a GeoJSON file or any
geometry, buffer it, split it into tiles, and use it to filter collections
or to say where to download a basemap.

.. code-block:: python

   region = BoundingBox.from_raster('eaton_patch_20250214.tiff')
   region.width, region.height, region.area
   tiles = region.tile_by_area(0.0001)   # sub-boxes of at most 0.0001 square degrees

Image assets
------------

An :class:`~rapidtools.core.ImageAsset` points at one image file. Its
``properties`` record where the image came from, for example the WGS84
bounds of an aerial crop or the panorama ID of a street-level view, and it
can carry semantic and instance masks. Extractors create image assets and
attach them to physical assets; analyzers and segmenters read them.

An image asset can exist before its file does
(``allow_missing_file=True``). Extractors use this to decide which images
are worth fetching before they spend any bandwidth.

Pipelines and stages
--------------------

A :class:`~rapidtools.processing.Pipeline` runs a list of steps over a
collection. A step is any callable that takes a collection and returns one.
The built-in steps also declare a :class:`~rapidtools.processing.Stage`,
which fixes where they run:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Stage
     - Built-in steps
   * - ``DETECT``
     - :class:`~rapidtools.processing.SAM3OrthoFeatureExtractor` finds new
       assets in a raster, and
       :class:`~rapidtools.processing.MapillaryFeatureExtractor` finds them
       along a street-level survey.
   * - ``EXTRACT_IMAGERY``
     - :class:`~rapidtools.processing.AerialImageryExtractor`,
       :class:`~rapidtools.processing.DroneImageExtractor`,
       :class:`~rapidtools.processing.GoogleStreetViewImageExtractor`,
       :class:`~rapidtools.processing.MapillaryImageExtractor` and
       :class:`~rapidtools.processing.MapillaryObjectImageExtractor` attach
       imagery to assets.
   * - ``REGULARIZE``
     - :class:`~rapidtools.processing.BuildingRegularizer` and
       :class:`~rapidtools.processing.RoadwayRegularizer` clean up
       geometries.
   * - ``SEGMENT``
     - :class:`~rapidtools.processing.SAM3ImageSegmenter` masks the attached
       images.
   * - ``VERIFY``
     - :class:`~rapidtools.processing.DetectionVerifier` and
       :class:`~rapidtools.processing.DuplicateResolver` screen detected
       objects with a model before they are analyzed.
   * - ``ANALYZE``
     - :class:`~rapidtools.processing.AssetAnalyzer` writes model answers to
       attributes.
   * - ``EXPORT`` / ``CUSTOM``
     - Your own callables. Steps without a stage run last.

The pipeline sorts its steps by stage before running them, so the order you
list them in does not matter: an analyzer listed before an extractor still
runs after it. Any plain function can be a step, which is the easiest way to
add your own filter or export.

.. code-block:: python

   def drop_unimaged(collection):
       return collection.filter_empty()

   pipeline = Pipeline([analyzer, extractor, drop_unimaged])
   result = pipeline.run(buildings)

Models and analyzers
--------------------

Every model wrapper, hosted or local, has the same
``run_inference(images, prompt)`` method and returns a
:class:`~rapidtools.models.ModelOutput`. You build one with
:func:`rapidtools.models.load` and a registry key, then hand it to
:class:`~rapidtools.processing.AssetAnalyzer`. Switching providers means
changing that key. The analyzer works out the rest itself, sending parallel
requests to hosted APIs and batching images for local models.

.. code-block:: python

   from rapidtools.models import GenerationConfig, load
   from rapidtools import AssetAnalyzer

   model = load('claude', api_key='sk-ant-...')
   analyzer = AssetAnalyzer(
       model,
       prompt='Return JSON with a damage_level key (0-4).',
       generation=GenerationConfig(json_mode=True, temperature=0.0),
   )

Answers are stored as attributes named after the provider
(``claude_damage_level``). Pass ``attribute_prefix=`` to choose your own
prefix.
