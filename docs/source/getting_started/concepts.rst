Concepts
========

Four ideas cover the whole library: assets and collections describe what you
are studying, image assets hold the pictures of them, pipeline steps do the
work, and models are interchangeable engines behind the analysis steps.

Assets and collections
----------------------

A :class:`~rapidtools.core.PhysicalAsset` is the digital twin of one
real-world object: a building, a road segment, a utility pole. It holds a
Shapely geometry, a dictionary of ``attributes``, and an
:class:`~rapidtools.core.ImageCollection` of the pictures taken of it.

.. code-block:: python

   from shapely.geometry import Point
   from rapidtools import PhysicalAsset

   pole = PhysicalAsset(
       id='pole_123',
       geometry=Point(-118.14, 34.19),
       attributes={'asset_type': 'utility_pole', 'material': 'wood'},
   )

A :class:`~rapidtools.core.PhysicalAssetCollection` is an ordered container
of assets with unique IDs and constant-time lookup by ID. It reads and writes
GeoJSON and Shapefiles, converts to a pandas DataFrame, filters by geometry or
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
A bounding box can be built from a raster's extent, a GeoJSON file or any
geometry, buffered, split into tiles, and used to filter collections or to
drive basemap downloads.

.. code-block:: python

   region = BoundingBox.from_raster('eaton_patch_20250214.tiff')
   region.width, region.height, region.area
   tiles = region.tile_by_area(0.0001)   # sub-boxes of at most 0.0001 square degrees

Image assets
------------

An :class:`~rapidtools.core.ImageAsset` points at one image file and carries
``properties`` such as the WGS84 bounds of an aerial crop or the panorama ID
of a street-level view, plus optional semantic and instance masks. Extractors
create image assets and attach them to physical assets; analyzers and
segmenters read them. An image asset can exist before its file is downloaded
(``allow_missing_file=True``), which lets extractors decide what to fetch
before spending bandwidth.

Pipelines and stages
--------------------

A :class:`~rapidtools.processing.Pipeline` runs a list of steps over a
collection. Each step is a callable that takes a collection and returns one,
and declares a :class:`~rapidtools.processing.Stage`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Stage
     - Components
   * - ``DETECT``
     - :class:`~rapidtools.processing.SAM3OrthoFeatureExtractor` discovers
       new assets in a raster.
   * - ``EXTRACT_IMAGERY``
     - :class:`~rapidtools.processing.AerialImageryExtractor`,
       :class:`~rapidtools.processing.GoogleStreetViewImageExtractor`,
       :class:`~rapidtools.processing.MapillaryImageExtractor` attach
       imagery to assets.
   * - ``REGULARIZE``
     - :class:`~rapidtools.processing.BuildingRegularizer`,
       :class:`~rapidtools.processing.RoadwayRegularizer` clean geometries.
   * - ``SEGMENT``
     - :class:`~rapidtools.processing.SAM3ImageSegmenter` masks the attached
       images.
   * - ``ANALYZE``
     - :class:`~rapidtools.processing.AssetAnalyzer` writes model answers to
       attributes.
   * - ``EXPORT`` / ``CUSTOM``
     - Your own callables; unknown steps run last.

Steps can be added in any order: the pipeline sorts them by stage before
running, so an analyzer added before an extractor still runs after it. Any
plain function can be a step, which is the easiest way to add a custom
export or filter.

.. code-block:: python

   def drop_unimaged(collection):
       return collection.filter_empty()

   pipeline = Pipeline([analyzer, extractor, drop_unimaged])
   result = pipeline.run(buildings)

Models and analyzers
--------------------

Every model wrapper, hosted or local, implements ``run_inference(images,
prompt)`` and returns a :class:`~rapidtools.models.ModelOutput`. Build one
with :func:`rapidtools.models.load` and its registry key, then hand it to
:class:`~rapidtools.processing.AssetAnalyzer`. Switching providers is a
one-word change; the analyzer picks threaded execution for APIs and batched
execution for local models on its own.

.. code-block:: python

   from rapidtools.models import GenerationConfig, load
   from rapidtools import AssetAnalyzer

   model = load('claude', api_key='sk-ant-...')
   analyzer = AssetAnalyzer(
       model,
       prompt='Return JSON with a damage_level key (0-4).',
       generation=GenerationConfig(json_mode=True, temperature=0.0),
   )

Answers are written as attributes prefixed with the provider name
(``claude_damage_level``); pass ``attribute_prefix=`` to choose your own.
