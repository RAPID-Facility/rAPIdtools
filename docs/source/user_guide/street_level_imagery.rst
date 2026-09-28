Street-level imagery
====================

Street-level views show façades, debris and recovery work that aerial imagery
cannot. rAPIdtools fetches them from Google Street View without a key, or
from Mapillary with a free token, and in both cases keeps only the views that
actually show the asset.

Google Street View
------------------

:class:`~rapidtools.processing.GoogleStreetViewImageExtractor` finds the
nearest official panorama within ``search_radius_m`` of each asset, downloads
its tiles, and crops the horizontal span that covers the footprint. It uses
the same public endpoints as the Google Maps web app, so no key or billing
account is needed.

.. code-block:: python

   from rapidtools import GoogleStreetViewImageExtractor, PhysicalAssetCollection

   buildings = PhysicalAssetCollection.from_geojson('buildings.geojson')
   street = GoogleStreetViewImageExtractor(
       'output/streetview',
       search_radius_m=50,
       max_images_per_asset=2,     # extra viewpoints from neighbouring panoramas
       vertical_crop=(0.2, 0.9),   # trim sky and the camera vehicle
       save_depth_map=True,        # also write the decoded depth map (.npy)
   )
   buildings = street(buildings)
   buildings['bldg_01'].image_assets[0].properties['pano_id']

``fov_buffer_deg`` adds a margin on both sides of the footprint and
``min_fov_deg`` stops crops from becoming slivers for small or distant
assets. Assets are processed concurrently (``max_workers``).

Mapillary
---------

:class:`~rapidtools.processing.MapillaryImageExtractor` fetches regional
image metadata once, indexes it spatially, and casts rays from each
candidate camera towards the asset to reject views blocked by a neighbouring
building. It then downloads one panorama per viewing direction: the four
principal axes of the footprint, plus the corners when ``cast_corner_rays``
is on.

.. code-block:: python

   from rapidtools import MapillaryImageExtractor

   extractor = MapillaryImageExtractor(
       access_token='MLY|...',
       save_directory='output/street_feb',
       start_date='2025-01-01',      # capture-date window
       end_date='2025-05-01',
       filter_rapid_only=True,       # only imagery uploaded by the RAPID Facility
       cast_corner_rays=False,
       smart_crop=True,              # remove sky and the collection vehicle
       max_images_per_asset=8,
   )
   buildings = extractor(buildings)

Date windows make longitudinal studies straightforward: run the extractor
once per survey into separate folders, then analyze each folder with an
``image_filter`` (see :doc:`../examples/street_level_recovery`).

Checking image contents before downloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``strict_content_filter=True`` the extractor reads Mapillary's semantic
segmentation for a candidate image and skips it when the asset type is not
visible, saving the heavy JPEG download. The asset type must be expressed in
Mapillary's label vocabulary, either through ``asset_type_mapping`` or by
letting a language model translate plain English with
:class:`~rapidtools.processing.MapillaryLabelMapper`:

.. code-block:: python

   from rapidtools import Gemma4Inference, MapillaryLabelMapper

   mapper = MapillaryLabelMapper(Gemma4Inference(model_id='google/gemma-4-E2B-it'))
   mapper.map_classes(['cars', 'utility poles'])
   # ['object--vehicle--car', 'object--support--utility-pole']

   extractor = MapillaryImageExtractor(
       access_token='MLY|...', save_directory='output/street',
       strict_content_filter=True, label_mapper=mapper,
   )

Bing Streetside
---------------

Bing Streetside cannot be accessed without a Bing Maps key, because its
bubble-lookup service rejects unauthenticated requests. Only Bing aerial
tiles are available keylessly.
