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

Discovering objects along a survey
----------------------------------

The extractors above start from an inventory you already have. When the
question is "where are all the vehicles, poles or hydrants along this
survey", :class:`~rapidtools.processing.MapillaryFeatureExtractor` builds the
inventory from the imagery itself, mirroring what
:class:`~rapidtools.processing.SAM3OrthoFeatureExtractor` does for rasters.

.. code-block:: python

   from rapidtools import (
       BoundingBox, MapillaryFeatureExtractor, MapillaryObjectImageExtractor,
       PhysicalAssetCollection, Pipeline,
   )

   spokane = BoundingBox(-117.5358, 47.6885, -117.4426, 47.7378)
   pipeline = Pipeline([
       MapillaryFeatureExtractor(
           classes=['vehicles', 'utility poles'],
           access_token='MLY|...',
           region=spokane,
           start_date='2025-08-01',
           frame_spacing_m=3,        # one frame every 3 m is enough
           min_observations=3,       # seen in at least three frames
       ),
       MapillaryObjectImageExtractor(
           'output/crops', access_token='MLY|...',
           max_images_per_asset=2, overlay_asset_outline=True, outline_shape='corners',
       ),
   ])
   objects = pipeline.run(PhysicalAssetCollection())
   objects[0].attributes['localization']      # 'triangulated', 'single_view' or 'map_feature'

How it works:

1. **Detections are metadata.** Mapillary segments every image it hosts.
   The extractor reads those polygons for the requested classes through the
   Graph API, so discovering objects across a city costs a few kilobytes per
   image and downloads no pixels. Plain-English class names such as
   ``'cars'`` or ``'utility poles'`` are translated to Mapillary labels; a
   :class:`~rapidtools.processing.MapillaryLabelMapper` can be supplied for
   names the built-in aliases miss. ``'vehicles'`` means motor vehicles
   (cars, trucks, buses, motorcycles); trailers, caravans, boats and
   Mapillary's catch-all "other vehicle" are separate classes, asked for as
   ``'trailers'``, ``'boats'`` or ``'all vehicles'``.
2. **The survey vehicle is removed.** The camera car appears at the same
   place in every frame of a sequence, while a parked car drifts across the
   frame. Detections that recur at the same position in most frames are
   dropped before anything else happens.
3. **Duplicate outlines are dropped frame by frame.** Mapillary's detector
   sometimes returns one object twice: a second outline nested in the
   first (a wheel, a window, the part visible past a tree) or the two
   halves of a panorama's seam. The smaller piece is dropped. Without this,
   each piece would start its own track and the two could never merge,
   since two detections in one frame normally are two objects. Adjacent
   pieces are not joined: two cars parked side by side look exactly like a
   split outline and are far more common.
4. **Each sighting becomes a ray.** The polygon's centre column and the full
   camera pose (Mapillary's ``computed_rotation``, camera type and lens
   parameters, so pitch and roll are accounted for) give an accurate bearing
   from the camera. The polygon's lowest point gives the elevation of the
   ground contact and, with the camera height, a rough range; that range is
   only a prior. Nothing is discarded for being small or distant at this
   stage: once a track has a distance, outlines narrower than
   ``min_object_width_m`` (1 m) are dropped as fragments, and positions
   farther than ``max_range_m`` (60 m) are not reported.
5. **Sightings are tracked, triangulated and merged.** Within each sequence,
   detections are linked from frame to frame by bearing continuity, where the
   data is precise, rather than by their noisy ground positions. How far a
   bearing may jump between frames is bounded physically: the outline's
   angular width says how close an object at least ``min_object_width_m``
   wide can be, and with the camera's displacement that caps the swing, so
   a speck far down the road cannot claim a car that appears beside it.
   Each track
   is triangulated robustly from all its rays; a track whose rays do not
   agree, or whose ground-contact ranges contradict the intersection, was a
   moving vehicle and is dropped, as is anything on the line the camera
   itself drove. Tracks and single views are then merged with a
   covariance-aware gate (narrow across a ray, wide along it), so one car
   seen from two passes becomes one object while two cars seen in the same
   frame never do, unless their positions are closer than a car is wide,
   which only a split outline can produce. A *weak* triangulation (never
   seen closer than 25 m, little parallax or a large uncertainty) is not
   compared by position at all, because a bearing bias from a tree hiding
   half the car moves a 40 m intersection metres along the line of sight:
   its rays are checked against the well-located objects they point at and
   it joins the nearest one within a fraction of its range, keeping the
   close pass's position. Two checks then remove what the geometry cannot
   vouch for. A single-view position whose nearest sighting
   is beyond ``max_single_view_range_m`` (30 m) is not reported: at that
   distance the ground-contact range is a guess and the track tends to hop
   between neighbouring cars in a distant row. And the frames in which the
   detector did *not* fire are used as negative evidence against weak
   estimates (single views, and triangulations never seen closer than 25 m
   or with little parallax): when same-day frames passed within
   ``witness_radius_m`` (12 m) of such an estimate with it in view and hold
   no detection of the class along its bearing, the object is not there and
   is dropped. A car triangulated from a close pass is trusted regardless,
   since the next pass often finds it hidden behind a hedge or another car.
   Objects carry ``localization='triangulated'`` or
   ``'single_view'`` (no parallax), plus ``position_sigma_m``,
   ``parallax_deg`` and ``n_images`` so you can filter by quality.

``localization_method='voting'`` swaps step 5 for a ray-voting baseline (rays
deposit votes on a ground grid; objects are the peaks), and ``'cluster'``
restores the earlier radius clustering, both useful to compare against.

Two shortcuts sit on top of this:

- **Static point classes come from Mapillary directly.** Utility poles, fire
  hydrants, street lights, traffic signs and the other classes Mapillary
  publishes as *map features* are already triangulated across every
  contributor's imagery, so in ``detection_source='auto'`` the extractor
  fetches them with :meth:`~rapidtools.data_sources.MapillaryClient.fetch_map_features`
  instead of computing them (``localization='map_feature'``, with
  ``aligned_direction``, ``first_seen`` and ``last_seen``). Vehicles are not
  map features and always go through the image route;
  ``detection_source='mapillary'`` forces the image route for everything.
- **False detections can be screened by a model you choose.** Mapillary's
  detector sometimes fires on things that are not the requested class.
  :class:`~rapidtools.processing.DetectionVerifier` is a pipeline step
  (stage ``VERIFY``, between cropping and analysis) that shows each object's
  closest crops to any model from :func:`rapidtools.models.load` and asks
  whether it is what it claims to be. The question is strict: a motor
  vehicle is not a trailer, boat, jet ski or lawn mower, and a wheel or a
  bumper on its own does not count. The model also reports how much of the
  object is visible, and objects below ``min_visible_fraction`` (half by
  default: hidden behind a fence, cut off, too distant to tell) are
  rejected whatever else it said. It writes ``verify_accepted``,
  ``verify_confidence``, ``verify_visible_fraction`` and ``verify_label``
  and removes the rejects (or keeps them flagged with
  ``keep_rejected=True``). A custom ``classifier`` callable can stand in
  for the model.
- **Repeated passes are reconciled by appearance.** With ``reid=True`` the
  extractor crops the closest view of objects from different sequences that
  lie within ``reid_max_distance_m`` of each other, embeds them with a
  DINOv2 backbone and merges look-alikes, which removes the double counts a
  second drive down the same street would otherwise leave
  (``reid_merged`` on the merged asset).

Every object is a point :class:`~rapidtools.core.PhysicalAsset` whose
attributes include the class, the most common label, the number of
sightings and images, and the sightings themselves as
:class:`~rapidtools.core.Observation` records. Those records are what lets
:class:`~rapidtools.processing.MapillaryObjectImageExtractor` fetch only the
closest one or two views of each object, at thumbnail resolution, and crop the
detection with a margin. From there,
:class:`~rapidtools.processing.AssetAnalyzer` works exactly as it does for
aerial crops.

A city-wide survey has tens of thousands of frames, and each frame's
Mapillary payload lists every label it contains. The extractor therefore
fetches frames in batches of ``frame_batch_size`` (200 by default), keeps
only the requested classes and drops the rest before the next batch, and
simplifies every outline with ``simplify_tolerance`` (about four pixels of a
2048-wide image by default, capped at 1 % of the outline's size so distant
objects keep their detail) so memory stays flat however large the region.
The cropper likewise downloads each source image once, serves every object
seen in it and releases it, holding at most ``max_workers`` images at a time.

Classes Mapillary does not segment, such as debris piles, can be detected
with SAM 3 on the same thumbnails: with ``detection_source='auto'`` (the
default) unresolved classes go to SAM 3, and the SAM 3 sightings enter the
same localisation and clustering. Use ``frame_spacing_m`` generously in that
mode, since each frame then costs an inference call.

Bing Streetside
---------------

Bing Streetside cannot be accessed without a Bing Maps key, because its
bubble-lookup service rejects unauthenticated requests. Only Bing aerial
tiles are available keylessly.
