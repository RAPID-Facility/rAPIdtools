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

   import rapidtools as rt

   spokane = rt.BoundingBox(-117.5358, 47.6885, -117.4426, 47.7378)
   pipeline = rt.Pipeline([
       rt.MapillaryFeatureExtractor(
           classes=['vehicles', 'buildings', 'utility poles'],
           region=spokane,
           start_date='2025-08-01',
           frame_spacing_m=3,        # one frame every 3 m is enough
           min_observations=3,       # seen in at least three frames
       ),
       rt.MapillaryObjectImageExtractor(
           'output/crops',
           max_images_per_asset=2, overlay_asset_outline=True, outline_shape='corners',
       ),
   ])
   objects = pipeline.run()                   # a detection pipeline needs no input
   objects[0].attributes['localization']      # 'triangulated', 'point_cloud', 'map_feature', ...

Neither step is given a token here: Mapillary components read the
``MAPILLARY_ACCESS_TOKEN`` environment variable when ``access_token`` is left
out. Passing the token, or the path of a text file holding it, works too.
The extractor's signature lists the arguments a first run needs; the options
that tune one method (fusion, structures, SAM 3, re-identification, the
earlier tracker) are passed as keywords in the same way and are documented,
by group, on its API page. A misspelt option raises an error that names the
closest match.

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
3. **Objects are placed by occupancy fusion.** Every outline is read as a
   small range scan, the way a robot reads a lidar sweep. Each image column
   the outline covers is a bearing from a known camera pose (Mapillary's
   ``computed_rotation``, camera type and lens parameters, so pitch and roll
   are accounted for). Where the outline's lower edge lies below the
   horizon, the column also gives the range at which the object meets the
   ground. Along that column the ground is *free* up to the object,
   *occupied* for a stretch behind the contact and *unknown* beyond it;
   columns without any outline are free as far as the frame is trusted to
   see (``fusion_range_m``, 30 m). All frames of a survey day vote on one
   ground grid, close frames counting most, and an object is a connected
   blob of cells that the votes mostly call occupied
   (``occupancy_threshold``). Nothing is tracked and no two sightings are
   ever compared, which is what makes the method the same for a hydrant and
   a bus:

   - An object seen from two streets, or on the way in and on the way out,
     lands on the same cells and is one object.
   - A moving vehicle never collects agreeing votes, and the ground the
     survey vehicle drove over is free by construction.
   - A far sighting has a long, uncertain range band and a low weight. It
     cannot create an object by itself and cannot move one placed from
     close up.
   - Frames in which the detector did not fire clear the ground they look
     at, so a few stray detections of an empty kerb do not survive the
     pass that drives right by it.

   Where frames keep seeing several outlines on one blob (two cars parked
   door to door), the blob is split along the outlines of the closest frame
   that separates them. Each object is then positioned by intersecting the
   centre bearings of its sightings, and its footprint is the part of the
   blob most of its outlines cover. Each survey day is fused by itself, so a
   car that left between two surveys is not voted away by the later one;
   objects of different days on the same spot are joined afterwards and
   list both days in ``survey_days``. Objects carry ``localization``
   (``'triangulated'``, or ``'fused'`` when the position is the centre of
   the cells), ``position_sigma_m``, ``parallax_deg``, ``n_images`` and the
   footprint's ``length_m``, ``width_m`` and ``footprint_area_m2``. The
   footprint of a small object is generous in depth, because a pass along
   one side cannot see how deep a car is; ``geometry='footprint'`` writes it
   as the asset's polygon instead of a point.
4. **Buildings come from the survey's own 3-D reconstruction.** A house is
   too large for the logic above: no frame shows all of it, Mapillary
   segments buildings as one surface rather than one outline per house, and
   only the front and part of the sides are ever seen from the street. What
   the survey does have is the sparse point cloud Mapillary aligned every
   panorama in (the image field ``sfm_cluster``), whose points lie on
   exactly the surfaces the survey saw. For classes under
   ``construction--structure--`` (``'buildings'``, ``'bridges'``) the
   extractor therefore:

   - *votes*: projects every reconstructed point into each frame within
     ``structure_range_m`` (40 m) and keeps the points most frames place
     inside an outline of the class;
   - *groups*: bins the points higher than 2 m above the road (fences,
     hedges and parked cars are lower) on the ground and joins dense cells
     closer together than ``min_gap_m`` into one structure, so houses
     separated by a side yard stay apart;
   - *outlines*: fits a rectangle along the facade, in the direction the
     wall points line up with, starting at the wall nearest the street and
     reaching back as far as roof and side-wall points were seen;
   - *writes sightings*: projects each structure's points back into the
     frames around it, which gives every building its own outline in every
     image that shows it. The per-image segmentation never had instance
     outlines for buildings; these are what the cropper and the models use.

   The front and the visible sides of a footprint are measured. The rear is
   not: where the survey saw less than ``assumed_depth_m`` (10 m) of a
   building, the footprint is closed at that depth and ``depth_source``
   reads ``'assumed'`` instead of ``'observed'``.
   ``structure_footprint='observed'`` draws only the convex outline of the
   points seen and assumes nothing. Each building also carries
   ``frontage_m``, ``depth_m``, ``observed_depth_m``, ``height_m`` (the
   highest points above the road), ``facade_bearing`` and ``n_points``, and
   ``structure_views`` sightings spread over the directions it was seen
   from. Point clouds are a few megabytes per hundred metres of driving and
   are cached under ``point_cloud_dir``. Frames Mapillary has not
   reconstructed have no point cloud; a class with none falls back to
   occupancy fusion with a warning.

``localization_method`` selects the method by hand: ``'auto'`` (the default)
is the split described above, ``'fusion'`` and ``'points'`` force one of the
two for every class, and ``'tracks'``, ``'voting'`` and ``'cluster'`` are the
earlier methods described at the end of this section.

What to expect, measured on the RAPID survey of Spokane (August and
September 2026, one panorama every 3 m):

- *Vehicles.* On a 500 m block the track-based method reported 240 objects
  and occupancy fusion 178. Fusion found 134 of the 146 objects the tracker
  had triangulated well. Of the tracker's 51 low-parallax objects fusion
  kept 12; a random sample of the others, checked against the panoramas,
  were empty kerb, lawn and driveway, or vehicles that drove past. Where
  the two methods disagreed by a few metres the tracker had usually left a
  second object between two real vehicles.
- *Buildings.* On a 1 km neighbourhood, 336 structures against 346
  OpenStreetMap footprints within 35 m of the route. 78 % of those
  footprints are overlapped by a structure, with a median intersection
  over union of 0.38 (the rear of most houses is assumed, and garages and
  porches differ between the two sources). About one house in six is
  reported as two structures, usually where a tree hides the middle of the
  facade, and 12 structures span two neighbouring houses. Treat the
  polygons as located facades with a plausible depth, not as surveyed
  footprints; when an aerial footprint layer exists, use it for the shape
  and the street-level structures for what each building looks like.

Further steps sit on top of this:

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
  crops to any model from :func:`rapidtools.models.load` and asks
  whether it is what it claims to be. The question is strict: a motor
  vehicle is not a trailer, boat, jet ski or lawn mower, and a wheel or a
  bumper on its own does not count. The model also reports how much of the
  object is visible, and objects below ``min_visible_fraction`` (half by
  default: hidden behind a fence, cut off, too distant to tell) are
  rejected whatever else it said. It writes ``verify_accepted``,
  ``verify_confidence``, ``verify_visible_fraction`` and ``verify_label``
  and removes the rejects (or keeps them flagged with
  ``keep_rejected=True``; ``reject_unverified=True`` also removes what the
  model gave no usable verdict for). A custom ``classifier`` callable can
  stand in for the model. Give it the widest views rather than the closest
  (``view_selection='widest'`` on the cropper): a close view can be an
  end-on sliver or a bumper, and on a labelled block the widest views let
  the model reject every piece of junk while keeping every vehicle, where
  the closest views let a bush through as a car.
- **A model settles the pairs geometry cannot.** One vehicle reported twice,
  a long vehicle seen as a cab and a box, a track split by a corner, two
  passes that geometry could not join, sits a few metres from its twin
  with good crops each, and no geometric rule separates such a pair from
  two cars parked side by side. :class:`~rapidtools.processing.DuplicateResolver`
  (stage ``VERIFY``, after the verifier) shows the model the crops of
  every pair of objects within ``max_distance_m`` (8 m) as two rows,
  tells it the viewpoints differ, and merges the pairs it calls one
  vehicle with at least ``min_confidence`` (0.9). Each crop is zoomed to
  its outline, drawn thick, so a neighbour in the frame is not what gets
  judged, and the model is made to name each vehicle's type and colour
  before answering. The object seen in the most frames is kept with the
  others' sightings and crops. On a labelled block of 39 vehicles the
  full chain, localisation, verifier and resolver, went from 81 %
  precision and 97 % recall to 95 % and 92 %: every duplicate gone, one
  junk report left, and two real vehicles lost, one to the verifier and
  one to a mistaken merge.
- **Repeated passes are reconciled by appearance.** With ``reid=True`` the
  extractor crops the widest view (from a 2048-pixel thumbnail,
  ``reid_image_size``) of objects from different sequences that lie within
  ``reid_max_distance_m`` of each other, embeds them with the
  image tower of CLIP ViT-B/16 (``reid_model`` accepts DINOv2 checkpoints
  too, with the merge threshold picked per backbone) and merges look-alikes,
  which removes the double counts a second drive down the same street would
  otherwise leave (``reid_merged`` on the merged asset). Appearance only
  ever adds merges: an object whose sightings are all narrower than
  ``reid_min_width_px`` (100 pixels on a 2048-pixel thumbnail) is not
  compared, because crops that small score as alike whether or not they
  show the same vehicle.

Every object is a :class:`~rapidtools.core.PhysicalAsset`, a point for
objects and a footprint polygon for structures (``geometry`` changes either),
whose attributes include the class, the most common label, the number of
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

The track-based method
~~~~~~~~~~~~~~~~~~~~~~

``localization_method='tracks'`` is the method rAPIdtools used before
occupancy fusion. It links detections from frame to frame, triangulates each
track and merges the results with a set of rules tuned on parked vehicles.
It remains available for comparison, and its arguments
(``max_single_view_range_m``, ``witness_radius_m``, ``close_range_m``,
``min_sighting_deg``, ``merge_pieces_of_neighbours``, ``abeam_window_deg``,
``report_single_views``, ``track_gap_frames`` and the merge distances) apply
only to it. After the first two steps above it works as follows:

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

``localization_method='voting'`` swaps the last step for a ray-voting
baseline (rays deposit votes on a ground grid; objects are the peaks), and
``'cluster'`` restores the earliest radius clustering.

Further rules of the track-based method:

- **Pieces of a neighbour are folded in.** Where vehicles line up along
  the line of sight, a driveway seen end-on or cars down the street, the
  tracker can slide from one to the next, and a detector can cut one car in
  two past a pole or into a front and a rear. Each leaves a second object a
  few metres from a well-located one, made of sightings that show the same
  vehicle. Because the two share frames, which the position merge takes as
  proof of two objects, they are judged on what their sightings show: when
  the weaker object's closest sightings point at the stronger one, at its
  range and within a quarter of a car of its centre, and no shared frame
  shows both objects properly at once, the weaker is folded into the
  stronger (``merge_pieces_of_neighbours``, on by default). An object never
  seen within 20 m has no dependable position of its own whatever its
  parallax, since ground-contact ranges read from that far are off by a
  third and a low-parallax intersection lands metres from where the
  sightings point; its closest sightings decide. When their ground points
  land on the close pass it joins it outright, and when only their
  bearings and the range band agree it joins if its triangulation lies
  within 30 % of its range. Its bearings may sit up to half a vehicle from
  the close pass's centre rather than a quarter, since the end of a long
  vehicle cut by a tree is what such a piece usually is, while a neighbour
  parked alongside would have close sightings of its own. The close pass
  keeps the position. What is left of an object never seen within 20 m is
  then dropped when its closest rays point at an object within 15 m that
  was seen up close, unless it is well triangulated regardless, with
  parallax of 45 degrees or more and uncertainty of half a metre or less
  (``close_range_m``, ``0`` to keep everything): such a far sighting is of
  a vehicle the close pass already placed. One whose rays point elsewhere,
  a car set back in a driveway behind the kerb or down a side street,
  stays. The bias is deliberate: the close pass is the dependable one.
- **What no one could identify is not reported.** An object must have been
  seen at least once with an outline at least ``min_sighting_deg`` wide or
  tall (6 degrees: 34 pixels on a 2048-pixel thumbnail, 225 on the
  original). A drive-by survey sees anything worth counting properly in
  some frame, so what was never more than a speck is dropped rather than
  guessed at. The specks still take part in tracking, where they lend
  parallax; culling them earlier would cut tracks at the moment of passing
  and leave short, ill-conditioned pieces. Single views, objects placed
  from ground-contact ranges alone, are not reported either unless
  ``report_single_views=True``: their positions are guesses.
- **An abeam-first mode exists, off by default.** With ``abeam_window_deg``
  set (75 is the sensible value) objects are built only from the sightings
  made while passing them, within that many degrees of abeam on either
  side of the direction of travel; the frames looking down the road ahead
  or behind, whose rays are nearly parallel to it, are attached afterwards
  to the objects they point at but never create one. The idea is sound
  for a drive-by survey, but the track classification was tuned on tracks
  that include the approach and a pass alone trips its moving-vehicle,
  parallax and size tests: on the Spokane survey a 75-degree window lost
  17 of 382 verified vehicles and removed few duplicates in return. It
  stays an option until the classifier is reworked for passes alone.

Bing Streetside
---------------

Bing Streetside cannot be accessed without a Bing Maps key, because its
bubble-lookup service rejects unauthenticated requests. Only Bing aerial
tiles are available keylessly.
