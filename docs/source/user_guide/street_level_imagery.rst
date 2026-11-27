Street-level imagery
====================

From the street you can see facades, debris and recovery work that never
show up from above. rAPIdtools fetches street-level views from Google Street
View, which needs no key, or from Mapillary, which needs a free token. It can
crop views of assets you already have on a map, or find the assets in a
Mapillary survey and build the inventory itself.

Google Street View
------------------

:class:`~rapidtools.processing.GoogleStreetViewImageExtractor` finds the
nearest official panorama within ``search_radius_m`` of each asset, downloads
its tiles and crops the slice of the view that covers the footprint. It
talks to the same public endpoints as the Google Maps web app, so you need
no key and no billing account.

.. code-block:: python

   from rapidtools import GoogleStreetViewImageExtractor, PhysicalAssetCollection

   buildings = PhysicalAssetCollection.from_geojson('buildings.geojson')
   street = GoogleStreetViewImageExtractor(
       'output/streetview',
       search_radius_m=50,
       max_images_per_asset=2,     # extra viewpoints from neighboring panoramas
       vertical_crop=(0.2, 0.9),   # trim sky and the camera vehicle
       save_depth_map=True,        # also write the decoded depth map (.npy)
   )
   buildings = street(buildings)
   buildings['bldg_01'].image_assets[0].properties['pano_id']

``fov_buffer_deg`` widens the crop on both sides of the footprint, and
``min_fov_deg`` keeps small or distant assets from turning into slivers.
Several assets are processed at once (``max_workers``).

Mapillary views of known assets
-------------------------------

:class:`~rapidtools.processing.MapillaryImageExtractor` downloads the image
metadata for the whole region once and indexes it. For each asset it then
casts rays from the candidate cameras toward the asset and throws out the
views that a neighboring building blocks. Finally it downloads one panorama
per viewing direction: the four main axes of the footprint, plus the corners
if ``cast_corner_rays`` is on.

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

The date window is what makes before-and-after studies easy. Run the
extractor once per survey, each into its own folder, then point the analyzer
at one folder at a time with an ``image_filter``
(see :doc:`../examples/street_level_recovery`).

Checking image contents before downloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mapillary publishes a semantic segmentation of every image it hosts. With
``strict_content_filter=True`` the extractor reads it first and skips an
image if the asset type is not visible in it, which saves downloading the
full JPEG. The asset type has to be written in Mapillary's label vocabulary.
You can provide the translation yourself with ``asset_type_mapping``, or let
a language model work it out from plain English with
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

Finding objects along a survey
------------------------------

The extractors above need an inventory to start from. Sometimes the question
is the inventory itself: where are all the vehicles, poles or hydrants along
this route? :class:`~rapidtools.processing.MapillaryFeatureExtractor`
answers it from the survey imagery, much as
:class:`~rapidtools.processing.SAM3OrthoFeatureExtractor` does for a raster.

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

Neither step gets a token here. When ``access_token`` is left out, the
Mapillary components read the ``MAPILLARY_ACCESS_TOKEN`` environment
variable. You can also pass the token itself, or the path of a text file
that holds it.

The extractor's signature lists only what a first run needs. Options that
tune a particular method (fusion, buildings, SAM 3, re-identification, the
earlier tracker) are passed as extra keywords, and its API page documents
them by group. If you misspell one, the error names the closest valid
option.

The rest of this section follows the detections through the extractor: how
they are read, how objects and buildings are placed, how accurate the
result is, and how to refine it.

Reading the detections
~~~~~~~~~~~~~~~~~~~~~~

The extractor does not run a detector of its own by default. Mapillary
already segments every image it hosts, and the extractor reads those
polygons for the requested classes through the Graph API. Finding objects
across a whole city therefore costs a few kilobytes per image and downloads
no pixels.

Class names can be plain English, such as ``'cars'`` or
``'utility poles'``; they are translated to Mapillary labels, and you can
pass a :class:`~rapidtools.processing.MapillaryLabelMapper` for names the
built-in aliases miss. ``'vehicles'`` means motor vehicles: cars, trucks,
buses and motorcycles. Trailers, caravans, boats and Mapillary's catch-all
"other vehicle" are separate classes, which you ask for as ``'trailers'``,
``'boats'`` or ``'all vehicles'``.

The first thing the extractor does with the detections is remove the survey
vehicle. The camera car sits at the same place in every frame of a
sequence, while a parked car moves across the frame as the camera passes.
Detections that recur at the same spot in most frames are dropped.

Placing objects: occupancy fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To put objects on the map, the extractor treats every outline as a small
range scan, the way a robot reads a lidar sweep. Each image column the
outline covers is a bearing from a known camera pose. The pose comes from
Mapillary's ``computed_rotation``, camera type and lens parameters, so pitch
and roll are accounted for. Where the bottom edge of the outline lies below
the horizon, the column also gives the distance at which the object meets
the ground. Along that column the ground is *free* up to the object,
*occupied* for a stretch behind the point of contact, and *unknown* beyond.
Columns with no outline count as free out to the distance the frame is
trusted to see (``fusion_range_m``, 30 m by default).

All frames from one survey day vote on a shared ground grid, with nearby
frames counting the most. An object is a connected patch of cells that the
votes mostly call occupied (``occupancy_threshold``). Nothing is tracked
from frame to frame and no two sightings are ever compared, which is why the
same method works for a hydrant and for a bus. It also takes care of the
usual problems without special rules:

- An object seen from two streets, or on the way in and on the way out,
  lands on the same cells and becomes one object.
- A moving vehicle never collects agreeing votes, and the road the survey
  vehicle drove along is free by construction.
- A distant sighting has a long, uncertain range and a low weight. It cannot
  create an object by itself, and it cannot move one that close frames have
  placed.
- Frames in which the detector did not fire clear the ground they look at.
  A few stray detections of an empty curb do not survive a pass that drives
  right by it.

When frames keep seeing several outlines on one patch, such as two cars
parked bumper to bumper, the patch is split along the outlines from the
closest frame that separates them. Each object is then positioned by
intersecting the center bearings of its sightings, and its footprint is the
part of the patch most of its outlines cover.

Each survey day is fused on its own, so a car that left between two surveys
is not voted away by the later one. Objects from different days on the same
spot are joined afterward and list both days in ``survey_days``.

Every object records how it was placed (``localization``: ``'triangulated'``,
or ``'fused'`` when the position is the center of its cells) and how well:
``position_sigma_m``, ``parallax_deg`` and ``n_images``. Its footprint is
described by ``length_m``, ``width_m`` and ``footprint_area_m2``. For small
objects the footprint is generous in depth, because a pass along one side
cannot see how deep a car is. ``geometry='footprint'`` stores it as the
asset's polygon instead of a point.

Placing buildings
~~~~~~~~~~~~~~~~~

Buildings need a different approach. No single frame shows all of a house,
only its front and part of its sides are ever seen from the street, and
Mapillary segments buildings as one continuous surface rather than one
outline per house. The extractor has three ways around this:

.. list-table::
   :header-rows: 1
   :widths: 26 18 56

   * - Method
     - ``localization_method``
     - What it needs
   * - From the point cloud (the default)
     - ``'points'``
     - Mapillary's 3-D reconstruction of the survey. No image downloads.
   * - One mask per house
     - ``'instances'``
     - The point cloud, plus SAM 3 on a thumbnail of every frame (a GPU).
   * - Matched to known footprints
     - ``'footprints'``
     - A footprint layer. SAM 3 is optional.

If you have a footprint layer, use it: matching is the most reliable of the
three by a wide margin. Without one, the SAM 3 masks keep neighbors apart
better than Mapillary's outlines if you have the GPU time, and the point
cloud method is the free fallback. The numbers behind this advice are in
`How accurate is it?`_.

From the point cloud
^^^^^^^^^^^^^^^^^^^^

Mapillary aligns every panorama of a survey in a sparse 3-D point cloud
(the image field ``sfm_cluster``), and the points lie on exactly the
surfaces the survey saw. For classes under ``construction--structure--``
(``'buildings'``, ``'bridges'``) the extractor works from those points:

1. **Vote.** It projects every point into each frame within
   ``structure_range_m`` (40 m) and keeps the points that most frames place
   inside an outline of the class. A frame only votes against a point it can
   actually see, so a street tree between the camera and a wall does not
   count against the wall.
2. **Group.** It keeps the points more than 2 m above the ground, which
   leaves out hedges and parked cars. The ground level is read from the
   cloud itself, because a lot can sit above or below the street. Dense
   ground cells closer together than ``min_gap_m`` form wall pieces. One
   house often gives several pieces, since the garage and the entrance stand
   at different setbacks and a garage door yields few points. A piece joins
   the larger piece it continues along the street when every frame in front
   of the two outlines them as one building. Two neighbors stay apart,
   because a frame facing the side yard between them outlines each one
   separately.
3. **Outline.** It draws the footprint along the facade, aligned with the
   longest wall, with a step wherever the wall changes setback, and closes
   it with a single straight line at the rear.
4. **Write sightings.** It projects each building's points back into the
   frames around it. That gives every building its own outline in every
   image that shows it, which Mapillary's segmentation never had. The
   cropper and the models use these outlines.

The front and the visible sides of a footprint are measured; the rear is
not. Where the survey saw less than ``assumed_depth_m`` (10 m) behind the
wall set farthest back, the footprint is closed at that depth and
``depth_source`` reads ``'assumed'`` instead of ``'observed'``. If you
prefer simpler shapes, ``structure_footprint='rectangle'`` draws one
rectangle around the whole facade, and ``'observed'`` draws only the convex
outline of the points that were seen and assumes nothing.

Each building also carries ``frontage_m``, ``depth_m``,
``observed_depth_m``, ``height_m`` (of the highest points, above the
ground), ``facade_bearing`` and ``n_points``. Its ``structure_views``
sightings are spread over the directions it was seen from.

Point clouds take a few megabytes per hundred meters of driving and are
cached under ``point_cloud_dir``. Frames that Mapillary has not
reconstructed have no point cloud; if a class has none at all, the extractor
falls back to occupancy fusion and logs a warning.

One mask per house
^^^^^^^^^^^^^^^^^^

Because Mapillary's building outline is one surface for a whole street
front, a fence or a shed between two houses can join them, and the point
cloud method then reports the two as one structure. A segmentation model
asked for houses draws each house on its own and leaves the fence out:

.. code-block:: python

   detector = rt.MapillaryFeatureExtractor(
       classes=['buildings', 'vehicles'],
       detection_source={'buildings': 'sam3'},   # vehicles stay on Mapillary
       sam3_prompts={'buildings': 'house'},      # SAM 3 wants a singular noun
       ...
   )

The model cannot tell which mask in the next frame is the same house, but
the point cloud can. Every mask collects the points its frame sees inside
it, and two masks that hold mostly the same points show the same building.
Links are made strongest first, and a frame shows each building only once,
so one mask that happens to cover two distant houses cannot tie their tracks
together.

A model often cuts a large house into wings, for example a gabled garage and
the rest. Two tracks are joined when their walls continue each other and
their masks touch in the frames in front of them; the masks of two
neighbors leave a gap there. With ``join_wings_by='prompt'`` the extractor
asks the model instead. Given a few points on one wing and asked for the
whole object, SAM 3 returns a mask that covers the other wing but not a
neighbor. On the Spokane test pairs this wrongly joined 2 of 84 neighbors,
where the touching rule joined 13. It costs a few extra model calls per
pair, about 1,700 on the test neighborhood.

Footprints, attributes and sightings are the same as with the point cloud
method, except that a building's sightings are its own masks. This method
is chosen automatically for buildings detected by a model.

It is not cheap. The extractor downloads a 1,024-pixel thumbnail of every
frame and runs the model on each, at about 0.4 s per frame on one GPU. It
also needs the frames dense: with every second frame left out, it merges
neighbors as often as the point cloud method does. The linking itself, in
:mod:`~rapidtools.processing.street_instances`, runs no model and accepts
outlines from any source. ``sam3_model`` takes the local SAM 3 or any other
model with the same ``run_inference``.

Matched to known footprints
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a footprint layer exists, the buildings do not have to be found,
located or told apart. The only question left is which outline in which
image shows which footprint:

.. code-block:: python

   footprints = rt.PhysicalAssetCollection.from_geojson('footprints.geojson')
   detector = rt.MapillaryFeatureExtractor(
       classes=['buildings'],
       footprints=footprints,                    # or the file's path
       detection_source={'buildings': 'sam3'},   # optional: better recall
       sam3_prompts={'buildings': 'house'},
       ...
   )

Each footprint is raised into a tall box and projected into every frame
within ``structure_range_m``. Nearer boxes hide farther ones, and what
remains of each box is where its building should appear; an outline found
there is a sighting of it. The wings of one house fall on the same
footprint and two neighbors fall on two, so nothing is split or merged.

The buildings come back as the footprints themselves, with their own IDs
and attributes, plus ``frames_expected`` (the frames the building should be
visible in) and ``frames_seen`` (those with an outline on it). A footprint
expected in many frames and seen in none is hidden, or no longer standing.

This method needs no point cloud, only the positions and rotations of the
images, and it works with either kind of outline. With Mapillary's building
surface it downloads no images at all, and the surface counts as a sighting
when it fills a fifth of the footprint's box. With one mask per house from a
model it sees a few more of the houses that trees partly hide. It is chosen
automatically for buildings whenever you pass ``footprints``. Without
footprints, the point cloud and the masks are what is left, and they remain
the default.

Buildings the layer does not have are still reported, marked ``unmapped``.
With a model's masks, the masks that fall on no footprint are placed by the
mask method. With Mapillary's outlines, the point cloud method runs as well,
and the structures it finds that lie off every footprint (less than a tenth
of their area on any one) are kept. Either way the frames need a point
cloud, so with Mapillary's outlines this adds the point cloud downloads and
about a minute per 1,000 frames. ``report_unmapped=False`` matches the layer
only.

Choosing the method by hand
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``localization_method='auto'``, the default, makes the choices described
above. To force one method for every class, pass one of these:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Value
     - Method
   * - ``'auto'``
     - Fusion for objects; for buildings, footprints when ``footprints`` is
       given, masks when a model detects them, the point cloud otherwise.
   * - ``'fusion'``
     - `Placing objects: occupancy fusion`_.
   * - ``'points'``
     - `From the point cloud`_.
   * - ``'instances'``
     - `One mask per house`_.
   * - ``'footprints'``
     - `Matched to known footprints`_.
   * - ``'tracks'``, ``'voting'``, ``'cluster'``
     - The earlier methods, described in `The earlier track-based method`_.

How accurate is it?
~~~~~~~~~~~~~~~~~~~

These figures come from the RAPID survey of Spokane, WA (August and
September 2026, one panorama every 3 m).

**Vehicles.** On a 500 m block the earlier track-based method reported 240
objects and occupancy fusion 178. Fusion found 134 of the 146 objects the
tracker had triangulated well, but kept only 12 of the tracker's 51
low-parallax objects. A random sample of the dropped ones, checked against
the panoramas, turned out to be empty curb, lawn and driveway, or vehicles
driving past. Where the two methods disagreed by a few meters, the tracker
had usually left a second object between two real vehicles.

**Buildings.** The test area is a 1 km neighborhood, checked against a
reference footprint layer. The table counts the 324 houses in the first row
along the street, the ones a street survey should see. Four more first-row
footprints in the layer are lots whose houses burned before the survey;
the notes below say what each method made of them.

.. list-table::
   :header-rows: 1
   :widths: 36 11 11 13 11 18

   * - Method
     - Houses found
     - Split in two
     - Structures over 2+ houses
     - Median IoU
     - Extra cost
   * - Footprints + SAM 3 masks
     - 324
     - 0
     - 0
     - n/a
     - SAM 3 on every frame
   * - Footprints + Mapillary outlines
     - 323
     - 0
     - 0
     - n/a
     - none
   * - SAM 3 masks, wings joined by prompt
     - 322
     - 18
     - 3
     - 0.61
     - SAM 3 on every frame, about 1,700 prompt calls
   * - SAM 3 masks, wings joined by touch
     - 321
     - 17
     - 7
     - 0.61
     - SAM 3 on every frame
   * - Point cloud (default)
     - 316
     - 14
     - 12
     - 0.55
     - none

IoU is the intersection over union of a structure with the reference
footprint. With known footprints the footprint *is* the building, so nothing
can be split or merged and the outline on the map is the layer's own; there,
"found" means seen in three or more frames. Checked against the footprint
that a mask's own reconstructed points stand on, 98% of those assignments
agree.

Some notes on reading the table:

- Matched to footprints, a burned lot can still pass for a house. With
  SAM 3 on every frame, 2 of the 4 lots were seen in three frames and
  reported; on every fifth frame, none. Mapillary's outlines reported 1.
  Every standing house was seen in 16 frames or more, so a low
  ``frames_seen`` against ``frames_expected`` flags such a lot. The touch
  and point cloud methods placed no structure on any of the four. The
  prompt row was counted before the burned lots were found and still
  includes them.
- With the point cloud method the rear of most houses is assumed. A house
  split in two usually has a tree hiding the middle of its facade.
  Structures that span neighbors mostly come from a fence that Mapillary
  segments as building and that bridges two houses.
- The first-row houses no method finds stand far back from the street or
  behind trees, and gave too few points or outlines. Buildings behind the
  first row are not expected to be found.
- The remaining splits with SAM 3 masks are large houses whose wings the
  model outlines separately, with a tree or a vehicle between them in the
  image.
- Running SAM 3 on the neighborhood's 5,251 frames took 33 minutes on one
  GPU, followed by 6 minutes of linking.
- To test the report of unmapped buildings, a tenth of the first-row houses
  (33) were taken out of the layer, three times over. With SAM 3 masks on
  every frame, 98 of the 99 were reported; on every fifth frame, 94. With
  Mapillary's outlines and the point cloud, 90. None of these reports
  repeated a house still in the layer.

Treat discovered building polygons as located facades with a plausible
depth, not as surveyed footprints. When an aerial footprint layer exists,
take the shape from it and use the street survey for what each building
looks like.

Refining the results
~~~~~~~~~~~~~~~~~~~~

The detections are only as good as Mapillary's detector, and geometry alone
cannot settle every case. Static objects can come straight from Mapillary's
own map instead. For the rest, a model can screen out false detections,
settle duplicates and merge objects seen on separate passes.

Map features for static objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Utility poles, fire hydrants, street lights, traffic signs and the other
classes Mapillary publishes as *map features* have already been
triangulated across every contributor's imagery. With the default
``detection_source='auto'`` the extractor simply fetches them with
:meth:`~rapidtools.data_sources.MapillaryClient.fetch_map_features` instead
of placing them itself. They come back with ``localization='map_feature'``
and the attributes ``aligned_direction``, ``first_seen`` and ``last_seen``.
Vehicles are not map features and always go through the images.
``detection_source='mapillary'`` sends every class through the images.

Screening false detections
^^^^^^^^^^^^^^^^^^^^^^^^^^

Mapillary's detector sometimes fires on things that are not the class you
asked for. :class:`~rapidtools.processing.DetectionVerifier` is a pipeline
step (stage ``VERIFY``, after cropping and before analysis) that shows each
object's crops to a model from :func:`rapidtools.models.load` and asks
whether the object is what it claims to be.

The question is strict. A motor vehicle is not a trailer, boat, jet ski or
lawn mower, and a wheel or bumper on its own does not count. The model also
reports how much of the object is visible, and anything below
``min_visible_fraction`` (half, by default) is rejected whatever else the
model said, whether it is hidden behind a fence, cut off at the edge or too
far away to tell.

The verifier writes ``verify_accepted``, ``verify_confidence``,
``verify_visible_fraction`` and ``verify_label``, and removes the rejected
objects. ``keep_rejected=True`` keeps them with the flag instead, and
``reject_unverified=True`` also removes objects the model gave no usable
answer for. A ``classifier`` callable can stand in for the model.

Give the verifier the widest views rather than the closest ones
(``view_selection='widest'`` on the cropper). A close view can be an end-on
sliver or just a bumper. On a labeled test block, the widest views let the
model reject every piece of junk while keeping every vehicle, whereas the
closest views let a bush through as a car.

Resolving duplicates
^^^^^^^^^^^^^^^^^^^^

Some duplicates survive the geometry: one vehicle reported twice, a long
vehicle seen as a cab and a separate box, a track broken at a corner, or two
passes that could not be joined. Each half sits a few meters from its twin
with good crops of its own, and no geometric rule tells such a pair from two
cars parked side by side.

:class:`~rapidtools.processing.DuplicateResolver` (stage ``VERIFY``, after
the verifier) shows a model the crops of every pair of objects within
``max_distance_m`` (8 m) as two rows of images, tells it the viewpoints
differ, and merges the pairs it calls one vehicle with a confidence of at
least ``min_confidence`` (0.9). Each crop is zoomed to its outline, drawn
thick, so the model judges the right vehicle and not a neighbor in the
frame, and the model has to name each vehicle's type and color before it
answers. The object seen in the most frames is kept and takes over the
sightings and crops of the others.

On a labeled block of 39 vehicles, the full chain of localization, verifier
and resolver went from 81% precision and 97% recall to 95% and 92%. Every
duplicate was gone and one junk report was left; two real vehicles were
lost, one to the verifier and one to a mistaken merge.

Re-identifying objects across passes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A second drive down the same street can leave double counts that geometry
did not join. With ``reid=True`` the extractor compares objects from
different sequences that lie within ``reid_max_distance_m`` (6 m) of each
other by appearance. It crops each object's widest view from a 2,048-pixel
thumbnail (``reid_image_size``), embeds the crops with the image encoder of
CLIP ViT-B/16, and merges look-alikes. ``reid_model`` also accepts DINOv2
checkpoints, with the merge threshold chosen per model. Merged objects are
marked with ``reid_merged``.

Appearance can only add merges, never split objects. Objects whose
sightings are all narrower than ``reid_min_width_px`` (100 pixels on a
2,048-pixel thumbnail) are left out of the comparison, because crops that
small look alike whether or not they show the same vehicle.

What you get back
~~~~~~~~~~~~~~~~~

Every object is a :class:`~rapidtools.core.PhysicalAsset`: a point for
objects and a footprint polygon for buildings (``geometry`` changes either).
Its attributes include the class, the most common label, the number of
sightings and images, and the sightings themselves as
:class:`~rapidtools.core.Observation` records. Those records let
:class:`~rapidtools.processing.MapillaryObjectImageExtractor` fetch only the
closest one or two views of each object, at thumbnail resolution, and crop
the detection with a margin. From there,
:class:`~rapidtools.processing.AssetAnalyzer` works exactly as it does with
aerial crops.

Large regions
~~~~~~~~~~~~~

A city-wide survey has tens of thousands of frames, and Mapillary's data
for each frame lists every label in it. To keep memory flat however large
the region, the extractor fetches frames in batches of ``frame_batch_size``
(200), keeps only the requested classes and discards the rest before the
next batch. It also simplifies every outline (``simplify_tolerance``, about
four pixels of a 2,048-pixel-wide image, capped at 1% of the outline's size
so distant objects keep their detail). The cropper likewise downloads each
source image once, cuts every object seen in it and releases it, holding at
most ``max_workers`` images at a time.

Classes Mapillary does not segment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some classes, such as debris piles, are not in Mapillary's vocabulary. With
``detection_source='auto'`` (the default) the extractor sends those classes
to SAM 3, which runs on the same thumbnails, and the SAM 3 outlines go
through the same placement steps as Mapillary's. Every frame then costs a
model call, so set ``frame_spacing_m`` generously in this mode.

The earlier track-based method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before occupancy fusion, rAPIdtools followed each object from frame to
frame, triangulated each track and merged the results using rules tuned on
parked vehicles. ``localization_method='tracks'`` still selects it, for
comparison. Two simpler baselines are also available:
``localization_method='voting'`` lets rays deposit votes on a ground grid
and takes the peaks as objects, and ``'cluster'`` restores the original
radius clustering.

After reading the detections and removing the survey vehicle, the tracker
works like this:

1. Where the detector returned one object twice in a frame, such as an
   outline nested inside another (a wheel, a window, the part visible past a
   tree) or the two halves at a panorama's seam, the smaller piece is
   dropped. Pieces side by side are not joined, because two cars parked
   side by side look exactly like a split outline and are far more common.
2. Each sighting becomes a ray: an accurate bearing from the outline's
   center column and the full camera pose, plus a rough distance from the
   ground contact and the camera height. The distance is only a starting
   guess.
3. Within each sequence, sightings are linked into tracks by how smoothly
   their bearings change, which is where the data is precise. The outline's
   angular width limits how close the object can be, which in turn limits
   how far its bearing can swing between frames, so a speck far down the
   road cannot claim a car that appears beside it.
4. Each track is triangulated from all its rays. A track whose rays do not
   agree, or whose ground-contact distances contradict the intersection,
   was a moving vehicle and is dropped, as is anything on the line the
   camera drove.
5. Tracks are merged with a gate that is narrow across a ray and wide along
   it. One car seen on two passes becomes one object, while two cars seen
   in the same frame stay two, unless they are closer together than a car
   is wide, which only a split outline can produce.
6. Weak triangulations, never seen closer than 25 m or with little parallax,
   are not merged by position, since a tree hiding half a car shifts a
   40 m intersection by meters. Their rays are checked against the
   well-placed objects they point at, and they join the nearest one within
   a fraction of their range. The close pass keeps its position. Likewise,
   when one object is a piece of a neighbor (a car seen end-on down a
   driveway, or cut in two by a pole), it is folded into the neighbor if
   its closest sightings point at it.
7. Frames in which the detector did *not* fire count against weak
   estimates. If frames from the same day passed close by with the spot in
   view and saw nothing of the class along that bearing, the estimate is
   dropped. A car triangulated from a close pass is kept regardless, since
   the next pass often finds it hidden behind a hedge or another car.
8. An object that was never seen larger than a speck is not reported, and
   neither is a single view placed only by its ground-contact distance,
   unless you ask for single views.

Objects carry ``localization='triangulated'``, or ``'single_view'`` when
there was no parallax, plus ``position_sigma_m``, ``parallax_deg`` and
``n_images`` for filtering by quality. These options apply only to the
track-based method:

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Option
     - Default
     - Effect
   * - ``max_range_m``
     - 60 m
     - Positions farther away are not reported.
   * - ``min_object_width_m``
     - 1 m
     - Once a track has a distance, outlines narrower than this are
       dropped as fragments.
   * - ``max_single_view_range_m``
     - 30 m
     - A single-view position whose nearest sighting is farther than this
       is not reported. At that distance the ground-contact range is a
       guess.
   * - ``witness_radius_m``
     - 12 m
     - How close a frame must pass to count against a weak estimate.
   * - ``close_range_m``
     - 20 m
     - An object never seen closer than this is dropped when its rays point
       at an object seen up close, unless it is well triangulated (45
       degrees of parallax or more, half a meter of uncertainty or less).
       ``0`` keeps everything.
   * - ``min_sighting_deg``
     - 6 degrees
     - An object must be seen at least once at this width or height (34
       pixels on a 2,048-pixel thumbnail). Smaller sightings still help
       tracking, where they add parallax.
   * - ``merge_pieces_of_neighbors``
     - ``True``
     - Fold pieces of a neighboring vehicle into it.
   * - ``report_single_views``
     - ``False``
     - Report objects placed from a single view.
   * - ``track_gap_frames``
     - 2
     - Frames a track may miss before it is closed.
   * - ``abeam_window_deg``
     - 0 (off)
     - Build objects only from sightings made while passing them, within
       this many degrees of abeam. Frames looking down the road are
       attached afterward but never create an object.

The abeam mode is sound in principle for a drive-by survey, but the
tracker's tests for moving vehicles, parallax and size were tuned on tracks
that include the approach. On the Spokane survey a 75-degree window lost 17
of 382 verified vehicles and removed few duplicates in return, so it stays
off until those tests are reworked.

Bing Streetside
---------------

Bing Streetside cannot be used without a Bing Maps key, because its lookup
service rejects unauthenticated requests. Only the Bing aerial tiles work
without a key.
