Drone imagery
=============

A drone flight photographs every building dozens of times: from above, from
each side, half out of frame, partly behind a neighbor. An orthomosaic keeps
only the roofs. The original perspective photos also show the walls, the
windows and the debris piled against them. rAPIdtools picks, for each
building, the few photos that together show the most of it, and attaches
them as a small multi-view record.

.. code-block:: python

   import rapidtools as rt

   buildings = rt.PhysicalAssetCollection.from_geojson('footprints.geojson')
   cameras = rt.read_drone_images('flight_01/')        # poses from EXIF/XMP

   extractor = rt.DroneImageExtractor(
       cameras, save_directory='output/drone_crops', max_views=4
   )
   buildings = extractor(buildings)

   house = buildings['bldg_17']
   house.attributes['drone_view_coverage']       # 0.84: share of the surface shown
   house.attributes['drone_unseen_sides']        # ['north']
   [img.properties['shows'] for img in house.image_assets]
   # [['roof', 'east', 'south'], ['west'], ['north']]

The crops are ordinary image assets, so
:class:`~rapidtools.processing.AssetAnalyzer` and the other analysis steps
use them as they are. Set the analyzer's ``max_images_per_asset`` to the
number of views so that the model sees the whole set at once.

Camera poses
------------

:func:`~rapidtools.data_sources.read_drone_images` reads a folder of photos
and returns one :class:`~rapidtools.data_sources.DroneCamera` per photo. For
DJI aircraft everything comes from the photo itself. The position comes from
the GPS tags; the viewing direction from the gimbal yaw, pitch and roll in
the ``drone-dji`` XMP block; the lens from the calibrated focal length or
the 35 mm equivalent. A photo without gimbal angles is treated as a
straight-down shot along the flight direction.

Two settings deserve a moment's thought.

Altitude
   Consumer drones measure height above the take-off point well and
   absolute altitude badly. The camera height is therefore the recorded
   relative altitude plus ``takeoff_elevation_m``. On flat ground, leave
   that at ``0``, which makes the take-off point the ground level
   everywhere. If your assets carry ground elevations (see below), give the
   elevation of the take-off point in the same datum.

Other aircraft
   Pass ``sidecar='poses.csv'``, or a GeoJSON of points, with the columns
   ``image``, ``lon``, ``lat``, ``altitude``, ``yaw`` and ``pitch``, and
   optionally ``roll`` and ``focal_px`` or ``hfov_deg``. Poses refined in
   photogrammetry software come in the same way.

A :class:`~rapidtools.data_sources.DroneCamera` is a pinhole camera.
``project_lonlat`` maps a point in the world to a pixel, ``pixel_to_ground``
maps a pixel back to the ground, and ``ground_footprint`` outlines the area
the photo covers.

How the views are chosen
------------------------

Each footprint is raised into a box, and the box's surface is divided into
patches: rows of wall patches along every edge and a grid of roof patches.
For every photo, each patch gets a score between 0 and 1. The score is zero
if the patch is outside the frame, faces away from the camera, or is hidden
by the building itself or a neighbor (rays are cast against the boxes
around it). Otherwise the score says how squarely and how sharply the photo
shows the patch.

The extractor then looks for the set of photos that maximizes the total,
over all patches, of the best score any chosen photo gives each patch. It
builds the set one photo at a time. The first pick is the photo that shows
the most on its own, usually an oblique shot over a corner with the roof and
two walls in view. Each following pick is the photo that adds the most to
what is already covered. A second frame from nearly the same spot adds
almost nothing and is never chosen, while a photo from the far side is.
Selection stops at ``max_views``, or earlier when the best remaining photo
would add less than 3% of the surface (``min_gain``). This greedy method is
guaranteed to reach at least 63% of the best possible score, and in practice
it comes within a few percent.

These are the arguments you are most likely to change:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Effect
   * - ``max_views``, ``min_gain``
     - How many photos each building gets.
   * - ``roof_weight``
     - How much roof area counts against wall area. ``0`` selects for
       facades only; values above ``1`` favor views of the roof.
   * - ``default_height_m``, ``height_attribute``
     - The height of the boxes. A ``height`` or story-count attribute on the
       asset is used when there is one.
   * - ``ground_elevation``
     - The ground level under the assets: a number, the name of an asset
       attribute, or a function ``(lon, lat) -> elevation`` such as a DEM
       lookup. This matters on sloping ground, since ten meters of height
       error moves a building ten meters in a 45-degree view.
   * - ``target_px_per_m``
     - The resolution at which a surface counts as fully resolved.
   * - ``obstacles``
     - Footprints of things that block the view but are not being
       analyzed, such as large trees or buildings outside the collection.
   * - ``position_error_m``, ``angle_error_deg``
     - The expected pose error. Crops grow by the shift such an error
       causes, so a pose that is a few meters off still frames the asset.
   * - ``max_crop_px``
     - Scales large crops down before saving. Models spend time and memory
       on every pixel, and 640 is plenty for one building.

Finding what the flight missed
------------------------------

Coverage is reported for each asset (``drone_view_coverage``,
``drone_coverage_roof``, ``drone_coverage_north`` and so on, plus
``drone_unseen_sides``). For planning a follow-up flight, it is also
available as geometry:

.. code-block:: python

   gaps = extractor.coverage_gaps()
   gaps.to_geojson('coverage_gaps.geojson')

Each feature is a wall that none of the chosen photos shows. It records the
direction the wall faces and a camera position from which it would be seen
squarely (``fly_to_lon``, ``fly_to_lat``, ``fly_height_m``,
``fly_yaw_deg``, ``fly_pitch_deg``). Load the file in a GIS or a flight
planner and you can see where the survey has holes while the crew is still
on site.

To run the selection without cropping anything, for example to check a
flight plan, use :class:`~rapidtools.processing.DroneViewSelector`
directly:

.. code-block:: python

   selector = rt.DroneViewSelector(cameras, max_views=4)
   view_sets = selector.select(buildings)
   view_sets['bldg_17'].coverage_by_side
   # {'roof': 0.97, 'north': 0.0, 'east': 0.81, 'south': 0.9, 'west': 0.74}

When the poses are off
----------------------

Consumer GPS is good to a few meters and a compass to a degree or two. In a
45-degree view from 70 m, that moves a house by about a third of its own
size in the photo. The choice of photos barely changes, but the crop and the
outline drawn on it drift off the building.

The extractor handles this in two ways. First, the crop margin grows with
``position_error_m`` and ``angle_error_deg`` (2 m and 1 degree by default),
which keeps the building inside the crop. Second, with
``refine_poses=True`` it asks SAM 3 where the buildings are in each photo it
is about to crop, and shifts that photo's projected footprints until they
line up best with the segmented buildings. A photo keeps its recorded pose
when the match is weak (few standing structures in view) or the shift would
gain little. The shift applied to each crop is stored as ``pose_shift_px``.

On a simulated flight over Altadena with 3 m and 2 degrees of noise added to
the poses, the default margin kept the whole building in 88% of the crops
(98% with the margin set to the true error). Refinement cut the median
offset of the outline from 28% to 15% of the building's size. With exact
poses it leaves nearly every photo alone, so it is safe to leave on, but you
do not need it with RTK positioning.

.. code-block:: python

   extractor = rt.DroneImageExtractor(
       cameras, 'output/drone_crops',
       position_error_m=3.0, angle_error_deg=2.0,
       refine_poses=True,                # needs a GPU; about a second per photo
   )

Limitations
-----------

- Buildings are modeled as boxes with flat roofs. Pitched roofs, overhangs
  and courtyards are ignored, so the scores are estimates rather than
  measurements.
- Trees hide walls, but they are only taken into account if you pass them
  as ``obstacles``.
- Poses are used as recorded. The crop margin absorbs a few meters and a
  degree or two. A compass that is tens of degrees off, which happens near
  steel structures, needs ``yaw_offset_deg`` or refined poses passed in
  through a sidecar file.

:doc:`../examples/building_inference_from_drone` shows a complete run, from
a simulated flight to the model's answers.
