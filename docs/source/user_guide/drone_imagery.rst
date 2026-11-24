Drone imagery
=============

A drone flight photographs every building dozens of times: from above, from
each side, half out of frame, behind a neighbor. An orthomosaic keeps only
the roofs. The perspective photos themselves keep the walls, the windows and
the debris against them, and rAPIdtools turns them into a compact multi-view
record of every asset: the few photos that *together* show the most of it.

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
work on them unchanged; set ``max_images_per_asset`` to the number of views
so the model sees the whole set.

Camera poses
------------

:func:`~rapidtools.data_sources.read_drone_images` reads a folder of photos
and returns one :class:`~rapidtools.data_sources.DroneCamera` per photo.
For DJI aircraft everything comes from the photo itself: position from the
GPS tags, direction from the gimbal yaw, pitch and roll in the ``drone-dji``
XMP block, the lens from the calibrated focal length or the 35 mm
equivalent. Photos without gimbal angles are taken as straight-down shots
along the flight yaw.

Two things are worth setting deliberately:

- **Altitude.** Consumer drones measure the height above the take-off point
  well and the absolute altitude badly, so by default the camera height is
  the recorded relative altitude plus ``takeoff_elevation_m``. Leave that at
  ``0`` on flat ground (the take-off point is then the ground level
  everywhere), or give the elevation of the take-off point when the assets
  carry ground elevations (below).
- **Other aircraft.** Pass ``sidecar='poses.csv'`` (or a GeoJSON of points)
  with ``image``, ``lon``, ``lat``, ``altitude``, ``yaw``, ``pitch`` and
  optionally ``roll`` and ``focal_px`` or ``hfov_deg`` columns; the same
  route takes poses refined by photogrammetry software.

A :class:`~rapidtools.data_sources.DroneCamera` is a pinhole camera:
``project_lonlat`` maps world points to pixels, ``pixel_to_ground`` maps a
pixel back to the ground, and ``ground_footprint`` outlines what the photo
covers.

How the views are chosen
------------------------

Each footprint is extruded to a box and its surface is sampled into patches:
rows of wall patches along every edge, and a grid of roof patches. For every
photo, each patch gets a quality between 0 and 1: zero when it is outside the
frame, faces away from the camera, or is hidden behind the building itself or
a neighbor (rays are cast against the extruded footprints around it);
otherwise how squarely and how sharply the photo shows it.

The view set is the one that maximizes the sum, over all patches, of the best
quality any chosen photo gives each patch. The first photo picked is the one
that shows the most on its own, usually an oblique over a corner with the
roof and two walls. Every further photo is the one that adds the most to
what is already covered, so a second frame from nearly the same spot gains
nothing and is never chosen, while the photo from the far side is. Selection
stops at ``max_views`` or when the best remaining photo would add less than
``min_gain`` (3%) of the surface. This greedy rule is provably within 63%
of the best possible set and in practice within a few percent.

The knobs that matter:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Effect
   * - ``max_views``, ``min_gain``
     - Size of the view set.
   * - ``roof_weight``
     - Importance of roof area against wall area: ``0`` selects for
       facades only, values above ``1`` favor views of the roof.
   * - ``default_height_m``, ``height_attribute``
     - Height of the boxes. A ``height`` or story-count attribute on the
       asset is used when present.
   * - ``ground_elevation``
     - Ground level under the assets: a number, the name of an asset
       attribute, or a function ``(lon, lat) -> elevation`` such as a DEM
       lookup. On sloping terrain this matters: ten meters of height error
       moves a building ten meters in a 45-degree view.
   * - ``target_px_per_m``
     - Resolution at which a surface counts as fully resolved.
   * - ``obstacles``
     - Footprints of things that block the view without being analyzed,
       such as large trees or buildings outside the collection.
   * - ``position_error_m``, ``angle_error_deg``
     - Expected pose error. Crops grow by the shift such an error causes,
       so a pose a few meters off still frames the asset.
   * - ``max_crop_px``
     - Scale large crops down before saving. A model spends time and
       memory per pixel; 640 is plenty for one building.

What the flight missed
----------------------

Coverage is reported per asset (``drone_view_coverage``,
``drone_coverage_roof``, ``drone_coverage_north`` and so on, and
``drone_unseen_sides``) and, for planning, as geometry:

.. code-block:: python

   gaps = extractor.coverage_gaps()
   gaps.to_geojson('coverage_gaps.geojson')

Each feature is a wall no chosen photo shows, with the direction it faces
and a suggested camera position (``fly_to_lon``, ``fly_to_lat``,
``fly_height_m``, ``fly_yaw_deg``, ``fly_pitch_deg``) from which it would be
seen squarely. Loaded in a GIS or a flight planner, the file shows where the
survey has holes while the crew is still on site.

To use the selection without cropping, for example to audit a flight plan,
call :class:`~rapidtools.processing.DroneViewSelector` directly:

.. code-block:: python

   selector = rt.DroneViewSelector(cameras, max_views=4)
   view_sets = selector.select(buildings)
   view_sets['bldg_17'].coverage_by_side
   # {'roof': 0.97, 'north': 0.0, 'east': 0.81, 'south': 0.9, 'west': 0.74}

Imperfect poses
---------------

Consumer GPS is good to a few meters and a compass to a degree or two. In a
45-degree view from 70 m that moves a house by about a third of its own size
in the photo: the choice of photos barely changes, but the crop and the mark
drift off the building. Two things deal with it.

The crop margin grows with ``position_error_m`` and ``angle_error_deg`` (2 m
and 1 degree by default), so the building stays inside the crop. And with
``refine_poses=True`` the extractor asks SAM 3 where the buildings are in
each photo it is about to crop, and slides that photo's projected footprints
to where they overlap the segmented buildings best. Photos where the match
is weak (few standing structures in view) or the gain is small keep the
recorded pose; the shift applied is stored on each crop as
``pose_shift_px``. On the simulated Altadena flight with 3 m and 2 degrees
of noise added to the poses, the default margin kept the whole building in
88% of the crops (98% with the margin set to the true error), and
refinement cut the median offset of the mark from 28% to 15% of the
building's size. With exact poses it leaves nearly every photo alone, so it
is safe to keep on, but it is not needed with RTK positioning.

.. code-block:: python

   extractor = rt.DroneImageExtractor(
       cameras, 'output/drone_crops',
       position_error_m=3.0, angle_error_deg=2.0,
       refine_poses=True,                # needs a GPU; about a second per photo
   )

Limits
------

- Buildings are boxes with flat roofs. Pitched roofs, overhangs and
  courtyards are not modeled, so qualities are estimates, not measurements.
- Trees hide walls and are only accounted for when passed as ``obstacles``.
- Poses are taken as recorded. The crop margin absorbs a few meters and a
  degree or two; a compass that is tens of degrees off (it happens near
  steel structures) needs ``yaw_offset_deg`` or refined poses through the
  sidecar route.

A complete run, from simulated flight to model answers, is in
:doc:`../examples/building_inference_from_drone`.
