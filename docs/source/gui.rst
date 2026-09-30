Graphical interface
===================

rAPIdtools ships a local web application that runs the detect-and-analyze
workflow without code. It is built on the Python standard library, so it
needs no extra dependencies, and it opens in your browser.

.. code-block:: bash

   rapidtools-gui                   # console script installed with the package
   python -m rapidtools.gui         # equivalent

.. code-block:: python

   from rapidtools.gui import launch_asset_analysis_app
   launch_asset_analysis_app(port=8765, open_browser=True)

The three steps
---------------

1. **Imagery.** Load a GeoTIFF from your computer, stitch a Bing or Google
   satellite basemap over a bounding box or the extent of a GeoJSON file (no
   API key; the tile count is checked before anything downloads), or
   download a UW RAPID sample. The image stays on screen for the rest of the
   session and streams native-resolution pixels as you zoom, so
   multi-gigabyte orthomosaics can be inspected without loading them into
   the browser. Its extent also defines the survey area for street-level
   discovery.
2. **Detect and analyze.** Three cards:

   *Assets.* Detect them in the image with SAM 3 (type ``building, tree,
   swimming pool``); the scan runs on a Bing or Google basemap stitched over
   the image extent, or on the image itself, and *Keep every detected
   instance separate* turns off overlap merging for countable objects such as
   vehicles. Or **discover objects along a street survey**: plain-English
   classes such as ``vehicles, utility poles, fire hydrants`` are located in
   every Mapillary image within the extent from Mapillary's own detections,
   the survey vehicle is removed, and repeated sightings are triangulated into
   one point per object (a Mapillary token is required; *RAPID token* fetches
   the one used by the examples). Or load a GeoJSON with asset locations.

   *Imagery for analysis.* Where the crops the model sees come from: aerial
   crops of the loaded image (with ``min_footprint_coverage`` and edge
   padding), Google Street View panoramas cropped to each footprint (no
   key), Mapillary panoramas of each footprint, or the closest Mapillary
   views of each discovered object. Where an outline can be drawn, choose
   its shape (geometry, bounding box, corner brackets, ...), buffer, width
   and colour.

   *Analysis model and prompt.* Every analysis model in rAPIdtools is
   available: the Gemini, Claude, OpenAI, Muse Spark and Qwen APIs; Gemma 4,
   Llama Vision, Muse Glimmer and Qwen locally; and any Hugging Face
   vision-language checkpoint with optional 4-bit loading. *Fetch models*
   lists the IDs your key can reach. Temperature, output length and JSON
   mode live under *Advanced options*. The prompt can come from a file, from
   the sample library (aerial CHS, street-level CHS, street-level recovery)
   or from the prompt builder described below.
3. **Results.** Hover over an asset to see its attributes, click one to view
   the exact crops the model analyzed, colour the overlay by any inferred
   attribute, browse the attribute table and download the GeoJSON outputs.

Progress and log output stream into the page while the work runs on a
background thread. **Cancel** stops detection or inference at the next tile
or batch and keeps the results produced so far.

The prompt builder
------------------

The sample prompts shipped with rAPIdtools share one layout: a ``TASK``
statement, numbered ``ANALYSIS STEPS``, ``ANALYSIS EDGE CASES``, a
``REQUIRED OUTPUT FORMAT`` whose ``Field: [options]`` lines the analyzer
parses into attributes, and a rubric that describes every class with its
key visual indicators. *Build a prompt* opens a dialog that walks through
those parts:

1. **Task** - the asset, the kind of imagery, what to decide, the expert
   role, what to ignore, and how the asset is marked on the crop (filled in
   from the outline settings).
2. **Output fields** - each field becomes one line of the output format and
   one attribute on every asset (``gemini_chs_level``, ...). Fields are
   categorical, numeric, free text or boolean; a JSON variant of the format
   is one checkbox away.
3. **Class indicators** - a name, a one-sentence meaning and 3-5 visual cues
   for every option of the main field.
4. **Steps and edge cases** - the order of work and what to answer when the
   image fits no class, with common edge cases one click away.
5. **Review** - optional background, the assembled prompt and the attribute
   names it will produce. *Use this prompt* inserts it into the editor.

The assembled prompt is previewed live. *Start from sample* loads any of
the registry prompts (aerial CHS, street-level CHS, street-level recovery)
into the builder as a specification to edit; describing a different task
and pressing *Repurpose for this task* has the assistant rewrite that
specification for it while keeping its structure and output format. The
assistant defaults to a local Gemma 4 model.

The assistant
~~~~~~~~~~~~~

The dialog's assistant uses any rAPIdtools model backend text-only: by
default the analysis model you selected (and its key), or a local Gemma 4
or Qwen checkpoint, which runs on your machine. It can

- **Draft** the whole specification from a one-line description ("grade
  roof damage of houses after a hurricane into four levels"),
- **Refine** the current specification as instructed ("add a confidence
  field from 1 to 3"),
- **Suggest** visual indicators for one class,
- **Review** the assembled prompt as a critical colleague, and
- **Import** a prompt from the editor into the builder's fields.

Requests run as cancellable jobs like everything else in the GUI; a local
model is loaded once and released before detection or inference needs the
GPU. Model replies are parsed leniently, and nothing reaches the prompt
editor until you press *Use this prompt*.

The map: find the place first
-----------------------------

The interface opens on a map, not on an empty image pane, and step 1 asks
how you want to start: **select an area on the map**, **load your own
imagery** (a GeoTIFF on this computer), or **download a UW RAPID sample**.
Only the chosen option's panel is shown.

Satellite tiles (Bing or Google) stream in as you pan and zoom, the way a
desktop GIS draws a basemap, fetched through the server so no key or
cross-origin access is needed. On top of them the cyan lines show where
street-level imagery exists on Mapillary: by default the surveys uploaded
by the UW RAPID Facility, or every Mapillary contributor with *RAPID
surveys only* off. *Heat* switches the routes to a glowing density style.
The layer panel sits on the map's top right and the zoom control on its
bottom right.

- **RAPID routes come from a local database.** On the first start the
  server reads the country-wide zoom-6 coverage tiles to find where RAPID
  surveys exist, then the zoom-13 coverage tiles of those places (173 of
  them for the continental United States, about 15 seconds), keeps only
  the RAPID routes and stores them gzipped as ``survey_routes_z13.json.gz``
  under ``~/.cache/rapidtools``. The map downloads that file once (under a
  megabyte) and draws every route from memory at every zoom, so panning
  and zooming make no coverage requests at all. The file is static until
  *refresh* in the layer panel (or ``POST /api/street/routes/rebuild``)
  rebuilds it, and it can be copied to another machine or replaced by a
  published one. ``GET /api/street/routes`` reports its status and
  ``GET /api/street/routes.json`` serves it.
- With *RAPID surveys only* off, routes of all contributors are read per
  view from the coverage tiles (``GET /api/street/sequences/<z>/<x>/<y>``
  from zoom 6 upwards, ``GET /api/street/overview`` below), which is
  slower and limited to the current view.
- *Search an address or place* uses OpenStreetMap's Nominatim
  (``GET /api/geocode?q=``) and jumps to the match; ``lat, lon`` works too.

Hold Shift and drag (or press *Draw box*) to choose the working area. The
*Selected area* panel shows its size, holds the coordinates (or a GeoJSON
file whose extent is used instead) and offers the two ways to work with
it: **Download satellite imagery here** stitches Bing or Google tiles over
the box into a GeoTIFF, and **Discover street-level objects here** goes
straight to step 2 with the box as the survey area, so no aerial image is
needed. The tile zoom level is chosen automatically from the size of the
box (the finest level, up to 19, that stays under the 2,500-tile limit)
and reported as a resolution in metres per pixel; *Adjust resolution*
overrides it. The box and the coordinate fields stay in sync, so typing
numbers moves the box.

Whatever the assets came from, they are drawn on the map in WGS84
(``GET /api/geo_overlay``) as well as on the raster views; the *Map* /
*Image* switch in the toolbar moves between them, and a checkbox shows the
stitched basemap a detection ran on instead of the loaded image.

Long runs: leave, come back, get notified
-----------------------------------------

Jobs run on the server, not in the browser tab, and the results stay there
until the server stops. You can watch the progress and log as before, close
the tab and reopen the same address later (the page resumes where the
server is), or ask to be told when the job ends. A bar appears under the
header once a download, detection or inference has been running for a
while. It never blocks: ignoring it changes nothing.

- **Notify me when it's done** sends one message when the current job
  finishes or fails, to an email address or a webhook URL (Slack, Teams or
  any endpoint accepting JSON; the payload carries a ``text`` field and the
  job name, status, message, run time, result files and a link back). The
  address is kept in the browser for next time; the server forgets it once
  the message is sent.
- **Browser alert** shows a desktop notification from the browser when the
  job ends. It needs no setup but the tab must stay open; a background tab
  is fine, and the tab title also changes.
- Email needs an SMTP relay on the server side::

      rapidtools-gui --smtp-host smtp.example.edu --smtp-user me@example.edu \
          --smtp-from me@example.edu --public-url https://gui.example.edu/

  The password is read from ``RAPIDTOOLS_SMTP_PASSWORD``; every flag has a
  ``RAPIDTOOLS_SMTP_*`` variable, and ``--smtp-ssl`` selects implicit TLS
  (port 465). ``--public-url`` sets the link put in messages when the
  server sits behind a proxy or tunnel. Without a relay the bar accepts
  webhook URLs only and says so.

Headless: ``POST /api/notify`` with ``{"target": ...}`` arms the message
for the running (or next) job, ``/api/state`` reports it under ``notify``
together with the last delivery result, and
:class:`~rapidtools.gui.Notifier` can be used directly.

Sharing with colleagues
-----------------------

Run the server on the machine with the GPU and the imagery, bind it to the
network, and protect it with a token:

.. code-block:: bash

   rapidtools-gui --host 0.0.0.0 --token auto --data-root /data/eaton --no-browser

The terminal prints a share link of the form ``http://<host>:8765/?token=...``.
Anyone with the link, or who enters the token on the sign-in page, can use
the app. ``--data-root`` confines the file browser and every path to one
directory, and ``RAPIDTOOLS_GUI_TOKEN`` can supply the token instead of the
flag. Everyone connected shares one workspace and one job queue, which suits
a small team taking turns. Keep the server on your institution's network or
a VPN: the token protects the app, not the transport. For a server that
colleagues reach over HTTPS, with Docker, notifications and a GPU, see
:doc:`deployment`.

Scripting the same workflow
---------------------------

The GUI drives :class:`~rapidtools.gui.AssetAnalysisWorkflow`, which is
usable headlessly with :class:`~rapidtools.gui.DetectionSettings`,
:class:`~rapidtools.gui.StreetDetectionSettings`,
:class:`~rapidtools.gui.RegionImagerySettings` and
:class:`~rapidtools.gui.InferenceSettings`:

.. code-block:: python

   from rapidtools.gui import (
       AssetAnalysisWorkflow, DetectionSettings, InferenceSettings,
       RegionImagerySettings, StreetDetectionSettings,
   )

   wf = AssetAnalysisWorkflow(progress_callback=print)
   raster = wf.download_basemap(RegionImagerySettings(
       output_dir='output', provider='google', zoom=19,
       min_lon=-117.489, min_lat=47.711, max_lon=-117.487, max_lat=47.7125,
   ))
   vehicles = wf.discover_street(StreetDetectionSettings(
       output_dir='output', classes=['vehicles'], access_token='MLY|...',
       raster_path=raster, frame_spacing_m=3, min_observations=2,
   ))
   graded = wf.analyze(vehicles.collection, InferenceSettings(
       raster_path=raster, output_dir='output',
       prompt='Condition: [intact, damaged, debris]\nJustification: [one sentence]',
       backend='gemma4', imagery='mapillary_objects', mapillary_token='MLY|...',
       outline_shape='corners', outline_color='#00ffff',
   ))

The prompt builder is available the same way through
:class:`~rapidtools.gui.PromptSpec`, :func:`~rapidtools.gui.assemble_prompt`
and :class:`~rapidtools.gui.PromptAssistant`:

.. code-block:: python

   from rapidtools.gui import OutputField, PromptAssistant, PromptSpec, assemble_prompt
   from rapidtools.models import load

   spec = PromptSpec(
       asset='vehicle', imagery='street',
       objective='classify the condition of the marked vehicle',
       fields=[OutputField('Condition', options=['intact', 'damaged', 'debris']),
               OutputField('Justification', kind='text', description='one sentence')],
   )
   print(assemble_prompt(spec))

   assistant = PromptAssistant(load('gemma4'))          # or load('gemini', api_key=...)
   spec = assistant.draft('grade roof damage of houses after a hurricane into four levels')
   spec.rubric[0].indicators = assistant.suggest_indicators(spec, spec.rubric[0].value)
   print(assistant.review(assemble_prompt(spec)))
