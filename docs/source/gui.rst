Graphical interface
===================

rAPIdtools includes a small web application that runs the detect-and-analyze
workflow without any code. It is built on the Python standard library, so it
needs no extra dependencies, and it opens in your browser.

.. code-block:: bash

   rapidtools-gui                   # console script installed with the package
   python -m rapidtools.gui         # the same thing

.. code-block:: python

   from rapidtools.gui import launch_asset_analysis_app
   launch_asset_analysis_app(port=8765, open_browser=True)

The interface walks you through three steps: choose a place and its imagery,
find and analyze the assets, then look at the results.

Step 1: the place and its imagery
---------------------------------

The interface opens on a map. Step 1 asks how you want to start: **select an
area on the map**, **load your own imagery** (a GeoTIFF on this computer) or
**download a UW RAPID sample**. Only the panel for the option you pick is
shown.

The map draws Bing or Google satellite tiles as you pan and zoom, the way a
desktop GIS draws a basemap. The tiles come through the server, so no key
is needed. On top of them, cyan lines show where Mapillary has street-level
imagery: by default only the surveys uploaded by the UW RAPID Facility, or
those of every contributor once you turn off *RAPID surveys only*. *Heat*
switches the lines to a glowing density view. The layer panel is at the top
right of the map and the zoom control at the bottom right. *Search an
address or place* jumps to a match from OpenStreetMap; typing
``lat, lon`` works too.

To choose a working area, hold Shift and drag, or press *Draw box*. The
*Selected area* panel shows the size of the box and its coordinates; typing
new numbers moves the box, and you can give a GeoJSON file whose extent is
used instead. The panel offers two ways forward:

- **Download satellite imagery here** stitches Bing or Google tiles over the
  box into a GeoTIFF. The zoom level is chosen for you from the size of the
  box (the finest level up to 19 that stays under the 2,500-tile limit) and
  shown as a resolution in meters per pixel. *Adjust resolution* lets you
  change it. The tile count is checked before anything downloads.
- **Discover street-level objects here** goes straight to step 2 with the
  box as the survey area, so you do not need an aerial image at all.

Whichever way you start, the image stays on screen for the rest of the
session. It streams full-resolution pixels as you zoom in, so you can
inspect a multi-gigabyte orthomosaic without loading it into the browser.
Its extent also becomes the survey area for street-level discovery.

Step 2: detect and analyze
--------------------------

Step 2 has three cards.

**Assets.** There are three ways to get assets:

- Detect them in the image with SAM 3 by typing what you are looking for,
  for example ``building, tree, swimming pool``. The scan runs on the image
  itself or on a Bing or Google basemap stitched over its extent. *Keep
  every detected instance separate* turns off the merging of overlapping
  detections, which you want for countable objects such as vehicles.
- Discover objects along a street survey. Type plain-English classes such
  as ``vehicles, utility poles, fire hydrants``, and the interface finds
  them in every Mapillary image within the extent, removes the survey
  vehicle and places each object on the map. This needs a Mapillary token;
  *RAPID token* fills in the one the examples use.
- Load a GeoJSON file with the asset locations.

**Imagery for analysis.** This decides what the model sees: aerial crops of
the loaded image (with ``min_footprint_coverage`` and edge padding), Google
Street View panoramas cropped to each footprint (no key needed), Mapillary
panoramas of each footprint, or the closest Mapillary views of each
discovered object. Where an outline can be drawn on the crop, you choose its
shape (the geometry, its bounding box, corner brackets and so on), buffer,
width and color.

**Analysis model and prompt.** Every analysis model in rAPIdtools is
available: the Gemini, Claude, OpenAI, Muse Spark and Qwen APIs; Gemma 4,
Llama Vision, Muse Glimmer and Qwen on your own GPU; and any Hugging Face
vision-language checkpoint, optionally loaded in 4-bit. *Fetch models* lists
the model IDs your key can reach. Temperature, output length and JSON mode
are under *Advanced options*. The output length follows the selected model
(2,048 tokens for most, 32,768 for Gemini 4, whose thinking counts against
the limit) until you type your own value. The prompt can come from a file,
from the sample library (aerial damage, street-level damage, street-level
recovery) or from the prompt builder described below.

While a job runs, its progress and log stream into the page. **Cancel**
stops detection or inference at the next tile or batch and keeps what has
been produced so far.

Step 3: results
---------------

Hover over an asset to see its attributes, or click it to see the exact
crops the model looked at. You can color the overlay by any attribute the
model produced, browse the attribute table and download the GeoJSON
results.

Assets are drawn both on the map and on the image. The *Map* / *Image*
switch in the toolbar moves between the two, and a checkbox shows the
basemap a detection ran on in place of the loaded image.

Writing a prompt
----------------

The sample prompts that come with rAPIdtools share one layout: a ``TASK``
statement, numbered ``ANALYSIS STEPS``, ``ANALYSIS EDGE CASES``, a
``REQUIRED OUTPUT FORMAT`` whose ``Field: [options]`` lines the analyzer
turns into attributes, and a rubric that describes each class by its key
visual signs. *Build a prompt* opens a dialog that walks you through those
parts:

1. **Task**: the asset, the kind of imagery, what to decide, the expert role
   to take, what to ignore, and how the asset is marked on the crop (filled
   in from your outline settings).
2. **Output fields**: each field becomes one line of the output format and
   one attribute on every asset (``gemini_chs_level``, ...). A field can be
   a category, a number, free text or yes/no, and one checkbox switches the
   format to JSON.
3. **Class indicators**: for every option of the main field, a name, a
   one-sentence meaning and three to five visual cues.
4. **Steps and edge cases**: the order in which to work, and what to answer
   when an image fits no class. Common edge cases are one click away.
5. **Review**: optional background, the assembled prompt and the attribute
   names it will produce. *Use this prompt* puts it in the editor.

The assembled prompt updates as you type. *Start from sample* loads one of
the registry prompts into the builder as a starting point. If you describe
a different task and press *Repurpose for this task*, the assistant rewrites
that prompt for the new task while keeping its structure and output format.

The assistant
~~~~~~~~~~~~~

The dialog's assistant can use any rAPIdtools model, text only. By default
it uses the analysis model you selected, with its key; it can also use a
local Gemma 4 or Qwen checkpoint on your own machine. It can:

- **Draft** a complete prompt from one line ("grade roof damage of houses
  after a hurricane into four levels").
- **Refine** the current prompt as you instruct ("add a confidence field from
  1 to 3").
- **Suggest** visual indicators for one class.
- **Review** the assembled prompt the way a critical colleague would.
- **Import** a prompt from the editor into the builder's fields.

Assistant requests run as cancelable jobs like everything else in the
interface. A local model is loaded once and released before detection or
inference needs the GPU. Nothing reaches the prompt editor until you press
*Use this prompt*.

Long runs: leave and come back
------------------------------

Jobs run on the server, not in the browser tab, and their results stay on
the server until it stops. You can watch the progress, close the tab and
open the same address later (the page picks up where the server is), or ask
to be told when the job ends. Once a download, detection or inference has
been running for a while, a bar appears under the header offering:

- **Notify me when it's done.** One message when the current job finishes
  or fails, sent to an email address or to a webhook URL (Slack, Teams or
  any endpoint that accepts JSON). The browser remembers the address for
  next time; the server forgets it once the message is sent.
- **Browser alert.** A desktop notification from the browser when the job
  ends. It needs no setup, but the tab has to stay open (a background tab
  is fine). The tab title changes too.

Ignoring the bar changes nothing. Email needs an SMTP relay on the server
side:

.. code-block:: bash

   rapidtools-gui --smtp-host smtp.example.edu --smtp-user me@example.edu \
       --smtp-from me@example.edu --public-url https://gui.example.edu/

The password is read from ``RAPIDTOOLS_SMTP_PASSWORD``. Every flag also has
a ``RAPIDTOOLS_SMTP_*`` environment variable, and ``--smtp-ssl`` selects
implicit TLS (port 465). ``--public-url`` sets the link included in messages
when the server sits behind a proxy or tunnel. Without a relay, the bar
accepts webhook URLs only and says so. A webhook receives a JSON body with a
``text`` field plus the job name, status, message, run time, result files
and a link back.

Sharing with colleagues
-----------------------

Run the server on the machine that has the GPU and the imagery, make it
listen on the network, and protect it with a token:

.. code-block:: bash

   rapidtools-gui --host 0.0.0.0 --token auto --data-root /data/eaton --no-browser

The terminal prints a share link of the form
``http://<host>:8765/?token=...``. Anyone with the link, or who types the
token on the sign-in page, can use the app. ``--data-root`` confines the
file browser, and every path, to one folder. ``RAPIDTOOLS_GUI_TOKEN`` can
supply the token instead of the flag.

Everyone connected shares one workspace and one job queue, which suits a
small team taking turns. Keep the server on your institution's network or a
VPN: the token protects the app, not the connection. For a server that
colleagues reach over HTTPS, with Docker, notifications and a GPU, see
:doc:`deployment`.

Job files
---------

The page never calls the analysis code directly. Every run it starts (a
basemap download, a detection, an inference, a prompt-assistant request) is
written as a JSON *job file* under ``<output folder>/jobs`` and handed to
:mod:`rapidtools.jobs`, which keeps a *result file* next to it up to date.
The log pane prints the path of the job file when a run starts.

That file is a complete record of the run, and you can rerun it without the
interface:

.. code-block:: bash

   rapidtools-job run output/jobs/20261124T154952Z-74fa-detection.json

API keys and tokens typed into the page are never written into job files.
They appear there as ``{"$secret": ...}`` references instead.
:doc:`json_interface` describes the format and every step type.

Scripting the same workflow
---------------------------

Behind the page is :class:`~rapidtools.gui.AssetAnalysisWorkflow`, which you
can also use from Python with
:class:`~rapidtools.gui.DetectionSettings`,
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

The prompt builder is available the same way, through
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

Server details
--------------

This section is for people running or automating the server.

Survey routes
~~~~~~~~~~~~~

The RAPID survey routes on the map come from a small local database. On its
first start the server reads Mapillary's country-wide coverage tiles at zoom
6 to find where RAPID surveys exist, then the zoom-13 tiles of those places
(173 of them for the continental United States, about 15 seconds). It keeps
only the RAPID routes and stores them, gzipped, as
``survey_routes_z13.json.gz`` under ``~/.cache/rapidtools``. The map
downloads that file once (it is under a megabyte) and draws every route
from memory, so panning and zooming make no further coverage requests.

The file does not change until you press *refresh* in the layer panel or
call ``POST /api/street/routes/rebuild``. You can copy it to another machine
or replace it with a published one. ``GET /api/street/routes`` reports its
status and ``GET /api/street/routes.json`` serves it.

With *RAPID surveys only* turned off, the routes of all contributors are
read for the current view from the coverage tiles
(``GET /api/street/sequences/<z>/<x>/<y>`` from zoom 6 up,
``GET /api/street/overview`` below that). This is slower and limited to
what is on screen.

Other endpoints
~~~~~~~~~~~~~~~

- ``GET /api/geocode?q=`` answers the address search, through
  OpenStreetMap's Nominatim.
- ``GET /api/geo_overlay`` returns the assets in WGS84 for the map.
- ``POST /api/notify`` with ``{"target": ...}`` arms a notification for the
  running (or next) job. ``GET /api/state`` reports it under ``notify``,
  together with the result of the last delivery.
  :class:`~rapidtools.gui.Notifier` can also be used directly.
