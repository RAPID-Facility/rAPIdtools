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

1. **Imagery.** Load a GeoTIFF from your computer, or download a UW RAPID
   sample. The image stays on screen for the rest of the session and streams
   native-resolution pixels as you zoom, so multi-gigabyte orthomosaics can
   be inspected without loading them into the browser.
2. **Detect and analyze.** Type the assets you want found (for example
   ``building, tree, swimming pool``) or load a file with their locations,
   then choose a model and write the prompt applied to every asset. Every
   analysis model in rAPIdtools is available: the Gemini, Claude, OpenAI,
   Muse Spark and Qwen APIs; Gemma 4, Llama Vision, Muse Glimmer and Qwen
   locally; and any Hugging Face vision-language checkpoint with optional
   4-bit loading. *Fetch models* lists the IDs your key can reach or the
   most downloaded open checkpoints. Detection thresholds, imagery source,
   asset size and the outline drawn on each crop live under *Advanced
   options*.
3. **Results.** Hover over an asset to see its attributes, click one to view
   the exact crops the model analyzed, colour the overlay by any inferred
   attribute, browse the attribute table and download the GeoJSON outputs.

Progress and log output stream into the page while the work runs on a
background thread. **Cancel** stops detection or inference at the next tile
or batch and keeps the results produced so far.

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
a VPN: the token protects the app, not the transport.

Scripting the same workflow
---------------------------

The GUI drives :class:`~rapidtools.gui.AssetAnalysisWorkflow`, which is
usable headlessly with :class:`~rapidtools.gui.DetectionSettings` and
:class:`~rapidtools.gui.InferenceSettings`:

.. code-block:: python

   from rapidtools.gui import AssetAnalysisWorkflow, DetectionSettings, InferenceSettings

   wf = AssetAnalysisWorkflow(progress_callback=print)
   detection = wf.detect(DetectionSettings(
       raster_path='eaton_patch_20250214.tiff', output_dir='output',
       assets=['building', 'tree'],
   ))
   inference = wf.analyze(detection.collection, InferenceSettings(
       raster_path='eaton_patch_20250214.tiff', output_dir='output',
       prompt='Describe the damage to the outlined asset.',
       backend='gemma4', outline_shape='corners', outline_buffer='2 m',
   ))
