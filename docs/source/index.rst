rAPIdtools
==========

.. rst-class:: hero-tagline

**Turn post-disaster imagery into structured, GIS-ready assessments.**
rAPIdtools crops aerial and street-level imagery around every building, road
or pole in your inventory, runs the crops through a vision-language model of
your choice, and writes the answers back to GeoJSON or Shapefile.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :fas:`download` Installation
      :link: getting_started/installation
      :link-type: doc

      ``pip install rapidtools``, then add the API keys or local weights for
      the models you want to use.

   .. grid-item-card:: :fas:`rocket` Quickstart
      :link: getting_started/quickstart
      :link-type: doc

      Grade every building in an orthomosaic in under twenty lines of code,
      with a local model or a hosted API.

   .. grid-item-card:: :fas:`map` Examples
      :link: examples/eaton_fire_aerial
      :link-type: doc

      Complete scripts on real data: damage and recovery after the 2025
      Eaton Fire, a street survey of Spokane, drone photos, footprints and
      roads.

   .. grid-item-card:: :fas:`book` API Reference
      :link: api/index
      :link-type: doc

      Every public class and function, one page each, with runnable
      examples.

What it does
------------

rAPIdtools handles the steps between "we have imagery" and "we have an answer
for every building":

- Read footprints from GeoJSON or Shapefiles and crop multi-gigabyte
  orthomosaics one window at a time.
- Fetch imagery when you have none. It stitches Bing or Google basemaps into
  GeoTIFFs and finds the Google Street View or Mapillary panoramas that
  actually show each asset, skipping views blocked by a neighbor. From a
  drone flight it picks the few photos that show the most of each building.
- Build the inventory when you have none. SAM 3 finds buildings, roads or
  trees in a raster from a text prompt, and a Mapillary survey yields
  vehicles, poles and houses from the detections Mapillary already
  publishes.
- Ask any vision-language model about each asset: Gemini, Claude, OpenAI,
  Muse Spark or Qwen through their APIs, or Gemma 4, Llama, Muse Glimmer,
  Qwen and other Hugging Face checkpoints on your own GPU. Batching and
  rate limits are handled for you.
- Work from a browser if you prefer. The bundled interface runs the same
  workflow without code, and any long job can be stopped without losing
  what it has already finished.

.. code-block:: python

   import rapidtools as rt

   rt.configure_logging()
   buildings = rt.PhysicalAssetCollection.from_geojson('buildings.geojson')
   model = rt.models.load('gemini', api_key='AIza...')  # or 'gemma4', 'claude', ...
   pipeline = rt.Pipeline([
       rt.AerialImageryExtractor('ortho.tif', save_directory='crops', buffer_m=20),
       rt.AssetAnalyzer(model, prompt='Rate the damage 0-5 as JSON.'),
   ])
   buildings = pipeline.run(buildings)

Feedback and feature requests
-----------------------------

rAPIdtools is shaped by the surveys it is used on. If a workflow does not fit
your data, a page is unclear or something is missing, please tell us. The
:doc:`feedback page <feedback>` has a form for feature requests, one for bug
reports and one for general feedback, plus an email address if you do not
use GitHub.

Citing rAPIdtools
-----------------

rAPIdtools is developed by the `NSF NHERI RAPID Facility
<https://www.uwrapid.org/>`_ at the University of Washington, supported by the
U.S. National Science Foundation, and released under the BSD 3-Clause license.
If it contributes to your research, please cite it; the repository's
*Cite this repository* button gives APA and BibTeX entries, and each release
is archived on Zenodo with a DOI.

.. code-block:: bibtex

   @software{rapidtools,
     author  = {Cetiner, Barbaros},
     title   = {rAPIdtools: A high-performance toolkit for performing large-scale AI inference and localization on post-disaster geospatial datasets},
     year    = {2026},
     doi     = {10.5281/zenodo.24542491},
     url     = {https://github.com/RAPID-Facility/rAPIdtools}
   }

This DOI covers every version and always resolves to the latest one. When you
report results, cite the release you used instead: add its version number and
use its own DOI, listed under *Versions* on the
`Zenodo page <https://doi.org/10.5281/zenodo.24542491>`_.

.. toctree::
   :hidden:
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :hidden:
   :caption: User Guide

   user_guide/aerial_imagery
   user_guide/street_level_imagery
   user_guide/drone_imagery
   user_guide/vision_language_models
   user_guide/feature_detection
   user_guide/cancellation_and_logging

.. toctree::
   :hidden:
   :caption: Examples

   examples/eaton_fire_aerial
   examples/street_level_recovery
   examples/vehicle_detection_from_street
   examples/building_footprints_from_street
   examples/building_inference_from_drone
   examples/building_footprints_from_bing
   examples/road_centerlines
   examples/single_image_inference

.. toctree::
   :hidden:
   :caption: Graphical Interface

   gui
   json_interface
   deployment
   deployment_aws

.. toctree::
   :hidden:
   :caption: Development

   development
   feedback

.. toctree::
   :hidden:
   :caption: Reference

   api/index
