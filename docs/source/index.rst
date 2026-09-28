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

      End-to-end walkthroughs on the 2025 Eaton Fire dataset: aerial damage,
      street-level recovery, footprint and road extraction.

   .. grid-item-card:: :fas:`book` API Reference
      :link: api/index
      :link-type: doc

      Every public class and function, one page each, with runnable
      examples.

What it does
------------

- **Ingest at scale.** Load footprints from GeoJSON or Shapefiles, filter
  them spatially, and read multi-gigabyte orthomosaics window by window.
- **Fetch imagery without API keys.** Stitch Bing or Google basemap tiles into
  georeferenced GeoTIFFs, and pull Google Street View or Mapillary panoramas
  that actually show the asset, with occluded views culled by ray casting.
- **Run any vision-language model.** One factory and one analyzer cover Gemini,
  Claude, OpenAI, Muse Spark and Qwen APIs, plus Gemma 4, Llama, Muse Glimmer,
  Qwen and any Hugging Face checkpoint locally, with batching, 4-bit loading
  and rate-limit back-off handled for you.
- **Discover and clean geometries.** Scan a raster for a text concept with
  SAM 3, then regularize the masks into building footprints or road
  centrelines.
- **Stop cleanly.** Every long-running component takes a cancel event, so
  notebooks and the bundled GUI can interrupt a job and keep partial results.

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
     author  = {Cetiner, Barbaros and {NSF NHERI RAPID Facility}},
     title   = {rAPIdtools: AI inference and localization for post-disaster geospatial datasets},
     year    = {2026},
     version = {0.2.0},
     url     = {https://github.com/RAPID-Facility/rAPIdtools}
   }

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
   user_guide/vision_language_models
   user_guide/feature_detection
   user_guide/cancellation_and_logging

.. toctree::
   :hidden:
   :caption: Examples

   examples/eaton_fire_aerial
   examples/street_level_recovery
   examples/building_footprints_from_bing
   examples/road_centerlines
   examples/single_image_inference

.. toctree::
   :hidden:
   :caption: Graphical Interface

   gui

.. toctree::
   :hidden:
   :caption: Reference

   api/index
