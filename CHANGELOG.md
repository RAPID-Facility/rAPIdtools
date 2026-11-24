# Changelog

All notable changes to rAPIdtools are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) with the usual 0.x caveat that
minor releases may change public APIs.

## [Unreleased]

## [0.2.1] - 2026-11-24

### Added

- **Street-level object discovery.** `MapillaryFeatureExtractor` (a `DETECT`
  step) finds objects such as `'vehicles'`, `'utility poles'` or
  `'buildings'` along a Mapillary survey from Mapillary's own detections,
  with no image downloads. Static classes such as hydrants and traffic signs
  come from Mapillary's map features, and classes outside its vocabulary can
  be detected with SAM 3. Objects are placed by occupancy fusion, where all
  frames of a survey day vote on a ground grid, so moving vehicles drop out
  and an object seen from several streets is reported once. The earlier
  track-and-triangulate method is one keyword away (`localization_method`).
- **Building footprints from a street survey.** `'buildings'` come back as
  polygons outlined from the survey's structure-from-motion point cloud, with
  `frontage_m`, `depth_m` and `height_m`. On a 1 km neighbourhood of Spokane,
  78 % of the 346 OpenStreetMap footprints near the route were found, with a
  median intersection over union of 0.38.
- **Screening detections with a model.** `MapillaryObjectImageExtractor`
  crops each discovered object, and a new `VERIFY` pipeline stage sits
  between cropping and analysis: `DetectionVerifier` drops false detections,
  `DuplicateResolver` merges objects a model is sure are one, and
  `MapillaryFeatureExtractor(reid=True)` merges repeated passes by
  appearance.
- **Inference from drone photos.** `DroneImageExtractor` picks, for every
  asset, the few perspective photos that together show the most of its walls
  and roof, and attaches the crops so `AssetAnalyzer` works unchanged.
  `coverage_gaps()` returns the walls no photo shows, and camera poses are
  read from EXIF and DJI metadata or a sidecar file (`read_drone_images()`).
  On a simulated flight over 179 Altadena buildings, four selected photos
  show 0.87 of a building's surface against 0.65 for the four nearest.
- **A JSON job interface** (`rapidtools.jobs`). A job file lists ordered
  steps (basemap download, aerial or street detection, model analysis, ...)
  with their inputs and options. `rapidtools-job run job.json` validates
  every step before the first one runs and keeps a result file up to date.
  API keys stay out of the file as `{"$secret": name}` or `{"$env": VAR}`
  references, and JSON Schemas are published with the documentation.
- **Shorter spellings.** `rt.AssetAnalyzer('gemini', prompt=...)` builds the
  model from a provider name, `rt.load`, `rt.GenerationConfig` and the other
  everyday names import from the top level, and `Pipeline.run()` needs no
  argument when the pipeline starts with a detection step.
- Mapillary tokens resolve like API keys: pass the token, the path of a file
  holding it, or nothing to read `MAPILLARY_ACCESS_TOKEN`.
- `PhysicalAssetCollection.filter(predicate)`,
  `to_geojson(include_images=False)` and a one-line `repr`.
- Error messages name the arguments a wrapper accepts, the closest match to
  a misspelt option, or the pipeline step at fault.
- **GUI rebuilt around a map.** The interface opens on a satellite map with
  RAPID's street-level survey routes and an address search. Drawing an area
  downloads Bing or Google imagery for it or starts street discovery
  directly, and inference can use aerial crops, Google Street View or
  Mapillary imagery.
- **GUI: prompt builder and notifications.** A *Build a prompt* dialog
  assembles a prompt from a task, output fields and per-class indicators and
  can ask any model to draft or review it. Long runs can send an email, a
  webhook message or a browser alert when they end.
- **Shared-server deployment.** `deploy/` holds a Docker and Caddy setup for
  a token-protected GUI server, with CUDA and CPU images published to
  `ghcr.io/rapid-facility/rapidtools-gui` for every release, and
  `deploy/aws/` adds a CloudFormation template with launch and teardown
  scripts.
- New examples for street-level vehicles, building footprints, detection
  maps and drone photos, job files under `examples/jobs/`, and documentation
  pages for the job interface, drone imagery, deployment, development and
  feedback.
- `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `ACCESSIBILITY.md`, issue forms and a pull request template.

### Changed

- **The GUI hands every run to the package as a job file** under
  `<output folder>/jobs/`, so any run can be repeated from a terminal. Keys
  typed into the page are not in those files.
- `import rapidtools` takes about half a second, and `Pipeline`, the
  extractors and the regularizers no longer load PyTorch and Transformers.
- The deprecated `*AssetAnalyzer` classes are no longer listed in
  `rapidtools.__all__`; they still import and warn.
- Hosted-model requests are retried only on 429, 503 and 529
  (`get_configured_session()` no longer retries POST by default), and
  provider rejections come back as a failed `ModelOutput` with `error` and
  `retryable`.
- `to_shapefile` writes images and full column names to a
  `<name>.rapidtools.json` sidecar that `from_shapefile` reads back, and
  `from_shapefile` keeps assets with invalid geometry.
- `numpy` and `scipy` are declared dependencies.
- CI runs `ruff` and `mypy`; the package type-checks clean.

### Removed

- `examples/mapillary_vehicle_detection.py`, superseded by
  `examples/vehicle_detection_from_street.py`.

### Fixed

- Image assets survive GeoJSON and shapefile round trips; loading a saved
  collection used to drop every image.
- SAM 3 polygons are no longer stretched or skewed on projected or rotated
  orthomosaics (0.56 to 1.00 IoU on a 20° rotated test raster).
- Provider requests are no longer re-sent after 500/502/504, which could
  double-bill, and refusals no longer pause every worker.
- One image SAM 3 could not load no longer shifts every later mask onto the
  wrong image.
- Crops of an asset covered by several rasters keep the georeferencing of
  the raster they came from.
- 16-bit RGBA orthomosaics no longer come out nearly black, and float
  rasters with NaN nodata get valid contrast.
- Street View depth maps are no longer mirrored relative to the panorama.
- `MapillaryClient.fetch_images_in_bbox()` returns only the images inside
  the box.
- Smaller fixes: asset IDs on shapefile read-back, Web Mercator indices at
  the grid edge, `load_mask(custom_path=...)`, a null
  `computed_compass_angle`, image media types for URL inputs,
  `search_hub_models` with `huggingface_hub` 1.x, and package log records in
  the GUI log pane.

### Security

- Hosted model wrappers no longer send the provider's API key to the host of
  an image given as a URL.
- The GUI server refuses to serve files from outside its data root and
  output folder; a crafted GeoJSON could previously read any file on disk.
- The Mapillary token no longer appears in logged error messages.
- A hard-coded Gemini API key was removed from the two example notebooks
  (the key has been revoked).
- `ImageAsset.save_interactive_html` escapes labels.

## [0.2.0] - 2026-09-28

### Added

- **Loading models and running analysis.** Use `rapidtools.models.load(provider, ...)`
  to get a model wrapper by its registry key, then hand it to `AssetAnalyzer(model, prompt, ...)`.
  This replaces the old per-provider analyzer classes. To see what's available without 
  importing PyTorch, call `list_providers()` or `catalog()` , or look at the `PROVIDERS` registry directly.
  `GenerationConfig` carries per-call options such as `json_mode` and `temperature`; 
  `RateLimitPolicy` gives hosted models a shared cooldown and retry passes so
  rate limits do not leave holes in the results.
- **New model backends:** OpenAI, Meta Muse Spark and Alibaba Qwen APIs; Meta
  Muse Glimmer, Qwen VL and any Hugging Face vision-language checkpoint
  (`HFVisionInference`) locally, with optional 4-bit loading. Every wrapper
  exposes `list_known_models()` and, where the provider supports it,
  `list_available_models(api_key)`.
- **Keyless Google imagery:** `GoogleAerialImageExtractor` and
  `GoogleOrthomosaicExtractor` stitch satellite tiles for polygons or regions,
  and `GoogleStreetViewImageExtractor` finds the nearest Street View panorama
  for each asset, crops the field of view covering the footprint and can save
  the decoded depth map. No API key or billing account required.
- **Graphical interface.** `rapidtools-gui` (or `python -m rapidtools.gui`)
  opens a local web app that loads imagery, detects assets with SAM 3, runs any
  supported model with a custom prompt, and explores the results on the image.
  It can be shared on a network with a token and driven headlessly through
  `AssetAnalysisWorkflow`.
- **Cooperative cancellation.** Every long-running component and the
  `Pipeline` accept a `cancel_event` and raise `OperationCancelled` at the next
  asset, tile or batch, keeping partial results.
- **Configurable asset outlines** in `AerialImageryExtractor`: `outline_shape`
  (`geometry`, `bbox`, `rotated_bbox`, `convex_hull`, `corners`),
  `outline_buffer` in pixels, percent or real-world units, `outline_width` in
  pixels or percent of the image, and `outline_color`.
- **Edge handling** in `AerialImageryExtractor`: `min_footprint_coverage` keeps
  assets whose footprint is only partly imaged and `pad_edges` pads their
  crops to full size. Coverage is judged from the raster mask, so alpha bands
  and internal masks count.
- `SAM3OrthoFeatureExtractor(merge_overlaps=False)` keeps one asset per
  detected instance with its confidence, for countable objects such as
  vehicles.
- `BoundingBox.from_raster()` and `BoundingBox.from_geojson()` build regions
  from a raster's extent or a GeoJSON file.
- `PhysicalAssetCollection.from_shapefile()` / `to_shapefile()`.
- `rapidtools.login()` and `ensure_huggingface_login()` authenticate with the
  Hugging Face Hub for gated weights; local wrappers call them automatically.
- `rapidtools.configure_logging()` for opt-in progress output; importing the
  package no longer touches logging configuration.
- Pipeline steps declare a `Stage`, so steps can be added in any order.
- Documentation site built with Sphinx and the Book theme: user guide, worked
  examples on the 2025 Eaton Fire dataset, GUI guide and a per-class API
  reference. Published through GitHub Pages.
- A test suite covering the core objects, data sources, models, processing
  components and the GUI, run on Linux, macOS and Windows for Python 3.11 to
  3.13.

### Changed

- `import rapidtools` is side-effect free and takes about two seconds; model
  and processing classes, and the `models`, `processing` and `gui`
  subpackages, load on first use.
- Constructor parameter names were harmonized across components
  (`save_directory`, `buffer_m`, `max_images_per_asset`, ...). Old names still
  work and emit a `DeprecationWarning`.
- `AerialImageryExtractor` accepts a folder or list of rasters and opens each
  raster once.
- `MapillaryImageExtractor` culls occluded views by ray casting against
  neighbouring footprints and can check Mapillary's semantic labels before
  downloading (`strict_content_filter`, `MapillaryLabelMapper`).
- `BuildingRegularizer` runs a second, lower-threshold pass on dropped assets
  to rescue shadowed buildings.
- Local model wrappers use dynamic batching, precision chosen for the
  hardware, and release GPU memory between batches.

### Deprecated

- The provider-specific analyzers (`GeminiAssetAnalyzer`,
  `ClaudeAssetAnalyzer`, `OpenAIAssetAnalyzer`, `MuseSparkAssetAnalyzer`,
  `QwenAssetAnalyzer`, `Gemma4AssetAnalyzer`, `LlamaVisionAssetAnalyzer`,
  `MuseGlimmerAssetAnalyzer`, `QwenVisionAssetAnalyzer`,
  `HFVisionAssetAnalyzer`). Use `AssetAnalyzer(load(provider, ...))`.
- Deprecated keyword aliases listed above.

### Fixed

- `rapidtools.models` and `rapidtools.processing` are reachable as attributes
  of the top-level package, as the README examples show.

## [0.1.2] - 2026-08-12

- Transformers syntax updated to the latest release.
- Bounding boxes can be extracted from a GeoJSON file.
- Post-disaster asset detection bug fixed.
- Generalized asset inference example added.

[0.2.0]: https://github.com/RAPID-Facility/rAPIdtools/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/RAPID-Facility/rAPIdtools/releases/tag/v0.1.2
