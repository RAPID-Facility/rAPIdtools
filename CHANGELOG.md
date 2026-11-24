# Changelog

All notable changes to rAPIdtools are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) with the usual 0.x caveat that
minor releases may change public APIs.

## [Unreleased]

## [0.2.1] - 2026-11-24

### Added

- **Street-level object discovery.** `MapillaryFeatureExtractor` (a `DETECT`
  step) finds objects of the requested classes along a Mapillary survey from
  Mapillary's own segmentation detections, read as metadata with no image
  downloads. Plain-English class names (`'vehicles'`, `'utility poles'`,
  `'buildings'`) are translated to Mapillary labels, classes outside the
  vocabulary can be detected with SAM 3 on thumbnails, and static point
  classes such as hydrants, poles and traffic signs come from Mapillary's
  pre-triangulated map features (`detection_source`). Detections of the
  survey vehicle itself are removed, and every object keeps its sightings as
  `Observation` records.
- **Localisation by occupancy fusion.** All frames of a survey day vote on a
  ground grid (free up to an object, occupied behind its ground contact), and
  an object is a blob of cells most frames call occupied. An object seen from
  several streets lands on the same cells, a moving vehicle never collects
  agreeing votes, and the same code places a hydrant and a bus. Objects carry
  `length_m`, `width_m` and `footprint_area_m2`. This is the default
  (`localization_method='auto'`); track-and-triangulate (`'tracks'`), ray
  voting (`'voting'`) and radius clustering (`'cluster'`) are one keyword
  away. Street-level geometry uses the full camera pose (rotation, camera
  type and lens parameters) for spherical, perspective and fisheye cameras.
- **Building footprints from a street survey.** For `'buildings'` the
  extractor labels the survey's own structure-from-motion point cloud with
  the detection outlines and returns one polygon per structure, with
  `frontage_m`, `depth_m`, `height_m` and `facade_bearing`. Where the survey
  did not see the rear, it is closed at `assumed_depth_m` and `depth_source`
  reads `'assumed'`. On a 1 km neighbourhood of Spokane: 336 structures
  against 346 OpenStreetMap footprints near the route, 78 % of them
  overlapped, median intersection over union 0.38, about one house in six
  reported in two pieces. `geometry='auto' | 'footprint' | 'point'` chooses
  the output.
- `MapillaryObjectImageExtractor` (an `EXTRACT_IMAGERY` step) crops the
  widest or closest views of each discovered object (`view_selection`), with
  the same `outline_*` options as the aerial extractor.
- **Screening detections with a model.** A new `VERIFY` pipeline stage runs
  between cropping and analysis. `DetectionVerifier` shows each object's
  crops to any `rapidtools.models.load()` model and drops what is not what it
  claims to be; `DuplicateResolver` merges pairs of nearby objects a model is
  sure are one; and `MapillaryFeatureExtractor(reid=True)` merges look-alike
  objects from repeated passes by appearance embedding
  (`rapidtools.processing.reid`, CLIP or DINOv2).
- **Inference from perspective drone photos.** `DroneImageExtractor` picks,
  for every asset, the few photos that together show the most of its walls
  and roof (`DroneViewSelector`), crops and marks them, and attaches the
  crops so `AssetAnalyzer` works unchanged. Assets gain
  `drone_view_coverage` and `drone_unseen_sides`; `coverage_gaps()` returns
  the walls no photo shows with a suggested camera position for the next
  flight. `read_drone_images()` and `DroneCamera` read camera poses from EXIF
  and DJI XMP metadata or a sidecar CSV/GeoJSON. On a simulated 452-photo
  flight over 179 Altadena buildings, four selected photos show 0.87 of a
  building's surface against 0.65 for the four nearest photos.
- **A JSON job interface** (`rapidtools.jobs`). Work is described in a job
  file: an ordered list of steps (`imagery.basemap`, `data.download`,
  `detect.aerial`, `detect.street`, `analyze.vlm`, `prompt.assist`), each
  with its inputs and options. `rapidtools-job run job.json` (or
  `rapidtools.jobs.run_job`) validates every step before the first one runs
  and keeps a result file up to date that can be polled. Steps can read what
  earlier steps wrote, and API keys stay out of the file as `{"$secret":
  name}` or `{"$env": VAR}` references. JSON Schemas (`rapidtools-job
  schema`) and the step reference in the documentation are generated from
  the same step declarations.
- **A shorter way to say the common things.** `rt.AssetAnalyzer('gemini',
  prompt=...)` builds the model from a provider name; `rt.load`,
  `rt.GenerationConfig`, `rt.ModelOutput`, `rt.list_providers`, `rt.catalog`,
  `rt.ImageCollection`, `rt.PipelineStep` and `rt.list_datasets` are
  importable from the top level; `Pipeline.run()` needs no argument when the
  pipeline starts with a detection step; and a `cancel_event` given to the
  pipeline is shared with every step that has none of its own.
- Mapillary tokens resolve like API keys: every Mapillary component accepts
  the token, the path of a file holding it, or nothing, in which case
  `MAPILLARY_ACCESS_TOKEN` (or `MAPILLARY_TOKEN`) is read.
- `PhysicalAssetCollection.filter(predicate)`,
  `to_geojson(include_images=False)` and a one-line `repr`.
- `MapillaryClient.fetch_detections()`, `fetch_map_features()`,
  `fetch_sequence_lines()`, `decode_detection_polygons()` and
  `get_image_url()`.
- Errors that say what to do: `load()` names the arguments a wrapper takes
  when given one it does not, misspelt extractor options suggest the closest
  match, `Pipeline` names the step that returned something other than a
  collection, and an empty Mapillary search says what was searched.
- **GUI: the map comes first.** The interface opens on a satellite map with
  RAPID's street-level survey routes drawn on top and an address search.
  Drawing an area downloads Bing or Google imagery for it or goes straight to
  street discovery, so no aerial image is needed. Detection can run on a
  Google basemap and keep every instance separate, and inference can use
  aerial crops, Google Street View, Mapillary panoramas or Mapillary object
  crops, with temperature, output length and JSON mode exposed.
- **GUI: prompt builder and assistant.** A *Build a prompt* dialog assembles
  a prompt from a task, output fields, per-class indicators, steps and edge
  cases, and can ask any model backend to draft, refine or review it
  (`rapidtools.gui.prompt_builder`).
- **GUI: notifications for long runs** by email, webhook (Slack, Teams) or
  browser alert when the running job ends (`rapidtools.gui.notify`, the
  `--smtp-*` and `--public-url` options of `rapidtools-gui`).
- **Shared-server deployment.** `deploy/` holds a Docker and Caddy setup for
  a token-protected GUI server, with CUDA and CPU images of
  `ghcr.io/rapid-facility/rapidtools-gui` published for every release tag
  and a systemd unit for installs without Docker. `deploy/aws/` adds a
  CloudFormation template for a GPU or CPU machine with HTTPS, an encrypted
  data disk, idle shutdown and `launch.sh` / `teardown.sh` wrappers.
- Examples: `vehicle_detection_from_street.py`,
  `building_footprints_from_street.py`, `street_detections_map.py`,
  `street_vehicle_condition_map.py`, `drone_flight_from_google_earth.py`,
  `building_inference_from_drone.py` and the job files under
  `examples/jobs/`.
- Documentation pages for the JSON job interface, drone imagery, deployment
  (shared server and AWS), development and testing, and feedback, plus
  example pages for the new street-level and drone workflows.
- `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md` and
  `ACCESSIBILITY.md`; issue forms for bug reports, feature requests and
  feedback; and a pull request template.

### Changed

- **The GUI hands every run to the package as a job file.** Downloads,
  detection, street discovery, inference and prompt-assistant requests are
  written to `<output folder>/jobs/` and executed through `rapidtools.jobs`,
  so any run can be repeated unchanged from a terminal. Keys typed into the
  page are not in those files. An inference started on its own now reads its
  assets from the GeoJSON the previous run wrote instead of from memory.
- `import rapidtools` takes about half a second, and using `Pipeline`, an
  extractor or a regularizer no longer loads PyTorch and Transformers.
- The deprecated `*AssetAnalyzer` classes are no longer listed in
  `rapidtools.__all__` or `dir(rapidtools)`; they still import and warn.
- `get_configured_session()` takes `allowed_methods` and `status_forcelist`,
  and its default no longer retries POST. API clients retry POST only on 429,
  503 and 529.
- `ModelOutput` gained `error`, `retryable`, `failed` and
  `ModelOutput.failure()`. Provider rejections come back as a failed output;
  transient failures are still a plain `None` from `run_inference`.
- `to_shapefile` writes images and the original names of truncated columns
  to a `<name>.rapidtools.json` sidecar, which `from_shapefile` reads back.
  `from_shapefile` keeps assets with invalid geometry and skips only empty
  ones, naming each record it skips.
- `DetectionSettings` has a `basemap` field (`'bing'`, `'google'` or
  `'recon'`), kept in sync with `detect_in_recon_imagery`.
- `filter_by_attribute` logs one line for all the assets that lack the
  attribute instead of a warning per asset.
- `numpy` and `scipy` are declared dependencies (both were already required
  transitively).
- CI runs `ruff check`, `ruff format --check` and `mypy`; the package
  type-checks clean.

### Removed

- `examples/mapillary_vehicle_detection.py`, superseded by
  `examples/vehicle_detection_from_street.py`.

### Fixed

- Image assets attached to a `PhysicalAsset` survive GeoJSON and shapefile
  round trips (`ImageAsset.to_dict()` / `from_dict()`); loading a saved
  collection used to drop every image.
- SAM 3 masks from `SAM3OrthoFeatureExtractor` and `BuildingRegularizer` are
  vectorized in the raster's own pixel grid and reprojected to WGS84.
  Polygons on projected or rotated orthomosaics were stretched and skewed: on
  a 20° rotated UTM test raster the old polygons reached 0.56 IoU against
  ground truth, the new ones 1.00.
- Provider POSTs are no longer re-sent automatically after 500/502/504, which
  could double-bill a request the provider had already processed. Refusals,
  safety blocks and 4xx responses no longer pause every worker for the
  rate-limit cooldown or get re-sent in the retry passes.
- One image SAM 3 could not load no longer shifts every later mask in the
  batch onto the wrong image.
- With several rasters covering one asset, the crop on disk and its recorded
  georeferencing now come from the same raster, and the winning raster is the
  same on every run.
- 16-bit RGBA orthomosaics no longer come out nearly black, float rasters
  with NaN nodata get valid contrast statistics, and `get_image_patch` no
  longer rejects every patch when the allowed missing-data ratio is 0.
- Decoded Street View depth maps are no longer mirrored relative to the
  panorama.
- `MapillaryClient.fetch_images_in_bbox()` returns only the images inside
  the box instead of everything in the 2.4 km tiles that cover it.
- A `computed_compass_angle` present with a null value falls back to
  `compass_angle` instead of discarding the frame.
- An asset attribute called `id` no longer replaces the asset ID when a
  shapefile is read back.
- Web Mercator pixel and tile indices are clamped at longitude 180 and the
  Mercator latitude limit instead of landing one past the grid.
- `ImageAsset.load_mask(custom_path=...)` reads the requested file instead of
  returning the mask cached for the default path.
- Image inputs given as URLs are sent to hosted models with a valid image
  media type when the host reports `binary/octet-stream` or adds a charset.
- `search_hub_models` works with `huggingface_hub` 1.x.
- GUI: progress and warnings from the package appear in the log pane when
  the server is started with `rapidtools-gui`.

### Security

- Hosted model wrappers no longer send the provider's API key to the host of
  an image given as a URL; images are fetched with a separate session that
  carries no credentials.
- The GUI server refuses to serve asset images from outside its data root
  and output folder. A crafted GeoJSON loaded on a shared server could
  previously read any file on disk.
- The Mapillary access token no longer appears in logged tile and Graph API
  error messages.
- The two asset-analysis notebooks under `examples/` no longer contain a
  hard-coded Gemini API key (the key has been revoked); they read
  `GOOGLE_API_KEY` from the environment.
- `ImageAsset.save_interactive_html` escapes labels in the generated markup
  and JavaScript.

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
