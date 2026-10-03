# Changelog

All notable changes to rAPIdtools are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) with the usual 0.x caveat that
minor releases may change public APIs.

## [Unreleased]

### Added

- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `CONTRIBUTING.md` and
  `SECURITY.md`, which asks for private vulnerability reports through GitHub
  Security Advisories (now enabled on the repository), plus a Contributing
  and Security section in the README.
- Issue forms for bug reports and feature requests, contact links to the
  security advisory form, docs and wiki, and a pull request template with the
  `CONTRIBUTING.md` checklist (under `.github/`).
- **Street-level object discovery.** `MapillaryFeatureExtractor` (a `DETECT`
  step) finds objects of the requested classes in every Mapillary image of a
  region from Mapillary's own segmentation detections, read as metadata with
  no image downloads. Plain-English class names (`'vehicles'`, `'utility
  poles'`) are translated to Mapillary labels; classes outside the vocabulary
  can be detected with SAM 3 on thumbnails. Detections of the survey vehicle
  are removed by their recurrence across a sequence, each sighting is placed
  from the camera pose and the detection's position in the frame, and objects
  seen from two or more cameras are triangulated. Results are point assets
  with `localization`, `n_observations`, `confidence` and the sightings as
  `Observation` records.
- `MapillaryObjectImageExtractor` (an `EXTRACT_IMAGERY` step) crops the
  closest distinct views of each discovered object from thumbnail-size
  downloads, with the same `outline_*` options as the aerial extractor.
- `rapidtools.core.Observation`, the record linking an asset to one sighting
  in a street-level image, and `rapidtools.processing.street_localization`
  with the bearing, range, triangulation, ego-mask, frame-thinning and
  clustering functions.
- `MapillaryClient.fetch_detections()`, `decode_detection_polygons()` and
  `get_image_url()`.
- `rapidtools.processing.outlines`, the outline-drawing helpers shared by the
  imagery extractors.
- Example `examples/vehicle_detection_from_street.py`: vehicles along the
  RAPID survey of Spokane, WA, with optional condition grading.
- **GUI: the newer components are now in the interface.** Step 1 can stitch
  a Bing or Google satellite basemap over a bounding box or a GeoJSON extent
  (`RegionImagerySettings`, `AssetAnalysisWorkflow.download_basemap`).
  Step 2 gains a third asset mode, *Discover along a street survey*, which
  runs `MapillaryFeatureExtractor` over the loaded image's extent
  (`StreetDetectionSettings`, `AssetAnalysisWorkflow.discover_street`); SAM 3
  detection can use a Google basemap and keep every instance separate
  (`merge_overlaps`); an *Imagery for analysis* card picks aerial crops
  (with `min_footprint_coverage` / `pad_edges`), Google Street View,
  Mapillary panoramas or Mapillary object crops (`InferenceSettings.imagery`
  and the street options); temperature, output length and JSON mode are
  exposed; and the sample-prompt button became a library of the three
  registry prompts. The results view hides observation records and counts
  analysed assets by the analyzer's marker attribute.
- **GUI: prompt builder and assistant.** `rapidtools.gui.prompt_builder`
  holds `PromptSpec` (task, output fields, per-class indicators, steps, edge
  cases, context), `assemble_prompt()` which renders it in the layout of the
  sample prompts (or as a JSON schema), and `PromptAssistant`, which uses any
  rapidtools model backend text-only to draft a specification from a brief,
  refine it, suggest indicators for a class, review a prompt or import an
  existing one. The GUI's *Build a prompt* dialog walks through those parts,
  previews the assembled prompt and the attribute names it will produce, and
  runs the assistant as a cancellable job (`AssistSettings`,
  `AssetAnalysisWorkflow.assist`); local assistant models are released before
  detection or inference needs the GPU. New endpoints: `/api/imagery/region`,
  `/api/street/discover`, `/api/prompt/assemble`, `/api/prompt/example`,
  `/api/prompt/assist`, `/api/mapillary_token`; `/api/prompt/sample` takes a
  `name`; `/api/detect` takes `basemap` and `merge_overlaps`.

- **GUI: the map comes first.** The interface opens on a streaming
  satellite map (Bing or Google tiles proxied by `GET /api/basemap_tile/...`)
  with Mapillary's street-level survey routes drawn on top, RAPID-only by
  default, as lines or a heat style: a country-wide index of RAPID routes
  at low zoom (`GET /api/street/overview`, built once from the zoom-6
  coverage tiles and cached under `~/.cache/rapidtools`) and the exact
  streets from the coverage tiles when zoomed in
  (`GET /api/street/sequences/<z>/<x>/<y>`,
  `MapillaryClient.fetch_sequence_lines()`). An address search
  (`GET /api/geocode`, OpenStreetMap Nominatim) jumps to a place. Drawing a
  box fills a *Selected area* card that downloads satellite imagery for it
  or goes straight to street discovery with the box as the survey area, so
  no aerial image is needed; the box is also mirrored into the coordinate
  fields of both steps. Assets are drawn on the map in WGS84
  (`GET /api/geo_overlay`). The RAPID and rapidtools logos sit in the
  header. `GET /api/street/coverage` returns image positions in a box.
- **GUI: RAPID route database.** The server builds a local database of
  every RAPID survey route at zoom-13 detail from Mapillary's coverage
  tiles (`GET /api/street/routes`, `GET /api/street/routes.json`, gzipped,
  `POST /api/street/routes/rebuild`), cached as
  `~/.cache/rapidtools/survey_routes_z13.json.gz`; the map draws from it
  at every zoom with no per-view requests. Step 1 asks how to start (map
  area, own imagery, or a sample) and shows one panel; the satellite tile
  zoom is chosen automatically from the area size. In step 2 the model ID
  is a dropdown of the backend's catalogue (with a custom entry), the
  imagery options sit under *Advanced options*, and the prompt builder
  can start from any sample prompt and repurpose it for another task,
  with Gemma 4 as the default assistant.
- **GUI: notifications for long runs.** A bar under the header offers to
  send one message (email through an SMTP relay, or any webhook such as
  Slack or Teams) when the running job ends, or to raise a browser alert;
  the page can be closed and reopened meanwhile. `rapidtools.gui.notify`
  (`NotificationConfig`, `Notifier`, `JobSummary`), `POST /api/notify`, the
  `notify` block of `/api/state`, and the `--smtp-host`, `--smtp-port`,
  `--smtp-user`, `--smtp-from`, `--smtp-ssl` and `--public-url` options of
  `rapidtools-gui` (also `RAPIDTOOLS_SMTP_*` / `RAPIDTOOLS_PUBLIC_URL`).

### Changed

- `MapillaryFeatureExtractor` reads frames in batches (`frame_batch_size`,
  default 200) and discards each image's full detection payload once the
  requested classes are read, and simplifies detection outlines on arrival
  (`simplify_tolerance`, default 0.002, capped at 1 % of each outline's
  size). `MapillaryObjectImageExtractor`
  downloads each source image once, crops every object seen in it and
  releases it instead of caching every image for the whole run. A city-wide
  run previously grew to tens of gigabytes of RAM; memory now stays flat.
  The ego-vehicle filter groups near-identical boxes before comparing them,
  so a survey-length sequence is linear instead of quadratic (the Spokane
  survey's 720,000 sightings now take two minutes to filter, localise and
  cluster). The example defaults to a neighbourhood of Spokane and shows
  the whole-survey box in a comment.
- `DetectionSettings.detect_in_recon_imagery` is kept in sync with the new
  `basemap` field (`'bing'`, `'google'` or `'recon'`); `DetectionResult`
  reports which `basemap` was used.
- `get_configured_session()` takes `allowed_methods` and `status_forcelist`;
  its default no longer retries POST. API clients use
  `rapidtools.models.api_base.api_session()`, which retries POST only on
  429, 503 and 529.
- `ModelOutput` gained `error`, `retryable`, `failed` and
  `ModelOutput.failure()`. Transient failures are still a plain `None` from
  `run_inference`; provider rejections now come back as a failed output.
- CI runs `ruff check`, `ruff format --check` and `mypy` in a lint job, and
  `make check` includes `typecheck`. The package type-checks clean.

### Removed

- `examples/mapillary_vehicle_detection.py`, superseded by the example above.
  Its geometry lives on in `street_localization`; its main flaw, counting the
  survey vehicle as a detection in every frame, is fixed by the ego filter.

### Fixed

- The two asset-analysis notebooks under `examples/` no longer contain a
  hard-coded Gemini API key (the key has been revoked); they read
  `GOOGLE_API_KEY` from the environment instead.
- Image assets attached to a `PhysicalAsset` survive GeoJSON and shapefile
  round trips. `to_geojson_feature` serialized the private image caches, so
  `from_geojson_feature` rejected every image with a warning, and
  `PhysicalAssetCollection.from_geojson` never rehydrated images at all.
  `ImageAsset.to_dict()` / `from_dict()` are the serialization pair both
  loaders use. `to_shapefile` writes the images and the original names of
  truncated columns to a `<name>.rapidtools.json` sidecar (the DBF keeps an
  `n_images` count instead of 254 characters of cut-off JSON), which
  `from_shapefile` reads back into `image_assets` and full attribute names.
- SAM 3 masks from `SAM3OrthoFeatureExtractor` and `BuildingRegularizer` are
  vectorized in the raster's own pixel grid and reprojected to WGS84
  (`rapidtools.processing.vectorize.mask_to_wgs84_polygons`,
  `OrthomosaicReader.generate_tiles(return_georef=True)` and
  `get_image_patch(return_georef=True)`, and the `'native_georef'` image
  property written by `AerialImageryExtractor`). Mapping pixels through the
  tile's lon/lat envelope stretched and skewed polygons on projected or
  rotated orthomosaics; on a 20° rotated UTM test raster the old polygons
  reached 0.56 IoU against ground truth, the new ones 1.00.
- Provider POSTs are no longer re-sent automatically after 500/502/504, which
  could double-bill a request the provider had already processed. Refusals,
  safety blocks and 4xx responses are returned with `retryable=False`, and
  `AssetAnalyzer` reports them as failed without pausing every worker for the
  rate-limit cooldown or re-sending them in the retry passes.
- The Mapillary access token no longer appears in logged tile and Graph API
  error messages.
- Web Mercator pixel and tile indices are clamped at longitude 180 and the
  Mercator latitude limit instead of landing one past the grid.
- `ImageAsset.load_mask(custom_path=...)` reads the requested file instead of
  returning the mask cached for the default path.
- `ImageAsset.save_interactive_html` escapes labels in the generated markup
  and JavaScript.
- A `computed_compass_angle` present with a null value falls back to
  `compass_angle` instead of discarding the frame.
- `from_shapefile` logs the records it skips for unreadable geometry instead
  of dropping them silently.

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
