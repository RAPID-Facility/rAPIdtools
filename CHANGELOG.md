# Changelog

All notable changes to rAPIdtools are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) with the usual 0.x caveat that
minor releases may change public APIs.

## [Unreleased]

### Added

- **One footprint per building from a tiled scan.**
  `SAM3OrthoFeatureExtractor(merge_overlaps=False)` no longer keeps the
  best-scoring copy of each detection: the views of an object from all the
  overlapping tiles are fused into one consensus outline
  (`rapidtools.processing.instance_fusion.fuse_view_detections`). Two
  detections from different tiles are the same object only if they agree
  inside the area both tiles covered, so a building cut by a tile edge
  still matches the whole building in the next tile; two instances from
  the same tile are never merged; and an outline made of two or more other
  outlines (a mask that took a house and its garage together) gives way to
  the parts. New options: `neck_width_m` cuts masks where they narrow below
  that width (a mask bleeding across the gap between two roofs), and
  `min_detection_rate` drops objects that most of the tiles seeing them
  missed. Assets carry a new `views` attribute. On the Eaton sample
  (`eaton_patch1`), scored against the 93 county footprints visible in the
  unburned half of the drone orthomosaic (4 more are hidden under trees or
  debris), the GUI's building detection (this scan with 50% tile overlap,
  then `BuildingRegularizer`) left 3, 4 and 0 polygons covering two or
  more buildings on Bing, Google and the RAPID drone orthomosaic, against
  6, 9 and 3 before, and found 84%, 81% and 99% of the buildings (68%, 63%
  and 95% before). The one RAPID building it did not match has a county
  outline shifted onto the neighboring carport; the detected outline
  follows the roof.
- **Square building outlines.** `FootprintRegularizer` (and
  `regularize_footprint` / `regularize_footprints` in
  `rapidtools.processing.postprocessing.footprints`) redraws mask-traced
  footprints with straight walls at right angles: each footprint is
  rasterized on a grid turned to its main wall direction (its own, its
  minimum rotated rectangle's, or the one its neighbors share), cleaned of
  bulges and dents narrower than about a meter, traced back, and cleared
  of short steps; near-rectangles become rectangles, round buildings are
  only simplified, and neighbors are made non-overlapping. It needs no
  model. On the Eaton sample the median footprint went from 150 to 10
  vertices (8 on the drone orthomosaic, from about 2,000), with 98-100% of
  the corners square, as in the county footprints.
- **Building scans need no settings and come out regularized.**
  `SAM3OrthoFeatureExtractor` recognizes prompts that name buildings
  (`building`, `house`, `structure`, also inside longer words; see
  `is_building_prompt`) and gives them the building scan for every setting
  left out: one footprint per building, tiles overlapping by half, masks
  cut at 1 m necks, buildings that fewer than 30% of the tiles seeing them
  detected dropped. The new `regularize` option, on by default for these
  prompts, gives the footprints straight, square walls with
  `FootprintRegularizer`; `regularize=False` returns the raw outlines.
  `scan_settings()` shows the settings a scan will use. Settings left out
  follow the prompt, so an extractor reused for another prompt switches to
  that prompt's defaults.
- `SAM3Inference.run_inference(max_mask_side=...)` and
  `SAM3ImageSegmenter(max_mask_side=...)` return the masks of large images
  at a reduced size; SAM 3 predicts masks at about 288 pixels, so nothing
  is lost. `SAM3ImageSegmenter` also stores the confidences of the masks in
  the `sam3_scores` attribute.

- **Newer models in the catalogs.** Hosted: Gemini 4 (`gemini-4-neon`,
  `gemini-4-carbon`, `gemini-4-argon`, `gemini-4-hydrogen`), GPT-6.2 Astra
  and Sol, GPT-6.1 Sol and Terra, Claude Fable 5.2, Opus 5.5 and Sonnet 5.5,
  and Muse Wonder (`muse-wonder-1.0`, through `MuseSparkInference`). Open
  weights: `Qwen/Qwen3.8-Flash-Next`, `zai-org/GLM-4.6V-Flash`,
  `zai-org/GLM-5.3-Flash`, `mistralai/Mistral-Large-4-1T-A52B` and
  `meta-models/Muse-Spark-Open-816B-A42B`. Only GLM-4.6V-Flash fits a single
  24 GB GPU; the others need a multi-GPU server. Default models are
  unchanged, apart from the `hf` provider (see Changed).
- **Per-model defaults.** Hosted wrappers built without `max_tokens` use
  the default of their model: 2,048 as before, 32,768 for Gemini 4, whose
  thinking counts against the limit. `set_model_defaults()` and
  `model_defaults()` in `rapidtools.models` change and show them. The
  `max_tokens` of the `analyze` job step now defaults to `null` (the
  model's default), and the GUI field follows the selected model until
  you type a limit.
- **Buildings from one mask per house.** `MapillaryFeatureExtractor` can
  place buildings from the masks of a segmentation model instead of
  Mapillary's building surface:
  `detection_source={'buildings': 'sam3'}, sam3_prompts={'buildings': 'house'}`.
  Masks in different frames are linked into one building by the
  reconstructed points they share
  (`localization_method='instances'`, chosen automatically for buildings
  detected this way), which keeps neighbors apart where a fence or a shed
  joins them in Mapillary's outline. On the 1 km Spokane neighborhood, of
  324 first-row houses: 321 found (316 without), 7 structures spanning two
  or more houses (12 without), median overlap with the mapped footprint
  0.61 (0.55 without); 17 houses come out in two pieces (14 without). It
  downloads a thumbnail of every frame and runs the model on each (about
  0.4 s per frame on one GPU), so it is opt-in. The linking lives in
  `rapidtools.processing.street_instances` and takes outlines from any
  source; `sam3_model` accepts any model with SAM 3's `run_inference`.
  `detection_source` now also takes a mapping from class to source, and
  `sam3_prompts` the text asked per class. A model often outlines the
  wings of a large house separately; they are rejoined when their outlines
  touch in the frames in front of them, or, with `join_wings_by='prompt'`,
  when SAM 3 asked for the whole object at a few points on one wing returns
  a mask that covers the other (new `SAM3PointInference` in
  `rapidtools.models.sam3`). The prompt test wrongly joined 2 of 84
  neighbor pairs where touching joined 13, and brought the structures
  spanning two or more houses down from 7 to 3, for about 1,700 further
  model calls on that neighborhood.
- **Buildings matched to known footprints.** Give
  `MapillaryFeatureExtractor` a footprint layer (`footprints=`: a
  `PhysicalAssetCollection`, a GeoJSON file or polygons) and buildings are
  no longer discovered: each footprint is projected into the frames around
  it and the outlines found there are its sightings
  (`localization_method='footprints'`, automatic when `footprints` is
  given). Buildings come back as the footprints themselves, with their own
  identifiers and attributes plus `frames_expected` and `frames_seen`, so
  nothing is split or merged and a footprint that should be visible but is
  not stands out. It needs no point cloud. On the 1 km Spokane
  neighborhood, of 324 first-row houses: all 324 matched with SAM 3 masks,
  323 with Mapillary's own outlines and no image download. Four more
  first-row footprints there are lots whose houses burned before the
  survey; SAM 3 on every frame reported 2 of them, each seen in three
  frames where every standing house was seen in 16 or more, and SAM 3 on
  every fifth frame reported none. Buildings the layer
  lacks are still placed and marked `unmapped` (`report_unmapped`): with a
  model, the masks on no footprint; with Mapillary's outlines, the point
  cloud structures that lie off every footprint. With a tenth of the
  first-row houses taken out of the layer, 99% of them came back with SAM 3
  and 91% with Mapillary's outlines. The matching lives in
  `rapidtools.processing.street_footprints`.

### Fixed

- **Out of memory regularizing drone crops.** `BuildingRegularizer` kept
  the masks of every crop at full resolution until the end; on the 1.9 cm
  Eaton orthomosaic (432 crops of about 2,900 pixels) that exhausted a
  32 GB machine. Masks are now kept at no more than 1,024 pixels a side and
  released as soon as they are traced.
- **SAM 3 no longer falls back to the CPU without a word.** With
  `device='auto'` the loader puts the weights wherever there is room; when
  the GPU was full, or unusable after a driver reset, that was the CPU,
  where SAM 3 is dozens of times slower and kept every core busy for hours
  (in the GUI, the building refinement loaded a second copy of SAM 3 next
  to the scan's and could end up there). The fallback stays, but
  `SAM3Inference` and `SAM3PointInference` now log a warning that says what
  happened when a GPU is present and the weights did not land on it, and
  then use at most half of the CPU cores so the computer stays usable;
  `device='cpu'` and machines without a GPU run on the CPU as before.
- **Scans no longer skip failed batches silently.** When SAM 3 failed on a
  batch of tiles, `SAM3OrthoFeatureExtractor` went on without those tiles.
  It now warns with the number of failed batches, stops after three failed
  batches in a row (a GPU that stopped responding fails every batch that
  follows), and raises when every batch failed instead of reporting that
  nothing was found.
- **Gemini answers cut off by the output limit.** A response that spent
  its whole limit on thinking used to come back as an empty success; it
  is now a failure that says to raise `max_tokens`, and a partly written
  answer logs a truncation warning.

### Changed

- **`BuildingRegularizer` keeps neighbors apart.** It used to dissolve
  every pair of masks that overlapped by half (so chains of overlaps
  merged whole rows), bridge gaps of up to 15 ft inside a mask, and cut
  overlapping buildings along the extension of a wall, which drew long
  straight lines through neighbors. It now fuses the views of each
  building from its own and its neighbors' crops like the tiled scan does
  (see Added), keeps the buildings that match a preliminary footprint (a
  footprint that covered two buildings yields both), regularizes the
  outlines with `FootprintRegularizer` and removes the overlaps. Outputs
  carry `confidence`, `views` and `source_ids` (the matched preliminary
  footprints). New options: `regularize_outlines`, `neck_width_m`,
  `min_area_m2` and `max_mask_side`.
- **Building scans in the GUI and job files.** The `detect.aerial` step
  scans prompts that name buildings (`building`, `house`, `structure`)
  object by object whatever `merge_overlaps` says, from tiles that overlap
  by at least half, with masks cut at 1 m necks and buildings that fewer
  than 30% of the tiles seeing them detected dropped, and always squares
  their walls. Its `regularize` option (the *Refine building footprints*
  box in the GUI, on by default) now decides whether the footprints are
  also refined on close-up crops with `BuildingRegularizer`, and whether
  roads are regularized. The refinement now reuses the scan's SAM 3 instead
  of loading a second copy (`BuildingRegularizer(model=...)` and
  `SAM3ImageSegmenter(model=...)` accept a loaded model) and segments four
  crops at a time instead of eight, so it fits an 8 GB GPU next to the
  display. Other prompts are unchanged.
- **New defaults for building prompts in `SAM3OrthoFeatureExtractor`.**
  `overlap_ratio`, `merge_overlaps`, `neck_width_m` and
  `min_detection_rate` now default to `None`, meaning the prompt's defaults
  (see Added); for prompts that do not name buildings these are the old
  values. A building scan therefore keeps neighbors apart and squares the
  walls without being asked, and with tiles overlapping by half it runs
  about three times as many tiles as before. Pass
  `merge_overlaps=True, overlap_ratio=0.15, regularize=False` for the old
  behavior.
- `SAM3OrthoFeatureExtractor` downsamples tiles larger than 1,024 pixels
  before inference (SAM 3 works at 1,008 pixels) and writes its temporary
  tiles at JPEG quality 95. Scanning the 1.9 cm Eaton orthomosaic took
  2.4 s per tile instead of 3.2 s on a GTX 1080.
- **New default for other open models.** `load('hf')`, `HFVisionInference()`
  and *Other open models* in the GUI now default to `zai-org/GLM-4.6V-Flash`
  (10B) instead of `Qwen/Qwen3.5-4B`. Its weights are about 20 GB; pass
  `load_in_4bit=True` on GPUs with less than 24 GB (the GUI does so by
  default). Pass `model_id='Qwen/Qwen3.5-4B'` to keep the old model.
- **Reasoning is kept out of local answers.** Models that write a
  `<think>...</think>` block before answering, GLM-4.6V among them, used to
  return it as part of the answer. `HFVisionInference` and the wrappers
  built on it now return only the answer in `text` (the full output stays
  in `raw_response`), and report a failure that says to raise `max_tokens`
  when the limit is reached before the answer.
- **Buildings from street level are found more often, and whole.** Changes
  to the point-cloud method behind
  `MapillaryFeatureExtractor(classes=['buildings'])`:
  - A house whose garage and entrance stand at different setbacks used to
    come out as two structures. Wall pieces now join when they continue each
    other along the street and every frame in front of them outlines them as
    one building; neighbors stay apart because a frame facing the side yard
    outlines each separately.
  - A frame votes against a reconstructed point only when it can see it, so
    houses behind street trees are no longer voted away.
  - Heights are measured above the ground read from the point cloud, not
    above the road, so houses on lots below the street keep their walls and
    fences on lots above it are no longer taken for walls. `height_m` is the
    height above that ground.
  - Footprints follow the setbacks of the facade
    (`structure_footprint='stepped'`, the new default) instead of one
    rectangle; `'rectangle'` and `'observed'` remain. The depth is taken
    from the bulk of the points behind the wall, not the deepest few.
  - A structure needs 40 reconstructed points instead of 80.

  On a 1 km neighborhood of the Spokane survey, checked against mapped
  footprints: of the 324 houses in the first row along the street, those
  found rose from 254 to 316; houses reported as two structures fell from
  62 to 14; the median overlap with the mapped footprint rose from 0.50 to
  0.55. Structures spanning two or more neighbors went from 10 to 12,
  mostly where a fence between houses is segmented as building.

### Deprecated

- `BuildingRegularizer(max_gap_bridge_ft=..., snap_tolerance_ft=...)`:
  both are ignored and warn. Bridging the gaps inside a mask is what merged
  neighboring buildings; `snap_tolerance_ft` was never used.

## [0.3.0] - 2026-11-25

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
  `frontage_m`, `depth_m` and `height_m`. On a 1 km neighborhood of Spokane,
  78% of the 346 OpenStreetMap footprints near the route were found, with a
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
  a misspelled option, or the pipeline step at fault.
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
  `ghcr.io/rapid-facility/rapidtools-gui` for every release (the CUDA image
  builds on PyTorch 2.15 with CUDA 13.2 and needs NVIDIA driver 580 or newer
  on the host), and
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
- **US English throughout.** Documentation, messages and the interface use
  US spelling. A job or step that was stopped has the status `canceled`.

### Deprecated

- `OperationCancelled`, `raise_if_cancelled`, `is_cancelled` and
  `rapidtools.gui.WorkflowCancelled` are now `OperationCanceled`,
  `raise_if_canceled`, `is_canceled` and `WorkflowCanceled`. The old names
  still work and warn.

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
  neighboring footprints and can check Mapillary's semantic labels before
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

[0.3.0]: https://github.com/RAPID-Facility/rAPIdtools/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/RAPID-Facility/rAPIdtools/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/RAPID-Facility/rAPIdtools/releases/tag/v0.1.2
