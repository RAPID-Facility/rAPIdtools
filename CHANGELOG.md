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
- A *Development and testing* page in the documentation (and the matching
  wiki section) describing the development setup, the lint, type and test
  gates, how the test suite is organised and how it stays offline. Test
  coverage of the package rose from 97% to 99%.
- **Street-level localisation rebuilt around tracking and triangulation.**
  `MapillaryFeatureExtractor` now links detections frame to frame by bearing
  (new module `rapidtools.processing.street_tracking`), triangulates each
  track robustly, drops moving vehicles (rays that do not agree,
  ground-contact ranges that contradict the intersection, or objects on the
  line the camera itself drove) and merges tracks and single views with a
  covariance-aware gate, instead of clustering noisy single-view ground
  positions by a fixed radius, which scattered detections and split one car
  into several. `localization_method` selects `'tracks'` (default), the
  `'voting'` ray-accumulation baseline or the previous `'cluster'`
  behaviour. Objects gain `position_sigma_m`, `parallax_deg` and `n_images`.
- Full camera pose for street-level geometry: frames carry Mapillary's
  `computed_rotation`, camera type and lens parameters, and
  `street_localization` turns polygons into world rays through spherical,
  perspective and fisheye camera models (`rotation_matrix`, `pixel_bearing`,
  `world_ray`, `observation_angles`), so camera pitch and roll no longer
  distort bearings and ranges. Frames without a rotation keep the
  level-camera fallback.
- `MapillaryClient.fetch_map_features()`, `MAP_FEATURE_VALUES` and
  `is_map_feature_value()`. In `detection_source='auto'`, static point
  classes (utility poles, hydrants, street lights, traffic signs, ...) now
  come from Mapillary's pre-triangulated map features
  (`localization='map_feature'`, with `aligned_direction`, `first_seen` and
  `last_seen`); `detection_source='map_features'` uses them exclusively and
  `'mapillary'` keeps everything on the image route.
- Street-level duplicates from split outlines and tracker slips are folded
  into the vehicle they show. A car cut in two by a pole or into a front and
  a rear, or a track that slid between vehicles lined up along the line of
  sight, used to leave a second object a few metres from the real one that
  no merge could touch because the two shared frames. `merge_pieces` in
  `rapidtools.processing.street_tracking` (on by default through
  `MapillaryFeatureExtractor(merge_pieces_of_neighbours=True)`) judges such
  pairs on what their sightings show instead and keeps the better-located
  object's position. An object never seen within 20 m is not trusted on its
  own whatever its parallax (ranges read from that far are off by a third,
  and a low-parallax intersection lands metres from where the sightings
  point): its closest sightings decide, and it joins the close pass they
  show, and what is left of it is dropped when its closest rays point at
  an object within 15 m that was seen up close, unless it is well
  triangulated regardless (parallax of 45 degrees or more, uncertainty of
  half a metre or less; `close_range_m`, 0 to keep everything). A far-only
  object whose rays point elsewhere, a car in a driveway behind the kerb,
  stays. An object never seen with an outline at least
  `min_sighting_deg` wide or tall (6 degrees) was never more than a speck
  and is not reported, and single views are no longer reported unless
  `report_single_views=True`. `MapillaryObjectImageExtractor(view_selection='widest')`
  keeps the widest views of each object instead of the closest, which is what
  `DetectionVerifier` should judge (on a labelled block it then rejected all
  junk and kept all vehicles, where the closest views let a bush through), and
  `DetectionVerifier(reject_unverified=True)` removes what the model gave no
  verdict for. `DuplicateResolver`, a `VERIFY` step to run after the
  verifier, shows a model the crops of every pair of objects within 8 m as
  two rows, tells it the viewpoints differ, and merges the pairs it is sure
  are one vehicle: the long vehicle seen as cab and box, the track split by
  a corner, the two passes geometry could not join. Crops are zoomed to
  their outline, which `MapillaryObjectImageExtractor` now records in crop
  pixels (`outline`), and the model names each vehicle's type and colour
  before answering. On the labelled block the full chain went from 81 %
  precision and 97 % recall to 95 % and 92 %, with no duplicates left. An abeam-first mode (`abeam_window_deg`,
  off by default) builds objects from the passing frames only and attaches
  the frames looking down the road afterwards; it loses real vehicles until
  the track classification is reworked for passes alone, and is kept as an
  option.
- `rapidtools.processing.reid`: appearance embeddings from a Hugging Face
  vision backbone (the image tower of CLIP ViT-B/16 by default, DINOv2
  supported) and `MapillaryFeatureExtractor(reid=True)`, which merges
  look-alike objects from different sequences within `reid_max_distance_m`,
  removing the double counts of a repeated pass. The merge threshold
  defaults per backbone (`recommended_min_similarity`), crops are cut from
  thumbnails of their own size (`reid_image_size`, 2048 by default, no
  longer the SAM 3 size), and only objects with a sighting at least
  `reid_min_width_px` wide are compared (`views_for_reid`), since on a
  Spokane survey crops narrower than 100 pixels at 2048 scored as alike
  whether or not they showed the same vehicle while CLIP separated the
  wider ones best.
- Example `examples/street_detections_map.py`: writes a self-contained HTML
  map of street-level detections on a Bing aerial basemap, with each
  object's attributes, its crops tiled, a link to every source image on
  Mapillary, lines from the cameras to the object and the collection route
  (from the sightings, or the complete drive when a Mapillary token is
  given); filters by localisation type, image count and position
  uncertainty.
- `DetectionVerifier`, a `VERIFY`-stage pipeline step (between cropping and
  analysis) that shows each object's closest crops to any model from
  `rapidtools.models.load()` (or a custom classifier callable) and asks
  whether it is what it claims to be; rejects are dropped (or kept flagged
  with `keep_rejected=True`). The street vehicle example uses it to screen
  Mapillary's false detections before the condition analysis, and draws the
  detection's rotated bounding box as a thin red line on the crops. The
  question it asks is strict (a motor vehicle is not a trailer, boat or
  jet ski, nor a wheel on its own), and the model's estimate of how much of
  the object is visible rejects heavily hidden ones below
  `min_visible_fraction` (`verify_visible_fraction` is recorded).
- `suppress_duplicate_sightings` in `street_tracking`: Mapillary's nested
  and seam-split outlines of one object are dropped within each frame
  before tracking (`discover_objects` reports them as
  `'duplicates'`), so one car no longer becomes two objects that can never
  merge. Two estimates closer than `MIN_SEPARATION_M` (1.5 m) merge even
  when they share a frame, as only a split outline can produce that.
- `'vehicles'` now means motor vehicles (car, truck, bus, motorcycle).
  Trailers, caravans, boats and Mapillary's "other vehicle" are no longer
  included; ask for `'trailers'`, `'boats'` or `'all vehicles'`
  (`ALL_VEHICLE_LABELS`). `MapillaryFeatureExtractor(min_observations=...)`
  counts distinct frames rather than sightings, and the vehicle example
  requires three. The tracker's gate for a track without a reliable
  position is bounded by physics (the closest an object at least
  `min_object_width_m` wide could be, given its angular size, and the
  camera's displacement) instead of a flat 45 degrees, which stops a far
  speck from claiming a near car and localising it from the wrong frames.
- Weak triangulations (never seen closer than 25 m, parallax under 15
  degrees or uncertainty over 1 m) now join the well-located object their
  rays point at, within a fraction of their range along the line of sight,
  instead of being compared by position (`is_weak_estimate`,
  `merge_by_rays(weak_range_m, weak_along_fraction)`). A car triangulated
  from 40 m down the street with a tree in front landed metres from where
  the close pass put it and was reported twice. Both merge stages now pick
  the best candidate rather than the first rule that fires, two solid
  triangulations are fused only within half an object length (rays) or
  2.5 m (positions), and the viewpoint floor is 0.2 of the object size:
  in a crowded driveway a car's second-pass twin used to be fused with the
  neighbour parked alongside it, after which the car itself was lost.
- Negative evidence in street-level discovery. `MapillaryFeatureExtractor`
  keeps a light `CameraFrame` record of every survey frame (including frames
  with no detection) and `street_tracking.prune_unwitnessed` drops a weak
  estimate (`is_weak_estimate`: a single view, or a triangulation never seen
  closer than 25 m, with parallax under 15 degrees or uncertainty over 1 m)
  when same-day frames within `witness_radius_m` (12 m) had its position in
  view and hold no detection of the class along its bearing; votes are
  tallied per survey day so a car seen up close on one day is kept even if
  it had left by the next.
  Single-view positions whose nearest sighting is beyond
  `max_single_view_range_m` (30 m) are no longer reported: distant rows of
  parked cars seen from a cross street produced objects whose track hopped
  between neighbours and whose range was a guess. `discover_objects` counts
  both as `'far_single_view'` and `'unwitnessed'`. The verifier's question
  also rejects outlines so small or loose that they take in several objects.
- `MapillaryObjectImageExtractor` removes its own crop files from earlier
  runs for the assets it is about to crop, and the street detections map
  only tiles crops whose source image is among the asset's sightings. Asset
  numbers restart with every run, so a reused output folder used to show
  yesterday's vehicle next to today's under the same ID.
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

- `numpy` and `scipy` are declared dependencies (both were already required
  transitively).
- Street-level detections are no longer filtered by pixel area.
  `min_area_fraction` is deprecated (a tiny internal floor remains for
  decoding noise); object size is judged in metres once a track's distance
  is known (`min_object_width_m`, 1 m), `max_range_m` defaults to 60 m and
  bounds the triangulated position rather than discarding sightings, and
  the moving-vehicle check is occlusion-aware (`object_height_m`): a track
  whose ground-contact ranges read too long is kept when another detection
  stands in front of it or its outline is too short for a full object, and
  dropped as moving only when the object is fully visible. On a sample of
  Spokane frames the old filters discarded four in five vehicle sightings
  before localisation, which is why driveway cars went missing.
- Track classification is vectorised and samples at most 48 ray pairs per
  track, so a 150-frame track takes milliseconds instead of seconds; the
  range bound now applies to the final position, so a track seen mostly
  from far-away cameras is no longer mistaken for a moving vehicle.
- Street-level estimates are associated by their lines of sight before any
  position-based merge (`merge_by_rays`): a single-view fragment joins the
  triangulated object its bearings point at instead of being placed from
  unreliable long ranges, and the two halves of a long vehicle seen from
  both ends merge into one object (the merge gate also grows with the
  object's measured size). Tracking survives the fast bearing swings of
  objects close to the camera (45-degree gate, four-frame gaps), and
  single-view positions use only the nearest frames.
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
