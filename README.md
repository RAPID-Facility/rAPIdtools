# rAPIdtools

[![PyPI version](https://img.shields.io/pypi/v/rapidtools.svg)](https://pypi.org/project/rapidtools/)
[![Python](https://img.shields.io/pypi/pyversions/rapidtools.svg)](https://pypi.org/project/rapidtools/)
[![Downloads](https://static.pepy.tech/badge/rapidtools/month)](https://pepy.tech/project/rapidtools)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

[![Tests](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml/badge.svg)](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/bacetiner/c890ae687368838a74c5e442b9ff5b94/raw/coverage.json)](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml)
[![Docs](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/docs.yml/badge.svg?branch=main)](https://rapid-facility.github.io/rAPIdtools/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typing](https://img.shields.io/pypi/types/rapidtools)](https://pypi.org/project/rapidtools/)

A high-performance toolkit for performing large-scale AI inference and localization on post-disaster geospatial datasets.

## Overview

The UW RAPID Facility collects terabytes of perishable, hyper-resolution data in the aftermath of natural disasters. As remote sensing technology has evolved, the core challenge in natural hazards engineering has shifted: the primary bottleneck is no longer data collection, but turning that raw data into actionable intelligence.

`rapidtools` is a high-performance Python package designed to eliminate this bottleneck. It delivers a seamless, object-oriented pipeline that connects raw spatial datasets with state-of-the-art Large Vision-Language Models (VLMs). Whether analyzing ten damaged homes or one hundred thousand regional assets, `rapidtools` equips researchers with the tools to automate complex feature extraction, pinpoint structural damage, and unlock engineering-grade insights at unprecedented speed.

## High-Level Impact & Key Features

**Large-scale geospatial ingestion**
Seamlessly fuse massive local orthomosaics, regional shapefiles, and street-view vector tiles. The `PhysicalAssetCollection` engine provides fast lookups, patial filtering, and native conversions between GeoJSON, ESRI Shapefiles, and Pandas DataFrames.

**Scalable AI Inference (Local & Cloud)**
Run deployments tailored to your resources. Deploy powerful local vision-language models directly on consumer hardware: Google's Gemma-4 (E2B to 31B), Meta's Llama 4 and Llama 3.2 Vision, Meta's open-weight Muse Glimmer, Alibaba's Qwen3.8 / Qwen3.6 / Qwen3.5 / Qwen3-VL, or any Hugging Face checkpoint (Gemma 3, LLaVA, InternVL, Pixtral, Granite Vision, ...) through `HFVisionAssetAnalyzer`, using dynamic batching, automated tensor precision scaling, and strict VRAM garbage collection to prevent Out-Of-Memory (OOM) crashes. Alternatively, scale instantly using built-in integrations for enterprise APIs (OpenAI GPT-5.x/GPT-6, Google Gemini 3.x, Anthropic Claude Opus 5 / Fable 5.1, Meta Muse Spark, Alibaba Qwen via Model Studio), which feature thread-safe global cooldowns and exponential backoff to handle rate limits automatically.

**Keyless Aerial and Street-Level Imagery**
Pull imagery for any footprint without provider API keys. `GoogleAerialImageExtractor` and `BingAerialImageExtractor` stitch satellite tiles for GeoJSON polygons (or yield raw tiles for ML pipelines), `GoogleOrthomosaicExtractor` / `BingOrthomosaicExtractor` synthesize georeferenced GeoTIFFs for a region, and `GoogleStreetViewImageExtractor` locates the nearest Google Street View panorama for each asset, downloads it, crops the field of view that covers the asset footprint (optionally with the decoded depth map) and attaches the result to the asset. Mapillary street-level imagery remains available through `MapillaryImageExtractor`.

**Street-Level Object Discovery**
Build an inventory from the survey itself. `MapillaryFeatureExtractor` reads Mapillary's segmentation detections as metadata (no image downloads), drops the survey vehicle, places every sighting from the camera geometry and triangulates objects seen from several positions; `MapillaryObjectImageExtractor` then crops the closest views of each object for a vision-language model. Classes outside Mapillary's vocabulary fall back to SAM 3 on thumbnails.

**Intelligent Feature Regularization**
Move beyond raw AI pixel masks. The toolkit includes sophisticated geometric regularizers that instantly translate semantic segmentations into usable, GIS-ready asset geometries.

**Advanced Line-of-Sight Localization**
Automate the extraction of the perfect viewing angle. Using KD-Trees, STRtrees, and ray-casting math, `rapidtools` can dynamically calculate asset principal axes and cull occluded perspectives (e.g., ignoring images where a target building is blocked by a neighboring structure) to guarantee your AI only analyzes the right data.

## Installation

You can install the latest stable release directly via pip:

```bash
pip install rapidtools
```

## Documentation

The published documentation lives at [rapid-facility.github.io/rAPIdtools](https://rapid-facility.github.io/rAPIdtools/).
It is built with Sphinx and the Book theme. Install the docs
extra once, then build:

```bash
pip install -e ".[docs]"
cd docs
make html
```

Open `docs/build/html/index.html` in your browser for the user guide, the
worked examples and the API reference (one page per class, generated from
the docstrings).

## Project Structure

Designed for flexibility and scale, `rapidtools` utilizes a cleanly decoupled architecture that makes extending workflows and managing complex data pipelines effortless:

* `rapidtools.core`: Domain models representing your data (`PhysicalAsset`, `PhysicalAssetCollection`, `ImageAsset`, `BoundingBox`).
* `rapidtools.data_sources`: Clients for fetching raw data from external APIs and massive local files (e.g., `MapillaryClient`, `GoogleStreetViewClient`, `OrthomosaicReader`, `BingAerialImageExtractor`, `GoogleAerialImageExtractor`).
* `rapidtools.models`: Base wrappers and handlers for executing ML models natively or via cloud APIs (`Gemma4Inference`, `SAM3Inference`, `GeminiInference`, `MuseSparkInference`, `QwenInference`, ...). See the table below.
* `rapidtools.processing`: High-level workflow components (Extractors, Segmenters, Analyzers, and Regularizers) designed to snap together effortlessly into the `Pipeline` engine.

## The rAPIdtools API in Five Lines

Every model is created through one factory and every analysis step is the same class, so switching providers is a one-word change:

```python
import rapidtools as rt

rt.configure_logging()                                   # opt in to progress output
buildings = rt.PhysicalAssetCollection.from_geojson('buildings.geojson')
model = rt.models.load('gemini', api_key='AIza...')      # or 'claude', 'openai', 'muse_spark', 'qwen',
                                                         #    'gemma4', 'llama', 'muse_glimmer', 'qwen_vl', 'hf'
pipeline = rt.Pipeline([
    rt.AerialImageryExtractor('ortho.tif', save_directory='crops', buffer_m=20),
    rt.AssetAnalyzer(model, prompt='Rate the damage 0-5 as JSON.'),
])
buildings = pipeline.run(buildings)
```

* `rapidtools.models.load(provider, **kwargs)` builds any wrapper from its registry key; `list_providers()`, `catalog(provider)` and `PROVIDERS` describe them without importing PyTorch.
* `AssetAnalyzer(model, prompt, ...)` replaces the provider-specific `XAssetAnalyzer` classes (still available, deprecated). Hosted models run threaded with a `RateLimitPolicy` (shared cooldown after failures, then retry passes for the assets that failed, so timeouts and 429s do not leave holes); local models run sequentially or in batches. Hosted wrappers take `timeout=` (default 60 s) for slow responses. Pass `generation=GenerationConfig(json_mode=True, temperature=0.0)` for per-call options.
* `AerialImageryExtractor` judges imagery coverage from the raster's mask (nodata value, alpha band or internal mask). By default a crop is skipped when more than `max_missing_data_ratio` of it is empty; pass `min_footprint_coverage=0.3` to keep every building whose footprint is at least 30% imaged, with edge crops padded so they keep their size.
* `SAM3OrthoFeatureExtractor(prompt, ...)` scans a whole orthomosaic for a text concept. By default touching detections dissolve into one polygon (roofs, roads, vegetation); pass `merge_overlaps=False` for countable objects such as vehicles to keep one asset per detected instance, with its SAM 3 `confidence`, and only remove cross-tile duplicates.
* Pipeline steps declare a `Stage` (`DETECT`, `EXTRACT_IMAGERY`, `REGULARIZE`, `SEGMENT`, `ANALYZE`, `EXPORT`), so steps can be added in any order.
* `import rapidtools` is side-effect free and fast (about 2 s); model and processing classes load on first use. Call `rapidtools.login()` to authenticate with the Hugging Face Hub for gated weights, or let local model wrappers do it when they load.

## Quick Start: Aerial Damage Detection Pipeline

Run state-of-the-art damage assessments completely offline. This example demonstrates how to download `rapidtools` sample datasets, extract building-specific image patches from a local drone orthomosaic, and analyze them using a local Gemma-4 vision model that does not require paid API usage or cloud tokens.

```python
from pathlib import Path
from rapidtools import (
    AerialImageryExtractor,
    Gemma4AssetAnalyzer,
    PhysicalAssetCollection,
    Pipeline,
    download_dataset,
)

# 1. Download required example datasets from the rapidtools registry
raster_path, footprint_path, prompt_path = download_dataset([
    'eaton_patch2',
    'altadena_sample_buildings',
    'aerial_chs_prompts'
])

image_save_dir = Path('eaton_fire_aerial_feb25/overlaid_imagery')

# 2. Load the regional building footprints
building_data = PhysicalAssetCollection.from_geojson(footprint_path)

# 3. Configure the Extractor
# Crops the orthomosaic around each asset and draws a reference outline.
# The outline can trace the footprint itself (default), its bounding box,
# rotated bounding box, convex hull or just corner brackets, optionally
# pushed away from the asset by a buffer, with any stroke width and colour.
extractor = AerialImageryExtractor(
    dataset=raster_path,
    save_directory=image_save_dir,
    overlay_asset_outline=True,
    outline_shape='rotated_bbox',
    outline_buffer='2 m',
    outline_width='1%',
    image_prefix='eaton_trinity_25',
    keep_multiple_copies=True,
)

# 4. Configure the AI Analyzer
# Ingests the newly cropped images and applies the configured prompt to evaluate damage
analyzer = Gemma4AssetAnalyzer(
    model_id='google/gemma-4-E2B-it',
    prompt=prompt_path,
    batch_size=8
)

# 5. Build and execute the pipeline
pipeline = Pipeline()
pipeline.add_step(extractor)
pipeline.add_step(analyzer)

print('Initiating processing pipeline...')
processed_collection = pipeline.run(building_data)

# Clean up empty assets and export the AI-enriched dataset for GIS mapping
final_collection = processed_collection.filter_empty()
print(f'Final inventory size: {len(final_collection)} assets processed.')

final_collection.to_geojson(
    'eaton_footprints_CHS_with_gemma4.geojson', 
    ignore_properties=['image_assets']
)
```

## Graphical Interface: Detect Assets and Run Inference Without Code

`rapidtools` ships with a local web application that wraps the end-to-end detect / extract / analyze workflow, for aerial imagery and for street-level surveys. It runs on the Python standard library (no extra dependencies) and opens in your browser.

```bash
rapidtools-gui            # console script installed with the package
python -m rapidtools.gui  # equivalent; add --port/--output-dir/--no-browser as needed
```

Or from Python:

```python
from rapidtools.gui import launch_asset_analysis_app

launch_asset_analysis_app()
```

The app walks through three steps while the aerial image stays on screen and is updated with every result:

1. **Imagery**: find the place on the map (satellite basemap, RAPID survey coverage, address search) and draw a box, then stitch a Bing or Google basemap over it (no API key), load a GeoTIFF from your computer, download a UW RAPID sample, or (coming soon) fetch it from TACC. Street-level discovery can start from the box alone.
2. **Detect & analyze**: find the assets, choose where their imagery comes from, and pick the model and prompt.
   - *Assets*: detect them in the image with SAM 3 (on a Bing or Google basemap, or the image itself, with an option to keep every instance of countable objects such as vehicles separate), **discover objects along a Mapillary street survey** (`vehicles, utility poles, fire hydrants` are located from Mapillary's own detections within the image extent, with the survey vehicle removed and sightings triangulated), or load a file with their locations.
   - *Imagery for analysis*: aerial crops from the loaded image (with edge handling), Google Street View panoramas cropped to each footprint (no key), Mapillary panoramas of each footprint, or the closest Mapillary views of each discovered object. The asset can be marked on every crop with an outline, bounding box or corner brackets.
   - *Model*: every analysis model in `rapidtools` is available: Google Gemini, Anthropic Claude, OpenAI, Meta Muse Spark, and Qwen through their APIs; Gemma-4, Llama Vision, Muse Glimmer, and Qwen locally; and, under *Other open models*, any Hugging Face vision-language checkpoint with optional 4-bit loading. *Fetch models* lists the model IDs your API key can access. Temperature, output length and JSON mode live under *Advanced options*.
   - *Prompt*: load one of the sample prompts (aerial CHS, street-level CHS, street-level recovery), a file, or open the **prompt builder**. The builder walks through task and role, output fields (which become the asset attributes), visual indicators for every class, analysis steps and edge cases, and assembles a prompt in the layout of the sample prompts. Its **assistant** uses any of the same model backends (the analysis model, or a local Gemma-4 / Qwen) to draft the whole specification from a one-line description, refine it on request, suggest indicators for a class, review the finished prompt, or import an existing prompt into the builder.
3. **Results**: hover over assets to inspect their attributes, click one to see the exact image crops the model analyzed, colour the overlay by any inferred attribute, browse the attribute table, and download the GeoJSON outputs.

### Sharing the app with colleagues

Run the server on the machine that has the GPU and the imagery, bind it to the network, and protect it with an access token:

```bash
rapidtools-gui --host 0.0.0.0 --token auto --data-root /data/eaton --no-browser
```

The terminal prints a share link of the form `http://<host>:8765/?token=...`. Anyone opening that link (or entering the token on the sign-in page) can use the app; everyone else gets the sign-in page. `--data-root` limits the file browser and every file path to one directory, and `RAPIDTOOLS_GUI_TOKEN` can supply the token instead of the flag. Everyone connected shares one workspace and one job queue, so this suits a small team taking turns rather than many simultaneous users. Keep the server on your institution's network or a VPN; the token protects the app, not the transport, which is plain HTTP.

The image panel streams the original pixels when you zoom in (toggle *Full resolution* to fall back to the lightweight preview), so multi-gigabyte orthomosaics can be inspected at native resolution without loading them into the browser.

Bounding boxes can be drawn on a satellite map that also shows where the RAPID street survey has images. Progress and log output stream into the page while the heavy lifting runs on a background thread, and **Cancel** stops detection or inference at the next tile or batch, keeping any results produced so far. Long runs offer to send an email or webhook message (or a browser alert) when they finish, and the tab can be closed and reopened in the meantime. The same workflow is available headlessly through `rapidtools.gui.AssetAnalysisWorkflow` for scripting, and the prompt builder through `rapidtools.gui.PromptSpec`, `assemble_prompt` and `PromptAssistant`.

## Supported Models

Every wrapper exposes `list_known_models()` (the curated catalogue below, no network) and `list_available_models(api_key)` (live listing from the provider where one exists). Pass any listed ID as `model_id`; the default is marked in bold.

| Provider / family | Wrapper · Analyzer | Key (env var) | Models |
|---|---|---|---|
| Google Gemini (API) | `GeminiInference` · `GeminiAssetAnalyzer` | `GOOGLE_API_KEY` | **gemini-3.8-flash**, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash, gemini-3.5-flash-lite (analyzer default), gemini-3.1-pro-preview, gemini-3.1-flash-lite, gemini-3.1-flash-image, gemini-3.1-flash-lite-image, gemini-3-flash-preview, gemini-3-pro-image, gemini-omni-1.1-flash |
| Anthropic Claude (API) | `ClaudeInference` · `ClaudeAssetAnalyzer` | `ANTHROPIC_API_KEY` | claude-fable-5-1, claude-fable-5, **claude-opus-5**, claude-opus-4-8, claude-opus-4-7, claude-opus-4-6, claude-sonnet-5, claude-sonnet-4-6, claude-haiku-4-5 |
| OpenAI (API) | `OpenAIInference` · `OpenAIAssetAnalyzer` | `OPENAI_API_KEY` | gpt-6-astra, gpt-6-sol, gpt-6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, **gpt-5.5**, gpt-5.5-pro, gpt-5.4, gpt-5.4-pro, gpt-5.4-mini, gpt-5.4-nano, gpt-5.2, gpt-5.1, o3, o3-pro, gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini |
| Meta Muse Spark (API) | `MuseSparkInference` · `MuseSparkAssetAnalyzer` | `MODEL_API_KEY` | **muse-spark-1.3**, muse-spark-1.3-contributor, muse-spark-1.2, muse-spark-1.2-contributor, muse-spark-1.1 |
| Alibaba Qwen (API, Model Studio) | `QwenInference` · `QwenAssetAnalyzer` | `DASHSCOPE_API_KEY` | **qwen3.8-max**, qwen3.8-omni-flash, qwen3.7-plus, qwen3.6-plus, qwen3.5-plus, qwen3-vl-plus, qwen3-vl-flash, qwen-vl-max, qwen-vl-plus |
| Google Gemma 4 (local) | `Gemma4Inference` · `Gemma4AssetAnalyzer` | — | google/gemma-4-31B-it, google/gemma-4-26B-A4B-it, google/gemma-4-12B-it, google/gemma-4-E4B-it, **google/gemma-4-E2B-it** |
| Meta Llama (local) | `LlamaVisionInference` · `LlamaVisionAssetAnalyzer` | — (gated download) | meta-llama/Llama-4-Scout-17B-16E-Instruct, meta-llama/Llama-4-Maverick-17B-128E-Instruct, **meta-llama/Llama-3.2-11B-Vision-Instruct**, meta-llama/Llama-3.2-90B-Vision-Instruct |
| Meta Muse Glimmer (local) | `MuseGlimmerInference` · `MuseGlimmerAssetAnalyzer` | — | **meta-models/Muse-Glimmer-30B** (4-bit fits a 24 GB GPU; needs `transformers>=5.15`) |
| Alibaba Qwen (local) | `QwenVisionInference` · `QwenVisionAssetAnalyzer` | — | Qwen/Qwen3.8-27B, Qwen/Qwen3.6-27B, Qwen/Qwen3.6-35B-A3B, Qwen/Qwen3.5-{397B-A17B, 122B-A10B, 35B-A3B, 27B, 9B, **4B**, 2B, 0.8B}, Qwen/Qwen3-VL-{235B-A22B, 32B, 30B-A3B, 8B, 4B, 2B}-Instruct, Qwen/Qwen2.5-VL-{72B, 32B, 7B, 3B}-Instruct, Qwen/Qwen2-VL-{7B, 2B}-Instruct |
| Any Hugging Face VLM (local) | `HFVisionInference` · `HFVisionAssetAnalyzer` | — | Curated list (`OPEN_VLM_CATALOG`) plus any repo with a chat template; default **Qwen/Qwen3.5-4B** |
| Meta SAM 3 (local segmentation) | `SAM3Inference` · `SAM3ImageSegmenter`, `SAM3OrthoFeatureExtractor` | — | **facebook/sam3** |

```python
from rapidtools import MuseSparkAssetAnalyzer, QwenVisionAssetAnalyzer

cloud = MuseSparkAssetAnalyzer(api_key='...', prompt='Return JSON with a damage_level key.')
local = QwenVisionAssetAnalyzer(prompt='Rate the fire damage 0-5.', model_id='Qwen/Qwen3.5-9B', load_in_4bit=True)
```

## Citing rAPIdtools

If rAPIdtools contributes to your research, please cite it. GitHub's
*Cite this repository* button (from `CITATION.cff`) gives APA and BibTeX
entries; each release is also archived on Zenodo with a DOI.

```bibtex
@software{rapidtools,
  author  = {Cetiner, Barbaros and {NSF NHERI RAPID Facility}},
  title   = {rAPIdtools: AI inference and localization for post-disaster geospatial datasets},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/RAPID-Facility/rAPIdtools}
}
```

rAPIdtools is developed at the NSF NHERI RAPID Facility, University of
Washington, supported by the U.S. National Science Foundation.

## License

This project is licensed under the BSD-3-Clause License. See the `LICENSE` file for details.
