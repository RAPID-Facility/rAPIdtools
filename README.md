# rAPIdtools

[![PyPI version](https://img.shields.io/pypi/v/rapidtools.svg)](https://pypi.org/project/rapidtools/)
[![Python](https://img.shields.io/pypi/pyversions/rapidtools.svg)](https://pypi.org/project/rapidtools/)
[![Downloads](https://static.pepy.tech/badge/rapidtools/month)](https://pepy.tech/project/rapidtools)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24542491.svg)](https://doi.org/10.5281/zenodo.24542491)<br>
[![Tests](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml/badge.svg)](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/bacetiner/c890ae687368838a74c5e442b9ff5b94/raw/coverage.json)](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml)
[![Docs](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/docs.yml/badge.svg?branch=main)](https://rapid-facility.github.io/rAPIdtools/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typing](https://img.shields.io/pypi/types/rapidtools)](https://pypi.org/project/rapidtools/)

A high-performance toolkit for performing large-scale AI inference and localization on post-disaster geospatial datasets.

## Overview

The UW RAPID Facility collects terabytes of perishable, hyper-resolution data in the aftermath of natural disasters. As remote sensing technology has evolved, the core challenge in natural hazards engineering has shifted: the primary bottleneck is no longer data collection, but turning that raw data into actionable intelligence.

`rapidtools` is a high-performance Python package designed to eliminate this bottleneck. It delivers a seamless, object-oriented pipeline that connects raw spatial datasets with state-of-the-art Large Vision-Language Models (VLMs). Whether analyzing ten damaged homes or one hundred thousand regional assets, `rapidtools` equips researchers with the tools to automate complex feature extraction, pinpoint structural damage, and unlock engineering-grade insights at unprecedented speed.

## Key Features

**Geospatial data in and out**
`PhysicalAssetCollection` holds the assets you are studying together with their geometry, attributes and images. It reads and writes GeoJSON and ESRI Shapefiles, converts to and from pandas DataFrames, and supports fast lookups and spatial filtering. Local orthomosaics, regional footprint files and street-level vector tiles all feed into it.

**Local and hosted models, same code**
Run a vision-language model on your own hardware or call one through an API. Local options are Google's Gemma 4 (E2B to 31B), Meta's Llama 4 and Llama 3.2 Vision, Meta's Muse Glimmer, Alibaba's Qwen 3.x and Qwen3-VL, and any Hugging Face checkpoint with a chat template (Gemma 3, LLaVA, InternVL, Pixtral, Granite Vision, ...). Local runs batch the images, choose tensor precision for the hardware and release GPU memory between batches, which keeps long jobs from running out of memory. Hosted options are Google Gemini, Anthropic Claude, OpenAI, Meta Muse Spark and Alibaba Qwen. Hosted runs are threaded, pause together after a rate-limit error and go back for the assets that failed. The full list is under [Supported Models](#supported-models).

**Aerial and street-level imagery without API keys**
`GoogleAerialImageExtractor` and `BingAerialImageExtractor` stitch satellite tiles for GeoJSON polygons, or yield the raw tiles for ML pipelines. `GoogleOrthomosaicExtractor` and `BingOrthomosaicExtractor` build a georeferenced GeoTIFF for a region. `GoogleStreetViewImageExtractor` finds the nearest Google Street View panorama for each asset, downloads it, crops the field of view that covers the footprint (optionally with the decoded depth map) and attaches the result to the asset. Mapillary street-level imagery is available through `MapillaryImageExtractor`, with a Mapillary access token.

**Object discovery from street-level surveys**
An inventory can come from the survey itself. `MapillaryFeatureExtractor` reads Mapillary's segmentation detections as metadata (no image downloads), drops the survey vehicle, places every sighting from the camera geometry and triangulates objects seen from several positions. `MapillaryObjectImageExtractor` then crops the closest views of each object for a vision-language model. Classes outside Mapillary's vocabulary fall back to SAM 3 on thumbnails.

**Cleaner footprints from segmentation masks**
Raw masks are jagged, sometimes merge two neighbors and occasionally invent a building. `FootprintRegularizer` redraws footprints with straight walls at right angles and keeps neighbors from overlapping, without a model; building scans run it automatically. `BuildingRegularizer` runs SAM 3 again on a high-resolution crop around each preliminary footprint, fuses the views of each building from its own and its neighbors' crops so touching neighbors stay apart, splits footprints that covered two buildings and regularizes the result. `RoadwayRegularizer` does the same job for roads, turning raw road polygons into centerlines and rebuilt polygons. The result is geometry you can load straight into a GIS.

**Views that actually show the asset**
When `rapidtools` picks street-level images of an asset, it checks that the asset is visible in them. It casts rays from each candidate camera toward the asset, using KD-trees and STR-trees to keep this fast, and rejects views in which a neighboring structure blocks the target. It then keeps one view per viewing direction, along the principal axes of the footprint.

## Installation

Install the latest release from PyPI (Python 3.11 or newer):

```bash
pip install rapidtools
```

## Documentation

The user guide, worked examples and API reference are at [rapid-facility.github.io/rAPIdtools](https://rapid-facility.github.io/rAPIdtools/). If you would rather not write code, start with the [graphical interface](https://rapid-facility.github.io/rAPIdtools/gui.html) page: `rapidtools-gui` opens a browser app that runs the same detect, extract and analyze workflow for aerial imagery and street-level surveys.

To build the documentation yourself, install the docs extra and run Sphinx:

```bash
pip install -e ".[docs]"
cd docs
make html
```

The pages are written to `docs/build/html`; open `index.html` there.

## Project Structure

The package is split into six subpackages, each with one job:

* `rapidtools.core`: The objects that represent your data (`PhysicalAsset`, `PhysicalAssetCollection`, `ImageAsset`, `BoundingBox`).
* `rapidtools.data_sources`: Clients that fetch data from external services and read large local files (e.g., `MapillaryClient`, `GoogleStreetViewClient`, `OrthomosaicReader`, `BingAerialImageExtractor`, `GoogleAerialImageExtractor`).
* `rapidtools.models`: Wrappers that run a model locally or through a hosted API (`Gemma4Inference`, `SAM3Inference`, `GeminiInference`, `MuseSparkInference`, `QwenInference`, ...). See the table below.
* `rapidtools.processing`: The workflow steps (extractors, segmenters, analyzers and regularizers) that you chain together in a `Pipeline`.
* `rapidtools.gui`: The browser application (`rapidtools-gui`) and the scriptable workflow behind it.
* `rapidtools.jobs`: The JSON job interface (`rapidtools-job`): a workflow written as a file that the GUI, a scheduler or another system hands to `rapidtools` to run.

## The rAPIdtools API in Five Lines

Every model is built by the same function and every analysis step is the same class, so switching providers means changing one word:

```python
import rapidtools as rt

rt.configure_logging()                                   # opt in to progress output
buildings = rt.PhysicalAssetCollection.from_geojson('buildings.geojson')
pipeline = rt.Pipeline([
    rt.AerialImageryExtractor('ortho.tif', save_directory='crops', buffer_m=20),
    rt.AssetAnalyzer('gemini', prompt='Rate the damage 0-5 as JSON.'),   # or 'claude', 'openai', 'muse_spark',
])                                                                      #    'qwen', 'gemma4', 'llama', 'qwen_vl', 'hf'
buildings = pipeline.run(buildings)
```

* `AssetAnalyzer('gemini', ...)` builds the model for you; hosted providers read their key from the environment (`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, ...) or take `api_key=`. `rt.load(provider, **kwargs)` builds a wrapper on its own, and `rt.list_providers()` and `rt.catalog(provider)` describe them without importing PyTorch.
* `AssetAnalyzer(model_or_provider, prompt, ...)` replaces the provider-specific `XAssetAnalyzer` classes (still importable, deprecated). Hosted models run threaded with a `RateLimitPolicy` (shared cooldown after failures, then retry passes for the assets that failed, so timeouts and 429s do not leave holes); local models run sequentially or in batches. Hosted wrappers take `timeout=` (default 60 s) for slow responses. Pass `generation=GenerationConfig(json_mode=True, temperature=0.0)` for per-call options.
* `AerialImageryExtractor` judges imagery coverage from the raster's mask (nodata value, alpha band or internal mask). By default a crop is skipped when more than `max_missing_data_ratio` of it is empty; pass `min_footprint_coverage=0.3` to keep every building whose footprint is at least 30% imaged, with edge crops padded so they keep their size.
* `SAM3OrthoFeatureExtractor(prompt, ...)` scans a whole orthomosaic for a text concept. By default touching detections dissolve into one polygon (roads, vegetation); pass `merge_overlaps=False` for countable objects such as vehicles to get one asset per object, fused from all the overlapping tiles that saw it, with its SAM 3 `confidence` and the number of tiles that detected it (`views`). Prompts that name buildings (`'building'`, `'house'`, `'structure'`) need no settings: one footprint per building, neighbors kept apart, walls squared by `FootprintRegularizer` (`regularize=False` for the raw outlines; `scan_settings()` lists the defaults).
* Pipeline steps declare a `Stage` (`DETECT`, `EXTRACT_IMAGERY`, `REGULARIZE`, `SEGMENT`, `ANALYZE`, `EXPORT`), so steps can be added in any order.
* `import rapidtools` is side-effect free and fast (about half a second); PyTorch and Transformers load only when a model is built, so pipelines of extractors and regularizers never pay for them. Mapillary components read `MAPILLARY_ACCESS_TOKEN` when no token is passed. Call `rapidtools.login()` to authenticate with the Hugging Face Hub for gated weights, or let local model wrappers do it when they load.

## Quick Start: Aerial Damage Detection Pipeline

This example runs the model on your own machine, so it needs no API key and no paid service. It downloads the `rapidtools` sample datasets, crops a drone orthomosaic around each building footprint and analyzes every crop with a local Gemma 4 model.

```python
import rapidtools as rt

rt.configure_logging()

# 1. Download the example datasets from the rapidtools registry
raster_path, footprint_path, prompt_path = rt.download_dataset(
    ['eaton_patch2', 'altadena_sample_buildings', 'aerial_chs_prompts']
)

# 2. Load the regional building footprints
buildings = rt.PhysicalAssetCollection.from_geojson(footprint_path)

pipeline = rt.Pipeline([
    # 3. Crop the orthomosaic around each building and draw a reference
    #    outline. The outline can trace the footprint itself (default), its
    #    bounding box, rotated bounding box, convex hull or corner brackets,
    #    optionally pushed away from the asset, in any width and color.
    rt.AerialImageryExtractor(
        raster_path,
        save_directory='eaton_fire_aerial_feb25/overlaid_imagery',
        overlay_asset_outline=True,
        outline_shape='rotated_bbox',
        outline_buffer='2 m',
        outline_width='1%',
    ),
    # 4. Apply the prompt to every crop with a local Gemma 4 model. Any
    #    provider name works here ('gemini', 'claude', 'openai', ...); hosted
    #    ones read their API key from the environment.
    rt.AssetAnalyzer('gemma4', prompt=prompt_path, batch_size=8),
])

# 5. Run it and export the AI-enriched inventory for GIS mapping
buildings = pipeline.run(buildings)
print(buildings)   # one line: how many assets, of which types
buildings.to_geojson('eaton_footprints_CHS_with_gemma4.geojson', include_images=False)
```

## Job Files: Run a Workflow From JSON

A whole workflow can be written down as a JSON *job file*: an ordered list of steps, where a later step can use what an earlier one produced. The graphical interface works this way too: every run it starts is saved as a job file under `<output folder>/jobs`. The same work can therefore be rerun from a terminal, queued by a scheduler or submitted by another system.

```json
{
  "schema": "rapidtools.job",
  "schema_version": "1.0",
  "output_dir": "output/eaton",
  "steps": [
    {"id": "sample", "type": "data.download", "params": {"dataset": "eaton_patch1"}},
    {
      "id": "detect",
      "type": "detect.aerial",
      "inputs": {"raster": {"step": "sample", "output": "file"}},
      "params": {"assets": ["building"], "basemap": "bing"}
    },
    {
      "id": "grade",
      "type": "analyze.vlm",
      "inputs": {
        "raster": {"step": "sample", "output": "file"},
        "assets": {"step": "detect", "output": "assets"}
      },
      "params": {
        "prompt": "Damage: [none, minor, major, destroyed]",
        "backend": "gemini",
        "api_key": {"$env": "GEMINI_API_KEY"}
      }
    }
  ]
}
```

```bash
rapidtools-job run job.json        # console script installed with the package
rapidtools-job validate job.json   # check a job without running it
rapidtools-job steps               # the step types and their options, as JSON
python -m rapidtools.jobs run job.json   # equivalent to the console script
```

Or from Python:

```python
from rapidtools.jobs import run_job

run_job('job.json').raise_for_status()
```

Every step is checked before the first one runs, and a result file (`job.result.json`, next to the job file) is rewritten at every change of state with each step's status, output files and summary. `run` exits with 0 when the job finished, 1 when a step failed, 2 when the job file is invalid and 3 when it was canceled. API keys stay out of job files: write `{"$env": "VARIABLE"}`, or `{"$secret": "name"}` with a separate `--secrets` file. The [JSON job interface](https://rapid-facility.github.io/rAPIdtools/json_interface.html) page is the full specification, with JSON Schemas for the three file types, and `examples/jobs` has jobs ready to run.

## Supported Models

Pass the name in the second column to `rt.load()` or `rt.AssetAnalyzer()`, and any model ID as `model_id`. `rt.catalog('gemini')` returns every curated ID for a provider without a network call; a wrapper's `list_available_models(api_key)` asks the provider what your key can reach.

**Hosted models** (the key is read from the environment variable shown, or passed as `api_key=`)

| Provider | Name | Key | Default model | Also available |
|---|---|---|---|---|
| Google Gemini | `gemini` | `GOOGLE_API_KEY` | `gemini-3.8-flash` | Gemini 4 (Neon, Carbon, Argon, Hydrogen), Gemini 3.x Flash, Flash-Lite and Pro |
| Anthropic Claude | `claude` | `ANTHROPIC_API_KEY` | `claude-opus-5` | Fable 5.x, Opus 5.5 and 4.x, Sonnet, Haiku 4.5 |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-5.5` | GPT-6.x, GPT-5.x, o3, GPT-4.x |
| Meta Muse | `muse_spark` | `MODEL_API_KEY` | `muse-spark-1.3` | Muse Wonder 1.0, earlier Muse Spark |
| Alibaba Qwen | `qwen` | `DASHSCOPE_API_KEY` | `qwen3.8-max` | Qwen 3.x Plus, Qwen-VL |

`AssetAnalyzer('gemini')` uses `gemini-3.5-flash-lite` unless you pass a `model_id`.

**Local models** (weights download from Hugging Face on first use)

| Family | Name | Default model | Also available |
|---|---|---|---|
| Google Gemma 4 | `gemma4` | `google/gemma-4-E2B-it` | E4B, 12B, 26B-A4B, 31B |
| Meta Llama | `llama` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Llama 3.2 90B, Llama 4 Scout and Maverick (gated downloads) |
| Meta Muse Glimmer | `muse_glimmer` | `meta-models/Muse-Glimmer-30B` | 4-bit loading fits a 24 GB GPU |
| Alibaba Qwen | `qwen_vl` | `Qwen/Qwen3.5-4B` | Qwen 3.8, 3.6 and 3.5 (0.8B to 397B), Qwen3-VL, Qwen2.5-VL, Qwen2-VL |
| Other open models | `hf` | `zai-org/GLM-4.6V-Flash` | GLM 5.3 Flash, Mistral Large 4, Muse Spark Open, Gemma 3, InternVL, LLaVA, Pixtral, Granite Vision, or any Hugging Face repository with a chat template |
| Meta SAM 3 (segmentation) | `sam3` | `facebook/sam3` | |

<details>
<summary>Every model ID in the catalogs (defaults in bold)</summary>

- **Google Gemini (API)** (`GeminiInference`): gemini-4-neon, gemini-4-carbon, gemini-4-argon, gemini-4-hydrogen, **gemini-3.8-flash**, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash, gemini-3.5-flash-lite (analyzer default), gemini-3.1-pro-preview, gemini-3.1-flash-lite, gemini-3.1-flash-image, gemini-3.1-flash-lite-image, gemini-3-flash-preview, gemini-3-pro-image, gemini-omni-1.1-flash
- **Anthropic Claude (API)** (`ClaudeInference`): claude-fable-5-2, claude-fable-5-1, claude-fable-5, claude-opus-5-5, **claude-opus-5**, claude-opus-4-8, claude-opus-4-7, claude-opus-4-6, claude-sonnet-5-5, claude-sonnet-5, claude-sonnet-4-6, claude-haiku-4-5
- **OpenAI (API)** (`OpenAIInference`): gpt-6.2-astra, gpt-6.2-sol, gpt-6.1-sol, gpt-6.1-terra, gpt-6-astra, gpt-6-sol, gpt-6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, **gpt-5.5**, gpt-5.5-pro, gpt-5.4, gpt-5.4-pro, gpt-5.4-mini, gpt-5.4-nano, gpt-5.2, gpt-5.1, o3, o3-pro, gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini
- **Meta Muse Spark / Muse Wonder (API)** (`MuseSparkInference`): muse-wonder-1.0, **muse-spark-1.3**, muse-spark-1.3-contributor, muse-spark-1.2, muse-spark-1.2-contributor, muse-spark-1.1
- **Alibaba Qwen (API, Model Studio)** (`QwenInference`): **qwen3.8-max**, qwen3.8-omni-flash, qwen3.7-plus, qwen3.6-plus, qwen3.5-plus, qwen3-vl-plus, qwen3-vl-flash, qwen-vl-max, qwen-vl-plus
- **Google Gemma 4 (local)** (`Gemma4Inference`): google/gemma-4-31B-it, google/gemma-4-26B-A4B-it, google/gemma-4-12B-it, google/gemma-4-E4B-it, **google/gemma-4-E2B-it**
- **Meta Llama (local)** (`LlamaVisionInference`): meta-llama/Llama-4-Scout-17B-16E-Instruct, meta-llama/Llama-4-Maverick-17B-128E-Instruct, **meta-llama/Llama-3.2-11B-Vision-Instruct**, meta-llama/Llama-3.2-90B-Vision-Instruct
- **Meta Muse Glimmer (local)** (`MuseGlimmerInference`): **meta-models/Muse-Glimmer-30B** (4-bit fits a 24 GB GPU; needs `transformers>=5.15`)
- **Alibaba Qwen (local)** (`QwenVisionInference`): Qwen/Qwen3.8-27B, Qwen/Qwen3.8-Flash-Next, Qwen/Qwen3.6-27B, Qwen/Qwen3.6-35B-A3B, Qwen/Qwen3.5-{397B-A17B, 122B-A10B, 35B-A3B, 27B, 9B, **4B**, 2B, 0.8B}, Qwen/Qwen3-VL-{235B-A22B, 32B, 30B-A3B, 8B, 4B, 2B}-Instruct, Qwen/Qwen2.5-VL-{72B, 32B, 7B, 3B}-Instruct, Qwen/Qwen2-VL-{7B, 2B}-Instruct
- **Any Hugging Face VLM (local)** (`HFVisionInference`): Curated list (`OPEN_VLM_CATALOG`, including **zai-org/GLM-4.6V-Flash**, zai-org/GLM-5.3-Flash, mistralai/Mistral-Large-4-1T-A52B and meta-models/Muse-Spark-Open-816B-A42B) plus any repo with a chat template
- **Meta SAM 3 (local segmentation)** (`SAM3Inference`, used by `SAM3ImageSegmenter` and `SAM3OrthoFeatureExtractor`): **facebook/sam3**

</details>

```python
import rapidtools as rt

cloud = rt.AssetAnalyzer('muse_spark', prompt='Return JSON with a damage_level key.')   # key from MODEL_API_KEY
local = rt.AssetAnalyzer('qwen_vl', prompt='Rate the fire damage 0-5.', model_id='Qwen/Qwen3.5-9B', load_in_4bit=True)
model = rt.load('claude')            # the wrapper itself, for single images: model.run_inference('roof.jpg', '...')
```

A hosted model built without `max_tokens` uses the default of its model:
2,048 tokens for most, 32,768 for Gemini 4, whose thinking counts against the
limit. `rapidtools.models.set_model_defaults('gemini-4-*', max_tokens=16384)`
changes it for a model or a family.

## Citing rAPIdtools

If rAPIdtools contributes to your research, please cite it. GitHub's
*Cite this repository* button (from `CITATION.cff`) gives APA and BibTeX
entries; each release is also archived on Zenodo with a DOI.

```bibtex
@software{rapidtools,
  author  = {Cetiner, Barbaros},
  title   = {rAPIdtools: A high-performance toolkit for performing large-scale AI inference and localization on post-disaster geospatial datasets},
  year    = {2026},
  doi     = {10.5281/zenodo.24542491},
  url     = {https://github.com/RAPID-Facility/rAPIdtools}
}
```

This DOI covers every version and always resolves to the latest one. When you
report results, cite the release you used instead: add its version number and
use its own DOI, listed under *Versions* on the
[Zenodo page](https://doi.org/10.5281/zenodo.24542491).

## Acknowledgments

rAPIdtools is developed at the RAPID Facility at the University of Washington, part of the Natural Hazards Engineering Research Infrastructure (NHERI) network. The RAPID Facility is supported by the U.S. National Science Foundation under awards CMMI-1611820 and CMMI-2130997, and by the National Institutes of Health. Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the funding agencies.
