# rAPIdtools

[![Tests](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml/badge.svg?label=Tests)](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/bacetiner/c890ae687368838a74c5e442b9ff5b94/raw/coverage.json)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://shields.io)](https://pypi.org/project/rapidtools/)
[![Typing](https://img.shields.io/pypi/types/rapidtools)](https://pypi.org/project/rapidtools/)


A high-performance toolkit for performing large-scale AI inference and localization on post-disaster geospatial datasets.

## Overview

The UW RAPID Facility collects terabytes of perishable, hyper-resolution data in the aftermath of natural disasters. As remote sensing technology has evolved, the core challenge in natural hazards engineering has shifted: the primary bottleneck is no longer data collection, but turning that raw data into actionable intelligence.

`rapidtools` is a high-performance Python package designed to eliminate this bottleneck. It delivers a seamless, object-oriented pipeline that connects raw spatial datasets with state-of-the-art Large Vision-Language Models (VLMs). Whether analyzing ten damaged homes or one hundred thousand regional assets, `rapidtools` equips researchers with the tools to automate complex feature extraction, pinpoint structural damage, and unlock engineering-grade insights at unprecedented speed.

## High-Level Impact & Key Features

**Large-scale geospatial ingestion**
Seamlessly fuse massive local orthomosaics, regional shapefiles, and street-view vector tiles. The `PhysicalAssetCollection` engine provides fast lookups, patial filtering, and native conversions between GeoJSON, ESRI Shapefiles, and Pandas DataFrames.

**Scalable AI Inference (Local & Cloud)**
Run deployments tailored to your resources. Deploy powerful local vision-language models (such as Google's Gemma-4 and Meta's Llama-Vision) directly on consumer hardware using dynamic batching, automated tensor precision scaling, and strict VRAM garbage collection to prevent Out-Of-Memory (OOM) crashes. Alternatively, scale instantly using built-in integrations for enterprise APIs (OpenAI, Google Gemini, Anthropic Claude), which feature thread-safe global cooldowns and exponential backoff to handle rate limits automatically.

**Intelligent Feature Regularization**
Move beyond raw AI pixel masks. The toolkit includes sophisticated geometric regularizers that instantly translate semantic segmentations into usable, GIS-ready asset geometries.

**Advanced Line-of-Sight Localization**
Automate the extraction of the perfect viewing angle. Using KD-Trees, STRtrees, and ray-casting math, `rapidtools` can dynamically calculate asset principal axes and cull occluded perspectives (e.g., ignoring images where a target building is blocked by a neighboring structure) to guarantee your AI only analyzes the right data.

## Installation

You can install the latest stable release directly via pip:

```bash
pip install rapidtools
```

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
# Crops the orthomosaic around each asset and draws a reference outline
extractor = AerialImageryExtractor(
    dataset=raster_path,
    save_directory=image_save_dir,
    overlay_asset_outline=True,
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

## Project Structure

Designed for flexibility and scale, `rapidtools` utilizes a cleanly decoupled architecture that makes extending workflows and managing complex data pipelines effortless:

* `rapidtools.core`: Domain models representing your data (`PhysicalAsset`, `PhysicalAssetCollection`, `ImageAsset`, `BoundingBox`).
* `rapidtools.data_sources`: Clients for fetching raw data from external APIs and massive local files (e.g., `MapillaryClient`, `OrthomosaicReader`, `BingAerialImageExtractor`).
* `rapidtools.models`: Base wrappers and handlers for executing ML models natively or via cloud APIs (`Gemma4Inference`, `SAM3Inference`, `GeminiInference`).
* `rapidtools.processing`: High-level workflow components (Extractors, Segmenters, Analyzers, and Regularizers) designed to snap together effortlessly into the `Pipeline` engine.

## Documentation

The official documentation is generated using Sphinx and can be built locally.

Navigate to the docs directory:
```bash
cd docs
make html
```
Open the file `docs/build/html/index.html` in your web browser to view the full API reference and advanced tutorials.

## License

This project is licensed under the BSD-3-Clause License. See the `LICENSE` file for details.
