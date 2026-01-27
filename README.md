# rAPIdtools

[![Tests](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml/badge.svg?label=Tests)](https://github.com/RAPID-Facility/rAPIdtools/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/bacetiner/c890ae687368838a74c5e442b9ff5b94/raw/coverage.json)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
<!-- [![Typing](https://img.shields.io/pypi/types/rapidtools)](https://pypi.org/project/rapidtools/) -->

A toolkit for damage detection in regional assets using AI and geospatial data.

## Overview

`rapidtools` is a Python package designed to streamline the process of analyzing infrastructure assets (such as buildings, bridges, and roads) from various imagery sources. It provides a robust, object-oriented framework for representing geospatial data and leverages powerful AI vision models to perform tasks like damage detection and description.

The library is built with a clean, decoupled architecture, making it easy to extend with new data sources, AI models, and analysis workflows.

## Key Features

*   **Domain-Driven Design:** A core set of intuitive, type-hinted classes (`InfrastructureAsset`, `ImageAsset`, `Region`, `BoundingBox`) to represent your data in a structured way.
*   **Geospatial Power:** Built on top of `shapely` for robust geometry handling, including automatic tiling of large bounding boxes for efficient processing.
*   **Multi-AI Provider Support:** Includes swappable clients for interacting with leading vision models, including:
    *   OpenAI (GPT-4o, etc.)
    *   Google (Gemini, Llama on Vertex AI)
    *   Anthropic (Claude 3.5 Sonnet, etc.)
    *   And more...
*   **Concurrent Batch Processing:** High-performance, multi-threaded utilities for processing entire directories of images with progress bars and automatic retries.
*   **Professional Tooling:** Comes with a complete setup for testing (`pytest`), linting (`ruff`), formatting (`black`), and type-checking (`mypy`).

## Installation

Currently, `rapidtools` is under development. To install it directly from the repository for use in your own projects:

```bash
pip install git+https://github.com/RAPID-Facility/rAPIdtools
```

## Project Structure

The project follows a clean, modern Python architecture to separate concerns:

*   `rapidtools/core/`: Contains the **core domain models** (`Region`, `InfrastructureAsset`, `ImageAsset`). These are the **"nouns"** of the application.
*   `rapidtools/data_sources/`: Contains **clients** for fetching data from external APIs (e.g., `MapillaryClient`).
*   `rapidtools/inference/`: Contains **wrappers** for running ML models (e.g., `DamagePredictor`).
*   `rapidtools/workflows/`: Contains the **workflows** that combines all components.

## Documentation

The official documentation is generated using [Sphinx](https://www.sphinx-doc.org/) and can be built locally.

1.  **Navigate to the docs directory:**
    ```bash
    cd docs
    ```

2.  **Build the HTML:**
    ```bash
    make html
    ```

3.  **Open the documentation:**
    Open the file `docs/build/html/index.html` in your web browser to view the site.

## License

This project is licensed under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause). See the [LICENSE](LICENSE) file for details.
