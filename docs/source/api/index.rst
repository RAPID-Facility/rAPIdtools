API Reference
=============

Everything below is importable from the top-level ``rapidtools`` package as
well as from the subpackage shown. Heavy dependencies load on first use.

Core objects
------------

.. currentmodule:: rapidtools.core

.. autosummary::
   :toctree: generated
   :nosignatures:

   PhysicalAsset
   PhysicalAssetCollection
   ImageAsset
   ImageCollection
   BoundingBox
   PolygonRegion
   OperationCancelled
   is_cancelled
   raise_if_cancelled

Pipeline and analysis
---------------------

.. currentmodule:: rapidtools.processing

.. autosummary::
   :toctree: generated
   :nosignatures:

   Pipeline
   Stage
   PipelineStep
   AssetAnalyzer
   RateLimitPolicy

Imagery extractors
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   AerialImageryExtractor
   BingOrthomosaicExtractor
   GoogleOrthomosaicExtractor
   GoogleStreetViewImageExtractor
   MapillaryImageExtractor
   MapillaryLabelMapper

Detection, segmentation and regularization
------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SAM3OrthoFeatureExtractor
   SAM3ImageSegmenter
   BuildingRegularizer
   RoadwayRegularizer

Provider-specific analyzers
---------------------------

These predate :class:`AssetAnalyzer` and remain for compatibility. New code
should use ``AssetAnalyzer(load(provider, ...))``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   GeminiAssetAnalyzer
   ClaudeAssetAnalyzer
   OpenAIAssetAnalyzer
   MuseSparkAssetAnalyzer
   QwenAssetAnalyzer
   Gemma4AssetAnalyzer
   LlamaVisionAssetAnalyzer
   MuseGlimmerAssetAnalyzer
   QwenVisionAssetAnalyzer
   HFVisionAssetAnalyzer

Models
------

The provider registry ``rapidtools.models.PROVIDERS`` maps each registry key
to a :class:`ModelInfo` record and is always importable without PyTorch.

.. currentmodule:: rapidtools.models

.. autosummary::
   :toctree: generated
   :nosignatures:

   load
   list_providers
   catalog
   get_model_class
   ModelInfo
   GenerationConfig
   SegmentationConfig
   ModelOutput
   BaseInferenceModel

Hosted models
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   GeminiInference
   ClaudeInference
   OpenAIInference
   MuseSparkInference
   QwenInference

Local models
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   Gemma4Inference
   LlamaVisionInference
   MuseGlimmerInference
   QwenVisionInference
   HFVisionInference
   SAM3Inference

Data sources
------------

.. currentmodule:: rapidtools.data_sources

.. autosummary::
   :toctree: generated
   :nosignatures:

   OrthomosaicReader
   GoogleAerialImageExtractor
   BingAerialImageExtractor
   GoogleStreetViewClient
   StreetViewPanorama
   MapillaryClient
   MapillaryLabels
   TileUtils

Datasets, configuration and authentication
------------------------------------------

.. currentmodule:: rapidtools

.. autosummary::
   :toctree: generated
   :nosignatures:

   datasets.download_dataset
   config.configure_logging
   config.get_configured_session
   config.resolve_alias
   auth.login
   auth.ensure_huggingface_login

Graphical interface
-------------------

.. currentmodule:: rapidtools.gui

.. autosummary::
   :toctree: generated
   :nosignatures:

   launch_asset_analysis_app
   AssetAnalysisWorkflow
   DetectionSettings
   InferenceSettings
