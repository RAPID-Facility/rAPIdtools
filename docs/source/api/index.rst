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
   Observation
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
   MapillaryObjectImageExtractor
   MapillaryLabelMapper

Detection, segmentation and regularization
------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SAM3OrthoFeatureExtractor
   MapillaryFeatureExtractor
   SAM3ImageSegmenter
   BuildingRegularizer
   RoadwayRegularizer
   vectorize.mask_to_wgs84_polygons

Street-level localisation
~~~~~~~~~~~~~~~~~~~~~~~~~

How sightings in street-level images become positioned objects: the camera
model, tracking and triangulation, covariance-aware merging, the ray-voting
baseline and appearance re-identification.

.. autosummary::
   :toctree: generated
   :nosignatures:

   street_localization.localize
   street_localization.observation_angles
   street_localization.pixel_bearing
   street_localization.world_ray
   street_localization.rotation_matrix
   street_localization.intersect_bearings
   street_localization.estimate_ego_mask
   street_localization.thin_frames
   street_tracking.discover_objects
   street_tracking.track_sequence
   street_tracking.triangulate_track
   street_tracking.merge_estimates
   street_tracking.vote_rays
   street_tracking.ObjectEstimate
   reid.AppearanceEmbedder
   reid.merge_by_appearance
   reid.pairs_to_compare
   reid.crop_observation

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
   orthomosaic_reader.PatchGeoref
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
   models.api_base.api_session
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
   StreetDetectionSettings
   RegionImagerySettings
   InferenceSettings
   AssistSettings
   NotificationConfig
   Notifier
   JobSummary
   PromptSpec
   OutputField
   RubricEntry
   assemble_prompt
   PromptAssistant
