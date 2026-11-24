API Reference
=============

The classes and factory functions below are importable from the top-level
``rapidtools`` package (``import rapidtools as rt``) as well as from the
subpackage shown; the lower-level functions listed by module name
(``street_fusion.fuse_observations``, ...) are imported from that module.
Heavy dependencies load on first use.

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
   DetectionVerifier
   DuplicateResolver

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
   DroneImageExtractor
   DroneViewSelector
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
model, occupancy fusion, structures from the survey's point cloud, the
earlier tracking and triangulation method with its covariance-aware merging
and ray-voting baseline, and appearance re-identification.

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
   street_fusion.fuse_observations
   street_fusion.scan_from_observation
   street_fusion.set_range_bands
   street_fusion.FusedObject
   street_structures.discover_structures
   street_structures.label_points
   street_structures.structures_from_points
   street_structures.footprint_rectangle
   street_structures.fetch_sfm_cluster
   street_structures.Structure
   street_structures.PointCloud
   street_tracking.discover_objects
   street_tracking.suppress_duplicate_sightings
   street_tracking.prune_unwitnessed
   street_tracking.CameraFrame
   street_tracking.track_sequence
   street_tracking.merge_by_rays
   street_tracking.triangulate_track
   street_tracking.merge_estimates
   street_tracking.merge_pieces
   street_tracking.sighting_shows
   street_tracking.vote_rays
   street_tracking.ObjectEstimate
   reid.AppearanceEmbedder
   reid.merge_by_appearance
   reid.pairs_to_compare
   reid.views_for_reid
   reid.polygon_width_px
   reid.crop_observation
   reid.recommended_min_similarity

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
   DroneCamera
   read_drone_images
   read_pose_table
   write_pose_metadata

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

JSON job interface
------------------

The file format is specified in :doc:`../json_interface`.

.. currentmodule:: rapidtools.jobs

.. autosummary::
   :toctree: generated
   :nosignatures:

   run_job
   JobRunner
   JobResult
   StepResult
   Job
   StepSpec
   new_job
   load_job
   Secrets
   JobError
   StepType
   Param
   Input
   Output
   StepContext
   register_step
   get_step_type
   job_schema
   result_schema
   secrets_schema
   describe_steps

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
