JSON job interface
==================

Everything rAPIdtools is asked to do can be written down as a JSON file. A
client (the graphical interface, a scheduler, a partner system) writes a
**job file** that lists the steps to run; rAPIdtools executes it and keeps a
**result file** next to it up to date. No Python objects cross that boundary,
so the client and rAPIdtools can run in different processes or on different
machines, as long as both see the same files.

.. code-block:: text

   client                          rAPIdtools
   ------                          ----------
   writes   job.json        --->   validates every step, then runs them in order
   polls    job.result.json <---   rewrites it at every change of state
   reads    the output files       named in job.result.json

The graphical interface works this way itself: every download, detection,
inference and prompt-assistant request it starts is a job file under
``<output folder>/jobs``, and any of those files reruns unchanged from a
terminal.

This page is the specification of interface version **1.0**. The three
document types have JSON Schemas (draft 2020-12) that can be used to validate
files before they are submitted:

- :download:`rapidtools-job-1.0.schema.json <_static/schemas/rapidtools-job-1.0.schema.json>`
- :download:`rapidtools-job-result-1.0.schema.json <_static/schemas/rapidtools-job-result-1.0.schema.json>`
- :download:`rapidtools-secrets-1.0.schema.json <_static/schemas/rapidtools-secrets-1.0.schema.json>`

Running a job
-------------

From a terminal:

.. code-block:: bash

   rapidtools-job run job.json
   rapidtools-job run job.json --secrets secrets.json   # credentials in a separate file
   rapidtools-job run job.json --result /shared/status/job-42.json

   rapidtools-job validate job.json    # check a job without running it
   rapidtools-job schema job           # print a JSON Schema: job | result | secrets
   rapidtools-job steps                # print the catalogue of step types as JSON

``python -m rapidtools.jobs`` is the same command for environments where the
console script is not on the path. ``run`` prints progress to the console,
writes the result file and exits with:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Exit status
     - Meaning
   * - ``0``
     - The job finished; every step succeeded.
   * - ``1``
     - A step failed. The result file names the step and the error.
   * - ``2``
     - The job file is invalid (or unreadable). Nothing was run.
   * - ``3``
     - The job was cancelled.

``SIGINT`` (Ctrl+C) or ``SIGTERM`` cancels cooperatively: the batch or tile
being processed finishes, partial results are written, the result file is
marked ``cancelled`` and the process exits with ``3``. A second signal ends
the process immediately.

From Python, the same thing is one call:

.. code-block:: python

   from rapidtools.jobs import run_job

   result = run_job('job.json', secrets='secrets.json')
   if result.ok:
       print(result.step('detect').outputs['assets'])   # a pathlib.Path
   else:
       print(result.status, result.message)

The job file
------------

A job file is one JSON object.

.. code-block:: json

   {
     "schema": "rapidtools.job",
     "schema_version": "1.0",
     "id": "eaton-detect-and-grade",
     "name": "Eaton fire: detect buildings and grade damage",
     "output_dir": "output/eaton",
     "steps": [
       {
         "id": "detect",
         "type": "detect.aerial",
         "inputs": {"raster": "eaton_patch_20250214.tiff"},
         "params": {"assets": ["building"], "basemap": "bing"}
       },
       {
         "id": "grade",
         "type": "analyze.vlm",
         "inputs": {
           "raster": "eaton_patch_20250214.tiff",
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

.. list-table:: Top-level fields
   :header-rows: 1
   :widths: 20 12 68

   * - Field
     - Required
     - Meaning
   * - ``schema``
     - yes
     - Always ``"rapidtools.job"``.
   * - ``schema_version``
     - yes
     - Interface version the file was written for, ``"1.0"``. See
       `Versioning`_.
   * - ``output_dir``
     - yes
     - Folder every step writes into. Created if missing.
   * - ``steps``
     - yes
     - The steps, run in order. At least one.
   * - ``id``
     - no
     - Identifier of the job, repeated in the result file. Letters, digits,
       ``.``, ``_`` and ``-``. Generated from the time and the name when
       omitted.
   * - ``name``
     - no
     - Human-readable name used in messages. Defaults to the ``id``.
   * - ``created_at``
     - no
     - UTC creation time, ISO 8601 (``2026-11-24T15:49:52Z``).
   * - ``created_by``
     - no
     - Free-form object naming the client, e.g.
       ``{"client": "rapidtools-gui", "version": "0.2.1"}``.
   * - ``metadata``
     - no
     - Free-form object for the client's own bookkeeping (ticket numbers,
       event names). Never interpreted.

.. list-table:: Fields of a step
   :header-rows: 1
   :widths: 20 12 68

   * - Field
     - Required
     - Meaning
   * - ``type``
     - yes
     - The step type, e.g. ``"detect.aerial"``. See `Step types`_.
   * - ``id``
     - no
     - Name of the step inside the job; later steps and the result file
       refer to it. Unique within the job. Defaults to ``s1``, ``s2``, ...
   * - ``inputs``
     - depends
     - The data the step reads: files, regions, or outputs of earlier steps.
   * - ``params``
     - depends
     - The options of the step type. Omitted options take the documented
       default.
   * - ``metadata``
     - no
     - Free-form object. Never interpreted.

Any other field, at the top level, in a step, under ``inputs`` or under
``params``, is an error. A misspelt option is reported with the closest
valid name instead of being silently ignored.

Paths
~~~~~

Every path in a job file (``output_dir`` and file inputs) is either absolute
or **relative to the folder that holds the job file**; ``~`` is expanded.
A job that uses relative paths can therefore be moved together with its data.
The process that runs the job must be able to read the inputs and write the
output folder: when the client and rAPIdtools run on different machines, the
files have to be on storage both can reach.

Inputs
~~~~~~

There are three kinds of input values.

A **file** (kinds ``raster``, ``assets`` and ``file``) is a path string, or a
reference to something an earlier step of the same job wrote:

.. code-block:: json

   {"raster": "/data/eaton/ortho.tiff"}
   {"assets": {"step": "detect", "output": "assets"}}

A reference names the ``id`` of an **earlier** step and one of the outputs
its step type declares. It is checked before the job starts. Assets are
exchanged as GeoJSON ``FeatureCollection`` files in WGS84 (EPSG:4326), one
feature per asset, attributes in ``properties``.

A **region** is a WGS84 bounding box or the extent of a GeoJSON file:

.. code-block:: json

   {"region": {"bbox": [-117.489, 47.711, -117.487, 47.7125]}}
   {"region": {"geojson": "study_area.geojson"}}

The box is ``[min_lon, min_lat, max_lon, max_lat]`` with longitudes in
[-180, 180] and latitudes in [-85, 85].

Secrets
~~~~~~~

API keys and access tokens are parameters like any other (they are marked
*secret* in the reference below), but they do not have to be written into
the job file. A secret parameter accepts three forms:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Form
     - Where the value comes from
   * - ``{"$secret": "mapillary_token"}``
     - The secrets file given to the runner; if the name is not there, the
       environment variable ``RAPIDTOOLS_SECRET_MAPILLARY_TOKEN`` (the name
       in upper case, other characters replaced by ``_``).
   * - ``{"$env": "GEMINI_API_KEY"}``
     - The environment variable of that exact name.
   * - ``"AIza..."``
     - The literal value. The job file must then be protected like the key.

A secrets file is a separate JSON document, passed with ``--secrets`` (or
``secrets=`` in Python):

.. code-block:: json

   {
     "schema": "rapidtools.secrets",
     "schema_version": "1.0",
     "secrets": {"mapillary_token": "MLY|...", "api_key": "AIza..."}
   }

Secret values are never written to the result file or to the log. A
reference that cannot be resolved fails the job before any step runs;
``rapidtools-job validate`` accepts unresolved references unless
``--require-secrets`` is given, so a job can be checked on a machine that
does not hold the credentials.

The result file
---------------

rAPIdtools writes ``<job file name without .json>.result.json`` next to the
job file (or wherever ``--result`` points). It is rewritten when the job
starts, when each step starts and ends, and when the job ends; a job started
with ``rapidtools-job run`` or ``run_job`` also refreshes ``message`` on
every progress line. Every write replaces the file in one atomic step, so a
reader always sees a complete document: polling it is safe.

.. code-block:: json

   {
     "schema": "rapidtools.job_result",
     "schema_version": "1.0",
     "job_id": "20261124T154952Z-74fa-download-bing-imagery",
     "job_name": "Download Bing imagery",
     "job_file": "/data/out/jobs/20261124T154952Z-74fa-download-bing-imagery.json",
     "status": "done",
     "message": "Download Bing imagery finished.",
     "started_at": "2026-11-24T15:49:52Z",
     "updated_at": "2026-11-24T15:49:54Z",
     "finished_at": "2026-11-24T15:49:54Z",
     "duration_s": 1.646,
     "rapidtools_version": "0.2.1",
     "steps": [
       {
         "id": "basemap",
         "type": "imagery.basemap",
         "status": "done",
         "started_at": "2026-11-24T15:49:52Z",
         "finished_at": "2026-11-24T15:49:53Z",
         "duration_s": 1.234,
         "outputs": {
           "raster": {
             "path": "/data/out/bing_z19_m117p48900_47p71100_m117p48700_47p71250.tiff",
             "kind": "raster"
           }
         },
         "files": [],
         "summary": {"provider": "bing", "zoom": 19}
       }
     ]
   }

.. list-table:: Top-level fields
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``status``
     - ``running``, ``done``, ``error`` or ``cancelled``. A job is finished
       when the status is anything but ``running``.
   * - ``message``
     - While running: the latest progress line. At the end: the outcome, or
       the error message.
   * - ``job_id``, ``job_name``, ``job_file``
     - Copied from the job. ``job_file`` is ``null`` for a job that was
       passed in memory.
   * - ``started_at``, ``updated_at``, ``finished_at``
     - UTC, ISO 8601. ``updated_at`` changes on every write; ``finished_at``
       is ``null`` until the job ends.
   * - ``duration_s``
     - Wall-clock seconds, ``null`` until the job ends.
   * - ``rapidtools_version``
     - The release that ran the job.
   * - ``steps``
     - One entry per step of the job, in the same order.

.. list-table:: Fields of a step result
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``id``, ``type``
     - Copied from the job.
   * - ``status``
     - ``pending`` (not reached yet), ``running``, ``done``, ``error``,
       ``cancelled``, or ``skipped`` (an earlier step ended the job).
   * - ``outputs``
     - The declared outputs that were written, by name: ``{"path": <absolute
       path>, "kind": "raster" | "geojson" | "file"}``. An output the step
       did not produce (nothing was detected, say) is absent.
   * - ``files``
     - Other files worth listing, each with ``label``, ``path`` and ``kind``,
       for example the per-asset-type GeoJSON files of a detection.
   * - ``summary``
     - Small facts about the run, such as ``{"n_assets": 179}``. The keys
       depend on the step type.
   * - ``data``
     - Present only for steps whose result is not a file
       (``prompt.assist``).
   * - ``error``
     - Present when the step failed or was cancelled:
       ``{"type": <exception name>, "message": <text>}``.
   * - ``started_at``, ``finished_at``, ``duration_s``
     - As above; ``null`` for steps that never started.

How a job ends
~~~~~~~~~~~~~~

- Every step is validated **before the first one runs**. An invalid job ends
  with ``status: "error"``, the offending step marked ``error`` and no step
  started.
- Steps run one after the other. The first step that fails or is cancelled
  ends the job; the steps after it are ``skipped``.
- A detection step that finds nothing succeeds, but a later step that needed
  its assets fails with a message saying so ("No assets were detected;
  skipping inference.").
- A cancelled ``analyze.vlm`` step keeps what it finished: the assets
  analyzed so far are written and listed as the ``partial_assets`` output.

Step types
----------

A step type is named ``<stage>.<variant>``. Version 1.0 defines six:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Type
     - What it does
   * - ``imagery.basemap``
     - Stitches a Bing or Google satellite basemap over a region.
   * - ``data.download``
     - Fetches a sample dataset from the rAPIdtools registry.
   * - ``detect.aerial``
     - Finds assets in an orthomosaic with SAM 3.
   * - ``detect.street``
     - Finds objects along a Mapillary street survey.
   * - ``analyze.vlm``
     - Asks a vision-language model a question about every asset.
   * - ``prompt.assist``
     - Asks a model to draft, refine or review a prompt.

The tables below are generated from the code that validates job files, and
``rapidtools-job steps`` prints the same catalogue as JSON for clients that
build their forms from it.

.. include:: _generated/job_steps.inc

Versioning
----------

``schema_version`` is ``MAJOR.MINOR``.

- A **minor** release only adds: new step types, new optional parameters,
  new optional fields, new keys in ``summary``. A file written for 1.0 keeps
  working with every 1.x release, with the same meaning.
- A **major** release may change or remove things. A runner refuses files of
  another major version.
- A runner accepts a file that declares a *newer* minor version than its
  own, but rejects any field or step type it does not know, with a message
  saying that rAPIdtools needs to be upgraded. A newer file that uses only
  1.0 features therefore still runs.
- Readers of result files should ignore fields they do not know, so that
  result files can grow in minor releases.

The JSON Schemas linked at the top of this page describe exactly one
version; ``rapidtools-job schema`` prints the ones matching the installed
release.

How the graphical interface uses it
-----------------------------------

The GUI server is a client of this interface and nothing more.

- **Where the files are.** Each run writes
  ``<output folder>/jobs/<UTC time>-<random>-<name>.json`` and the matching
  ``.result.json``. The log pane prints the path when a job starts, and both
  paths are in ``job.file`` and ``job.result_file`` of ``GET /api/state``.
- **Steps.** *Detect only*, *Run inference*, a basemap download and a
  prompt-assistant request are one-step jobs. *Detect + run inference* is
  one job with two steps, the second reading the ``assets`` output of the
  first.
- **Assets between runs.** An inference started on its own reads the assets
  from the file the previous run wrote (``assets_final.geojson``,
  ``street_objects.geojson``, ``assets_inferred.geojson``) or the file the
  user loaded. Running a second model therefore adds its attributes to the
  first model's.
- **Credentials.** Keys typed into the page are replaced by
  ``{"$secret": ...}`` in the job file. The values go to a secrets file in a
  private temporary folder outside the output folder, readable by the
  server's user only and deleted when the job ends.
- **What is not a job.** Loading a local image or an asset file for display,
  map tiles, address search, survey coverage and the list of models a key
  can access are answered by the GUI server directly; they prepare a request
  rather than run an analysis.

Adding a step type
------------------

The interface is extended by registering one record; the runner, the
schemas, the catalogue and the tables on this page follow from it.

.. code-block:: python

   from rapidtools.jobs import Input, Output, Param, StepType, register_step

   def build(step_type, params, inputs, output_dir, dry):
       """Return the settings the step runs with; raise ValueError if invalid."""
       return {'buffer_m': params['buffer_m'], 'assets': inputs['assets']}

   def run(ctx):
       assets = ctx.inputs['assets'].load()        # a PhysicalAssetCollection
       path = ctx.output_dir / 'assets_buffered.geojson'
       ...                                         # do the work, write the file
       ctx.outputs['assets'] = path                # what later steps can read
       ctx.summary['n_assets'] = len(assets)

   register_step(StepType(
       name='postprocess.buffer',
       title='Buffer asset footprints',
       doc='Grows every footprint by a fixed distance.',
       build=build,
       run=run,
       inputs=(Input('assets', 'assets', 'The assets to buffer.', required=True),),
       params=(Param('buffer_m', 'number', 'Distance in meters.', default=2.0,
                     minimum=0),),
       outputs=(Output('assets', 'geojson', 'The buffered assets.'),),
   ))

Limits of version 1.0
---------------------

These are deliberate omissions rather than oversights; each has a place in
the design where it can be added in a minor release.

- **Shared storage is assumed.** Inputs and outputs are paths. There is no
  upload, download or object-store URL scheme yet.
- **Execution is local and sequential.** The runner executes a job in its
  own process, one step at a time. The GUI runs one job at a time. Queuing,
  parallel steps and remote execution belong to the system that calls
  ``rapidtools-job run``.
- **Progress is a message.** The result file carries the latest progress
  line and the state of each step, not a percentage.
- **Cancellation is a signal.** A job is cancelled by signalling the process
  that runs it; there is no cancel file to drop next to the job.
- **No sandbox.** The runner reads and writes whatever paths the job names
  and its user may access. The folder restriction of a shared GUI server
  (``--data-root``) is enforced by the GUI before it writes a job file, not
  by the runner; a system that accepts job files from untrusted users should
  confine the process itself.
- **Resources are not declared.** A job does not say how much GPU memory or
  time it needs; the local model steps need a GPU with about 16 GB for
  SAM 3 and the default Gemma 4 checkpoint.
