Development and testing
=======================

This page is for people changing rAPIdtools itself: how to set up a
development checkout, what a change must pass, and how the test suite is
built so that new tests fit in. The repository's ``CONTRIBUTING.md`` has the
same material in short form together with the review and release process.

Setting up
----------

rAPIdtools needs Python 3.11 or newer. Clone the repository and install it in
editable mode with the development extras, which add pytest, Ruff, mypy and
Sphinx to the runtime dependencies:

.. code-block:: bash

   git clone https://github.com/RAPID-Facility/rAPIdtools.git
   cd rAPIdtools
   make install          # python -m pip install -e ".[dev]"

On a machine without a GPU, install CPU-only PyTorch first so the environment
stays small. ``torch`` and ``torchvision`` must come from the same index, or
every model module fails to import:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

``make help`` lists every developer target; ``PYTHON`` can be overridden for
any of them (``make test PYTHON=python3.12``).

The three gates
---------------

Continuous integration runs the same three checks on every push and pull
request; ``make check`` runs them locally in one go.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Check
     - Command
     - What it expects
   * - Style
     - ``make lint`` (``make format`` to fix)
     - Ruff is the style gate for linting and formatting alike: line length
       88, single quotes and the rule sets ``E``, ``W``, ``F``, ``I``, ``UP``,
       ``B`` and ``Q`` from ``[tool.ruff]`` in ``pyproject.toml``.
   * - Types
     - ``make typecheck``
     - ``mypy rapidtools`` is clean on ``main``. Keep public signatures
       annotated and avoid ``# type: ignore`` unless a third-party stub forces
       it.
   * - Tests
     - ``make test``
     - Every test passes on Linux, macOS and Windows with Python 3.11 to
       3.13. The suite is offline: it never downloads model weights or
       contacts a provider, a tile server or the Hugging Face Hub.

How the test suite is organised
-------------------------------

The suite lives in ``tests/``: 44 modules and about 1,200 tests, run with
``python -m pytest`` (``make test``). ``[tool.pytest.ini_options]`` in
``pyproject.toml`` adds ``--cov=rapidtools --cov-report=term-missing``, so
every run ends with a coverage table that lists the uncovered lines of each
module.

Most modules test one component and are named after it:
``test_mapillary_client.py``, ``test_street_tracking.py``,
``test_image_analyzers.py``, ``test_gui_server.py`` and so on. A few cut
across components:

- ``test_edge_cases.py`` holds the error branches and fallbacks of the
  street-level modules (duplicate resolver, verifier, re-identification,
  localisation and tracking helpers): unreadable crops, failed model calls,
  cancellation, parallel rays, sightings without a ground position, rays that
  meet beyond the range limit, singular merge gates.
- ``test_coverage_gaps.py`` collects the small fallbacks of otherwise
  well-covered modules (outline parsing, GUI server helpers, workflow model
  listing, prompt builder, image asset parsing, provider rejections in the
  analyzer).
- ``test_api_harmonization.py`` checks the public names, the lazy imports and
  the deprecation shims of ``rapidtools``; ``test_cancellation.py`` checks
  that every long-running component honours its ``cancel_event``.

Put a test next to the component it exercises; use the cross-cutting modules
only for a branch that needs a fixture the component's own module does not
have.

Keeping the suite offline
-------------------------

Nothing in the suite reaches the network or loads a checkpoint. The pieces
that make this hold are worth knowing before writing a test:

- ``tests/conftest.py`` has two autouse fixtures. ``_no_huggingface_login``
  replaces ``rapidtools.auth.get_token`` with a stub that returns a test token
  and ``rapidtools.models.local_base.ensure_huggingface_login`` with a stub
  returning ``True``; because a token is always "cached", the real
  ``ensure_huggingface_login`` (which the re-identification embedder imports
  from ``rapidtools.auth``) also returns ``True`` without logging in.
  ``_restore_package_logger`` undoes any ``configure_logging()`` call a test
  or the GUI command made, so logger state does not leak between tests.
- HTTP traffic to providers, tile servers and the Mapillary Graph API is
  answered by ``requests-mock`` (the ``requests_mock`` fixture) or by small
  fake clients such as the ``FakeClient`` in ``test_street_objects.py``,
  which serves canned images, detections and pixels for a synthetic survey.
- Local models never load weights. ``test_models_local.py`` monkeypatches the
  processor and model loaders with fakes; ``test_street_objects.py`` passes a
  ``FakeSAM3`` that returns fixed masks. ``test_edge_cases.py`` goes one step
  further for the re-identification embedder: it injects fake ``torch`` and
  ``transformers`` modules into ``sys.modules`` that expose only what the
  loader touches (``AutoConfig``, ``AutoModel``,
  ``CLIPVisionModelWithProjection``, ``AutoImageProcessor``,
  ``inference_mode``), so both the CLIP and the DINOv2 branches run in
  milliseconds.
- Vision-language models are stand-ins with a ``run_inference`` method that
  answer from the image file name or a script (``FakeModel`` in
  ``test_verification.py`` and ``test_image_analyzers.py``); a ``judge``
  callable does the same for the duplicate resolver.
- Rasters, GeoJSON, shapefiles and crops are built in ``tmp_path``; the
  only committed fixtures are the small files under ``tests/mapillary_images``.

Coverage
--------

Coverage sits at 99% of the package's statements, and the CI badge in the
README is updated from the Ubuntu / Python 3.11 cell of the test matrix. The
lines still uncovered are the ones that need a service to misbehave in a
specific way: the GUI server's survey overview and route builders, the
building regularisation fallbacks for invalid geometry, cancellation inside a
thread pool, and a few far-sighting rules of the tracker that only fire on
real survey geometry.

To see what a change left uncovered, read the ``Missing`` column of the
summary, or build the HTML report:

.. code-block:: bash

   make coverage         # writes htmlcov/index.html

A pull request should not lower the total. When a new branch is hard to reach
through the public API, test the helper directly, as the cross-cutting
modules do, rather than leaving it uncovered.

Writing a test
--------------

- Patch the network boundary with ``requests-mock`` or ``monkeypatch``; never
  call a real service.
- Replace model and processor loaders with fakes; never let ``transformers``
  load a checkpoint.
- Keep fixtures deterministic and small, and build data in ``tmp_path``.
- Assert on behaviour the user sees (attributes, files, log messages,
  exceptions), not on how a helper got there.
- Add a test for every bug fix that would have caught it.
- Run ``make check`` before opening the pull request; CI runs the same
  commands.

Documentation
-------------

Public classes and functions carry Google-style docstrings with an
``Example:`` block, and the API reference is generated from them; a new
public object also needs an entry in ``docs/source/api/index.rst``. Build the
site with ``make docs`` and open ``docs/build/html/index.html``. User-facing
changes get a line in the ``Unreleased`` section of ``CHANGELOG.md``.
