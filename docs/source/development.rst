Development and testing
=======================

This page is for people working on rAPIdtools itself. It covers setting up a
development checkout, the checks every change has to pass, and how the test
suite is put together so that new tests fit in. ``CONTRIBUTING.md`` in the
repository has the same material in short form, along with the review and
release process.

Setting up
----------

rAPIdtools needs Python 3.11 or newer. Clone the repository and install it in
editable mode with the development extras, which add pytest, Ruff, mypy and
Sphinx on top of the runtime dependencies:

.. code-block:: bash

   git clone https://github.com/RAPID-Facility/rAPIdtools.git
   cd rAPIdtools
   make install          # python -m pip install -e ".[dev]"

On a machine without a GPU, install the CPU-only build of PyTorch first to
keep the environment small. ``torch`` and ``torchvision`` must come from the
same index, or none of the model modules will import:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

``make help`` lists every developer target. Each one accepts a different
interpreter through ``PYTHON``, for example ``make test PYTHON=python3.12``.

The three checks
----------------

Continuous integration runs three checks on every push and pull request.
``make check`` runs all three locally.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Check
     - Command
     - What it expects
   * - Style
     - ``make lint`` (``make format`` to fix)
     - Ruff, for both linting and formatting: line length 88, single quotes,
       and the rule sets ``E``, ``W``, ``F``, ``I``, ``UP``, ``B`` and ``Q``
       configured under ``[tool.ruff]`` in ``pyproject.toml``.
   * - Types
     - ``make typecheck``
     - ``mypy rapidtools`` reports no errors. Keep public signatures
       annotated, and use ``# type: ignore`` only when a third-party stub
       leaves no choice.
   * - Tests
     - ``make test``
     - Every test passes on Linux, macOS and Windows with Python 3.11 to
       3.13. The suite runs offline: it never downloads model weights or
       contacts a model provider, a tile server or the Hugging Face Hub.

How the test suite is organized
-------------------------------

The tests live in ``tests/``, about fifty modules in all, and run with
``python -m pytest`` (``make test``). ``pyproject.toml`` adds
``--cov=rapidtools --cov-report=term-missing`` to every run, so each run ends
with a coverage table listing the uncovered lines of each module.

Most modules test one component and are named after it, such as
``test_mapillary_client.py``, ``test_street_tracking.py``,
``test_image_analyzers.py`` or ``test_gui_server.py``. A few cut across
components:

- ``test_edge_cases.py`` collects the error paths and fallbacks of the
  street-level modules (duplicate resolver, verifier, re-identification,
  localization and tracking helpers): unreadable crops, failed model calls,
  cancellation, parallel rays, sightings without a ground position, rays
  that meet beyond the range limit and singular merge gates.
- ``test_coverage_gaps.py`` collects small fallbacks in modules that are
  otherwise well covered (outline parsing, GUI server helpers, model
  listing in the workflow, the prompt builder, image asset parsing, provider
  rejections in the analyzer).
- ``test_api_harmonization.py`` checks the public names, the lazy imports
  and the deprecation shims of ``rapidtools``, and ``test_cancellation.py``
  checks that every long-running component honors its ``cancel_event``.

Put a new test next to the component it exercises. Use the cross-cutting
modules only when a branch needs a fixture that the component's own module
does not have.

Keeping the suite offline
-------------------------

Nothing in the suite reaches the network or loads a real checkpoint. Before
writing a test, it helps to know how that is arranged.

``tests/conftest.py`` has two fixtures that apply to every test.
``_no_huggingface_login`` replaces ``rapidtools.auth.get_token`` with a stub
that returns a test token, and
``rapidtools.models.local_base.ensure_huggingface_login`` with a stub that
returns ``True``. Since a token always appears to be cached, the real
``ensure_huggingface_login`` (which the re-identification embedder imports
from ``rapidtools.auth``) also returns ``True`` without logging in.
``_restore_package_logger`` undoes any ``configure_logging()`` call made by a
test or the GUI command, so logger settings do not leak from one test to the
next.

HTTP requests to model providers, tile servers and the Mapillary Graph API
are answered by ``requests-mock`` (the ``requests_mock`` fixture) or by small
fake clients. The ``FakeClient`` in ``test_street_objects.py``, for example,
serves canned images, detections and pixels for a synthetic survey.

Models are fakes too. ``test_models_local.py`` swaps the processor and model
loaders for stand-ins, and ``test_street_objects.py`` passes a ``FakeSAM3``
that returns fixed masks. For the re-identification embedder,
``test_edge_cases.py`` goes a step further and puts fake ``torch`` and
``transformers`` modules into ``sys.modules``, exposing only what the loader
touches (``AutoConfig``, ``AutoModel``, ``CLIPVisionModelWithProjection``,
``AutoImageProcessor``, ``inference_mode``), so both the CLIP and the DINOv2
code paths run in milliseconds. Vision-language models are replaced by
objects with a ``run_inference`` method that answer from the image file name
or from a script (``FakeModel`` in ``test_verification.py`` and
``test_image_analyzers.py``), and a ``judge`` callable does the same for the
duplicate resolver.

Rasters, GeoJSON files, Shapefiles and crops are created in ``tmp_path``. No
data files are committed as fixtures.

Coverage
--------

Tests cover 99% of the package's statements. The coverage badge in the
README comes from the Ubuntu / Python 3.11 job of the CI matrix. The lines
still uncovered are the ones that need a service to fail in a particular
way: the GUI server's survey overview and route builders, the fallbacks for
invalid geometry in building regularization, cancellation inside a thread
pool, and a few rules of the tracker for distant sightings that only fire on
real survey geometry.

To see what a change left uncovered, read the ``Missing`` column of the
summary, or build the HTML report:

.. code-block:: bash

   make coverage         # writes htmlcov/index.html

A pull request should not lower the total. If a new branch is hard to reach
through the public API, test the helper directly, as the cross-cutting
modules do, rather than leaving it uncovered.

Writing a test
--------------

- Patch the network boundary with ``requests-mock`` or ``monkeypatch``.
  Never call a real service.
- Replace model and processor loaders with fakes. Never let
  ``transformers`` load a checkpoint.
- Keep fixtures small and deterministic, and build data in ``tmp_path``.
- Assert on what a user would see (attributes, files, log messages,
  exceptions), not on how a helper got there.
- Every bug fix gets a test that would have caught the bug.
- Run ``make check`` before opening a pull request; CI runs the same
  commands.

Documentation
-------------

Public classes and functions have Google-style docstrings with an
``Example:`` block, and the API reference is generated from them. A new
public object also needs an entry in ``docs/source/api/index.rst``. Build
the site with ``make docs`` and open ``docs/build/html/index.html``.
User-facing changes get a line in the ``Unreleased`` section of
``CHANGELOG.md``.
