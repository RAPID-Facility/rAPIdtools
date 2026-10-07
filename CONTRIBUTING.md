# Contributing to rAPIdtools

Thank you for your interest in rAPIdtools. Bug reports, documentation fixes,
new examples and code are all welcome. This page explains how to set up a
development environment, what a change must pass before it is merged, and how
we review pull requests. Participation is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md); security problems should be reported as
described in our [Security Policy](SECURITY.md), not in a public issue.

## Ways to contribute

Not every contribution is code. Feedback, questions, bug reports and feature
ideas all help, and there are two ways to send them depending on whether you
use GitHub. Documentation fixes and code go through pull requests, described
further down.

### Feedback without a GitHub account

Email is all you need: [uwrapid@uwrapid.org](mailto:uwrapid@uwrapid.org?subject=rAPIdtools%20feedback).
Send feedback, a question, a bug or a feature idea; say what you were doing,
what you expected and what happened. For a bug, include the rapidtools
version (`python -c "import rapidtools; print(rapidtools.__version__)"`),
your operating system and Python version, the smallest script that
reproduces the problem and the full error output. A maintainer will answer
and, with your agreement, turn the message into a GitHub issue so others can
follow it. Never include API keys, Mapillary tokens or Hugging Face tokens;
redact them.

### Feedback with a GitHub account

The three forms below open an issue on the repository, where other users can
see it, add to it and follow the fix; each asks only for what we need to act
on it. **They need a GitHub account**: clicking a link asks you to sign in
first, and GitHub offers to create a free account on that page. If you would
rather not, use the email route above.

| I want to... | Open |
|---|---|
| Report something that fails or gives a wrong result | [Bug report](https://github.com/RAPID-Facility/rAPIdtools/issues/new?template=bug_report.yml) |
| Ask for a capability that is missing | [Feature request](https://github.com/RAPID-Facility/rAPIdtools/issues/new?template=feature_request.yml) |
| Ask a question, or say what worked and what did not | [Feedback or question](https://github.com/RAPID-Facility/rAPIdtools/issues/new?template=feedback.yml) |

A feature request should describe the use case before the solution, so we
can agree on the approach before any code is written; the
[wiki](https://github.com/RAPID-Facility/rAPIdtools/wiki) and the
[documentation](https://rapid-facility.github.io/rAPIdtools/) describe how
the existing pieces fit together. The same redaction rule applies: never
paste API keys or tokens into an issue.

### Documentation and code

- **Improve the documentation.** Docstrings, the Sphinx pages under
  `docs/source/`, the examples under `examples/` and the wiki can all be
  edited; small fixes do not need an issue first.
- **Submit code.** Fork the repository, branch from `main`, and open a pull
  request. The rest of this page covers what that involves.

## Development setup

rAPIdtools requires Python 3.11 or newer. Clone your fork and install the
package in editable mode with the development extras:

```bash
git clone https://github.com/<your-user>/rAPIdtools.git
cd rAPIdtools
make install          # python -m pip install -e ".[dev]"
```

On a machine without a GPU, install CPU-only PyTorch first so the environment
stays small. `torch` and `torchvision` must come from the same index, or every
model module fails to import:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

`make help` lists every developer target. `PYTHON` can be overridden for any
of them, for example `make test PYTHON=python3.12`.

## Before you open a pull request

Run the same checks as continuous integration:

```bash
make check            # ruff lint and format check, mypy, pytest
```

The three parts, and what they expect:

| Check | Command | Expectation |
| --- | --- | --- |
| Style | `make lint` (`make format` to fix) | Ruff is the style gate: line length 88, single quotes, rule sets `E`, `W`, `F`, `I`, `UP`, `B`, `Q` (see `[tool.ruff]` in `pyproject.toml`). |
| Types | `make typecheck` | `mypy rapidtools` is clean on `main`; keep it that way and annotate public signatures. Avoid `# type: ignore` unless a third-party stub forces it, and say why in a comment. |
| Tests | `make test` | All tests pass. The suite is offline: it never downloads model weights or contacts a provider, tile server or the Hugging Face Hub. |

### Writing tests

Tests live in `tests/`, one module per component, plus a few cross-cutting
modules: `test_edge_cases.py` (error branches of the street-level modules),
`test_coverage_gaps.py` (small fallbacks of otherwise well-covered modules),
`test_api_harmonization.py` (public names and deprecation shims) and
`test_cancellation.py`. Put a test next to the component it exercises. Follow
the patterns already there:

- Patch the network boundary with `requests-mock` or `monkeypatch`; never call
  a real service.
- Replace model and processor loaders with fakes, as `tests/test_models_local.py`
  does, so `transformers` never loads a checkpoint. `tests/conftest.py` already
  stubs the Hugging Face token lookup and login for every test. Where a loader
  must run end to end, inject fake `torch` and `transformers` modules into
  `sys.modules`, as the re-identification test in `tests/test_edge_cases.py`
  does.
- Keep fixtures deterministic and small; build rasters and GeoJSON in
  `tmp_path` rather than committing data files.
- Add a test for every bug fix that would have caught it.
- Coverage sits at 99% and every run prints the uncovered lines
  (`--cov-report=term-missing` is in the pytest options); do not let a change
  lower the total. When a branch is hard to reach through the public API,
  test the helper directly rather than leaving it uncovered.

The documentation's *Development and testing* page describes how the suite
stays offline in more detail.

### Documentation and changelog

- Public classes and functions carry Google-style docstrings with an `Example:`
  block. The API reference is generated from them, so a new public object also
  needs an entry in `docs/source/api/index.rst`.
- Build the docs with `make docs` and check the result in
  `docs/build/html/index.html` when you change user-facing behaviour.
- Add a line to the `Unreleased` section of `CHANGELOG.md` under `Added`,
  `Changed`, `Deprecated`, `Removed` or `Fixed`. The file follows
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

### Compatibility

- Keep the 0.x promise honest: minor releases may change public APIs, but a
  renamed parameter or class should keep the old name working with a
  `DeprecationWarning` for at least one release, as the 0.2.0 renames did.
- All geometries exchanged through `PhysicalAssetCollection` are WGS84
  (EPSG:4326); rasters may be in any CRS and are reprojected on read.
- Heavy imports (`torch`, `transformers`) stay inside the model modules so
  that `import rapidtools` remains fast and API-only users are not forced to
  load them.

## Commits and pull requests

- Commit subjects follow the repository's pattern
  `bc - <Sentence describing the change>`, for example
  `bc - Making the ego-vehicle filter linear in the sequence length`. Put the
  reasoning in the body.
- Keep a pull request to one topic. Describe what changed, why, and how you
  verified it; link the issue it closes.
- Continuous integration must be green: the lint and type-check job plus the
  test matrix on Linux, macOS and Windows with Python 3.11 to 3.13.
- A maintainer reviews every pull request. Expect questions about correctness
  on real data (projected rasters, antimeridian edges, provider failure modes)
  and about memory use on large regions; those are where most bugs have lived.
- Do not commit credentials, tokens, model weights, large rasters or generated
  outputs. `.gitignore` already excludes the usual locations; if you add an
  example that needs a key, read it from an environment variable.

## Releases

Maintainers cut releases by moving the `Unreleased` entries in `CHANGELOG.md`
under a new version heading, setting the same version in `pyproject.toml` and
`CITATION.cff`, and pushing a `v<version>` tag. The `Release` workflow creates
the GitHub Release from the changelog section and refuses a tag that does not
match `pyproject.toml`.

## License

rAPIdtools is released under the [BSD-3-Clause license](LICENSE). By
contributing, you agree that your contributions are licensed under the same
terms. Source files carry the license header, a `Contributors:` list and a
`Last updated:` date; add yourself to the list when you make a substantial
change to a file.

## Questions

Open a [discussion or issue](https://github.com/RAPID-Facility/rAPIdtools/issues)
on GitHub. For anything that should not be public, including suspected
security problems, email <uwrapid@uwrapid.org>.
