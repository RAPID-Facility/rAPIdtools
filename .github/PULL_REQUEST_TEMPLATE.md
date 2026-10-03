<!-- Thanks for contributing. Keep one topic per pull request and fill in the sections below;
     CONTRIBUTING.md explains each item. Delete anything that does not apply. -->

## Summary

<!-- What changes and why. Link the issue it closes, e.g. "Closes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New feature or component
- [ ] Behaviour change to an existing component (describe the migration below)
- [ ] Documentation, examples or wiki
- [ ] Tests, tooling or CI

## How it was verified

<!-- Commands you ran and what they showed. For geospatial or model changes, say which data
     (sample dataset, projected raster, provider) you tried it on. -->

## Checklist

- [ ] `make check` passes locally (ruff lint and format, mypy, pytest).
- [ ] Tests cover the change and stay offline (no network, no model weights).
- [ ] Public functions and classes have Google-style docstrings with an `Example:` block; new public objects are listed in `docs/source/api/index.rst`.
- [ ] `CHANGELOG.md` has an entry under `Unreleased`.
- [ ] Renamed or removed parameters keep working with a `DeprecationWarning`.
- [ ] No credentials, tokens, model weights, rasters or generated outputs are included.
- [ ] Commit subjects follow `bc - <Sentence describing the change>`.
