# Implementation CI Suite

This directory contains implementation-oriented pytest cases (post-HLS reports and full bitfile flow checks).

## Pipeline separation

The project keeps one GitLab entrypoint (`.gitlab-ci.yml`) but has two pipeline modes:

- default mode (`CI_PIPELINE_MODE` unset): runs normal pytest CI only
- implementation mode (`CI_PIPELINE_MODE=implementation`): runs only the implementation child pipeline

Implementation mode is dispatched to:

- `test/pytest/implementation/pytests.yml` (static job list)

Both normal and implementation templates reuse the shared base setup:

- `test/pytest/ci-base-template.yml`

and keep their behavior-specific logic in:

- `test/pytest/ci-template.yml` (normal matrix)
- `test/pytest/implementation/ci-template.yml` (implementation suite)

Tool exposure policy:

- normal template uses `vivado_hls`
- implementation template exposes `vivado_hls`, `vivado`, `vitis-run`, `v++`, and `vitis`
- implementation jobs select tool-exposure mode by extending a script profile in
  `implementation/pytests.yml` (`.implementation.script.tools-*`)
- implementation dataset output path is controlled by `IMPLEMENTATION_DATASET_DIR`
  (default in CI: `test/pytest/implementation`)

## Adding new implementation tests

1. Add a new file in this folder matching `test_*.py`.
2. Add a matching static CI job entry in `test/pytest/implementation/pytests.yml`.
3. Set per-job tool versions (`VIVADO_VERSION`, `VITIS_VERSION`) in `pytests.yml`.
4. Select a per-job tool wrapper script profile in `pytests.yml`.
5. Keep tests collect-only: validate full flow success and emit dataset artifacts.
