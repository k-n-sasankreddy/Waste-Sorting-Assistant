# Project progress

## Phase 0 — Scope, environment, and contracts

- Added Python packaging, a development dependency lock, and Git ignore rules.
- Added validated local defaults in `config/dev.yaml` and shared gateway and
  worker payload schemas under `src/waste_assistant/shared/`.
- Added a deterministic test-only predictor and tests for configuration and
  synthetic score isolation.
- Documented setup, proposed resource limits, the one-process gateway
  constraint, and the first real-model milestone in the README.

Development host was Windows x64 with an Intel Core Ultra 7 258V, eight
logical processors, and Python 3.12.14. The checks use CPU only. GPU and LAN
capacity remain unverified.

Verified in a fresh virtual environment using the locked dependencies and
the README install commands. Package import and configuration loading worked.
`pip check`, all five pytest cases, and Ruff lint and formatting checks passed.

Phase 1 starts on request. Verify TrashNet provenance and terms, audit files
and duplicate groups, then generate and verify reproducible stratified splits.
