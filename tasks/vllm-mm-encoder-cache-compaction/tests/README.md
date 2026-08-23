# Verifier layout

`required/` contains the core range, cache/scheduler, physical-storage, and
gather/merge behavior that every valid solution must address.

`heldout/` contains profiling/connector integration, lifecycle regressions,
hardened edge cases, and trusted scorer self-tests. These tests do not add
requirements beyond `instruction.md`; they vary execution paths and inputs to
reject partial or hard-coded implementations. The formal Harbor reward is
binary; continuous capability completion is retained in `scoring.json` only.

`test.sh` is the only Harbor entrypoint. It clears prior outputs, verifies the
frozen test and environment manifests, runs each candidate-bound pytest node in
isolation, and writes `reward.json`, diagnostic `scoring.json`, and compatibility `reward.txt` under
`/logs/verifier`.

The dispatcher starts Python in isolated, no-bytecode mode and disables the
pytest cache provider and third-party plugin autoloading. Candidate-controlled
`PYTHONPATH`, user-site packages, `conftest.py`, and stale verifier outputs are
not grading inputs.
