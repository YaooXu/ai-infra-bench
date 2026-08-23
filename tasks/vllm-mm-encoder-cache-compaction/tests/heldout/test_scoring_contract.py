"""Trusted self-tests for the correctness reward aggregator.

These tests validate verifier logic and are not candidate-scored nodes.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


SCORING_PATH = Path(os.environ.get("BENCH_TESTS_DIR", "/tests")) / "scoring.py"
spec = importlib.util.spec_from_file_location("trusted_bench_scoring", SCORING_PATH)
assert spec is not None and spec.loader is not None
scoring = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scoring
spec.loader.exec_module(scoring)
_TestResult = scoring.TestResult
calculate_scores = scoring.calculate_scores


MANIFEST = {
    "score_groups": {"first": 0.6, "second": 0.4},
    "core_cap": {"value": 0.35, "apply_when_all_fail": ["A", "B"]},
    "requirements": [
        {"id": "A", "score_group": "first", "test_node_ids": ["a"]},
        {"id": "B", "score_group": "second", "test_node_ids": ["b"]},
    ],
}


def test_all_pass_is_one():
    score = calculate_scores(
        MANIFEST,
        [_TestResult("a", "passed"), _TestResult("b", "passed")],
    )
    assert score["reward"] == 1.0
    assert score["raw_correctness"] == 1.0
    assert score["core_cap_applied"] == 0


def test_partial_failure_is_binary_failure_with_local_diagnostics():
    score = calculate_scores(
        MANIFEST,
        [_TestResult("a", "passed"), _TestResult("b", "failed")],
    )
    assert score["reward"] == 0.0
    assert score["raw_correctness"] == 0.6
    assert score["validity_gate"] == 1


def test_all_core_fail_applies_cap():
    score = calculate_scores(
        MANIFEST,
        [_TestResult("a", "failed"), _TestResult("b", "error")],
    )
    assert score["reward"] == 0.0
    assert score["raw_correctness"] == 0.0
    assert score["core_cap_applied"] == 1


def test_validity_gate_zero_forces_zero():
    score = calculate_scores(
        MANIFEST,
        [_TestResult("a", "passed"), _TestResult("b", "passed")],
        validity_gate=0,
    )
    assert score["reward"] == 0.0
    assert score["raw_correctness"] == 0.0


def test_invalid_validity_gate_is_rejected():
    with pytest.raises(ValueError, match="validity_gate"):
        calculate_scores(MANIFEST, [], validity_gate=2)


def test_weights_must_sum_to_one():
    invalid = {**MANIFEST, "score_groups": {"first": 0.7, "second": 0.4}}
    with pytest.raises(ValueError, match="sum to one"):
        calculate_scores(invalid, [])


def test_unknown_or_duplicate_results_are_rejected():
    with pytest.raises(ValueError, match="undeclared"):
        calculate_scores(MANIFEST, [_TestResult("unknown", "passed")])
    with pytest.raises(ValueError, match="duplicate test node"):
        calculate_scores(
            MANIFEST,
            [_TestResult("a", "passed"), _TestResult("a", "failed")],
        )


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="unknown test status"):
        calculate_scores(MANIFEST, [_TestResult("a", "flaky")])
