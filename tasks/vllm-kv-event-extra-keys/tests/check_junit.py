#!/usr/bin/env python3
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


root = ET.parse(sys.argv[1]).getroot()
suites = list(root.iter("testsuite"))
tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
names = {case.attrib.get("name") for case in root.iter("testcase")}
required = {
    "test_null_blocks_do_not_break_event_alignment",
    "test_prompt_embeddings_use_compact_per_range_fingerprints",
    "test_msgpack_round_trip_preserves_extra_keys",
    "test_router_can_reconstruct_hidden_combination",
}
assert tests >= 19, f"expected at least 19 tests, got {tests}"
assert failures == 0 and errors == 0 and skipped == 0
assert required <= names, f"missing required tests: {required - names}"
