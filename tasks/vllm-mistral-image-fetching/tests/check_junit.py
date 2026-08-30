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
    "test_nested_batch_preserves_structure",
    "test_local_png_path_loads_real_pixels",
    "test_nested_invalid_value_raises_instead_of_being_silently_kept",
    "test_fetched_images_continue_through_existing_encoder",
}
assert tests >= 20, f"expected at least 20 tests, got {tests}"
assert failures == 0 and errors == 0 and skipped == 0
assert required <= names, f"missing required tests: {required - names}"
