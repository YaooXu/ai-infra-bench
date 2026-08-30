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
    "test_varied_page_sizes_share_one_correct_backing",
    "test_simple_cpu_capacity_counts_backing_once",
    "test_connector_registers_one_canonical_backing",
    "test_partially_strided_layout_is_rejected",
}
assert tests >= 22, f"expected at least 22 tests, got {tests}"
assert failures == 0 and errors == 0 and skipped == 0
assert required <= names, f"missing required tests: {required - names}"
