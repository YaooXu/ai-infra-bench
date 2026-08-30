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
    "test_mla_registration_uses_stride_for_address_and_payload_for_copy",
    "test_group_identity_and_null_blocks_drive_transfer_addresses",
    "test_hybrid_remote_decode_truncates_once",
    "test_shared_backing_is_registered_once",
}
assert tests >= 23, f"expected at least 23 tests, got {tests}"
assert failures == 0 and errors == 0 and skipped == 0
assert required <= names, f"missing required tests: {required - names}"
