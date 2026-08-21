import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path("/app")
LOG_DIR = Path("/logs/verifier")
JUNIT_PATH = LOG_DIR / "pytest-junit.xml"
UPSTREAM_TEST_PATCH = Path("/tests/heldout/upstream-tests.patch")

CORRECTNESS_WEIGHTS = {
    "correctness_ordering": 0.30,
    "correctness_lifecycle": 0.25,
    "correctness_recovery": 0.30,
    "correctness_regressions": 0.15,
}
PERFORMANCE_WEIGHTS = {
    "performance_hotpath": 0.70,
    "performance_timing": 0.30,
}


def classify(classname: str, test_name: str) -> str:
    identity = f"{classname} {test_name}"

    if "test_blocked_scheduler_cpu_microbenchmark" in identity:
        return "performance_timing"
    if any(
        marker in identity
        for marker in (
            "test_performance",
            "test_remote_kv_update_work_is_event_driven",
            "test_blocked_requests_do_not_churn_the_schedulable_queue",
        )
    ):
        return "performance_hotpath"

    if any(
        marker in identity
        for marker in (
            "test_behavior",
            "test_priority_ordering",
            "test_fcfs_mixed_blocked_waiting_types_keep_order",
            "test_priority_order_is_preserved_across_blocked_and_ready_requests",
            "test_blocked_high_priority_request_does_not_starve_ready_work",
            "test_remote_kv_promotion_keeps_fcfs_with_fsm_prefix",
            "test_fcfs_mixed_skipped_waiting_types_keep_order",
        )
    ):
        return "correctness_ordering"

    if any(
        marker in identity
        for marker in (
            "test_queue_isolation",
            "test_async_remote_kv_requests_leave_the_schedulable_waiting_queue",
            "test_blocked_requests_are_counted_reported_and_cleaned_on_abort",
            "test_remote_prefill_lifecycle",
            "test_abort_request_waiting_for_remote_kvs",
            "test_abort_request_finished_recving",
        )
    ):
        return "correctness_lifecycle"

    if any(
        marker in identity
        for marker in (
            "test_error_propagation",
            "test_invalid_blocks_correctness",
            "test_kv_load_failure_recovery",
        )
    ):
        return "correctness_recovery"

    return "correctness_regressions"


def zero_scores() -> dict[str, float]:
    return {name: 0.0 for name in (*CORRECTNESS_WEIGHTS, *PERFORMANCE_WEIGHTS)}


def apply_upstream_test_patch() -> tuple[bool, str]:
    check = subprocess.run(
        [
            "git",
            "apply",
            "--check",
            "--whitespace=nowarn",
            str(UPSTREAM_TEST_PATCH),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check.returncode == 0:
        applied = subprocess.run(
            [
                "git",
                "apply",
                "--whitespace=nowarn",
                str(UPSTREAM_TEST_PATCH),
            ],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return applied.returncode == 0, applied.stdout

    reverse_check = subprocess.run(
        [
            "git",
            "apply",
            "--reverse",
            "--check",
            "--whitespace=nowarn",
            str(UPSTREAM_TEST_PATCH),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if reverse_check.returncode == 0:
        return True, "Upstream PR tests were already present.\n"

    return False, (
        "Unable to install the hidden upstream PR tests.\n"
        f"Forward check:\n{check.stdout}\n"
        f"Reverse check:\n{reverse_check.stdout}\n"
    )


def run_tests() -> tuple[dict[str, float], str]:
    patch_ok, patch_output = apply_upstream_test_patch()
    if not patch_ok:
        return zero_scores(), patch_output

    command = [
        "python3",
        "-m",
        "pytest",
        "-q",
        "-s",
        f"--junitxml={JUNIT_PATH}",
        "/tests/required/test_behavior.py",
        "/tests/required/test_queue_isolation.py",
        "/tests/heldout/test_priority_ordering.py",
        "/tests/heldout/test_performance.py",
        "tests/v1/core/test_scheduler.py::test_remote_kv_promotion_keeps_fcfs_with_fsm_prefix",
        "tests/v1/core/test_scheduler.py::test_fcfs_mixed_skipped_waiting_types_keep_order",
        "tests/v1/core/test_scheduler.py::test_abort_request_waiting_for_remote_kvs",
        "tests/v1/core/test_scheduler.py::test_abort_request_finished_recving",
        "tests/v1/kv_connector/unit/test_error_propagation.py",
        "tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py",
        "tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py",
        "tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py",
        "tests/v1/core/test_scheduler.py::test_schedule",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    try:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1200,
            check=False,
        )
        output = patch_output + result.stdout
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or b""
        output = (
            captured.decode(errors="replace")
            if isinstance(captured, bytes)
            else captured
        )
        output = patch_output + output
        output += "\nVerifier timed out after 1200 seconds.\n"
        return zero_scores(), output

    group_names = (*CORRECTNESS_WEIGHTS, *PERFORMANCE_WEIGHTS)
    counts = {name: 0 for name in group_names}
    failures = {name: 0 for name in group_names}
    if JUNIT_PATH.exists():
        root = ET.parse(JUNIT_PATH).getroot()
        for case in root.iter("testcase"):
            group = classify(
                case.attrib.get("classname", ""),
                case.attrib.get("name", ""),
            )
            counts[group] += 1
            if case.find("failure") is not None or case.find("error") is not None:
                failures[group] += 1

    scores = {
        name: (
            (counts[name] - failures[name]) / counts[name] if counts[name] > 0 else 0.0
        )
        for name in group_names
    }
    output += f"\nGrouped cases: {counts}\nGrouped failures: {failures}\n"
    return scores, output


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    return sum(scores[name] * weight for name, weight in weights.items())


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    scores, output = run_tests()
    (LOG_DIR / "pytest.log").write_text(output)
    print(output)

    correctness = weighted_score(scores, CORRECTNESS_WEIGHTS)
    performance = weighted_score(scores, PERFORMANCE_WEIGHTS)
    reward = 0.65 * correctness + 0.35 * performance
    rewards = {
        "reward": reward,
        "correctness": correctness,
        "performance": performance,
        **scores,
    }
    (LOG_DIR / "reward.json").write_text(json.dumps(rewards, indent=2) + "\n")


if __name__ == "__main__":
    main()
