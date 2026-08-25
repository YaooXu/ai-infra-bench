#!/usr/bin/env python3
"""Trusted verifier dispatcher for the frozen required/heldout suites.

The dispatcher validates the read-only verifier bytes, probes infrastructure
without trusting candidate output, collects every declared pytest node, and
runs nodes independently.  JUnit XML is the grading source of truth; human
pytest output is retained only through diagnostic hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from scoring import (
    TestResult,
    calculate_scores,
    diagnostic_evidence,
    semantic_evidence,
    semantic_hash,
)


TESTS_DIR = Path(os.environ.get("BENCH_TESTS_DIR", "/tests")).resolve()
WORKSPACE = Path(os.environ.get("BENCH_WORKSPACE", "/app")).resolve()
LOG_DIR = Path(os.environ.get("BENCH_LOG_DIR", "/logs/verifier")).resolve()
PYTHON = os.environ.get("BENCH_PYTHON", sys.executable)
PER_TEST_TIMEOUT = int(os.environ.get("BENCH_TEST_TIMEOUT", "180"))
ENVIRONMENT_SPEC = Path(
    os.environ.get(
        "BENCH_ENVIRONMENT_SPEC", "/opt/bench/lock/environment_spec.json"
    )
).resolve()
LOCK_ROOT = Path(os.environ.get("BENCH_LOCK_ROOT", "/opt/bench/lock")).resolve()
SCRIPTS_ROOT = Path(
    os.environ.get("BENCH_SCRIPTS_ROOT", "/opt/bench/scripts")
).resolve()
ENVIRONMENT_ROOT = Path(
    os.environ.get("BENCH_ENVIRONMENT_ROOT", "/opt/bench")
).resolve()
INTEGRITY_ANCHOR = Path(
    os.environ.get(
        "BENCH_INTEGRITY_ANCHOR", "/opt/bench/verifier-integrity.sha256"
    )
).resolve()


def sha256_bytes(value: bytes | str | None) -> str:
    if value is None:
        value = b""
    if isinstance(value, str):
        value = value.encode("utf-8", errors="replace")
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def safe_test_path(relative_value: str) -> Path:
    file_part = relative_value.split("::", 1)[0]
    relative = Path(file_part)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe pytest node ID: {relative_value!r}")
    candidate = (TESTS_DIR / relative).resolve()
    if TESTS_DIR not in candidate.parents:
        raise ValueError(f"pytest node escapes tests directory: {relative_value!r}")
    return candidate


def verify_integrity(
    tests_dir: Path = TESTS_DIR,
    environment_spec: Path = ENVIRONMENT_SPEC,
    environment_root: Path = ENVIRONMENT_ROOT,
    integrity_anchor: Path = INTEGRITY_ANCHOR,
) -> list[str]:
    """Verify every frozen verifier file and the environment contract."""
    path = tests_dir / "integrity.json"
    if not path.is_file() or path.is_symlink():
        return ["missing or unsafe tests/integrity.json"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid tests/integrity.json: {exc}"]

    expected = manifest.get("test_files")
    if not isinstance(expected, dict) or not expected:
        return ["tests/integrity.json has no test_files"]

    errors: list[str] = []
    actual: dict[str, Path] = {}
    for candidate in sorted(tests_dir.rglob("*")):
        if candidate.is_symlink():
            errors.append(f"symlink is forbidden in verifier: {candidate}")
        elif candidate.is_file():
            relative = candidate.relative_to(tests_dir).as_posix()
            if (
                relative != "integrity.json"
                and "__pycache__" not in candidate.parts
                and candidate.suffix not in {".pyc", ".pyo"}
            ):
                actual[relative] = candidate

    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        errors.append(
            f"verifier tree differs: missing={missing[:10]} extra={extra[:10]}"
        )

    for relative, expected_sha in expected.items():
        candidate = actual.get(relative)
        if candidate is None:
            continue
        if sha256_bytes(candidate.read_bytes()) != expected_sha:
            errors.append(f"test integrity mismatch: {relative}")

    executable_files = manifest.get("executable_files", [])
    if not isinstance(executable_files, list):
        errors.append("executable_files must be a list")
    else:
        for relative in executable_files:
            candidate = actual.get(str(relative))
            if candidate is None or not candidate.stat().st_mode & 0o111:
                errors.append(f"required executable bit is absent: {relative}")

    expected_environment = manifest.get("environment_spec_sha256")
    if not isinstance(expected_environment, str) or len(expected_environment) != 64:
        errors.append("environment_spec_sha256 is absent or invalid")
    elif not environment_spec.is_file() or environment_spec.is_symlink():
        errors.append("fixed environment spec is absent or unsafe")
    elif sha256_bytes(environment_spec.read_bytes()) != expected_environment:
        errors.append("fixed environment spec integrity mismatch")

    expected_environment_files = manifest.get("environment_files")
    if not isinstance(expected_environment_files, dict) or not expected_environment_files:
        errors.append("tests/integrity.json has no environment_files")
    else:
        for relative, expected_sha in expected_environment_files.items():
            relative_path = Path(str(relative))
            if (
                relative_path.is_absolute()
                or not relative_path.parts
                or ".." in relative_path.parts
            ):
                errors.append(f"unsafe environment integrity path: {relative}")
                continue
            candidate = (environment_root / relative_path).resolve()
            if environment_root not in candidate.parents:
                errors.append(f"environment integrity path escapes root: {relative}")
            elif not candidate.is_file() or candidate.is_symlink():
                errors.append(f"protected environment file is absent: {relative}")
            elif sha256_bytes(candidate.read_bytes()) != expected_sha:
                errors.append(f"protected environment file changed: {relative}")

    anchor_required = os.environ.get("BENCH_REQUIRE_INTEGRITY_ANCHOR", "0") == "1"
    if integrity_anchor.is_file() and not integrity_anchor.is_symlink():
        fields = integrity_anchor.read_text(encoding="utf-8").strip().split()
        actual_manifest_sha = sha256_bytes(path.read_bytes())
        if len(fields) < 1 or fields[0] != actual_manifest_sha:
            errors.append("root-owned verifier integrity anchor mismatch")
    elif anchor_required:
        errors.append("root-owned verifier integrity anchor is absent")
    return errors


def controlled_env() -> dict[str, str]:
    allowed = {
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "VLLM_VERSION_OVERRIDE",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(
        {
            "HOME": "/tmp/verifier-home",
            "BENCH_WORKSPACE": str(WORKSPACE),
            "BENCH_TESTS_DIR": str(TESTS_DIR),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def isolated_python(code: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, "-B", "-I", "-c", code],
        cwd=WORKSPACE,
        env=controlled_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def verify_locked_environment() -> list[str]:
    """Revalidate immutable control bytes from the independent image."""
    verifier = SCRIPTS_ROOT / "verify_artifacts.py"
    if not verifier.is_file():
        return ["public-build artifact verifier is absent from independent image"]
    proc = subprocess.run(
        [
            PYTHON,
            "-B",
            "-I",
            str(verifier),
            "--lock-root",
            str(LOCK_ROOT),
            "--control-only",
        ],
        cwd=WORKSPACE,
        env=controlled_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        return ["public-build environment integrity failed: " + (proc.stderr + proc.stdout)[-2000:]]
    return []


def run_scorer_self_tests() -> list[str]:
    """Run trusted reward-contract tests without candidate paths on sys.path."""
    code = (
        "import pathlib,sys,pytest; "
        f"root=pathlib.Path({str(TESTS_DIR)!r}); "
        "sys.path.insert(0,str(root)); "
        "raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--confcutdir',str(root),"
        "str(root/'heldout'/'test_scoring_contract.py')]))"
    )
    proc = subprocess.run(
        [PYTHON, "-B", "-I", "-c", code],
        cwd=TESTS_DIR,
        env=controlled_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        return ["trusted scorer self-tests failed: " + (proc.stdout + proc.stderr)[-2000:]]
    return []


def probe_environment() -> tuple[dict[str, object], list[str], list[str]]:
    infra_errors = verify_locked_environment() + run_scorer_self_tests()
    candidate_errors: list[str] = []
    metadata: dict[str, object] = {}

    no_gpu = isolated_python(
        "import json,torch; "
        "available=bool(torch.cuda.is_available()); "
        "count=int(torch.cuda.device_count()); "
        "assert not available, available; assert count==0, count; "
        "print(json.dumps({'gpu_model':None,'cuda':torch.version.cuda,"
        "'torch':torch.__version__,'cuda_device_count':count,"
        "'physical_gpu_visible':available}))"
    )
    if no_gpu.returncode != 0:
        infra_errors.append(
            "no-physical-GPU probe failed: " + no_gpu.stderr[-1000:]
        )
    else:
        try:
            metadata.update(json.loads(no_gpu.stdout.strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError) as exc:
            infra_errors.append(f"device probe returned invalid JSON: {exc}")

    network = isolated_python(
        "import socket; s=socket.socket(); s.settimeout(2); "
        "s.connect(('1.1.1.1',443)); print('EGRESS_SUCCEEDED')",
        timeout=5,
    )
    metadata["network_probe_returncode"] = network.returncode
    if network.returncode == 0:
        infra_errors.append("runtime network policy failed: outbound TCP succeeded")

    source_probe = isolated_python(
        "import importlib.util,json,pathlib,sys; root=pathlib.Path("
        + repr(str(WORKSPACE))
        + "); "
        "sys.path.insert(0,str(root)); import vllm; "
        "from vllm.multimodal.inputs import PlaceholderRange; "
        "from vllm.v1.core.encoder_cache_manager import EncoderCacheManager; "
        "source=pathlib.Path(vllm.__file__).resolve(); "
        "native_spec=importlib.util.find_spec('vllm._C'); "
        "native=(pathlib.Path(native_spec.origin).resolve() if native_spec and native_spec.origin else None); "
        "assert (root/'vllm/__init__.py').is_file(); "
        "source.relative_to(root); "
        "assert native is None or native.is_relative_to(root); "
        "print(json.dumps({'candidate_init':str(root/'vllm/__init__.py'),"
        "'candidate_import':str(source),'native_extension':str(native) if native else None,"
        "'target_python_imports_loaded':True}))"
    )
    if source_probe.returncode != 0:
        candidate_errors.append("global vLLM import probe failed: " + source_probe.stderr[-1500:])
    else:
        try:
            workspace_probe = json.loads(source_probe.stdout.strip().splitlines()[-1])
            imported = Path(str(workspace_probe["candidate_import"])).resolve()
            imported.relative_to(WORKSPACE)
            native_value = workspace_probe.get("native_extension")
            if native_value is not None:
                Path(str(native_value)).resolve().relative_to(WORKSPACE)
            if workspace_probe.get("target_python_imports_loaded") is not True:
                raise ValueError("target Python imports were not loaded")
            metadata["workspace_probe"] = workspace_probe
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            candidate_errors.append("vLLM import did not resolve to candidate workspace")

    # Loading the CUDA extension is diagnostic rather than a validity gate.
    # Missing driver-provided libcuda.so.1 in a gpus=0 container does not make
    # the Python correctness contract invalid.
    native_probe = isolated_python(
        "import json,pathlib,sys; "
        + "sys.path.insert(0," + repr(str(WORKSPACE)) + "); "
        "import vllm._C; print(json.dumps({'native_extension_loaded':True}))"
    )
    metadata["native_extension_loaded_without_gpu"] = native_probe.returncode == 0
    if native_probe.returncode != 0:
        metadata["native_extension_import_error"] = native_probe.stderr[-1500:]
    return metadata, infra_errors, candidate_errors


def collected_node_ids(output: str) -> list[str]:
    nodes: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "::" not in line:
            continue
        file_value, suffix = line.split("::", 1)
        path = Path(file_value)
        try:
            if path.is_absolute():
                file_value = path.resolve().relative_to(TESTS_DIR).as_posix()
            else:
                file_value = path.as_posix()
                if file_value.startswith("tests/"):
                    file_value = file_value[len("tests/") :]
        except ValueError:
            continue
        nodes.append(f"{file_value}::{suffix}")
    return nodes


def collect_node(node_id: str, env: dict[str, str]) -> tuple[bool, str]:
    test_path = safe_test_path(node_id)
    selector = str(test_path) + node_id[len(node_id.split("::", 1)[0]) :]
    proc = subprocess.run(
        [
            PYTHON,
            "-B",
            "-I",
            str(TESTS_DIR / "pytest_entry.py"),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--confcutdir",
            str(TESTS_DIR),
            selector,
        ],
        cwd=TESTS_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    collected = collected_node_ids(proc.stdout)
    ok = proc.returncode == 0 and collected == [node_id]
    detail = json.dumps(
        {
            "returncode": proc.returncode,
            "expected": node_id,
            "collected": collected,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        },
        sort_keys=True,
    )
    return ok, detail


def parse_junit(path: Path) -> str:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return "error"
    cases = list(root.iter("testcase"))
    if len(cases) != 1:
        return "error"
    case = cases[0]
    if case.find("failure") is not None:
        return "failed"
    if case.find("error") is not None:
        return "error"
    if case.find("skipped") is not None:
        return "skipped"
    return "passed"


def run_node(node_id: str, env: dict[str, str]) -> TestResult:
    safe_name = hashlib.sha256(node_id.encode()).hexdigest()[:16]
    junit = LOG_DIR / f"junit-{safe_name}.xml"
    junit.unlink(missing_ok=True)
    test_path = safe_test_path(node_id)
    selector = str(test_path) + node_id[len(node_id.split("::", 1)[0]) :]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [
                PYTHON,
                "-B",
                "-I",
                str(TESTS_DIR / "pytest_entry.py"),
                "-q",
                "-p",
                "no:cacheprovider",
                "--confcutdir",
                str(TESTS_DIR),
                "--junitxml",
                str(junit),
                selector,
            ],
            cwd=TESTS_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=PER_TEST_TIMEOUT,
            check=False,
        )
        status = parse_junit(junit) if junit.exists() else "error"
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout, stderr = exc.stdout, exc.stderr
    duration = time.monotonic() - started
    return TestResult(
        node_id=node_id,
        status=status,
        duration_seconds=duration,
        stdout_sha256=sha256_bytes(stdout),
        stderr_sha256=sha256_bytes(stderr),
    )


def suite_evidence(
    manifest: dict[str, object], results: tuple[TestResult, ...]
) -> dict[str, object]:
    suites: dict[str, dict[str, object]] = {}
    result_by_node = {item.node_id: item for item in results}
    for requirement in manifest["requirements"]:  # type: ignore[index]
        suite = str(requirement["suite"])
        entry = suites.setdefault(
            suite,
            {
                "requirement_ids": [],
                "collected_test_ids": [],
                "failed_test_ids": [],
            },
        )
        entry["requirement_ids"].append(requirement["id"])
        for node_id in requirement["test_node_ids"]:
            if node_id not in entry["collected_test_ids"]:
                entry["collected_test_ids"].append(node_id)
            if not result_by_node[node_id].passed:
                entry["failed_test_ids"].append(node_id)
    for entry in suites.values():
        entry["passed"] = not entry["failed_test_ids"]
    return suites


def failure_scoring(*, infra_error: int, reason: str) -> dict[str, object]:
    return {
        "reward": 0.0,
        "raw_correctness": 0.0,
        "validity_gate": 0,
        "infra_error": infra_error,
        "release_eligible": 0 if infra_error else 1,
        "failure_reason": reason,
    }


def record_failure(*, infra_error: int, reason: str) -> None:
    """Write a fresh binary Harbor reward and separate diagnostic record."""
    write_json(LOG_DIR / "reward.json", {"reward": 0.0})
    write_json(
        LOG_DIR / "scoring.json",
        failure_scoring(infra_error=infra_error, reason=reason),
    )


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    reward_path = LOG_DIR / "reward.json"
    scoring_path = LOG_DIR / "scoring.json"
    reward_path.unlink(missing_ok=True)
    scoring_path.unlink(missing_ok=True)

    integrity_errors = verify_integrity()
    if integrity_errors:
        write_json(LOG_DIR / "integrity_failure.json", {"errors": integrity_errors})
        record_failure(infra_error=0, reason="verifier_integrity_failure")
        return 1

    try:
        manifest = json.loads(
            (TESTS_DIR / "requirements.json").read_text(encoding="utf-8")
        )
        node_ids = sorted(
            {
                node
                for requirement in manifest["requirements"]
                for node in requirement["test_node_ids"]
            }
        )
        if not node_ids:
            raise ValueError("no scored test nodes")
        for node_id in node_ids:
            safe_test_path(node_id)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        write_json(LOG_DIR / "manifest_failure.json", {"error": str(exc)})
        record_failure(infra_error=0, reason="invalid_requirement_manifest")
        return 1

    env_metadata, infra_errors, candidate_errors = probe_environment()
    env_metadata["image_digest"] = os.environ.get("BENCH_IMAGE_DIGEST")
    env_metadata["structural_assertions"] = {}
    if infra_errors:
        write_json(LOG_DIR / "infra_error.json", {"errors": infra_errors})
        record_failure(infra_error=1, reason="infrastructure_preflight_failed")
        return 2
    if candidate_errors:
        write_json(LOG_DIR / "global_failure.json", {"errors": candidate_errors})
        record_failure(infra_error=0, reason="candidate_global_import_failure")
        return 1
    write_json(LOG_DIR / "workspace_probe.json", env_metadata["workspace_probe"])

    env = controlled_env()
    collection_errors: dict[str, str] = {}
    for node_id in node_ids:
        try:
            ok, detail = collect_node(node_id, env)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            ok, detail = False, str(exc)
        if not ok:
            collection_errors[node_id] = detail
    if collection_errors:
        write_json(LOG_DIR / "collection_errors.json", collection_errors)
        record_failure(infra_error=0, reason="pytest_collection_failed")
        return 1

    results = tuple(run_node(node_id, env) for node_id in node_ids)
    try:
        score = calculate_scores(manifest, results, validity_gate=1)
        if not math.isfinite(float(score["reward"])):
            raise ValueError("non-finite reward")
    except (KeyError, TypeError, ValueError) as exc:
        write_json(LOG_DIR / "scoring_failure.json", {"error": str(exc)})
        record_failure(infra_error=1, reason="trusted_scorer_failed")
        return 2

    score.update({"infra_error": 0, "release_eligible": 1})
    semantic = semantic_evidence(score, results, env_metadata)
    write_json(LOG_DIR / "semantic_evidence.json", semantic)
    write_json(LOG_DIR / "diagnostic_evidence.json", diagnostic_evidence(results))
    write_json(LOG_DIR / "semantic_hash.json", {"sha256": semantic_hash(semantic)})
    write_json(LOG_DIR / "requirement_scores.json", score["requirement_scores"])
    write_json(scoring_path, score)
    write_json(
        LOG_DIR / "verifier_result.json",
        {
            "schema_version": "1.0",
            "status": "complete",
            "environment": env_metadata,
            "suites": suite_evidence(manifest, results),
            "score": score,
        },
    )
    write_json(reward_path, {"reward": score["reward"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
