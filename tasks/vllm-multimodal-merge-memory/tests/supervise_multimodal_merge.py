#!/usr/bin/env python3
"""Trusted parent which owns case enumeration, grading, and reward output.

The supervisor never imports candidate-controlled code.  Every observation is
produced by an unprivileged, nonce-bound worker subprocess whose raw output is
interpreted here.  A worker that exits early, terminates itself, or reports an
incomplete observation cannot make this parent write a passing reward.
"""

from __future__ import annotations

import json
import os
import pwd
import secrets
import stat
import subprocess
import sys
from pathlib import Path


REWARD = Path("/logs/verifier/reward.txt")
# Harbor parses every key of ``reward.json`` as a reward channel, so the
# grading summary is written to a separate diagnostics file instead.
SUMMARY = Path("/logs/verifier/grading-summary.json")
WORKER = Path("/tests/verify_multimodal_merge.py")
RESULT_PREFIX = "AI_INFRA_OBSERVATION="

# Peak temporary CUDA allocation, expressed as a multiple of the target
# ``inputs_embeds`` tensor.  The measured curve on the task image (A100/H20
# class, torch 2.10+cu129, 8192x512 tokens, five repeats each) is:
#
#   implementation   cpu-mask f16/bf16   cpu-mask f32
#   base             ValueError (bug)    ValueError (bug)
#   oracle           1.003               0.500
#   alternative      1.003               0.500
#
# while every implementation that keeps materialising the full mask on the
# device measures 5.509 (f16/bf16) and 2.588 (f32).  The gate below sits far
# above the two independent passing designs and far below the wasteful one, so
# it separates "temporaries proportional to the placeholders" from "temporaries
# proportional to the whole request" without prescribing an exact curve.
MAX_PEAK_RATIO = 4.0

MERGE_CASES = (
    ("merge_cpu_mask_float16", "cpu", "float16"),
    ("merge_cpu_mask_bfloat16", "cpu", "bfloat16"),
    ("merge_cpu_mask_float32", "cpu", "float32"),
    ("merge_cuda_mask_bfloat16", "cuda", "bfloat16"),
)

# Cardinality mismatches on the declared CPU-mask path.  ``1 -> 3`` is the
# broadcastable shape: a single embedding row can be silently stretched across
# three placeholders by indexed assignment, so an implementation that relies on
# the assignment raising is caught here.  The remaining shapes are
# non-broadcastable in both directions.
CARDINALITY_CASES = (
    ("cardinality_cpu_1_to_3", "cpu", 1, 3),
    ("cardinality_cpu_1_to_9", "cpu", 1, 9),
    ("cardinality_cpu_5_to_3", "cpu", 5, 3),
    ("cardinality_cpu_2_to_4", "cpu", 2, 4),
    ("cardinality_cpu_2_to_6", "cpu", 2, 6),
)

# No cardinality case is graded on the same-device (CUDA mask) path.  Measured
# on the task image, the unmodified base and the declared alternative both
# accept a 5 -> 3 same-device mismatch silently, while a device-side assert is
# what surfaces the non-broadcastable shapes.  Demanding a clean error there
# would grade one particular design rather than the declared contract, so the
# same-device path is covered by ``merge_cuda_mask_bfloat16`` for
# compatibility and correctness only.

OTHER_CASES = ("empty_identity", "model_interface_path")

CASES: tuple[str, ...] = (
    tuple(name for name, _, _ in MERGE_CASES)
    + tuple(name for name, _, _, _ in CARDINALITY_CASES)
    + OTHER_CASES
)

CASE_REQUESTS: dict[str, dict[str, object]] = {}
for _name, _mask_device, _dtype in MERGE_CASES:
    CASE_REQUESTS[_name] = {
        "kind": "merge",
        "mask_device": _mask_device,
        "dtype": _dtype,
    }
for _name, _mask_device, _num_embeddings, _num_placeholders in CARDINALITY_CASES:
    CASE_REQUESTS[_name] = {
        "kind": "cardinality",
        "mask_device": _mask_device,
        "num_embeddings": _num_embeddings,
        "num_placeholders": _num_placeholders,
    }
for _name in OTHER_CASES:
    CASE_REQUESTS[_name] = {"kind": _name}


def write_reward(value: int, *, exclusive: bool = False) -> None:
    flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_CREAT
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    descriptor = os.open(REWARD, flags, 0o600)
    try:
        os.fchown(descriptor, 0, 0)
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, f"{value}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_reward() -> None:
    verifier_dir = REWARD.parent
    try:
        current = verifier_dir.lstat()
    except FileNotFoundError:
        verifier_dir.mkdir(parents=True, mode=0o700)
    else:
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
            verifier_dir.unlink()
            verifier_dir.mkdir(parents=True, mode=0o700)
    os.chown(verifier_dir, 0, 0)
    verifier_dir.chmod(0o700)
    for path in (REWARD, SUMMARY, REWARD.with_name("reward.json")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    # Default to a failing reward before any candidate code runs, so an
    # interrupted or killed supervisor still leaves a zero behind.
    write_reward(0, exclusive=True)


def write_summary(payload: dict[str, object]) -> None:
    flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_CREAT | os.O_TRUNC
    descriptor = os.open(SUMMARY, flags, 0o600)
    try:
        os.fchown(descriptor, 0, 0)
        os.write(descriptor, (json.dumps(payload) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def trusted_file(path: Path) -> bool:
    info = path.stat()
    return (
        info.st_uid == 0
        and stat.S_ISREG(info.st_mode)
        and not (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    )


def observation_passes(case: str, value: object) -> tuple[bool, str]:
    request = CASE_REQUESTS[case]
    kind = request["kind"]

    if kind == "merge":
        if not isinstance(value, dict):
            return False, f"expected a merge observation, got {value!r}"
        for flag, detail in (
            ("returned_input_tensor", "merge must return the mutated input tensor"),
            ("first_segment_placed", "first nested embedding segment is misplaced"),
            ("second_segment_placed", "second nested embedding segment is misplaced"),
            ("text_rows_preserved", "non-placeholder embeddings were modified"),
        ):
            if value.get(flag) is not True:
                return False, detail
        if value.get("mask_copied_to_cuda"):
            return False, (
                "the placeholder mask itself was materialised on CUDA: "
                f"{value.get('mask_copied_to_cuda')!r}"
            )
        ratio = value.get("peak_ratio")
        if not isinstance(ratio, (int, float)):
            return False, f"missing peak allocation ratio: {value!r}"
        if request["mask_device"] == "cpu" and ratio >= MAX_PEAK_RATIO:
            return False, f"temporary CUDA allocation is excessive: {ratio:.3f}"
        return True, f"peak_ratio={ratio:.3f}"

    if kind == "cardinality":
        if not isinstance(value, dict):
            return False, f"expected a cardinality observation, got {value!r}"
        num_embeddings = request["num_embeddings"]
        num_placeholders = request["num_placeholders"]
        if not value.get("raised_value_error"):
            return False, (
                f"cardinality mismatch {num_embeddings}!={num_placeholders} was "
                f"accepted ({value.get('outcome')!r})"
            )
        message = str(value.get("message", ""))
        lowered = message.lower()
        if (
            str(num_embeddings) not in message
            or str(num_placeholders) not in message
            or not any(w in lowered for w in ("multimodal", "embedding", "token"))
            or not any(w in lowered for w in ("placeholder", "mask", "expected"))
        ):
            return False, f"uninformative cardinality error: {message}"
        return True, "informative ValueError"

    if kind == "empty_identity":
        valid = isinstance(value, dict) and value.get("identity") is True
        return valid, f"empty multimodal input must be an identity operation: {value!r}"

    if kind == "model_interface_path":
        if not isinstance(value, dict):
            return False, f"expected an interface observation, got {value!r}"
        if value.get("replacements_placed") is not True:
            return False, "production model interface misplaced replacements"
        if value.get("text_rows_preserved") is not True:
            return False, "production model interface changed text embeddings"
        return True, "production interface honoured the CPU mask"

    return False, f"unknown case {case}"


def run_case(python_bin: Path, agent: pwd.struct_passwd, case: str) -> tuple[bool, str]:
    nonce = secrets.token_hex(32)
    command = [
        "/usr/bin/setpriv",
        f"--reuid={agent.pw_uid}",
        f"--regid={agent.pw_gid}",
        "--init-groups",
        "--no-new-privs",
        str(python_bin),
        "-I",
        str(WORKER),
    ]
    request = json.dumps({"case": case, "nonce": nonce, **CASE_REQUESTS[case]}) + "\n"
    try:
        result = subprocess.run(
            command,
            cwd="/workspace/repo",
            env={
                **os.environ,
                "PYTHONPATH": "/workspace/repo",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HOME": agent.pw_dir,
                "XDG_CACHE_HOME": f"{agent.pw_dir}/.cache",
            },
            input=request,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "candidate observation process timed out"

    lines = [
        line.removeprefix(RESULT_PREFIX)
        for line in result.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    # An early ``SystemExit(0)`` or ``os._exit(0)`` inside candidate code yields
    # a zero exit status with no observation line; both conditions fail here.
    if result.returncode != 0 or len(lines) != 1:
        return False, (
            "candidate observation did not complete "
            f"(exit={result.returncode}, observations={len(lines)})\n{result.stdout}"
        )
    try:
        observation = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return False, f"malformed observation: {exc}"
    if (
        observation.get("case") != case
        or observation.get("nonce") != nonce
        or observation.get("error") is not None
    ):
        return False, f"invalid observation envelope: {observation!r}"
    return observation_passes(case, observation.get("value"))


def main() -> int:
    if os.geteuid() != 0:
        print("FAIL: verifier supervisor must run as root")
        return 0
    prepare_reward()
    if len(sys.argv) != 2:
        print("FAIL: trusted Python path was not supplied")
        return 0
    python_bin = Path(sys.argv[1]).resolve()
    supervisor = Path(__file__).resolve()
    if not all(trusted_file(path) for path in (python_bin, supervisor, WORKER)):
        print("FAIL: verifier executable or scripts are not root-owned/read-only")
        return 0

    agent = pwd.getpwnam("agent")
    completed: list[str] = []
    failures: list[str] = []
    for case in CASES:
        passed, detail = run_case(python_bin, agent, case)
        if passed:
            completed.append(case)
            print(f"PASS: {case}: {detail}")
        else:
            failures.append(f"{case}: {detail}")
            print(f"FAIL: {failures[-1]}")

    graded = len(completed) == len(CASES) and not failures
    if graded:
        write_reward(1)
        print(f"PASS: trusted parent graded all {len(CASES)} cases")
    else:
        print(f"FAIL: trusted parent completed {len(completed)}/{len(CASES)} cases")
    write_summary(
        {
            "reward": 1 if graded else 0,
            "cases_total": len(CASES),
            "cases_passed": len(completed),
            "failures": failures,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
