#!/usr/bin/env python3
"""Trusted owner of lifecycle cases, grading, and reward output."""

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
# Resolved next to this script so the worker follows it when test.sh stages
# a root-owned copy outside the read-only /tests mount.
WORKER = Path(__file__).resolve().parent / "verify_retention.py"
RESULT_PREFIX = "AI_INFRA_OBSERVATION="
CASES = (
    "candidate_source",
    "live_request_retained",
    "normal_completion_releases",
    "waiting_cancel_releases",
    "running_cancel_releases",
    "streaming_wait_retained",
    "streaming_end_releases",
    "initial_prefix_hashes",
    "append_prefix_hashes",
    "streaming_continuation_hashes",
    "no_prefix_cache_completion",
)


def write_reward(value: int, *, exclusive: bool = False) -> None:
    flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_CREAT
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    descriptor = os.open(REWARD, flags, 0o644)
    try:
        os.fchown(descriptor, 0, 0)
        os.fchmod(descriptor, 0o644)
        os.write(descriptor, f"{value}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_reward() -> None:
    directory = REWARD.parent
    try:
        info = directory.lstat()
    except FileNotFoundError:
        directory.mkdir(parents=True, mode=0o755)
    else:
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            directory.unlink()
            directory.mkdir(parents=True, mode=0o755)
    # Mode 0755, not 0700: Harbor bind-mounts /logs from the host and reads
    # verifier/reward.txt from outside the container, where it runs as an
    # unprivileged user (uid 1001 on GitHub runners), so the directory has to
    # stay traversable and the reward file world-readable -- Harbor's
    # _parse_reward_text does stat() then read_text(), and a 0600 file makes the
    # read raise PermissionError even though the stat succeeds. This costs
    # nothing: both are root-owned with no group or other write bit, so
    # candidate code still cannot create, rename, unlink, symlink over,
    # hardlink, chmod, chown or rmtree anything here. Read access is harmless
    # because the candidate runs in a separate container that never mounts
    # /logs, and every write path below rewrites the file unconditionally.
    os.chown(directory, 0, 0)
    directory.chmod(0o755)
    try:
        REWARD.unlink()
    except FileNotFoundError:
        pass
    write_reward(0, exclusive=True)


def untrusted_reason(path: Path) -> str | None:
    """Return why *path* is not trusted, or None when it is."""
    info = path.stat()
    if info.st_uid != 0:
        return f"{path} is owned by uid {info.st_uid}, not root"
    if not stat.S_ISREG(info.st_mode):
        return f"{path} is not a regular file"
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return f"{path} is group/other writable (mode {stat.S_IMODE(info.st_mode):04o})"
    return None


def observation_passes(case: str, value: object) -> tuple[bool, str]:
    if case == "candidate_source":
        valid = isinstance(value, str) and value.startswith("/workspace/repo/vllm/")
        return valid, f"candidate import escaped repository: {value!r}"
    if case in {
        "normal_completion_releases",
        "waiting_cancel_releases",
        "running_cancel_releases",
        "streaming_end_releases",
        "no_prefix_cache_completion",
    }:
        expected = {"feature_alive": False, "request_alive": False, "owned": False}
        return value == expected, f"expected released state {expected!r}, got {value!r}"
    if case in {"live_request_retained", "streaming_wait_retained"}:
        expected = {"feature_alive": True, "request_alive": True, "owned": True}
        return value == expected, f"expected live state {expected!r}, got {value!r}"
    expected_counts = {
        "initial_prefix_hashes": [2],
        "append_prefix_hashes": [2, 3],
        "streaming_continuation_hashes": [2, 3, 4],
    }
    expected = expected_counts[case]
    valid = (
        isinstance(value, dict)
        and value.get("counts") == expected
        and value.get("prefixes_preserved") is True
        and value.get("unique_hashes") == expected[-1]
    )
    return valid, f"unexpected production prefix hashes: {value!r}"


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
    try:
        result = subprocess.run(
            command,
            cwd="/workspace/repo",
            env={**os.environ, "PYTHONPATH": "/workspace/repo", "HOME": agent.pw_dir},
            input=json.dumps({"case": case, "nonce": nonce}) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "observation process timed out"
    lines = [
        line.removeprefix(RESULT_PREFIX)
        for line in result.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if result.returncode != 0 or len(lines) != 1:
        return False, (
            f"observation incomplete (exit={result.returncode}, count={len(lines)})\n"
            + result.stdout
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
    # Execute the selected path itself so a trusted virtual-environment
    # interpreter retains its prefix; separately validate its resolved target.
    python_bin = Path(sys.argv[1])
    python_target = python_bin.resolve()
    for path in (python_bin, python_target, Path(__file__).resolve(), WORKER):
        reason = untrusted_reason(path)
        if reason is not None:
            print(
                "FAIL: verifier executable or scripts are not "
                f"root-owned/read-only: {reason}"
            )
            return 0
    agent = pwd.getpwnam("agent")
    completed: list[str] = []
    for case in CASES:
        passed, detail = run_case(python_bin, agent, case)
        if not passed:
            print(f"FAIL: {case}: {detail}")
            continue
        completed.append(case)
        print(f"PASS: {case}")
    if len(completed) != len(CASES):
        print(f"FAIL: trusted parent completed {len(completed)}/{len(CASES)} cases")
        return 0
    write_reward(1)
    print(f"PASS: trusted parent graded all {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
