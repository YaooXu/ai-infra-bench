#!/usr/bin/env python3
"""Behavioral Model Runner V2 selection and GPUWorker-consumer contract."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "/workspace/repo")

import torch
import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, VllmConfig


MODEL_FIXTURE = Path("/tests/fixtures/qwen3")


def set_override(value: str | None) -> None:
    if value is None:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = value


def check_env(failures: list[str]) -> None:
    for value, expected in ((None, None), ("0", False), ("1", True)):
        set_override(value)
        actual = envs.VLLM_USE_V2_MODEL_RUNNER
        if actual is not expected:
            failures.append(f"env {value!r}: expected {expected!r}, got {actual!r}")


def make_config(*, unsupported: bool = False) -> VllmConfig:
    """Build through the same public configuration path used at startup."""

    model_config = ModelConfig(
        model=str(MODEL_FIXTURE),
        skip_tokenizer_init=True,
        dtype="float16",
    )
    cache_config = CacheConfig(kv_sharing_fast_prefill=unsupported)
    return VllmConfig(model_config=model_config, cache_config=cache_config)


def construct_worker(config: VllmConfig) -> object:
    """Execute production GPUWorker.__init__, replacing no selection logic."""

    import vllm.distributed.elastic_ep.elastic_execute as elastic_execute
    from vllm.v1.worker.gpu_worker import Worker

    # Elastic EP is orthogonal control-plane state. This prevents it from
    # requiring a distributed process group while leaving Worker.__init__ and
    # its actual selection consumer line intact.
    with patch.object(
        elastic_execute,
        "ElasticEPScalingExecutor",
        side_effect=lambda _worker: object(),
    ):
        return Worker(
            vllm_config=config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
        )


def selected_by_worker(*, override: str | None, unsupported: bool = False) -> bool:
    set_override(override)
    config = make_config(unsupported=unsupported)
    return bool(construct_worker(config).use_v2_model_runner)


def check_selection(failures: list[str]) -> None:
    cases = (
        (None, False, True, "supported dense Qwen3 defaults to V2"),
        (None, True, False, "unsupported configuration defaults to V1"),
        ("0", False, False, "explicit 0 selects V1"),
        ("1", False, True, "explicit 1 selects V2"),
    )
    for override, unsupported, expected, label in cases:
        try:
            actual = selected_by_worker(
                override=override,
                unsupported=unsupported,
            )
        except Exception as exc:
            failures.append(f"{label}: raised {type(exc).__name__}: {exc}")
            continue
        if actual is not expected:
            failures.append(f"{label}: expected {expected}, got {actual}")

    # This exercises the public startup/config-construction boundary. It does
    # not require any particular helper, property, or private method name.
    set_override("1")
    try:
        make_config(unsupported=True)
    except (ValueError, AssertionError) as exc:
        message = str(exc).lower()
        if not message or not any(
            word in message for word in ("unsupported", "support", "kv sharing")
        ):
            failures.append("forced-V2 startup error did not explain incompatibility")
    except Exception as exc:
        failures.append(
            "forced-V2 unsupported startup raised the wrong exception: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        failures.append("forced V2 silently accepted an unsupported configuration")


def main() -> int:
    failures: list[str] = []
    try:
        if not torch.cuda.is_available():
            failures.append(
                "CUDA is unavailable; production GPUWorker consumption cannot be tested"
            )
        if not (MODEL_FIXTURE / "config.json").is_file():
            failures.append("local Qwen3 fixture is missing")
        else:
            check_env(failures)
            check_selection(failures)
    finally:
        set_override(None)

    if failures:
        print("FAIL: Model Runner V2 selection/consumer contract is incomplete")
        for failure in failures:
            print(f" - {failure}")
        return 1
    props = torch.cuda.get_device_properties(0)
    print("PASS: public config startup and real GPUWorker agree in both directions")
    print(f"gpu={props.name} capability={props.major}.{props.minor} uuid={props.uuid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
