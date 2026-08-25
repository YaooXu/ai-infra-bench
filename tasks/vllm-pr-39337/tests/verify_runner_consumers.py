#!/usr/bin/env python3
"""Behavioral Model Runner V2 selection and GPUWorker-consumer contract."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import vllm.envs as envs
from vllm.config import VllmConfig


def model(
    architecture: str,
    *,
    runner: str = "generate",
    moe: bool = False,
    quantized: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        model="local/synthetic-config",
        architectures=[architecture],
        runner_type=runner,
        is_moe=moe,
        is_quantized=quantized,
    )


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


def check_defaults(failures: list[str], selector: object) -> None:
    cases = (
        (model("Qwen3ForCausalLM"), True, "dense Qwen3"),
        (model("OPTForCausalLM"), False, "non-whitelisted"),
        (model("Qwen3MoeForCausalLM", moe=True), False, "MoE"),
        (model("Qwen3ForCausalLM", quantized=True), False, "quantized"),
        (model("Qwen3ForCausalLM", runner="pooling"), False, "pooling"),
        (model("Qwen3_5ForConditionalGeneration"), False, "later Qwen family"),
    )
    for model_config, expected, label in cases:
        actual = selector(SimpleNamespace(model_config=model_config))
        if actual is not expected:
            failures.append(f"default {label}: expected {expected}, got {actual}")


class ConfigHarness:
    """Binds production VllmConfig selection methods to local configs."""

    def __init__(
        self,
        model_config: SimpleNamespace,
        *,
        unsupported: tuple[str, ...] = (),
    ) -> None:
        self.model_config = model_config
        self.unsupported = list(unsupported)

        # Fields consumed by the real GPU Worker base initializer.
        self.cache_config = SimpleNamespace()
        self.lora_config = None
        self.load_config = SimpleNamespace()
        self.parallel_config = SimpleNamespace(rank=0)
        self.scheduler_config = SimpleNamespace()
        self.device_config = SimpleNamespace()
        self.speculative_config = None
        self.observability_config = SimpleNamespace()
        self.kv_transfer_config = None
        self.compilation_config = SimpleNamespace()
        self.profiler_config = SimpleNamespace(profiler=None)
        self.weight_transfer_config = None

    def _get_v2_model_runner_unsupported_features(self) -> list[str]:
        return list(self.unsupported)


def bind_production_methods() -> bool:
    prop = getattr(VllmConfig, "use_v2_model_runner", None)
    selector = getattr(VllmConfig, "_is_default_v2_model_runner_model", None)
    validator = getattr(VllmConfig, "_validate_v2_model_runner", None)
    if not isinstance(prop, property) or selector is None or validator is None:
        return False
    ConfigHarness.use_v2_model_runner = prop
    ConfigHarness._is_default_v2_model_runner_model = selector
    ConfigHarness._validate_v2_model_runner = validator
    return True


def check_property(failures: list[str]) -> None:
    dense = ConfigHarness(model("Qwen3ForCausalLM"))
    unsupported = ConfigHarness(
        model("Qwen3ForCausalLM"), unsupported=("synthetic unsupported feature",)
    )
    for value, config, expected in (
        ("0", dense, False),
        ("1", unsupported, True),
        (None, dense, True),
        (None, unsupported, False),
    ):
        set_override(value)
        actual = config.use_v2_model_runner
        if actual is not expected:
            failures.append(
                f"config env={value!r} unsupported={bool(config.unsupported)}: "
                f"expected {expected}, got {actual}"
            )

    set_override("1")
    try:
        unsupported._validate_v2_model_runner()
    except ValueError as exc:
        if "synthetic unsupported feature" not in str(exc):
            failures.append("forced-V2 error omitted unsupported feature")
    else:
        failures.append("forced V2 silently accepted an unsupported feature")


def construct_worker(config: ConfigHarness) -> object:
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


def check_gpu_worker_consumer(failures: list[str]) -> None:
    dense = ConfigHarness(model("Qwen3ForCausalLM"))
    unsupported = ConfigHarness(
        model("Qwen3ForCausalLM"), unsupported=("synthetic unsupported feature",)
    )
    for value, config, expected in (
        (None, dense, True),
        (None, unsupported, False),
        ("0", dense, False),
        ("1", unsupported, True),
    ):
        set_override(value)
        worker = construct_worker(config)
        actual = worker.use_v2_model_runner
        if actual is not expected:
            failures.append(
                f"GPUWorker env={value!r} unsupported={bool(config.unsupported)}: "
                f"expected {expected}, got {actual!r}"
            )


def main() -> int:
    failures: list[str] = []
    try:
        if not torch.cuda.is_available():
            failures.append(
                "CUDA is unavailable; production Triton eligibility cannot be tested"
            )
        check_env(failures)
        selector = getattr(VllmConfig, "_is_default_v2_model_runner_model", None)
        if selector is None or not bind_production_methods():
            failures.append("VllmConfig model-runner selection API is unavailable")
        else:
            check_defaults(failures, selector)
            check_property(failures)
            check_gpu_worker_consumer(failures)
    finally:
        set_override(None)

    if failures:
        print("FAIL: Model Runner V2 selection/consumer contract is incomplete")
        for failure in failures:
            print(f" - {failure}")
        return 1
    props = torch.cuda.get_device_properties(0)
    print("PASS: tri-state selection and real GPUWorker consumption agree in both directions")
    print(f"gpu={props.name} capability={props.major}.{props.minor} uuid={props.uuid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
