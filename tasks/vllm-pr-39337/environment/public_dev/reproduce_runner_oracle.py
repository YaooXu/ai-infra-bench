import os
from types import SimpleNamespace

import vllm.envs as envs
from vllm.config import VllmConfig


def model(architecture: str, *, runner: str = "generate", moe: bool = False,
          quantized: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model="local/synthetic-config",
        architectures=[architecture],
        runner_type=runner,
        is_moe=moe,
        is_quantized=quantized,
    )


def check_env_contract(failures: list[str]) -> None:
    cases = ((None, None), ("0", False), ("1", True))
    for value, expected in cases:
        if value is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = value
        actual = envs.VLLM_USE_V2_MODEL_RUNNER
        if actual is not expected:
            failures.append(
                f"env {value!r}: expected {expected!r}, got {actual!r}"
            )


def check_default_matrix(failures: list[str]) -> None:
    selector = getattr(VllmConfig, "_is_default_v2_model_runner_model", None)
    if selector is None:
        failures.append("default model selector is unavailable")
        return

    cases = (
        (model("Qwen3ForCausalLM"), True, "dense Qwen3 generation"),
        (model("OPTForCausalLM"), False, "non-whitelisted architecture"),
        (model("Qwen3MoeForCausalLM", moe=True), False, "Qwen3 MoE"),
        (model("Qwen3ForCausalLM", quantized=True), False, "quantized Qwen3"),
        (model("Qwen3ForCausalLM", runner="pooling"), False, "pooling"),
    )
    for model_config, expected, label in cases:
        actual = selector(SimpleNamespace(model_config=model_config))
        if actual is not expected:
            failures.append(f"{label}: expected {expected}, got {actual}")


def check_property_and_fallback(failures: list[str]) -> None:
    property_object = getattr(VllmConfig, "use_v2_model_runner", None)
    selector = getattr(VllmConfig, "_is_default_v2_model_runner_model", None)
    validator = getattr(VllmConfig, "_validate_v2_model_runner", None)
    if not isinstance(property_object, property) or selector is None:
        failures.append("runner oracle property is unavailable")
        return

    class Harness:
        use_v2_model_runner = property_object
        _is_default_v2_model_runner_model = selector
        unsupported: list[str]

        def __init__(self, model_config, unsupported=()):
            self.model_config = model_config
            self.unsupported = list(unsupported)

        def _get_v2_model_runner_unsupported_features(self):
            return list(self.unsupported)

    dense = Harness(model("Qwen3ForCausalLM"))
    unsupported = Harness(model("Qwen3ForCausalLM"), ["synthetic unsupported feature"])

    cases = (("0", dense, False), ("1", unsupported, True), (None, dense, True),
             (None, unsupported, False))
    for value, config, expected in cases:
        if value is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = value
        actual = config.use_v2_model_runner
        if actual is not expected:
            failures.append(
                f"oracle env={value!r} unsupported={bool(config.unsupported)}: "
                f"expected {expected}, got {actual}"
            )

    if validator is None:
        failures.append("forced-V2 validator is unavailable")
        return
    try:
        validator(unsupported)
    except ValueError as exc:
        if "synthetic unsupported feature" not in str(exc):
            failures.append("validator omitted the unsupported feature name")
    else:
        failures.append("forced V2 accepted an unsupported feature")


def main() -> None:
    failures: list[str] = []
    try:
        check_env_contract(failures)
        check_default_matrix(failures)
        check_property_and_fallback(failures)
    finally:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)

    if failures:
        print("FAIL: model-runner selection contract is incomplete")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print("PASS: tri-state overrides, default selection, and fallback all work")


if __name__ == "__main__":
    main()
