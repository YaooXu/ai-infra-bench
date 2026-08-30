import pytest
import torch
from pydantic import ValidationError

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _request(**kwargs):
    return ResponsesRequest(model="test-model", input="test input", **kwargs)


@pytest.mark.parametrize(
    "field,value,sampling_field,expected",
    [
        ("repetition_penalty", 1.2, "repetition_penalty", 1.2),
        ("seed", 42, "seed", 42),
        ("ignore_eos", True, "ignore_eos", True),
        ("vllm_xargs", {"custom": "value"}, "extra_args", {"custom": "value"}),
    ],
)
def test_individual_responses_extensions_reach_sampling_params(
    field,
    value,
    sampling_field,
    expected,
):
    params = _request(**{field: value}).to_sampling_params(default_max_tokens=1000)
    assert getattr(params, sampling_field) == expected


@pytest.mark.parametrize(
    "stop,expected",
    [("STOP", ["STOP"]), (["END", "STOP"], ["END", "STOP"]), (None, []), ([], [])],
)
def test_stop_accepts_string_list_and_empty_forms(stop, expected):
    params = _request(stop=stop).to_sampling_params(default_max_tokens=1000)
    assert params.stop == expected


def test_all_new_controls_compose():
    params = _request(
        repetition_penalty=1.15,
        seed=1234,
        stop=["END", "STOP"],
        ignore_eos=True,
        vllm_xargs={"scale": 0.75, "ids": [1, 2, 3]},
    ).to_sampling_params(default_max_tokens=1000)

    assert params.repetition_penalty == 1.15
    assert params.seed == 1234
    assert params.stop == ["END", "STOP"]
    assert params.ignore_eos is True
    assert params.extra_args == {"scale": 0.75, "ids": [1, 2, 3]}


@pytest.mark.parametrize("defaults,expected", [({}, 1.0), ({"repetition_penalty": 1.35}, 1.35)])
def test_omitted_repetition_penalty_respects_defaults(defaults, expected):
    params = _request().to_sampling_params(
        default_max_tokens=1000,
        default_sampling_params=defaults,
    )
    assert params.repetition_penalty == expected


@pytest.mark.parametrize("seed", [torch.iinfo(torch.long).min, torch.iinfo(torch.long).max])
def test_seed_accepts_torch_long_boundaries(seed):
    assert _request(seed=seed).seed == seed


@pytest.mark.parametrize("seed", [torch.iinfo(torch.long).min - 1, torch.iinfo(torch.long).max + 1])
def test_seed_rejects_values_outside_torch_long(seed):
    with pytest.raises(ValidationError):
        _request(seed=seed)


@pytest.mark.parametrize(
    "xargs",
    [
        {"text": "value", "count": 3, "ratio": 0.5},
        {"strings": ["a", "b"], "numbers": [1, 2], "floats": [0.1, 0.2]},
    ],
)
def test_vllm_xargs_accepts_supported_scalar_and_list_values(xargs):
    params = _request(vllm_xargs=xargs).to_sampling_params(default_max_tokens=1000)
    assert params.extra_args == xargs


def test_existing_responses_sampling_fields_remain_unchanged():
    params = _request(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_output_tokens=50,
    ).to_sampling_params(default_max_tokens=1000)
    assert (params.temperature, params.top_p, params.top_k, params.max_tokens) == (
        0.7,
        0.9,
        40,
        50,
    )
