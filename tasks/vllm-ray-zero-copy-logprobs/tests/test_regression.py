from types import SimpleNamespace

import numpy as np
from vllm.v1.executor import ray_executor, ray_utils
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, ModelRunnerOutput


def _output(*, readonly: bool = True) -> ModelRunnerOutput:
    token_ids = np.array([[1, 2], [3, 4]], dtype=np.int32)
    logprobs = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    ranks = np.array([1, 2], dtype=np.int32)
    if readonly:
        token_ids.setflags(write=False)
        logprobs.setflags(write=False)
        ranks.setflags(write=False)

    return ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        logprobs=LogprobsLists(token_ids, logprobs, ranks, [0, 2]),
        prompt_logprobs_dict={"req-0": LogprobsTensors.empty_cpu(1, 2)},
    )


def _assert_detached(output: ModelRunnerOutput, original: LogprobsLists) -> None:
    detached = output.logprobs
    assert detached is not None
    assert detached is not original
    for actual, before in zip(detached[:3], original[:3]):
        assert actual is not before
        assert actual.flags.writeable
        np.testing.assert_array_equal(actual, before)
    assert detached.cu_num_generated_tokens is original.cu_num_generated_tokens


class _Ref:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Dag:
    def __init__(self, refs):
        self.refs = refs

    def execute(self, _inputs):
        return self.refs


class _FakeRay:
    @staticmethod
    def get(refs, timeout=None):
        del timeout
        if isinstance(refs, list):
            return [ref.value for ref in refs]
        return refs.value


class _Aggregator:
    def __init__(self):
        self.outputs = None

    def aggregate(self, outputs, output_rank=0):
        self.outputs = outputs
        return outputs[output_rank]


def test_blocking_executor_detaches_direct_result():
    output = _output()
    original = output.logprobs
    assert original is not None
    executor = SimpleNamespace(
        forward_dag=_Dag([_Ref(output)]),
        has_connector=False,
    )

    result = ray_executor.RayDistributedExecutor._execute_dag(
        executor, object(), None, non_block=False
    )

    assert result is output
    _assert_detached(output, original)


def test_nonblocking_future_detaches_direct_result(monkeypatch):
    output = _output()
    original = output.logprobs
    assert original is not None
    monkeypatch.setattr(ray_utils, "ray", _FakeRay)

    result = ray_utils.FutureWrapper(_Ref(output)).result()

    assert result is output
    _assert_detached(output, original)


def test_connector_paths_detach_every_result(monkeypatch):
    outputs = [_output(), _output()]
    originals = [output.logprobs for output in outputs]
    assert all(original is not None for original in originals)
    refs = [_Ref(output) for output in outputs]
    aggregator = _Aggregator()
    executor = SimpleNamespace(
        forward_dag=_Dag(refs),
        has_connector=True,
        kv_output_aggregator=aggregator,
    )
    monkeypatch.setattr(ray_executor, "ray", _FakeRay)

    result = ray_executor.RayDistributedExecutor._execute_dag(
        executor, object(), None, non_block=False
    )

    assert result is outputs[0]
    for output, original in zip(outputs, originals):
        assert original is not None
        _assert_detached(output, original)

    later_outputs = [_output(), _output()]
    later_originals = [output.logprobs for output in later_outputs]
    monkeypatch.setattr(ray_utils, "ray", _FakeRay)
    later_aggregator = _Aggregator()
    future = ray_utils.FutureWrapper(
        [_Ref(output) for output in later_outputs], later_aggregator
    )
    assert future.result() is later_outputs[0]
    for output, original in zip(later_outputs, later_originals):
        assert original is not None
        _assert_detached(output, original)


def test_writable_arrays_and_unrelated_fields_keep_identity(monkeypatch):
    output = _output(readonly=False)
    original_logprobs = output.logprobs
    original_prompt = output.prompt_logprobs_dict["req-0"]
    monkeypatch.setattr(ray_utils, "ray", _FakeRay)

    result = ray_utils.FutureWrapper(_Ref(output)).result()

    assert result.logprobs is original_logprobs
    assert result.prompt_logprobs_dict["req-0"] is original_prompt
