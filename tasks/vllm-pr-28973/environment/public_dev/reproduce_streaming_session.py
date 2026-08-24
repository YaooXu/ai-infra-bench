import asyncio
import sys

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs


MODEL = "/opt/models/tiny-streaming"


async def run() -> None:
    try:
        from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput
    except ImportError:
        print("FAIL: session-based streaming input API is unavailable")
        raise SystemExit(1)

    async def input_chunks():
        yield StreamingInput(prompt="A short session starts.")
        await asyncio.sleep(0)
        yield StreamingInput(prompt="A short session starts. continues now.")

    engine_args = AsyncEngineArgs(
        model=MODEL,
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=4,
        gpu_memory_utilization=0.1,
        disable_log_stats=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    outputs = []
    try:
        params = SamplingParams(
            max_tokens=2,
            temperature=0.0,
            ignore_eos=True,
        )
        async for output in engine.generate(
            input_chunks(), params, request_id="streaming-session-dev"
        ):
            outputs.append(output)
    finally:
        engine.shutdown()
        await asyncio.sleep(0.1)

    if not outputs:
        print("FAIL: streaming session produced no request outputs")
        raise SystemExit(1)
    if not outputs[-1].finished:
        print("FAIL: streaming session did not finish when its input closed")
        raise SystemExit(1)
    if any(output.finished for output in outputs[:-1]):
        print("FAIL: streaming session finished before its input closed")
        raise SystemExit(1)
    token_ids = [
        token_id
        for output in outputs
        for completion in output.outputs
        for token_id in completion.token_ids
    ]
    if not token_ids:
        print("FAIL: streaming session produced no tokens")
        raise SystemExit(1)
    print(f"PASS: one streaming session produced {len(token_ids)} token(s)")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"FAIL: streaming session request raised {type(exc).__name__}: {exc}")
        sys.exit(1)
