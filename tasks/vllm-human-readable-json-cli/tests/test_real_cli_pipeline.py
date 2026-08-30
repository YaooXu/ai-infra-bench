#!/usr/bin/env python3
"""Parse the documented JSON and dotted forms through the real CLI parser."""

from __future__ import annotations

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def main() -> int:
    try:
        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        args = parser.parse_args(
            [
                "--model",
                "Qwen/Qwen3-0.6B",
                "--kv-transfer-config",
                '{"kv_connector":"OffloadingConnector",'
                '"kv_connector_extra_config":{"cpu_bytes_to_use":80m},'
                '"kv_role":"kv_both"}',
                "--compilation-config.max_cudagraph_capture_size",
                "1k",
            ]
        )
        engine_args = EngineArgs.from_cli_args(args)
        assert (
            engine_args.kv_transfer_config.kv_connector_extra_config[
                "cpu_bytes_to_use"
            ]
            == 80_000_000
        )
        assert engine_args.compilation_config.max_cudagraph_capture_size == 1_000
        print(
            {
                "entrypoint": "EngineArgs.add_cli_args and from_cli_args",
                "cpu_bytes_to_use": 80_000_000,
                "max_cudagraph_capture_size": 1_000,
                "forms": ["nested JSON", "dotted override"],
            },
            flush=True,
        )
        return 0
    except Exception as exc:
        lines = str(exc).splitlines()
        print(
            {
                "error": type(exc).__name__,
                "message": lines[0] if lines else "no exception message",
            },
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
