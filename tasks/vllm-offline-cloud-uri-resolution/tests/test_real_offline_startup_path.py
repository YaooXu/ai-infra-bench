#!/usr/bin/env python3
"""Drive offline EngineArgs and object-storage preparation in one process."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import huggingface_hub

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs


def main() -> int:
    try:
        model_uri = "s3://production-models/model/"
        tokenizer_uri = "gs://production-tokenizers/tokenizer/"
        with (
            patch.object(huggingface_hub.constants, "HF_HUB_OFFLINE", True),
            patch("vllm.engine.arg_utils.get_model_path", side_effect=AssertionError("unexpected Hugging Face access")),
        ):
            args = EngineArgs(model=model_uri, tokenizer=tokenizer_uri)
        assert args.model == model_uri
        assert args.tokenizer == tokenizer_uri

        config = SimpleNamespace(model_weights=None)
        with patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files") as pull:
            ModelConfig.maybe_pull_model_tokenizer_for_runai(
                config,
                args.model,
                args.tokenizer,
            )
        assert [entry.args[0] for entry in pull.call_args_list] == [model_uri, tokenizer_uri]
        print(
            {
                "entrypoints": ["EngineArgs.__post_init__", "ModelConfig.maybe_pull_model_tokenizer_for_runai"],
                "hf_hub_offline": True,
                "model_uri": model_uri,
                "tokenizer_uri": tokenizer_uri,
                "pull_order": [entry.args[0] for entry in pull.call_args_list],
            },
            flush=True,
        )
        return 0
    except Exception as exc:
        lines = str(exc).splitlines()
        print({"error": type(exc).__name__, "message": lines[0] if lines else "no exception message"}, flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
