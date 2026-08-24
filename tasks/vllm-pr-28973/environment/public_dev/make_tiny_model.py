from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


DESTINATION = Path("/opt/models/tiny-streaming")


def main() -> None:
    torch.manual_seed(0)
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "A": 4,
        "short": 5,
        "session": 6,
        "starts": 7,
        "continues": 8,
        "now": 9,
        ".": 10,
        "answer": 11,
        "next": 12,
        "token": 13,
        "stream": 14,
        "input": 15,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.model_max_length = 64

    config = LlamaConfig(
        vocab_size=len(vocabulary),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        torch_dtype="float16",
    )
    model = LlamaForCausalLM(config).half()

    DESTINATION.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(DESTINATION)
    model.save_pretrained(DESTINATION, safe_serialization=True)


if __name__ == "__main__":
    main()
