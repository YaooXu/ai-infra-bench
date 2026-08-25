#!/usr/bin/env python3
"""Verify that encoder-cache capacity is measured in embedding rows."""

from __future__ import annotations

import json

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager


PLACEHOLDER_TOKENS = 100
EMBEDDING_ROWS = 8


class SparseRequest:
    request_id = "sparse-placeholder"
    has_encoder_inputs = True

    def __init__(self) -> None:
        mask = torch.zeros(PLACEHOLDER_TOKENS, dtype=torch.bool)
        mask[torch.tensor([5, 15, 25, 35, 45, 55, 65, 75])] = True
        self.mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="sparse-item",
                mm_position=PlaceholderRange(0, PLACEHOLDER_TOKENS, mask),
            )
        ]

    def get_num_encoder_tokens(self, _input_id: int) -> int:
        return PLACEHOLDER_TOKENS

    def get_num_encoder_embeds(self, input_id: int) -> int:
        return self.mm_features[input_id].mm_position.get_num_embeds


def main() -> None:
    request = SparseRequest()
    position = request.mm_features[0].mm_position
    assert position.get_num_embeds == EMBEDDING_ROWS
    assert position.get_embeds_indices_in_range(0, PLACEHOLDER_TOKENS) == (
        0,
        EMBEDDING_ROWS,
    )

    manager = EncoderCacheManager(cache_size=EMBEDDING_ROWS)
    assert manager.can_allocate(request, 0, EMBEDDING_ROWS, 0)
    manager.allocate(request, 0)
    assert manager.num_free_slots == 0
    assert request.mm_features[0].identifier in manager.cached

    print(
        json.dumps(
            {
                "placeholder_tokens": PLACEHOLDER_TOKENS,
                "embedding_rows": EMBEDDING_ROWS,
                "cache_capacity": EMBEDDING_ROWS,
                "allocation_passed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

