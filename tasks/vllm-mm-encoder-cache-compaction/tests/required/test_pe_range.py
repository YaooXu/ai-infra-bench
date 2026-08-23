from __future__ import annotations

import random

import torch

from vllm.multimodal.inputs import PlaceholderRange


def _count(position: PlaceholderRange) -> int:
    value = position.get_num_embeds
    return int(value() if callable(value) else value)


def _actual_range(position: PlaceholderRange, start: int, end: int) -> tuple[int, int]:
    return tuple(position.get_embeds_indices_in_range(start, end))


def _oracle(mask: list[bool], start: int, end: int) -> tuple[int, int]:
    # Independent P+1 prefix oracle; it does not call candidate range code.
    prefix = [0]
    for item in mask:
        prefix.append(prefix[-1] + int(item))
    return prefix[start], prefix[end]


def test_sparse_and_generated_range_mapping():
    fixed = [False, True, False, True, True]
    position = PlaceholderRange(0, len(fixed), torch.tensor(fixed))
    assert _count(position) == 3
    for start, end in ((0, 2), (1, 4), (3, 5), (0, 5), (4, 5)):
        assert _actual_range(position, start, end) == _oracle(fixed, start, end)

    random_source = random.Random(30475)
    for length in range(1, 18):
        for _ in range(8):
            mask = [bool(random_source.getrandbits(1)) for _ in range(length)]
            candidate = PlaceholderRange(7, length, torch.tensor(mask))
            assert _count(candidate) == sum(mask)
            start = random_source.randrange(length + 1)
            # The public core contract includes ordinary and nonzero empty
            # ranges. The explicitly hardened [0, 0) case has its own lower-
            # weight requirement and must not silently contaminate this score.
            end = random_source.randrange(max(start, 1), length + 1)
            assert _actual_range(candidate, start, end) == _oracle(mask, start, end)


def test_dense_mask_absence_is_one_to_one():
    position = PlaceholderRange(offset=3, length=5, is_embed=None)
    assert _count(position) == 5
    for start, end in ((0, 5), (2, 4), (5, 5), (0, 0)):
        assert _actual_range(position, start, end) == (start, end)


def test_nonzero_empty_interval():
    mask = [True, False, True, False]
    position = PlaceholderRange(0, 4, torch.tensor(mask))
    assert _actual_range(position, 2, 2) == (1, 1)


def test_nonempty_all_false_has_zero_embeddings():
    mask = torch.zeros(8, dtype=torch.bool)
    position = PlaceholderRange(0, 8, mask)
    assert _count(position) == 0
    assert _actual_range(position, 0, 8) == (0, 0)
    assert _actual_range(position, 3, 7) == (0, 0)


def test_zero_origin_empty_interval():
    position = PlaceholderRange(0, 4, torch.tensor([True, False, True, False]))
    assert _actual_range(position, 0, 0) == (0, 0)


def test_zero_length_placeholder():
    position = PlaceholderRange(0, 0, torch.zeros(0, dtype=torch.bool))
    assert _count(position) == 0
    assert _actual_range(position, 0, 0) == (0, 0)
    assert position.extract_embeds_range() == []
