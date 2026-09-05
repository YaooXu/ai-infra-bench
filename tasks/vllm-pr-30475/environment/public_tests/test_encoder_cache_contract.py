"""Public examples for prompt-space versus embedding-space accounting."""

import inspect

import torch

from vllm.multimodal.inputs import PlaceholderRange


def _read_num_embeds(position, expected):
    """Accept either a property or method; the representation is not scored."""

    for name in dir(position):
        lowered = name.lower()
        if name.startswith("_") or "embed" not in lowered:
            continue
        if "num" not in lowered and "count" not in lowered:
            continue
        member = getattr(position, name)
        if callable(member):
            try:
                if len(inspect.signature(member).parameters) != 0:
                    continue
                value = member()
            except (TypeError, ValueError):
                continue
        else:
            value = member
        if isinstance(value, int) and not isinstance(value, bool) and value == expected:
            return value
    raise AssertionError(f"no public embedding-count behavior returned {expected}")


def test_sparse_placeholder_counts_embedding_rows():
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[5, 15, 25, 35, 45, 55, 65, 75]] = True
    position = PlaceholderRange(offset=0, length=100, is_embed=mask)
    assert _read_num_embeds(position, 8) == 8


def test_mask_free_and_empty_masks():
    assert _read_num_embeds(PlaceholderRange(0, 5, None), 5) == 5
    assert (
        _read_num_embeds(
            PlaceholderRange(0, 5, torch.zeros(5, dtype=torch.bool)), 0
        )
        == 0
    )
