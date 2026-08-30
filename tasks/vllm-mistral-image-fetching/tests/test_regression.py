# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.transformers_utils.processors.pixtral import MistralCommonImageProcessor


@pytest.fixture
def processor() -> MistralCommonImageProcessor:
    return MistralCommonImageProcessor(mm_encoder=None)


def assert_same_nested_identity(actual, expected) -> None:
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_same_nested_identity(actual_item, expected_item)
    else:
        assert actual is expected


def test_decoded_rgb_image_passes_through(processor) -> None:
    image = Image.new("RGB", (7, 5), (10, 20, 30))
    assert processor.fetch_images(image) is image


def test_decoded_grayscale_image_passes_through(processor) -> None:
    image = Image.new("L", (3, 9), 71)
    assert processor.fetch_images(image) is image


def test_decoded_rgba_image_passes_through(processor) -> None:
    image = Image.new("RGBA", (4, 6), (1, 2, 3, 4))
    assert processor.fetch_images(image) is image


def test_numpy_image_passes_through(processor) -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    assert processor.fetch_images(image) is image


def test_torch_image_passes_through(processor) -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
    assert processor.fetch_images(image) is image


def test_flat_batch_preserves_order_and_identity(processor) -> None:
    images = [
        Image.new("RGB", (2, 3), "red"),
        Image.new("RGB", (5, 7), "green"),
        Image.new("RGB", (11, 13), "blue"),
    ]
    result = processor.fetch_images(images)
    assert_same_nested_identity(result, images)


def test_nested_batch_preserves_structure(processor) -> None:
    images = [
        [Image.new("RGB", (2, 2), "red")],
        [
            Image.new("RGB", (3, 3), "green"),
            [Image.new("RGB", (4, 4), "blue")],
        ],
    ]
    result = processor.fetch_images(images)
    assert_same_nested_identity(result, images)


def test_tuple_inputs_preserve_nesting_as_lists(processor) -> None:
    images = (
        Image.new("RGB", (3, 4), "yellow"),
        (Image.new("RGB", (5, 6), "purple"),),
    )
    result = processor.fetch_images(images)
    assert_same_nested_identity(result, images)


def test_empty_batches_preserve_empty_structure(processor) -> None:
    assert processor.fetch_images([]) == []
    assert processor.fetch_images([[], [()]]) == [[], [[]]]


def test_mixed_decoded_image_types_preserve_identity(processor) -> None:
    pil_image = Image.new("RGB", (3, 3), "orange")
    numpy_image = np.zeros((4, 5, 3), dtype=np.uint8)
    torch_image = torch.zeros((3, 6, 7), dtype=torch.uint8)
    images = [[pil_image, numpy_image], [torch_image]]
    assert_same_nested_identity(processor.fetch_images(images), images)


def test_batch_input_is_not_mutated(processor) -> None:
    first = Image.new("RGB", (2, 2), "black")
    second = Image.new("RGB", (2, 2), "white")
    images = [first, [second]]
    snapshot = [images[0], list(images[1])]
    result = processor.fetch_images(images)
    assert images[0] is snapshot[0]
    assert images[1][0] is snapshot[1][0]
    assert result is not images
    assert result[1] is not images[1]


def test_local_png_path_loads_real_pixels(processor, tmp_path) -> None:
    path = tmp_path / "camera-frame.png"
    Image.new("RGB", (6, 4), (17, 89, 203)).save(path)
    result = processor.fetch_images(str(path))
    assert isinstance(result, Image.Image)
    assert result.size == (6, 4)
    assert result.convert("RGB").getpixel((2, 1)) == (17, 89, 203)


def test_nested_paths_and_decoded_images_are_resolved_in_place(processor, tmp_path) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (3, 2), (200, 10, 20)).save(first_path)
    Image.new("RGB", (4, 5), (5, 90, 170)).save(second_path)
    decoded = Image.new("RGB", (8, 9), (40, 50, 60))
    result = processor.fetch_images([[str(first_path), decoded], [str(second_path)]])
    assert result[0][0].size == (3, 2)
    assert result[0][0].getpixel((0, 0)) == (200, 10, 20)
    assert result[0][1] is decoded
    assert result[1][0].size == (4, 5)


@pytest.mark.parametrize("invalid", [None, 42, 3.14, {"url": "x"}])
def test_unsupported_single_values_raise_type_error(processor, invalid) -> None:
    with pytest.raises(TypeError, match="single or a list"):
        processor.fetch_images(invalid)


def test_nested_invalid_value_raises_instead_of_being_silently_kept(processor) -> None:
    valid = Image.new("RGB", (2, 2))
    with pytest.raises(TypeError, match="single or a list"):
        processor.fetch_images([valid, [object()]])


def test_existing_patch_count_behavior_is_unchanged() -> None:
    class PatchCounter:
        def _image_to_num_tokens(self, image):
            assert image.size == (19, 11)
            return 5, 3

    processor = MistralCommonImageProcessor(mm_encoder=PatchCounter())
    assert processor.get_number_of_image_patches(11, 19) == (15, 3, 5)


def test_fetched_images_continue_through_existing_encoder(processor, tmp_path) -> None:
    class RecordingEncoder:
        def __init__(self):
            self.sizes = []

        def __call__(self, chunk):
            self.sizes.append(chunk.image.size)
            return SimpleNamespace(image=np.asarray(chunk.image, dtype=np.float32))

    encoder = RecordingEncoder()
    processor.mm_encoder = encoder
    path = tmp_path / "input.png"
    Image.new("RGB", (9, 7), (12, 34, 56)).save(path)
    fetched = processor.fetch_images(str(path))
    output = processor(fetched, return_tensors="pt")
    assert encoder.sizes == [(9, 7)]
    assert len(output["images"]) == 1
    assert tuple(output["images"][0].shape) == (7, 9, 3)
