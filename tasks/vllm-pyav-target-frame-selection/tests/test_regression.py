import io

import av
import numpy as np
from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoBackend

NUM_FRAMES = 72
FPS = 24
HEIGHT = 64
WIDTH = 64


def _long_gop_video() -> bytes:
    """Create a deterministic clip whose green channel identifies each frame."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=FPS)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = NUM_FRAMES
        stream.codec_context.max_b_frames = 0
        stream.codec_context.options = {
            "x264-params": (f"scenecut=0:keyint={NUM_FRAMES}:min-keyint={NUM_FRAMES}")
        }

        for frame_index in range(NUM_FRAMES):
            image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            image[:, :, 1] = frame_index
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _frame_markers(frames: np.ndarray) -> list[int]:
    return [int(frame[HEIGHT // 2, WIDTH // 2, 1]) for frame in frames]


def _assert_targets(frames: np.ndarray, actual_indices: list[int], targets: list[int]):
    assert actual_indices == targets
    assert frames.shape == (len(targets), HEIGHT, WIDTH, 3)

    markers = _frame_markers(frames)
    assert len(set(markers)) == len(markers), (
        f"decoder collapsed target positions onto keyframes: {markers=} {targets=}"
    )
    for marker, target in zip(markers, targets):
        assert abs(marker - target) <= 10, (
            f"decoded the wrong temporal frame: {marker=} {target=}"
        )


def test_uniform_sampling_returns_temporal_targets():
    data = _long_gop_video()
    loader = VIDEO_LOADER_REGISTRY.load("opencv")
    frames, metadata = loader.load_bytes(data, num_frames=5, backend="pyav")
    targets = list(metadata["frames_indices"])

    assert targets == sorted(targets)
    _assert_targets(frames, targets, targets)


def test_decoder_handles_nonuniform_forward_targets():
    data = _long_gop_video()
    targets = [0, 11, 37, 64]
    with av.open(io.BytesIO(data)) as container:
        source = VideoBackend.get_metadata(container)
        frames, valid_indices = VideoBackend.decode_frames(
            container,
            targets,
            source.original_fps,
            source.duration,
        )

    _assert_targets(frames, valid_indices, targets)
