# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import http.server
import tempfile
import threading
from pathlib import Path

from PIL import Image
from transformers import ProcessorMixin

from vllm.transformers_utils.processors.pixtral import MistralCommonImageProcessor


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mistral-image-pipeline-") as raw_dir:
        directory = Path(raw_dir)
        path = directory / "scene.png"
        Image.new("RGB", (13, 9), (31, 97, 211)).save(path)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            image_processor = MistralCommonImageProcessor(mm_encoder=None)
            bridge = object.__new__(ProcessorMixin)
            bridge.image_processor = image_processor
            url = f"http://127.0.0.1:{server.server_port}/{path.name}"
            decoded = Image.new("RGB", (5, 4), (7, 8, 9))
            images, text, videos, audio = ProcessorMixin.prepare_inputs_layout(
                bridge,
                images=[url, decoded],
            )
            assert text is None and videos is None and audio is None
            assert isinstance(images, list) and len(images) == 2
            assert isinstance(images[0], Image.Image)
            assert images[0].size == (13, 9)
            assert images[0].convert("RGB").getpixel((4, 3)) == (31, 97, 211)
            assert images[1] is decoded
            print("REAL_TRANSFORMERS_IMAGE_PIPELINE_OK images=2 remote=1 decoded=1")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    main()
