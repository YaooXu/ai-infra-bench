# Environment lock

The current development environment is pinned by the Dockerfile's base image, vLLM base commit, and explicit Torch version. Producing a portable published image and its final dependency/image digest is outside this format-and-Git-sanitization pass.
