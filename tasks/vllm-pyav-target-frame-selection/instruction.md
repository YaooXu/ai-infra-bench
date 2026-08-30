# Return the requested frames when decoding video with PyAV

When vLLM samples frames from a video with a long group of pictures, the PyAV
backend can return an earlier keyframe for multiple requested positions while
reporting those positions as successfully decoded. The result has the expected
shape and metadata but contains the wrong temporal content.

Make PyAV decoding return the frame at each requested temporal position. Keep
the output and metadata ordering aligned and preserve the existing public
video-loading interface.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
