While checking that the PyAV video backend is compatible with the existing OpenCV backend, I decoded `/opt/video/sintel-trailer.mp4` with `num_frames=8` through both backends.

Both backends report the same eight target indices and return arrays with the same shape and dtype. OpenCV returns the requested moments, but several PyAV frames come from different moments even though they carry the same position labels.

Find out why these frames disagree and make PyAV return the requested moments. Keep the PyAV backend on its own decoding path rather than redirecting it to OpenCV, and leave the rest of the video-loading interface working as before.
