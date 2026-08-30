We transcribe audio longer than the model’s window, so vLLM splits one request into several model calls and merges their text. With Cohere Transcribe and Qwen3 ASR, adjacent chunks such as `"Hello, this"` and `"is vLLM"` currently become `"Hello, thisis vLLM"`. Long transcripts show the same problem at every chunk boundary, for example `"superstar.A founding"` and `"server.But I guess"`.

Whisper usually hides the bug because its chunk text already starts with whitespace. The correction must work for both streaming and non-streaming transcription and translation. English-like languages need a boundary space without adding a leading space; languages such as Chinese and Japanese must continue joining chunks without an inserted space.

Investigate and fix the multi-chunk merge behavior through the public speech-to-text paths. Single-chunk responses, timestamps and metadata, stream ordering, and model-specific language behavior must remain unchanged.
