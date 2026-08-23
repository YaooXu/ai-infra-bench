# Compact multimodal encoder-cache accounting and storage

A multimodal placeholder span in the decoder input can contain both positions
that receive encoder embeddings and decoder-side special or text tokens. Let
`P` be the full decoder placeholder length and `E` be the number of positions
that actually receive encoder embeddings. In sparse inputs, `E` may be much
smaller than `P`.

Update vLLM so that encoder computation, encoder-cache capacity and lifecycle
accounting, physically cached encoder-output rows, profiling limits, and
external encoder-cache transfer use `E`. Decoder positions, placeholder
overlap, and chunk boundaries must continue to use `P`.

For every legal half-open decoder interval `[start, end)`, select exactly the
encoder rows corresponding to embedding positions inside that interval.
Preserve their order and place them back only at the matching decoder
positions; special or text positions must not be overwritten.

The implementation must support dense placeholders whose embedding mask is
absent, sparse and alternating masks, empty intervals including `[0, 0)`, a
zero-length placeholder, and a non-empty all-false mask. A chunk containing
only non-embedding positions must not run the encoder or allocate
encoder-cache capacity. Cache hits, eviction, shared hashes, preemption,
multiple multimodal items, profiling, and external cache transfer must remain
consistent in embedding units. Text-only and dense multimodal behavior must
not regress.

The evaluation runs without network access on one NVIDIA A100 GPU. All
dependencies are already present; do not download packages, source code,
models, or other assets. Make the implementation changes in the vLLM source
tree under `/app`. Do not modify tests, verifier files, reward files, or the
runtime isolation configuration.

This task grades observable correctness and structural resource semantics.
Efficiency is checked by facts such as cache rows and accounting units being
`E`, not by a fixed wall-clock latency or memory-number threshold.
