# Construction and validation record

## Status

`HARDWARE+MODEL-BLOCKED-BEFORE-BUILD`

The source solution is stable and accepted, but the assigned A100 worker
cannot execute the real GLM-5.2-FP8 workload. No Dockerfile was constructed,
no Docker build or pull was started, and no image tag was created. This
directory must not be published as a runnable Agent benchmark.

## Read-only worker evidence

The assigned A100 host reported eight devices:

```text
0..7, NVIDIA A100-SXM4-40GB, 40960 MiB each
```

This is 327,680 MiB (320 GiB) aggregate nominal device memory. The exact
PR-time model revision contains 755,632,050,320 bytes (about 703.73 GiB) of
safetensors weight shards. The weights alone exceed aggregate device memory by
more than a factor of two. The runtime would additionally need memory for KV
cache, activations, MTP state, CUDA Graph captures, communication buffers, and
allocator headroom.

The A100's compute capability is 8.0. Exact-base vLLM reports native FP8
support only at compute capability 8.9 or newer, and the FP8 KV-cache selected
by the original command requires native `fp8e4nv` on SM89+. The upstream PR
discussion also explicitly says A100 cannot run GLM-5.2 and recommends H200
or, ideally, B200/B300.

No dependency-only or structural probe image was built because it could not
advance the task past either hard gate. In particular, it would be misleading
to mark an environment ready after testing only imports, constructor branches,
or a miniature synthetic model.

## Original symptom and missing public oracle

The PR author ran:

```bash
VLLM_DEEP_GEMM_WARMUP=skip vllm serve zai-org/GLM-5.2-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 5 \
  --max-num-seqs 32 \
  --cudagraph-capture-sizes 1 2 4 8 16 32 \
  --kv-cache-dtype fp8_e4m3
```

The reported result was an increase in mean acceptance length from roughly 3
to roughly 4, mean acceptance rate to 60%, and unchanged IFBench score 74.62.
The PR does not publish the exact request corpus, request order, random seeds,
run count, warm-up policy, or raw metric trace. Therefore those headline
numbers are context, not a reproducible acceptance threshold.

## Conditions for a future real verifier

Resume this task only on a worker with at least eight H200-class GPUs and
enough aggregate free memory for the 703.73-GiB checkpoint plus runtime
headroom; B200 or B300 is preferred, matching upstream guidance. Then:

1. Build the Agent Base from accepted squash parent
   `9ea3a4015b412d146d38ee1b697aafe92979c6ae` and the isolated evaluator
   Oracle from `ab666069935c1f23e8ef56038b4659ac9e8f19f8` using the same
   digest-pinned official runtime and exact Python source.
2. Mark native extensions as same-release donors unless rebuilt from the exact
   source. The candidate path is Python, so Base/Oracle must share identical
   native binaries and expose that native scope in machine-readable labels.
3. Keep the solved source, model, and verifier out of the Agent image. Mount
   model revision `a0b55e88465d1a06afece97bc8d6b366aff39089` read-only from an
   externally digest-verified cache.
4. Run Base and Oracle with the original tensor-parallel, MTP, FP8 KV-cache,
   and CUDA Graph settings under `--network none`.
5. Use a fixed, published prompt corpus, tokenizer revision, deterministic
   sampling settings, request order, warm-up, and run count. Record raw
   accepted-token traces, not only aggregate summaries.
6. Hard-gate correctness by comparing greedy token IDs from MTP decoding with
   non-speculative decoding. Only after correctness passes, compare paired
   same-host acceptance length/rate and throughput across repeated runs.
7. Add evaluator-side structural checks for both contracts: skipped backbone
   indexers still receive the shared Top-K buffer, and the proposer recycles
   post-final-norm hidden state. These checks supplement rather than replace
   the end-to-end model run.

Without the fixed prompt corpus and raw metric definition, even an H200 run
would remain `reproduction-incomplete` rather than verifier-ready.

## Construction-guide feedback

- For squash-merged PRs, derive the authoritative golden mapping from the
  merged commit's parent to the merged commit; do not assume the recorded PR
  head is a merge parent.
- Perform both architecture and aggregate-memory gates before downloading a
  model. Compare checkpoint bytes with total device memory, then reserve
  explicit headroom for runtime state.
- A model-specific acceptance-rate bug cannot be validated by a tiny config
  that reveals the changed branches. If the real model does not fit, publish a
  hardware/model-blocked audit rather than a narrowed substitute task.
- PR headline metrics are not an oracle unless the request corpus, seeds,
  sampling, warm-up, run count, and raw traces are available.
