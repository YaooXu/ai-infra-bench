# Upstream investigation for task 1.4.0

The task retains its request-lifecycle and prefix-cache correctness scope. The instruction now explicitly mentions multi-image reuse failing during decoding; it does not disclose a hash layout, helper, or repair algorithm. Base remains `e94ec597334d9a3e9b0d04bc17152e2747c83d51`. The reference patch combines the original lifecycle fix with curator adaptations for defects already reachable at Base; `oracle_commit` identifies the original upstream source, not the complete task solution. The executable reference is `solution/oracle.patch`.

## Sources and decisions

Checked on 2026-09-13 using GitHub PR and issue bodies, linked discussions, diffs, and repository issue searches for `49377`, `"multimodal" "hash"`, and `34183`. The broad multimodal search returned 402 results and only the first 100 were retrieved; follow-up investigation focused on the request, scheduler, and prefix-cache paths below. This is a bounded investigation, not proof that no other bugs exist.

| Source | Status at inspection | Applicability and decision |
| --- | --- | --- |
| [PR 34183](https://github.com/vllm-project/vllm/pull/34183) | Merged | Original request-reference-cycle fix; normal completion, cancellation, and streaming end must release ownership while live sessions retain it. |
| [Issue 49377](https://github.com/vllm-project/vllm/issues/49377), [issue 49449](https://github.com/vllm-project/vllm/issues/49449) | Open | Streaming replacement can retain discarded-token identities. Reachable through Base's session continuation; include valid reuse and rejection of stale prefixes. |
| [PR 49448](https://github.com/vllm-project/vllm/pull/49448), [PR 49619](https://github.com/vllm-project/vllm/pull/49619) | Closed, unmerged | Related streaming repairs; closure does not establish a fix landed. Adapt the behavior, not their exact implementation. |
| [PR 52806](https://github.com/vllm-project/vllm/pull/52806), [PR 53156](https://github.com/vllm-project/vllm/pull/53156) | Open, unmerged | Text-only suffix invalidation and combined text/MM streaming proposals. Cover the common failure plus existing media in the recomputed region; no particular invalidation strategy required. |
| [PR 51694](https://github.com/vllm-project/vllm/pull/51694) | Open, unmerged | Ordinary decoding can complete a block containing more than the final media item. Base and the previous task Oracle reproduce the missed reuse; include this in the scored lifecycle. |
| [PR 36708](https://github.com/vllm-project/vllm/pull/36708) | Merged | Adjacent range-boundary and media-position defects also exist at Base. Include matching-prefix reuse after a media range ends and non-aliasing when media positions differ. No fixed hash format is required. |
| [PR 50114](https://github.com/vllm-project/vllm/pull/50114) | Open, unmerged | Skips unnecessary hasher calls; exclude the performance optimization because the task specifies correctness and release, not a hasher-call count. |

External KV connector, offload/event provenance, Mamba truncation, media preprocessing, and compile-cache reports surfaced by the search are outside this task's request-lifecycle/local-prefix-cache boundary. Do not import their new APIs, hardware requirements, or benchmarks. Upstream material and private reproducers remain outside the agent image. Current-main status is not an acceptance criterion for the frozen task.

## Bidirectional coverage

| Instruction behavior | Scored scenario and observation |
| --- | --- |
| Finished requests and payloads become reclaimable without explicit GC | `lifecycle`: normal completion, waiting/running cancellation, queued streaming end, and resumed-session end; weak-reference release and absence of GC during these transitions. `memory_rounds`: root parent observes the real worker's RSS over live and released batches. |
| Live requests and waiting streams retain media | `lifecycle` checks ownership before completion, while waiting, and after continuation. |
| Prefix caching and multimodal input remain enabled | Real cache hits for identical multimodal inputs, misses for changed inputs, and live media retention; caching-disabled normal completion is an adjacent lifecycle regression. |
| Initial and appended prefixes remain correct | `cache_cases`: initial, append, and streaming lookup checks. `incremental_media_cases`: actual decoding completes partial blocks with two media items; matching full prefixes reuse, changing either media item prevents reuse from its affected block onward. |
| Streaming updates preserve valid prefixes and reject discarded-token prefixes | Existing streaming check plus `incremental_media_cases` through actual session add/schedule/update and stream end; query the same KV manager without manually inserting cache blocks. |
| Media-dependent cache identities remain correct at boundaries | `incremental_media_cases` with media ending exactly at a block boundary; subsequent decoded blocks still reuse. `media_position_case` checks matching versus moved media with identical placeholder token IDs. These follow from initial/later prefix-cache correctness. |

The semantic boundary is request admission or continuation → real Request, Scheduler, encoder ownership, and KV cache lifecycle → returned tokens, object release, and actual cached-token counts. Fixed media shapes and sampled token IDs substitute for model computation; no full Qwen model, HTTP server, GPU kernel, or output-accuracy comparison is claimed. Expected reuse comes from computed full blocks and the first changed input, never from Oracle-generated hash values. Validating a range/position distinction does not prescribe the representation of that distinction.

## Reference and controls

The reference avoids binding the request into its own callback, rebuilds streaming identities after replacement, scans all relevant media, and distinguishes media boundaries and positions. The alternative preserves Base's bound callback while the request is live and releases it at the real completion boundary; it receives the same cache checks. A renamed-private-helper control checks that the verifier does not depend on that scheduler helper's name.

The previous task Oracle is retained as `old-oracle.patch`. `last-media-only.patch`, `media-boundary.patch`, and `media-position-alias.patch` each restore one cache defect while retaining the other reference fixes. `stream-stale.patch` restores stale streaming identities while retaining the lifecycle and media fixes. Existing partial-lifecycle, forced-GC, fake-hash, forged-report, and early-success-exit controls remain. Outcomes must be recorded against final patch and verifier hashes; previous passing rollouts do not validate this revision.

Raw logs and upstream snapshots belong in the external evidence ZIP, not this task directory. The evidence index identifies the checked artifacts, results, scope, and archive checksum.
