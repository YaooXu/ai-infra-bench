# Review remediation matrix

All results below were produced on the final task image
`ai-infra-bench/vllm-multimodal-merge-memory:base-36d7f1989784`
(image id `sha256:d1432a560ed68bdf8cd9e2ca9088e7bbbce3c588bcf4208e8f53a9a37fc597fb`,
matching `task.toml`'s `image_digest`), on a real CUDA device with
torch 2.10.0+cu129. No score from an earlier snapshot is reused.

| Review finding | Remediation | Current state |
| --- | --- | --- |
| Meaningful task identity | Uses the descriptive task directory and task name. | Complete |
| Human task statement | First-person operational request contains no test, file, helper, or Oracle hints. | Complete |
| Environment isolation | Pinned image unchanged; donor and image contents audited for future-source leakage (see below). | Audited on the final image |
| Behavioral / E2E boundary | Production multimodal interface and CUDA merge behaviour are observed; no source inspection. | Executed on the final image |
| Implementation independence | Transfers, synchronization, allocation, ordering and errors are observed. No cardinality strictness is required on the same-device path, because the unmodified base and the declared alternative both accept a broadcastable same-device mismatch. | Executed on the final image |
| Harbor alignment | 10-hour Agent budget, offline phases, separate verifier, explicit artifacts, accelerator/workdir metadata, verifier mount. | Static validator passes |
| Scoring integrity (P0) | Reward is no longer keyed off an exit code. A root-owned trusted parent owns the case list and expectations, creates the reward file exclusively with a leading `0`, and runs candidate code only in unprivileged nonce-bound observation subprocesses. | Negative controls executed; see below |
| Oracle cardinality strictness (P1) | The Oracle now validates cardinality *before* the indexed assignment on the CPU-mask path, instead of relying on the assignment raising. The Oracle's own upstream test file gains `test_merge_multimodal_embeddings_cardinality_mismatch`, and the verifier adds broadcastable `1 -> 3` and `1 -> 9` regressions. | Fixed and re-measured on CUDA |
| Resource threshold justification | `MAX_PEAK_RATIO = 4.0` is calibrated against measured peak temporary allocation; see the table below. | Measured, five repeats each |
| Controls | One production-only alternative patch and one incomplete patch are declared. | Executed on the final image |
| Calibration | Base / Oracle / controls all re-measured through the real grading entry point on the final image. | Complete |

## P1: cardinality behaviour on CUDA (real device, per-case isolated processes)

A non-broadcastable mismatch against a CUDA mask raises a device-side assert
that poisons the CUDA context, so each shape is measured in its own process.

| shape / mask device | base | oracle (before fix) | oracle (after fix) | alternative | incomplete |
| --- | --- | --- | --- | --- | --- |
| `3 -> 3` CPU (normal merge) | ValueError (the reported bug) | accepted | accepted | accepted | **ValueError** |
| `1 -> 3` CPU (broadcastable) | ValueError | **silently broadcast to 3 rows** | ValueError | ValueError | ValueError |
| `5 -> 3` CPU | ValueError | ValueError | ValueError | ValueError | ValueError |
| `2 -> 4` CPU | ValueError | ValueError | ValueError | ValueError | ValueError |
| `2 -> 6` CPU | ValueError | ValueError | ValueError | ValueError | ValueError |
| `5 -> 3` CUDA (same device) | accepted | ValueError | ValueError | accepted | accepted |

The review's finding is confirmed on a real device: the original Oracle was the
only implementation that lost the cardinality error for the broadcastable
`1 -> N` shape, which `instruction.md` requires not to regress.

The last row is why the same-device path is not graded for strictness: the
unmodified base already accepts that mismatch, so requiring an error there
would grade one particular design rather than the declared contract.

### The Oracle's own upstream test now covers the finding

`solution/oracle.patch` adds `test_merge_multimodal_embeddings_cardinality_mismatch`
next to the upstream `test_merge_multimodal_embeddings_no_sync`, so the fix is
self-evidencing inside the patch rather than only in the hidden verifier. The
task image ships runtime dependencies only (no `pytest`), so both test bodies
were driven directly with an explicit `vllm.__file__.startswith("/workspace/repo/")`
binding assertion, on a real CUDA device:

| variant | `no_sync` | `cardinality_mismatch` |
| --- | --- | --- |
| base (unmodified) | FAIL — `ValueError: Error during masked scatter operation` | PASS (the base is strict, it is the merge that is broken) |
| oracle (after fix) | **PASS** — no D2H sync, merge completed | **PASS** — `Attempted to assign 1 = 1 multimodal tokens to 3 placeholders` |
| oracle with only the new guard disabled | PASS | **FAIL** — `no error raised; 3 rows written (broadcast)` |

The last row is the mutation control: with nothing but the new cardinality
guard neutralised, the added test reproduces exactly the silent `1 -> 3`
broadcast the review reported. The test therefore covers the finding rather
than passing incidentally.

## Resource threshold calibration

`peak_ratio` = peak temporary CUDA allocation during the merge, divided by the
size of the target `inputs_embeds` tensor (8192 x 512). Five repeats each; every
repeat produced an identical value.

| implementation | CPU mask f16/bf16 | CPU mask f32 | CUDA mask f16/bf16 | CUDA mask f32 |
| --- | --- | --- | --- | --- |
| base | ValueError | ValueError | 5.509 | 2.588 |
| oracle (after fix) | **1.003** | **0.500** | 1.003 | 0.500 |
| alternative | **1.003** | **0.500** | 5.509 | 2.588 |
| incomplete | ValueError | 0.500 | 5.509 | 0.500 |

Two independently written passing designs both measure 1.003 / 0.500, while
every design that keeps materialising the full request-length mask on the
device measures 5.509 / 2.588. The gate at 4.0 separates "temporaries
proportional to the placeholders" from "temporaries proportional to the whole
request" with a wide margin on both sides, without prescribing an exact curve.

This is the observable resource goal the instruction asks for: it is measured
directly, it is stable across repeats, and it does not rest on the base's
device-mismatch failure.

## P0: scoring integrity, measured through the real grading entry point

`tests/test.sh` no longer derives the reward from an exit status. It locates a
root-owned interpreter and `exec`s the root-owned trusted parent
`tests/supervise_multimodal_merge.py`, which holds the case list and the
expectations, creates `/logs/verifier/reward.txt` exclusively with a leading
`0` (root-owned, mode 0600, in a root-owned 0700 directory), and executes
candidate code only inside unprivileged `setpriv --reuid=1000 --no-new-privs`
observation subprocesses. Each subprocess must return exit status 0 **and**
print exactly one observation line carrying the parent's per-case nonce.

Every row below was run as `bash /tests/test.sh` on the final image. The
bypasses are layered on the **unmodified base**, so reward 1 would require
defeating the architecture, not fixing the bug.

| case | `test.sh` exit | cases passed | reward | expected |
| --- | --- | --- | --- | --- |
| base (unmodified) | 0 | 7/11 | **0** | 0 |
| oracle (after fix) | 0 | 11/11 | **1** | 1 |
| alternative (different design, different location) | 0 | 11/11 | **1** | 1 |
| incomplete (float32 only) | 0 | 8/11 | **0** | 0 |
| `os._exit(0)` at `vllm` import | 0 | 0/11 | **0** | 0 |
| `raise SystemExit(0)` at `vllm` import | 0 | 0/11 | **0** | 0 |
| forged observation envelopes for all 11 cases | 0 | 0/11 | **0** | 0 |
| reward write / unlink / rmtree / symlink / `SIGKILL` parent | 0 | 7/11 | **0** | 0 |

The two premature-termination controls are the review's explicit acceptance
criterion: both terminate with status 0 before any required check runs, and both
score 0 because a missing observation line is a failure, not a pass. The forged
envelope control is the completion-evidence criterion: the candidate cannot
produce evidence the parent will accept, because the nonce is unguessable and
only one line per subprocess is read. The tampering control confirms the reward
file itself is out of reach: as uid 1000 every write, unlink, rmtree and symlink
attempt fails with EPERM/EACCES, and killing the parent leaves the initial `0`
in place.

### Repeatability

Base / Oracle / alternative / incomplete were each run three times through the
same entry point: 12/12 runs reproduced the expected reward and the identical
passed-case count, with no jitter.

## Donor / image leakage audit (on the final image)

- Donor is `vllm-0.19.0.dist-info`. Its installed
  `_merge_multimodal_embeddings` is still the old `masked_scatter_`
  implementation, so the donor does not carry the answer.
- Whole-filesystem grep for the Oracle-specific string
  `Error during index put operation`: no hits outside the candidate repo.
- `inputs_embeds[is_multimodal]` appears in the donor only inside a comment.
- `gemma4_mm.py:1224`'s `is_multimodal.to(input_ids.device)` is pre-existing
  upstream code, unrelated to this task's fix.
- No vLLM wheel or source archive, no pip cache, empty `/home/agent/.cache`.
- Candidate repo: no remotes, no reflog, `git fsck --unreachable` empty,
  zero commits reachable after the `2026-04-01` cutoff, HEAD dated
  `2026-04-01 13:42:27 +0800`.
- `tests/models/test_utils.py` does not contain the Oracle's added
  `test_merge_multimodal_embeddings_no_sync`.
- All eight native `.so` objects come from the single v0.19.0 donor, verified
  against `environment/lock/native.sha256` at build time.

No future-source leakage was found.
