# Source and eligibility lock

This survey item is derived from open vLLM PR
[#41518](https://github.com/vllm-project/vllm/pull/41518), “Reduce
`time vllm --help` and `time vllm serve --help` to <1s warm”. It is
**blocked before environment construction** and must not be published as an
Agent task yet.

## Why the survey snapshot is not a valid Base/Oracle pair

The survey snapshot taken on 2026-08-08 recorded:

| Role | Commit | Git tree | Archive bytes | GitHub archive SHA-256 |
|---|---|---|---:|---|
| Survey `base_sha` | `228bcc436b0f09cf1824f00fd44f8f9c94060940` | `f0dd83024e35b359d6a930f0951ada7f390d2468` | 36,490,144 | `c03d010aa3e27d62d36b56408f7175d7d774fb891888d326730652e2d87c27f4` |
| Snapshot PR head | `7d85640f2ddb613b83449982d5d6d5a7dcf14293` | `03756750d607d898cb06dd9250e3d9896170d002` | 36,495,258 | `8ddebf1bfc18554ef3a908376706a0c30b40f62f19d57b463eb0d539af3bf57d` |

GitHub's compare API reports that this pair is **diverged**, with the head 14
commits ahead and one commit behind the recorded base. Their actual merge base
is:

| Role | Commit | Git tree | Archive bytes | GitHub archive SHA-256 |
|---|---|---|---:|---|
| Snapshot merge base | `3d3ba460a20a7e313d278c2df710a9c0b4bcb00e` | `bf39feb5dafc68dff586960ad77a0f60a68cd575` | 36,491,828 | `292765faafa4c711f574accab96656fcc0a5000ffdcf3e59fac72031a48da6ee` |

Consequently, a direct survey-base-to-head tree diff contains three unrelated
reversions in `CMakeLists.txt`, `csrc/libtorch_stable/ops.h`, and
`csrc/libtorch_stable/torch_bindings.cpp`. That direct diff is not the PR
solution and cannot serve as a golden patch.

The archive hashes above were measured from these immutable public endpoints:

- `https://github.com/vllm-project/vllm/archive/228bcc436b0f09cf1824f00fd44f8f9c94060940.tar.gz`
- `https://github.com/netanel-haber/vllm/archive/7d85640f2ddb613b83449982d5d6d5a7dcf14293.tar.gz`
- `https://github.com/vllm-project/vllm/archive/3d3ba460a20a7e313d278c2df710a9c0b4bcb00e.tar.gz`

These archives were used only for read-only eligibility analysis. They are not
Docker inputs, and no Dockerfile or Agent image was created.

## Open-PR drift

The survey snapshot described 14 commits, 39 changed files, 863 additions and
462 deletions. On 2026-08-25, GitHub reported that the still-open PR had moved
again to base `7ca49fbe4bab019e55d57cdc4b7fd3d55c67c1a6` and head
`b8bc7028dd8a5eb44d56eb1ef7385218fb47bfaa`, with 20 commits, 43 changed
files, 997 additions and 481 deletions. It was last updated on 2026-08-22 and
had another merge-conflict/rebase notice. There is no merged commit and no
maintainer-approved stable solution.

Review discussion also remains design-sensitive. Maintainers questioned the
maintenance cost of hand-written lazy imports and requested a narrower
dispatch shape; later discussion explicitly mentioned an overlapping narrower
patch. This makes the current head unsuitable as an authoritative golden
solution even independently of the SHA divergence.

## Atomicity and solution mapping

The intended user-visible task is coherent: preserve CLI behavior while making
top-level `vllm` and `vllm serve` help/error feedback fast. `vllm bench` is
explicitly out of scope. The implementation is nevertheless a broad import
architecture refactor across configuration, platform, quantization, tool
parser, and CLI modules. It depends on maintainer judgment about where lazy
boundaries belong.

If this item becomes eligible later, first freeze an accepted or merged head
and use its true merge base as the Agent baseline. The only valid solution
mapping for the surveyed snapshot would have been
`3d3ba460a20a7e313d278c2df710a9c0b4bcb00e` to
`7d85640f2ddb613b83449982d5d6d5a7dcf14293`; the surveyed `base_sha` must not
be substituted. A future accepted head will require a fresh mapping and fresh
archive/tree hashes.

## Hardware, dependencies, and data

This is a CPU-visible CLI startup task. It requires no model, dataset, GPU, or
CUDA Graph workload. A GPU-enabled official vLLM image may still be useful as
a dependency-complete runtime, but A100 execution cannot make an unstable
source pair eligible. The nearest official release before the snapshot merge
base was v0.22.1; it is only a prospective dependency donor and was not
selected or validated as an Agent environment.
