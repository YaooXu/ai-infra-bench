# Environment lock: intentionally blocked

No publishable Agent Dockerfile or dependency lock is emitted for
`vllm__issue__27433`.

The survey item is an open umbrella issue and supplies no `base_sha`, `head_sha`,
or closing solution. Its linked implementation PRs have different source bases,
hardware targets, models, topologies, and behavior contracts. Selecting any one
of them would silently redefine the issue-level task; selecting repository HEAD
at issue creation or survey snapshot time would be an ungrounded inferred base.
Neither choice can satisfy an exact-base/canonical-tree contract.

The absence of a Dockerfile is deliberate. It prevents an environment-only GPU
probe or an already-packaged child PR from being published as an exact Agent
image for the umbrella program. No solved code, model weights, or hidden Oracle
material is present in this task directory.

The task can be unblocked only after curation provides:

- one atomic child issue/contract;
- an explicit candidate `base_sha` and expected canonical tree;
- one mapped solution SHA or closing PR;
- a model artifact digest and supported GPU/topology;
- a two-sided behavioral Base/Oracle outcome.

The environment and dependency choices can then be locked against that source
cutoff. Existing survey tasks for PR #29345 and PR #40408 are valid child-level
work; they are not substitutes for this umbrella issue.
