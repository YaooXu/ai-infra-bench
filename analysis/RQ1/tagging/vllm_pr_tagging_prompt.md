You are labeling exactly one merged pull request from `vllm-project/vllm`. Never classify the repository as a whole.

## Inputs

- Repository: `__REPOSITORY__`
- Read-only repository workspace at the PR base commit: `__WORKSPACE__`
- PR metadata and compact CI summary: `__METADATA_PATH__`
- Complete raw CI evidence, for lookup only when the summary is insufficient: `__CI_EVIDENCE_PATH__`
- Complete PR patch: `__PATCH_PATH__`
- Required result JSON path: `__RESULT_PATH__`

Read the metadata summary and patch first. Inspect the repository, nearby tests, CODEOWNERS, and matching Buildkite definitions when necessary. Open the raw CI file only to resolve an unclear job, result, timing, device alias, rerun, or command. Treat the workspace as read-only.

## Evidence rules

- Classify this PR, not vLLM in general.
- Determine `change_type` from the intended outcome; determine `project_scope` from the role of materially changed artifacts.
- `affected_platforms` describes hardware backends deliberately targeted by materially changed production code, build support, tests, benchmarks, CI, or documentation. It does not describe the reproduction environment or a runner that merely happened to execute backend-agnostic validation.
- Backend-specific support work counts: a CUDA-only test fix, XPU CI change, ROCm build dependency, or TPU support document targets that backend even without production-code changes. Use `backend_agnostic` only when no changed artifact targets a backend.
- A backend counts only when the patch materially introduces, removes, or modifies its behavior or support. Do not tag a backend from an unchanged downstream consumer, incidental runner, comparison-only reference implementation, or reproduction route.
- A changed benchmark that directly executes or measures a backend implementation targets that backend. A backend used only as an unchanged comparison reference does not.
- `torch.cuda`, `device="cuda"`, `.cuda()`, and similar generic PyTorch CUDA-namespace usage are not sufficient NVIDIA evidence because ROCm uses the same API namespace. Require a changed NVIDIA-specific path, guard, library, architecture, or CI target.
- A `cpu_test` marker, CPU-runnable test, host-side orchestration, or mock is not by itself evidence that the CPU backend is affected. Require a changed CPU-specific implementation, build/test contract, CI target, or documentation contract.
- Test-file changes do not prove that tests ran.
- Use only concrete validation evidenced before merge. Post-merge runs do not prove pre-merge validation.
- Consider only checks relevant to the changed behavior. Do not let unrelated repository jobs determine `verification.tested`.
- A generic pre-commit, lint, or build success does not make a functional production bugfix, feature, or performance change `passed`. Ignore that unrelated check. Static/build validation may count when it directly validates the primary deliverable itself.
- For reruns of the same check, use the latest conclusive pre-merge result. A later successful rerun supersedes its earlier failure.
- Skipped, cancelled, neutral, or inconclusive jobs do not by themselves make the result mixed.
- `verification.methods` must contain `performance_benchmark` exactly when `verification.performance_benchmark` is `run_with_result` or `run_without_result`.
- An exact command written in PR prose is not by itself proof that it ran. Use `claim_only` when execution or success is asserted without an inspectable execution record, CI result, artifact, log, or measurement.

## Reproduction rules

- Return exactly one primary reproduction environment.
- First decide whether CPU-only execution can adequately verify the important changed behavior. If yes, `reproduction.platform` must be `cpu`.
- Do not require an accelerator merely because the repository is vLLM, accelerator CI exists, or the patch mentions CUDA, ROCm, kernels, models, or an accelerator-related path.
- Select an accelerator only when CPU-only execution cannot adequately verify the important changed behavior. State explicitly why CPU is insufficient in both `reproduction.reasoning` and `reproduction.platform.reasoning`.
- Always select a concrete configurable `reproduction.platform`; `unknown` is not available. For backend-agnostic GPU behavior, choose the closest relevant successful repository CI route (for example an evidenced H100/H200 NVIDIA route) while keeping `affected_platforms=backend_agnostic`.
- When an accelerator is necessary, prefer the exact matching successful pre-merge CI route, then an explicit PR command or benchmark environment, then the nearest repository-defined route that exercises the change.
- The selected model is a concrete machine choice for reproduction, not a claim that the implementation requires that exact SKU.
- Use `other` for an explicit unlisted model and `unknown` when an accelerator is selected but no exact model is evidenced.
- Keep platform, model, topology, count, and memory consistent. Never infer allocated memory from physical device capacity.
- Use `unknown` for host CPU count or host memory unless a CI resource declaration, runner profile, or explicit PR statement provides evidence.
- `commands` contains at most five exact commands, test targets, or CI scripts supported by evidence.
- `software_requirements` contains at most eight concise, explicit requirements. Do not invent versions, drivers, CPU models, models, datasets, or environment variables.
- Use `platform=none` only when no executable command or software environment is needed; then both `commands` and `software_requirements` must be empty. If reproduction executes even a static command, use `cpu` or an evidenced accelerator.

## Output and reasoning rules

- Every selected single-label value, every selected multi-label value, and every open-code item must contain exactly `value` and `reasoning`.
- `reasoning` must be concise Chinese and cite conclusion-relevant evidence such as a changed path, patch behavior, PR statement, CI job, runner route, command, or repository configuration.
- The structural `verification` and `reproduction` objects must also contain a concise Chinese `reasoning` summarizing the evidence used for that group.
- Do not use English-only reasoning, generic restatements of the label, unsupported guesses, or private chain-of-thought.
- A good reasoning is normally one or two Chinese sentences.

## Consistency examples

These are fragments that clarify edge cases. Still emit the complete required output object below.

Backend-specific support-only change:

```json
{
  "project_scope": [{"value": "tests", "reasoning": "补丁只修改 CUDA 上执行的融合测试。"}],
  "architecture": [{"value": "support_only", "reasoning": "没有生产代码变更。"}],
  "affected_platforms": [{"value": "nvidia_cuda", "reasoning": "被修改测试只覆盖 CUDA 后端；这不是由偶然的 CI runner 推断。"}]
}
```

Executed benchmark consistency:

```json
{
  "verification": {
    "reasoning": "PR 在合并前实际执行了延迟测量并报告结果。",
    "methods": [{"value": "performance_benchmark", "reasoning": "PR 报告了实际执行的延迟测量。"}],
    "performance_benchmark": {"value": "run_with_result", "reasoning": "报告包含可检查的前后延迟数值。"}
  }
}
```

Accelerator selection consistency:

```json
{
  "reproduction": {
    "reasoning": "CPU 无法执行被修改的 CUDA 内核，因此采用合并前成功的 H200 路线。",
    "platform": {"value": "nvidia_cuda", "reasoning": "CPU 无法覆盖真实 CUDA 内核执行路径。"}
  }
}
```

Backend-agnostic change with a concrete GPU reproduction route:

```json
{
  "affected_platforms": [
    {"value": "backend_agnostic", "reasoning": "补丁修改通用 GPU 逻辑，没有改变厂商专用支持契约。"}
  ],
  "reproduction": {
    "reasoning": "CPU 无法覆盖真实 GPU 执行；选用合并前成功的 H200 测试路线。",
    "platform": {"value": "nvidia_cuda", "reasoning": "CPU 无法执行关键 GPU 路径，H200 路线有直接成功记录。"},
    "accelerator_model": {"value": "nvidia_h200", "reasoning": "相关成功 CI 明确使用 H200。"}
  }
}
```

Unrelated static check for a functional production change:

```json
{
  "verification": {
    "reasoning": "只有通用 pre-commit 成功，没有执行被修改的 TPU 权重加载行为。",
    "test_assets": {"value": "none", "reasoning": "补丁未修改测试资产。"},
    "tested": {"value": "no_evidence", "reasoning": "通用静态检查不能证明关键生产行为被测试。"},
    "methods": [{"value": "none", "reasoning": "没有与关键行为直接相关的具体执行证据。"}],
    "performance_benchmark": {"value": "not_applicable", "reasoning": "该功能修复不以性能为目标。"}
  }
}
```

Review-only change with no executable environment:

```json
{
  "reproduction": {
    "reasoning": "该文档措辞变更只需人工审阅，不需要可执行环境。",
    "platform": {"value": "none", "reasoning": "没有需要执行的验证命令。"},
    "accelerator_model": {"value": "none", "reasoning": "不执行加速器验证。"},
    "topology": {"value": "none", "reasoning": "不分配加速器。"},
    "accelerator_count": {"value": "none", "reasoning": "不分配加速器。"},
    "accelerator_memory": {"value": "none", "reasoning": "没有显存需求。"},
    "host_cpu_architecture": {"value": "none", "reasoning": "不需要可执行主机环境。"},
    "host_cpu_count": {"value": "none", "reasoning": "不需要可执行主机环境。"},
    "host_memory": {"value": "none", "reasoning": "不需要可执行主机环境。"},
    "commands": [],
    "software_requirements": [],
    "confidence": {"value": "high", "reasoning": "补丁与审阅范围明确。"}
  }
}
```

## Tagging schema

`__TAGGING_SCHEMA__`

## Compact CI hardware catalog

`__HARDWARE_CATALOG__`

## Required output shape

Write exactly one JSON object to `__RESULT_PATH__`:

```json
{
  "change_type": {"value": "bugfix", "reasoning": "该 PR 修复了补丁中可见的既有行为回归。"},
  "project_scope": [
    {"value": "production_code", "reasoning": "补丁修改了实际发布的 vLLM 运行时代码。"},
    {"value": "tests", "reasoning": "补丁同时修改了用于验证该行为的可执行测试。"}
  ],
  "architecture": [
    {"value": "engine", "reasoning": "改动影响引擎生命周期和内部控制流。"}
  ],
  "affected_platforms": [
    {"value": "backend_agnostic", "reasoning": "该引擎修复不改变任何特定硬件后端的支持契约。"}
  ],
  "verification": {
    "reasoning": "合并前相关测试有可检查的成功记录，未发现相关最终失败结果。",
    "test_assets": {"value": "modified", "reasoning": "补丁修改了现有测试文件，但没有新增或删除测试。"},
    "tested": {"value": "passed", "reasoning": "相关合并前测试的最终有效运行结果均为成功。"},
    "methods": [
      {"value": "unit_test", "reasoning": "CI 执行了补丁对应的聚焦测试目标。"}
    ],
    "performance_benchmark": {"value": "not_applicable", "reasoning": "该修复不改变性能目标，性能基准与关键行为无关。"}
  },
  "reproduction": {
    "reasoning": "CPU 足以执行相关聚焦测试，因此不需要为该改动配置加速器。",
    "platform": {"value": "cpu", "reasoning": "相关测试不执行必须依赖真实加速器的路径。"},
    "accelerator_model": {"value": "none", "reasoning": "CPU 已能充分验证关键改动。"},
    "topology": {"value": "none", "reasoning": "所选验证环境不使用加速器。"},
    "accelerator_count": {"value": "none", "reasoning": "所选验证环境不分配加速器。"},
    "accelerator_memory": {"value": "none", "reasoning": "所选验证环境没有加速器显存需求。"},
    "host_cpu_architecture": {"value": "not_arch_specific", "reasoning": "补丁与测试没有表现出主机架构特定要求。"},
    "host_cpu_count": {"value": "unknown", "reasoning": "现有 CI 和 PR 证据没有给出主机 CPU 分配数量。"},
    "host_memory": {"value": "unknown", "reasoning": "现有 CI 和 PR 证据没有给出主机内存分配量。"},
    "commands": [
      {"value": "pytest -q tests/example.py", "reasoning": "该命令对应补丁修改的聚焦测试文件。"}
    ],
    "software_requirements": [],
    "confidence": {"value": "medium", "reasoning": "测试目标明确，但主机资源分配信息缺失。"}
  }
}
```

Output JSON only. Do not wrap it in Markdown.
