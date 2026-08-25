# vLLM Survey Docker 构建审计

本轮审计覆盖 `data/vllm_survey_results.jsonl` 的全部 24 项，使用 A100 隔离 Docker
daemon 进行真实构建或硬件资格检查。

结果：10 个 `validated`、7 个 `environment-only`、7 个 `blocked`。

| # | Task | 状态 | 核心结论 |
|---:|---|---|---|
| 1 | vllm-pr-34183 | validated | 稳定复现 Request/payload 保留 |
| 2 | vllm-issue-29595 | environment-only | A100 是原问题负对照 |
| 3 | vllm-pr-30475 | validated | 稳定复现容量错误拒绝 |
| 4 | vllm-pr-28973 | environment-only | project-scale，native ABI 近似 |
| 5 | vllm-pr-29999 | validated | 真实 Triton MoE 告警双对照 |
| 6 | vllm-pr-35781 | validated | scheduler callback/queue 结构复现 |
| 7 | vllm-pr-40408 | blocked | 目标 FP8 kernel 需要 SM89+ |
| 8 | vllm-pr-29345 | validated | correctness + A100 配对性能 |
| 9 | vllm-pr-32892 | validated | exact `_moe_C` + 大 batch scaling |
| 10 | vllm-pr-21476 | validated | exact `_C`，A100 约 2.46–2.50x |
| 11 | vllm-pr-39832 | environment-only | constructor contract，缺 solved pass |
| 12 | vllm-pr-30282 | environment-only | 真实 Triton 后触发 config contract |
| 13 | vllm-pr-40841 | environment-only | node-local supervisor，不含 K8s E2E |
| 14 | vllm-pr-32618 | validated | 两 rank NCCL token path |
| 15 | vllm-pr-39337 | environment-only | config contract，不含 consumers |
| 16 | vllm-pr-20087 | blocked | DeepGEMM v2 需要 SM100/B200 |
| 17 | vllm-pr-34179 | validated | DCP Triton slot mapping + graph replay |
| 18 | vllm-issue-27433 | blocked | open umbrella，无唯一 Base/Oracle |
| 19 | vllm-issue-41286 | blocked | open migration umbrella，多 Base |
| 20 | vllm-pr-34246 | environment-only | merge primitive 通过，mixed native 风险 |
| 21 | vllm-pr-42430 | validated | Mamba FULL-CG metadata contract |
| 22 | vllm-pr-41518 | blocked | open gold 漂移且 Base/Head 分叉 |
| 23 | vllm-pr-45895 | blocked | 753B 模型与 FP8 路径超出 A100 能力 |
| 24 | flashinfer-pr-3803 | blocked | 目标 native target 仅支持 Blackwell |

每项的 source/digest、构建时间、镜像 ID、Base/Oracle 结果和剩余风险见对应目录下的
`validation/docker-build.md`。`blocked-before-build` 项没有伪造 Dockerfile 或镜像。

其中 10 个 `validated` 项已进一步封装为统一 Harbor task；格式、资源与统一
Base/Oracle 门禁见 [vLLM Survey Harbor 封装审计](VLLM_HARBOR_TASK_AUDIT.md)。
