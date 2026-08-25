# vLLM environment-only 二次验收

本轮重新审计最初标记为 `environment-only` 的 7 项。目标不是把“能启动的镜像”包装成
Task，而是为每项重新寻找可被上游 accepted Oracle 解释的最小真实行为边界，并要求
Harbor Base reward 为 `0`、同一 verifier 上 Oracle reward 为 `1`。

最终 6 项升级为可评分 Harbor task，1 项保持 blocked。

## 升级结果

| Task | Scored behavior | 资源 | Base | Oracle | Harbor image |
|---|---|---:|---:|---:|---|
| `vllm-pr-28973` | GPU runner streaming continuation 原地更新 cached request | CPU | 0 | 1 | `94ca1f6c7342…` |
| `vllm-pr-30282` | modular MoE 显式配置所有权、真实 Triton 与两个 production consumer | 1×A100 | 0 | 1 | `f232f10352aa…` |
| `vllm-pr-34246` | CPU mask 合并 CUDA multimodal embeddings，无同步且分配受限 | 1×A100 | 0 | 1 | `816a69704cc0…` |
| `vllm-pr-39337` | Model Runner V2 三态选择与真实 GPUWorker consumer | 1×A100 | 0 | 1 | `b98f42baebd8…` |
| `vllm-pr-39832` | KV connector 三参数 lifecycle 与 TypeError 边界 | CPU | 0 | 1 | `f19cf5a1c619…` |
| `vllm-pr-40841` | node-local DP supervisor 的进程、HTTP、故障传播与端口清理 | CPU | 0 | 1 | `5edc57698af5…` |

CPU task 仍在 A100 节点独立执行 candidate native import 与 CUDA allocation 作为镜像
完整性证据，但这些探针不进入 scored behavior，也不让每次评分无意义地占用 GPU。

每项均包含 `task.toml`、`instruction.md`、`solution/`、`tests/` 和最终
`validation/docker-build.md`。Harbor `0.22.0` 能加载全部 6 项，且
`harbor run -p <task> -a nop --print-config` 均成功。

## 仍然 blocked：vllm-issue-29595

29595 是 Hopper 上 Qwen3-VL compiled grounding 退化。上游报告明确 A100、A800 和
Jetson 不出现该退化；已有 A100 上的真实 vLLM 服务、真实图片、`torch.compile` 与
CUDA Graph 运行也是稳定负对照。accepted 解法落在 Triton/PTXAS 依赖层，Base 与修复
后的 Triton 在 A100 上都应保持正确，因此无法形成可信的 `0 → 1`。

它保留为 `BLOCKED-BEFORE-HARBOR-PACKAGING` 资格审计，不创建伪 `task.toml`、静态
字符串 verifier 或小模型替代症状。重新启动该任务至少需要受影响的 H20/H100/H200、
固定 Qwen3-VL 模型与图片集，以及 exact Triton Base/Oracle build。

## 本轮统一修正

- Docker build context 固定为每项的 `environment/`；Dockerfile 的本地 `COPY` 路径
  也以此为根。
- 删除历史 `environment/public_dev`，Agent 镜像默认不含 `public_dev`、`tests` 或
  `solution`，避免轻量根因脚本泄漏搜索空间。
- Oracle 只提取 accepted behavior 所需生产文件；大 PR 中未验证的 Scheduler、
  FlashInfer/NIXL、Kubernetes 或完整模型服务不进入任务声明。
- factory/config 类任务降为 CPU 资源；A100 只做独立 provenance probe。
- verifier 执行 production method、consumer、CUDA/Triton 或真实子进程生命周期，
  不以 patch、文件名或源码字符串相似度评分。

各项的完整 source、ancestry、镜像 digest、build 时间、Base/Oracle 原始输出和残余范围
见对应 `tasks/<task>/validation/docker-build.md`。
