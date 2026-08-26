# Claude Opus 5：16 个 vLLM Harbor task 运行审计

本文汇总 Claude Opus 5 在首批 16 个可评分 vLLM Harbor task 上的单次运行结果。运行于
A100 节点，Agent 使用 Claude Code；每项均从未修复 Base 开始，只向 Agent提供
`instruction.md`、工作仓库和 Claude 配置，不提供隐藏 `tests/`、Oracle `solution/`
或 Docker socket。

运行日期为 2026-08-25。结果以 A100 上保留的原始 trajectory、Agent patch、runner
metadata、隐藏 verifier 输出和两项 native focused reverify 为准。

## 总结

- 最终通过：`4 / 16`，通过率 `25%`。
- 原始 runner 直接通过：`3 / 16`；`vllm-pr-32892` 在重建 candidate native 后由
  `0` 更正为 `1`。
- 累计 Agent 运行时间：`16,644s`，即 `4h 37m 24s`。
- 可恢复的 15 项合计：`528` turns、`$69.610107`；`vllm-pr-21476` 在 3600 秒
  超时前留下 174 个 user events 和 475 个 assistant events，但没有最终 result 事件，
  因此其 turns 和费用不能可靠恢复。
- 16 项的 patch 导出均成功；只有 `vllm-pr-21476` 的 Agent 进程超时退出
  `124`。
- 所有任务 `SECRET_LEAK=0`。Agent 容器没有挂载 `/tests`、`/solution` 或 Docker
  socket。

## 逐项结果

`Patch` 列是原始 `agent.patch` 的文件数和 numstat。`vllm-pr-21476` 的原始 patch
包含 10 个 `.deps` 构建噪音文件；focused reverify 过滤后，实际生产源码范围为 4 个
文件、`+39/-1`。

| Task | 最终 reward | 时间 | Turns / 费用 | Patch | 结果或首个失败契约 |
|---|---:|---:|---:|---:|---|
| `vllm-pr-34183` | **1** | 203s | 16 / $1.077489 | 1, +10/-2 | 用 weakref closure 断开 Request 自引用环；Request 和多模态 payload 生命周期全部通过。 |
| `vllm-pr-30475` | 0 | 854s | 37 / $4.349728 | 4, +33/-9 | API 形态错误：`get_num_embeds` 被实现/使用成方法，隐藏契约要求稳定属性值。 |
| `vllm-pr-29999` | 0 | 230s | 16 / $1.756092 | 1, +10/-6 | 只把全局 config lookup 移到 `__init__`；合法 profile 生命周期仍产生 `Current vLLM config is not set`。 |
| `vllm-pr-35781` | 0 | 1318s | 49 / $9.256628 | 1, +91/-48 | 大幅重构 scheduler，但没有 accepted `skipped_waiting` 状态契约。 |
| `vllm-pr-29345` | 0 | 220s | 10 / $0.787838 | 1, +164/-5 | 自创 `_bmm_kernel_persistent`，没有 production/accepted 入口 `bmm_kernel`。 |
| `vllm-pr-32892` | **1** | 707s | 24 / $2.938832 | 3, +44/-30 | 原始 runner 未重建 CUDA native 而误报 0；focused rebuild 后 correctness `7/7`，4096/512 ratio `2.723`。 |
| `vllm-pr-21476` | 0 | 3621s | 未知 / 未知 | 14, +105/-55 | Agent 超时；focused `_C` 构建成功，但导出成 `per_token_group_int8_quant`，accepted 名称为 `per_token_group_quant_int8`。 |
| `vllm-pr-32618` | 0 | 3428s | 71 / $16.360253 | 2, +86/-24 | 缺少 GPUModelRunner 的 `_pp_broadcast_prev_sampled_token_ids` 和 `_pp_receive_prev_sampled_token_ids_to_input_batch`。 |
| `vllm-pr-34179` | 0 | 846s | 38 / $5.103628 | 5, +93/-5 | 使用分散的替代实现，但没有 accepted `prepare_dcp_local_seq_lens`，真实 CUDA slot mapping 无法准备。 |
| `vllm-pr-42430` | **1** | 1023s | 29 / $5.189346 | 1, +37/-0 | 真实 Mamba FULL-CG metadata 路径通过：prior-state 单 token 为 decode，首 token 保持 prefill。 |
| `vllm-pr-28973` | **1** | 338s | 23 / $2.851325 | 1, +38/-0 | production `_update_states` 原地更新 streaming-session cached request，CPU 隐藏 verifier 通过。 |
| `vllm-pr-30282` | 0 | 595s | 48 / $3.455342 | 5, +11/-28 | 接收了显式 config，但没有保存 `self.moe_parallel_config`；ordinary/DP+EP identity 契约失败。 |
| `vllm-pr-34246` | 0 | 907s | 33 / $4.435635 | 3, +28/-21 | CPU mask 路径执行 `positions.to(cuda)`，在 sync-debug 下触发同步 CUDA 操作。 |
| `vllm-pr-39337` | 0 | 835s | 58 / $4.526032 | 3, +67/-10 | 实现了另一种 resolved field 形态；隐藏 verifier 无法取得 `VllmConfig` 的 model-runner selection API。 |
| `vllm-pr-39832` | 0 | 1034s | 38 / $4.875634 | 7, +63/-119 | current consumer 和内部 TypeError 传播通过；legacy connector 应在构造前抛 `ValueError`，候选却抛 `TypeError`。 |
| `vllm-pr-40841` | 0 | 485s | 38 / $2.646304 | 1, +458/-0 | 从头实现了不同的 supervisor API；accepted helper `_build_vllm_dp_server_args` 缺失。 |

## 失败模式

12 个失败任务中，主要问题不是环境无法运行，而是 Agent 偏离 accepted behavior：

- **API/solution mapping 偏离**：`30475`、`29345`、`30282`、`32618`、`34179`、
  `39337`、`40841`，以及 native symbol 命名错误的 `21476`。
- **生命周期或错误边界理解错误**：`29999`、`39832`。
- **过度重构但遗漏状态契约**：`35781`。
- **GPU 同步语义错误**：`34246`。

几个任务中，Agent 的自测或最终陈述声称成功，但隐藏 verifier 在第一个 accepted contract
上就失败。这说明 verifier 需要继续以 production API、对象身份、错误类型、真实
CUDA/NCCL 路径和性能硬门为准，不能接受“功能近似”的替代入口。

## Runner 协议修正

最初 runner 在 Agent 修改 CUDA/C++ 源码后，直接在 Base image 中运行 verifier，没有
先重建并固化 candidate native。这会把源码 patch 与旧 `.so` 混用。因此对两项 native
patch 做了 focused reverify：

- `vllm-pr-32892`：重建成功，candidate native correctness 与性能门通过，最终从原始
  reward `0` 更正为 `1`。
- `vllm-pr-21476`：focused `_C` 构建耗时 `1662.69s` 且 import 成功；随后 verifier
  仍因 native export symbol 词序错误失败，最终保持 `0`。

后续 runner 已改为：应用 patch后构建/提交 candidate image，再在离线新容器中执行隐藏
verifier。native 任务必须保留并验证新构建的扩展，不能复用 Base `.so`。

## 基础设施与隔离审计

- 除 `vllm-pr-21476` 的 Agent timeout 外，没有 Agent 基础设施退出失败。
- `vllm-pr-34246` 的 verifier exit `1` 是候选触发真实同步 CUDA RuntimeError，不是
  基础设施故障。
- 其余 verifier 进程退出码为 `0`；reward `0` 来自隐藏行为断言。
- `vllm-pr-32618` 发生一次 API retry，随后 Agent 正常完成。
- 每项 Agent 容器只挂载 Claude settings、`instruction.md` 和 Claude CLI；隐藏测试、
  Oracle 与 Docker socket 均不可见。
- 16 项 secret scan 全部为 `SECRET_LEAK=0`，settings/API token 未进入 trajectory、
  patch 或 verifier 日志。

## 原始轨迹索引

所有原始运行资产保留在 A100：

```text
/data/ai-infra-bench/claude16-eval/runs/<task>/attempt-1/
```

16 个完整运行目录也已统一打包：

```text
/data/ai-infra-bench/claude16-eval/claude-opus-5-vllm-16-full-runs-20260825.zip
```

- 大小：`261,410,703` bytes；
- SHA-256：`67490e50d02a7401e020ea0e50b76cd6dad52996e8dd5749e2678ba967b12ae5`；
- `unzip -tq`：`No errors detected in compressed data`。

每个目录至少包含 `trajectory.raw.jsonl`、`agent.patch`、`metadata.env`、Agent/patch/status/
verifier exit code、verifier stdout/stderr、container inspect 和 `secret-scan.txt`。轨迹不
直接提交到 Git；下表给出字节数和 SHA-256，便于核验远端资产未变化。

| Task | Bytes | SHA-256 |
|---|---:|---|
| `vllm-pr-21476` | 1,008,192 | `c60774d86c94cf92c2d6d373c920486d51a15a9e80cada3782d1dd0a4af0eb36` |
| `vllm-pr-28973` | 753,063 | `4db48fc106fc176571ac05f2c094cba9a6d4d42935b51b99dfdd697a561b55b5` |
| `vllm-pr-29345` | 172,325 | `a594653307d0ed4c71c62cdf9f705299182dd571aa46b87c4f3a0fe612cc7923` |
| `vllm-pr-29999` | 450,167 | `a02d49ab52e339e3a5e1b1707f8a503c0b50c7094253a5ee9f160a6e3d9dc786` |
| `vllm-pr-30282` | 904,867 | `a5b719dce9ef15d034fc9d85883d2166fc99f4dcdde70aa80197de9c4742458e` |
| `vllm-pr-30475` | 460,901 | `3c220b41fdcdc894b341b5b984dab9c20ef0cfcfa23afa7cfabdef9a8c04f73f` |
| `vllm-pr-32618` | 2,216,382 | `94da9e01d0443201466dbd87bcf6485c1b2419e06d2f26b0f79ffcf7c3312fe5` |
| `vllm-pr-32892` | 265,325 | `39b96e1bed930b15f790a7732068caeb056818d194f4c295f071f39b6daf253e` |
| `vllm-pr-34179` | 786,043 | `6dc868b2ac21a6a66d5c150d93028018b52bcb54364272ab1b927564febfd36b` |
| `vllm-pr-34183` | 118,150 | `70145ae425b921f15bc1cdced45b873f99b73ea5627fa185c6d03844aae6af90` |
| `vllm-pr-34246` | 428,818 | `66f0fafbd7853dcf160e64c883b1e2f2a10923478a01cc9fbc07c81abd5afcdc` |
| `vllm-pr-35781` | 2,132,814 | `37ea8e38811a4dca185dc93b9799a2bcbc2fcc035eee33238f1aff9d531e66ef` |
| `vllm-pr-39337` | 1,308,737 | `2e297ac005445f84074928d8b409019d1f164ea544550b88189a388f8d95e204` |
| `vllm-pr-39832` | 481,171 | `dc5094e63d99b80145644ff4ccf9d74a0da9fa04543a75566187351ad22a4204` |
| `vllm-pr-40841` | 342,850 | `fe9e88527d00107e4c3b093069d53a33327a870c5275e56c9c55bfc1b7f69bea` |
| `vllm-pr-42430` | 357,470 | `78e69da122ca24908a3c06ee35367cb4835be1b1fabb6fd02429d215b7bad9a3` |

## 对后续评测的建议

1. native/CUDA patch 必须在 candidate image 中重新编译并记录扩展 hash。
2. Agent timeout 后仍应导出 patch，但必须把“Agent 未完成”和“patch 可重建验证”分开记录。
3. 结果表同时保留 raw reward 与协议修正后的 final reward。
4. 失败报告记录第一个 accepted contract，而不是只记录最终 `reward=0`。
5. 继续保持 solution/tests 运行时只读注入，并对 Agent trajectory 做 secret scan。
