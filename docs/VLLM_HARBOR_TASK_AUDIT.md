# vLLM Survey Harbor 封装审计

首轮把 24 项 survey 审计中标记为 `validated` 的 10 项发布为 Harbor task。随后对 7 项
`environment-only` 做二次行为审计，其中 6 项补齐 accepted Oracle 与 Base `0` / Oracle
`1` 后升级；29595 因 A100 是明确负对照而保持 blocked。当前共 16 项可评分 Harbor
task。二次验收细节见
[vLLM environment-only 二次验收](VLLM_ENVIRONMENT_ONLY_RETRY_AUDIT.md)。
首轮 Claude Opus 5 实跑结果、Agent 轨迹索引和失败分类见
[16-task Agent 运行审计](CLAUDE_OPUS_5_16_TASK_RUN_AUDIT.md)。

## 统一格式

每个任务均包含：

- `task.toml`：Harbor schema、资源、超时和离线运行约束；
- `instruction.md`：只描述问题、可观察症状和允许修改的仓库范围；
- `environment/Dockerfile`：固定 Base commit、依赖与运行环境；
- `solution/solve.sh`、`solution/fix.patch`：从已接受的上游合并提交提取 Oracle；
- `tests/test.sh`、`tests/verify_*.py`：运行时注入的隐藏 verifier，写入
  `/logs/verifier/reward.txt`。

Docker 镜像中不复制 `tests/` 或 `solution/`。运行阶段关闭网络；GPU、CPU、内存、
存储与超时均在 `task.toml` 中显式声明。

## 纳入的任务

| Task | 资源 | 评分核心 |
|---|---:|---|
| `vllm-pr-34183` | CPU | Request 与多模态 payload 生命周期 |
| `vllm-pr-30475` | CPU | 稀疏多模态 embedding 容量核算 |
| `vllm-pr-29999` | 1×A100 | 真实 Triton MoE profile 的告警双对照 |
| `vllm-pr-35781` | CPU | remote-KV blocked queue/callback 次数 |
| `vllm-pr-29345` | 1×A100 | Triton BMM 正确性与相对性能 |
| `vllm-pr-32892` | 1×A100 | exact `_moe_C` 正确性与大 batch scaling |
| `vllm-pr-21476` | 1×A100 | exact `_C` INT8 quant 原生算子与正确性 |
| `vllm-pr-32618` | 2×A100 | 两 rank NCCL sampled-token 路径 |
| `vllm-pr-34179` | 1×A100 | DCP Triton slot mapping 与 CUDA Graph replay |
| `vllm-pr-42430` | 1×A100 | Mamba FULL-CG metadata 语义 |

二次升级的 6 项为 `vllm-pr-28973`、`vllm-pr-30282`、`vllm-pr-34246`、
`vllm-pr-39337`、`vllm-pr-39832` 和 `vllm-pr-40841`；其资源与评分核心见二次验收文档。

## 验收门槛

发布前必须同时满足：

1. 当前 Harbor SDK 能解析全部 task；
2. Dockerfile 的 build context 自洽，不依赖 curator 工作目录中的临时文件；
3. Oracle patch 能干净应用到锁定 Base；
4. 未修复 Base 的 verifier reward 为 `0`；
5. 应用 Oracle 后同一个 verifier reward 为 `1`；
6. GPU task 在 A100 上执行真实 CUDA/Triton/NCCL 路径，不用 mock 或静态检查替代；
7. verifier 与 Oracle 不进入 Agent 镜像，运行时只读注入。

## A100 Base/Oracle 结果

相同 verifier 在未修复 Base 和上游已接受 Oracle 上分别运行。CPU 任务也在同一
A100 节点的隔离 daemon 中执行，但不申请 GPU。

| Task | Base reward | Oracle reward | Oracle 关键证据 |
|---|---:|---:|---|
| `vllm-pr-34183` | 0 | 1 | Request/payload retained `0/16` |
| `vllm-pr-30475` | 0 | 1 | `P=100, E=8, capacity=8` 可分配 |
| `vllm-pr-29999` | 0 | 1 | 合法 lifecycle 无警告；真实缺配置仍警告 |
| `vllm-pr-35781` | 0 | 1 | 24 blocked、5 idle rounds、callback scans `0` |
| `vllm-pr-29345` | 0 | 1 | correctness 全过；A100 相对 legacy `6.09x` |
| `vllm-pr-32892` | 0 | 1 | 7 组 correctness；4096/512 ratio `2.882` |
| `vllm-pr-21476` | 0 | 1 | native op 存在；三 dtype 正确；`2.39–2.44x` |
| `vllm-pr-32618` | 0 | 1 | 两 rank NCCL token/mapping/discard/placeholder 全过 |
| `vllm-pr-34179` | 0 | 1 | production Triton slot mapping + CUDA Graph replay |
| `vllm-pr-42430` | 0 | 1 | prior-state 单 token 为 decode，首 token 保持 prefill |

结构验证使用 Harbor `0.22.0`：当前 16 项均能被 SDK 加载，且
`harbor run -p <task> -a nop --print-config` 均成功。具体构建证据、完整输出与
残余范围见各任务的 `validation/docker-build.md`；本表只记录 Harbor 封装后的
统一评分门结果。
