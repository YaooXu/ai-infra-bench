# AI Infra Task 构建指南

这份短指南用于从真实 Issue / PR 构建可交给 Agent 的基础设施任务。

## 先做资格检查

写 Dockerfile 前先确认：

1. 问题有唯一、可观察的行为边界；
2. Base commit 已冻结，并且是选定 Head 的祖先；
3. Oracle 可以从已接受的 PR、squash commit 或独立参考实现中映射；
4. 当前 GPU 型号、数量、显存和依赖能够触发目标路径；
5. 模型和数据成本可接受，或最小子集已经证明 Base 失败、Oracle 通过。

Open umbrella、持续漂移的 open PR、没有唯一 Base/Oracle、硬件不符或模型装不下时，
应发布 `blocked-before-build` 审计，不要任选一个 child commit 或微型模型改变题目。

## 一个可发布 Task 的最小组成

```text
tasks/<task-id>/
├── instruction.md
├── environment/
│   ├── Dockerfile
│   ├── lock/
│   └── public_dev/
├── tests/                 # 独立的最终验证
├── solution/              # Agent 不可见
└── validation/
```

本轮 survey 目录只负责环境构造与资格审计，因此尚未补齐 `instruction/tests/solution` 的
项目必须保持 `environment-only`，不能直接称为 release-ready。

## Docker 环境硬要求

- 按 source cutoff 选择最近的官方发布镜像，并锁 amd64 digest；
- 锁 exact source archive、SHA-256 和 canonical Git tree；
- 工作树只有一个 synthetic commit、没有 remote，运行用户非 root 且可写；
- runtime 使用 `--network none`，镜像配置和 history 不保留代理或凭据；
- 不使用 `VLLM_TARGET_DEVICE=empty`，至少执行真实 native import 和 GPU tensor/op；
- Candidate Python、关键模块和 native extension 必须从 Agent 可修改工作树加载。

若复用 release native artifact，必须记录 Torch/CUDA/Python ABI、来源版本、相对路径和
hash。Post-cutoff donor 只能通过 multi-stage 白名单复制，不能带入未来 Python、tests 或
staging。出现 duplicate registration 或 ABI 近似时，要下调环境声明范围。

涉及 C++/CUDA 的任务优先从 exact source 构建完整目标 extension。可以 focused build，
但必须恢复临时 CMake 修改，核对目标架构 cubin，并明确其他 extension 不是 exact。

## 测试分三层

1. `public_dev`：Agent 可运行，一条命令，离线，Base 稳定失败；
2. Oracle control：同一环境正向通过，但补丁、head 文件和验证脚本不进入 Agent 镜像；
3. hidden verifier：覆盖 heldout、原功能回归和常见偷懒解法，处于独立信任边界。

测试应执行真实服务、CUDA、通信或目标生产分支。Constructor mock、source-string 检查和
import smoke 只能作为前置探针，不能冒充症状复现。新增功能题还必须有隔离 Oracle 正向
通过，不能只凭 Base 的 missing symbol 退出。

## 数据、性能与硬件

- 大数据集可固定小型代表子集，但不能先下载完整数据再裁剪；
- 小模型只有证明同一 Base-fail/Oracle-pass 后才能替代原模型；
- OOM 可缩到明确的 causal primitive，但要测 allocator 增量、正确性和同步，不只等 OOM；
- 性能使用同 GPU、同输入的 Base/Oracle 配对测量，正确性先于计时；
- upstream 在 H100/B200 的绝对数字不能直接成为 A100 阈值；
- 硬件不满足时记录 capability、目标 dispatch/translation unit 和未来 verifier 条件。

## 状态必须分开写

至少分别记录：

- environment 是否可构建；
- pipeline smoke 是否通过；
- 原症状是否复现；
- solution 是否唯一映射；
- Base/Oracle 是否形成两侧对照；
- hidden verifier 是否完成；
- 完整模型、数据和拓扑还缺什么。

Docker build 成功只说明环境可构建，不代表 Task 已经是权威评测。
