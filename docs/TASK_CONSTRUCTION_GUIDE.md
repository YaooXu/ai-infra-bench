# AI Infra Bench Task 完整构造手册

本文说明如何从真实 Issue / PR 构造一个可交给 Coding Agent、可离线运行、可由
Harbor 评分的 AI 基础设施任务。目标不是复刻 PR diff，而是冻结一个真实工程问题，
使未修复 Base 稳定失败、合法修复通过，并让浅层绕过无法得分。

本仓库 10 个已封装任务的实际 Base/Oracle 结果见
[vLLM Survey Harbor 封装审计](VLLM_HARBOR_TASK_AUDIT.md)。

## 1. 最终交付标准

一个可发布 Task 至少要回答清楚六件事：

1. 问题是什么，外部可观察行为是什么；
2. 未修复 Base 是哪个不可变 commit；
3. Oracle 是什么，为什么它代表被上游接受的正确行为；
4. Docker 环境如何绑定源码、依赖和硬件；
5. Agent 如何开发，最终 verifier 如何判断行为而不是比较 patch；
6. 哪些原始模型、数据、硬件或拓扑没有被覆盖。

完成 Docker build 只代表环境可构建，不代表 Task 已可发布。发布硬门是：

```text
Base reward = 0
Oracle reward = 1
no-op reward = 0
关键错误解法 reward = 0
```

## 2. 第一步：资格检查

写 Dockerfile 前完成资格审计。以下条件缺一不可：

- 有唯一、可观察的行为边界；
- Base commit 已冻结；
- Base 是选定 Oracle 的祖先；
- 有稳定且被接受的 Oracle；
- 当前硬件能进入目标生产路径；
- 模型、数据、依赖和运行成本可承受，或能够证明一个等价的缩小范围；
- 任务能够离线交付给 Agent，不依赖临时账号或持续变化的外部服务。

以下情况不要强行构造：

- Open umbrella 同时包含多个独立问题；
- Open PR 的 Head 持续漂移；
- Survey Base 与 PR Head 已分叉；
- 没有 closing/fix PR，也没有独立参考实现；
- A100 是目标现象的负对照；
- 目标 kernel 只支持 SM90/SM100，而构建机是 SM80；
- 模型权重下界已经超过整机显存；
- 用小模型后不再执行受影响代码。

这种情况记录 `blocked-before-build` 审计，写明证据和未来所需条件，不创建一个看似
可运行、实际改变了问题的 Dockerfile。

## 3. 冻结 Source、Base 与 Oracle

### 3.1 Source

保存 Issue、PR、review、关联 PR、模型卡、环境信息和复现命令的不可变快照或标识。
记录 source cutoff；cutoff 之后出现的信息不能倒灌进 Agent instruction。

### 3.2 Base

Base 是 Agent 开始工作的仓库状态。必须记录：

- 完整 40 字符 commit SHA；
- source archive SHA-256；
- canonical Git tree；
- commit 日期和对应 vLLM version；
- 与 Oracle 的 ancestry/merge-base 关系。

不要仅因为 Survey 写了 `base_sha` 就相信它。必须实际检查 Base 是否为 Oracle 祖先。

### 3.3 Oracle

Oracle 是用于证明任务可解的参考行为，通常从上游已接受的 Golden Solution 提取。
它不是评分时要求 Agent 生成的唯一 patch。

优先级如下：

1. 已合并 commit 的真实 parent → merged commit；
2. 已接受 PR 的稳定 Base → Head；
3. 维护者确认的独立参考实现。

GitHub squash merge 只有一个 parent。权威映射应使用：

```text
squash commit 的 parent -> squash commit
```

不要默认使用贡献分支的旧 Base → Head；分支可能合入 main、发生回退或包含无关改动。
Open PR、分叉分支和仍在变化的 Head 不能作为正式 Oracle。

Oracle patch 应从不可变 commit 生成，并执行：

```bash
git apply --check solution/fix.patch
```

## 4. 确定任务范围

真实 PR 经常包含模型、分布式拓扑、kernel、配置和测试的组合。构造时可以缩小运行
成本，但不能改变被验证的因果关系。

### 4.1 三种可接受的复现深度

1. **完整症状复现**：原始模型、服务、数据和拓扑直接触发问题；
2. **真实生产路径复现**：缩小输入或拓扑，但仍执行受影响的 production class、CUDA
   kernel、collective 或 scheduler branch；
3. **因果 primitive 复现**：只验证已被 Issue/PR 实验证明为根因边界的 primitive，
   同时明确不声称端到端复现。

Constructor mock、source-string 搜索、import smoke 和静态断言只能作为环境探针，不能
单独充当症状复现。

### 4.2 如何缩小模型或数据

- 优先固定一个能够触发相同行为的小样本；
- 数据集过大时，只下载所需样本，不先下载完整数据再裁剪；
- 固定样本内容、revision、digest 和预处理结果；
- 小模型必须通过相同的 Base-fail/Oracle-pass 证明等价性；
- 如果原问题依赖模型结构、head size、量化或 GPU 架构，小模型不能删掉这些条件；
- OOM 可以缩到 causal primitive，但要验证正确性、allocator 增量和同步行为，不能只
  人为制造一个小 OOM。

### 4.3 先给根因还是从症状诊断

任务初版可以是“已知根因后的代码修复题”，instruction 给出受影响功能和行为约束。
如果 Agent 很容易靠局部修改通过，再升级为“从真实症状开始诊断并修复”：

- instruction 只描述用户可观察现象；
- 不暴露目标文件、函数或轻量复现脚本的内部定位；
- verifier 继续检查相同生产行为；
- Agent 可见 Dev 测试与 hidden verifier 使用不同输入。

轻量脚本如果直接展示了引用环、目标函数或精确错误分支，会显著缩小搜索空间。此时它
适合作为 curator/verifier 资产，不一定适合直接交给 Agent。

## 5. 编写 instruction / query

`instruction.md` 应包含：

- 工作目录；
- 可观察症状；
- 期望行为；
- 必须保持的原功能；
- 合法修复边界；
- 必要的运行或资源约束。

不要包含：

- Oracle commit、PR 号或最终 diff；
- 原 Issue 未公开的目标文件和函数；
- hidden verifier 的输入、阈值和断言；
- “修改某一行即可”式修复提示。

合法修复边界用于阻止浅层绕过。例如：

- 修复告警时，合法配置不警告，但真正缺配置仍必须警告；
- 修复内存保留时，不能强制 `gc.collect()`、禁用 prefix cache 或提前删除 payload；
- 修复性能时，不能跳过计算、缓存固定答案或改变数值语义；
- 修复分布式路径时，不能用 CPU object broadcast 替代要求的 GPU/NCCL 路径。

## 6. Docker 环境构造

### 6.1 基础镜像选择

Issue 没写基础镜像时，按 Base 中的 vLLM version 选择时间上最近、ABI 合理的官方
同版本发布镜像，并锁定 linux/amd64 manifest digest。不要使用 `latest`。

选择依据必须记录：

- Base/source 日期；
- 官方镜像 tag 和 manifest digest；
- Python、Torch、CUDA 版本；
- 为什么没有引入修复之后的 Python 实现。

如果 Issue 没有 GPU 要求，不要人为把 query 写成某个平台问题；但镜像仍要记录实际
验证硬件。平台是运行合同，不应凭空成为问题定义。

### 6.2 Source binding 与 Git 防泄漏

镜像必须真正从 Agent 工作树加载 candidate source：

- Python module 位于 `/workspace/repo` 或 `/app`；
- 关键 native extension 也从该 candidate tree 加载；
- 不允许 future site-packages 抢先导入；
- 删除上游 `.git`、remote、tag、reflog 和可恢复 Oracle 的对象；
- 用 exact Base tree 创建一个 synthetic commit；
- 最终仓库只有一个 commit、没有 remote、Agent 非 root 且可写。

推荐构造顺序：

```text
锁定 source archive -> 校验 SHA/tree -> 导出工作树
-> 补充必要 generated/native artifact
-> 创建 synthetic Git -> chown 给 agent
```

### 6.3 不使用 empty target

不要设置 `VLLM_TARGET_DEVICE=empty`。它会跳过真实设备相关依赖和扩展，只适合纯源码
分析，无法支撑后续端到端或 GPU verifier。

至少执行：

- `vllm._C` 或目标 native module import；
- `torch.cuda.is_available()`；
- GPU tensor allocation；
- 与任务相关的 CUDA/Triton/NCCL 生产路径。

### 6.4 Native extension

涉及 C++/CUDA 的任务优先从 exact Base source 编译完整目标 extension。

可以使用 focused build，但必须：

- 只缩小 extension 集合，不缩小该 candidate extension 内部受影响对象；
- 临时修改 `setup.py`/CMake 后恢复 canonical source；
- 构建前后逐文件核对源码未被改写；
- 记录 `cuobjdump`/cubin 架构；
- 保留增量 build tree，支持 Agent 修复后重编；
- Oracle 使用同一工具链和构建流程。

如果只能复用官方 release native artifact：

- 记录 donor tag、digest、Torch/CUDA/Python ABI；
- 使用 multi-stage 白名单复制；
- 为每个 artifact 记录相对路径、类型和 SHA-256；
- 禁止复制 future Python、tests、`.git` 或 staging；
- 计算 donor artifact 与目标改动文件的交集；
- Post-cutoff donor、duplicate registration 或 ABI approximation 必须降低任务声明范围。

### 6.5 网络、代理与凭据

Docker pull 使用 Docker daemon 的网络。Shell 中执行代理脚本，并不自动让 daemon 能
拉取 Docker Hub。必要时由机器管理员为 daemon 配置代理或预先导入锁定 digest 的镜像。

构建依赖下载可以使用 A100 的
`/data/akg_kernel_bench_lite/A100_proxy.sh` 或宿主缓存，但代理属于构建基础设施：

- 不把账号、密码写进 Dockerfile；
- 不把代理凭据写入 build ARG、ENV、image history 或 lock 文档；
- 能直连的 apt/codeload 不强制经过失败的代理；
- 推荐宿主预下载 + SHA-256 校验 + build 时离线消费；
- 最终运行一律 `--network none`。

冷 pull 时间、Dockerfile build 时间、失败重试和缓存重建时间要分别记录。缓存命中不能
冒充冷构建时间。

### 6.6 模型与数据

- 记录模型 revision、文件清单、总大小和 digest；
- runtime 不从 Hugging Face 临时下载；
- 大数据集使用能触发问题的最小固定子集；
- 数据不需要时明确写 `N/A`，不要为了填字段增加无意义下载；
- 模型或数据不能进入 Agent 可修改目录后被当作 verifier 答案读取。

## 7. 测试与评分设计

### 7.1 三层测试

1. **Agent-visible Dev test（可选但推荐）**：离线、一条命令、能指导开发；
2. **Oracle control（必需）**：同一环境应用 Oracle 后正向通过；
3. **Hidden verifier（必需）**：运行时只读注入，覆盖 heldout 和错误修复。

Agent-visible Dev test 不是 Harbor schema 硬要求。如果它会泄漏根因，可只提供症状级
入口或上游测试；不能直接把最终 hidden verifier 暴露给 Agent。

### 7.2 公开复现脚本与本地开发测试

“可执行公开复现”是指能够在 Base 环境中实际触发现象的脚本，不只是描述命令。它可以
来自原 PR，也可以由 curator 构造，但必须核对：

- 原 PR 脚本是否只验证作者实现，而不是验证任务行为；
- 是否依赖不可公开数据、外部服务或特定大集群；
- 是否泄漏修复位置；
- 未修复 Base 是否真的失败；
- Oracle 是否用同一个行为合同通过。

上游提交者提供的测试可以复用，但不能因此省略 heldout。最终 verifier 应加入不同输入、
边界条件和反偷懒对照。

### 7.3 防偷懒与合法边界

不能只在 instruction 中写“不要偷懒”。必须用 tests 固定最终行为。例如：

- 警告题：合法状态无警告 + 真缺配置仍警告；
- cache 题：有 mask、无 mask、部分选择、分配和 eviction 使用同一单位；
- 生命周期题：不调用全局 GC，正常 owner 释放后对象立即不可达；
- 性能题：正确性先过，再计时；
- distributed 题：检查真实 backend、rank 和传输 tensor；
- 新功能题：Base missing/失败 + isolated Oracle 正向通过。

固定 `tests/test.sh` 作为最终启动入口，统一写：

```text
/logs/verifier/reward.txt
```

不要根据 patch 相似度、文件名或字符串出现与否评分。

### 7.4 性能任务

- Base 和 Oracle 在同一 GPU、同一输入相邻运行；
- 固定 seed、shape、dtype、warmup、iterations 和聚合方式；
- correctness 是硬门；
- 使用多组代表 shape，不依赖单个 noisy 点；
- 使用 median 或稳健聚合；
- 阈值来自本机 Base/Oracle margin，不照搬 PR 中另一种 GPU 的绝对数字；
- 如果改进不明显大于噪声，该任务不能使用性能评分。

## 8. Harbor 目录与配置

标准目录：

```text
tasks/<task-id>/
├── task.toml
├── instruction.md
├── environment/
│   ├── Dockerfile
│   └── lock/
├── solution/
│   ├── solve.sh
│   └── fix.patch
├── tests/
│   ├── test.sh
│   └── verify_*.py
└── validation/
    └── docker-build.md
```

`task.toml` 至少声明：

- schema、task name/version；
- source IDs、repository、Base 和 Oracle curator metadata；
- agent/verifier user 与 timeout；
- CPU、内存、存储、GPU 数量和型号；
- runtime `network_mode = "no-network"`；
- grader 类型。

注意：Oracle 标识是否公开取决于 release 策略。正式盲测发布前，可将 Oracle metadata
保留在 curator 私有 manifest，而不随 Agent task 分发。

Docker build context 只使用 `environment/` 内存在的文件。不得依赖 curator 临时目录，
也不得把 `solution/` 或 `tests/` COPY 进 Agent 镜像。

## 9. A100 构建与验收流程

建议使用隔离 Docker daemon，避免不同 curator 污染默认 daemon。每项按以下顺序执行。

### 9.1 Preflight

- 检查 GPU SKU、compute capability、数量和显存；
- 检查目标 kernel dispatch 是否包含 SM80；
- 检查模型权重与 KV/cache 预算；
- 检查 Base/Oracle ancestry；
- 检查所有外部 archive、gitlink/submodule 和 digest。

硬件 gate 应在拉取十几 GB 镜像、下载模型或运行长编译之前完成。

### 9.2 Build

记录完整命令和时间：

```bash
docker build --network host \
  -t ai-infra-bench/<task-id>:base \
  tasks/<task-id>/environment
```

最终还要验证一次从锁定依赖出发的可重复构建。调试 build、缓存 build 和 final build
分别记录，不覆盖失败证据。

### 9.3 Base

在没有 solution 的 Base 镜像运行 verifier：

```bash
docker run --rm --network none \
  --gpus '"device=0"' \
  -v "$PWD/tasks/<task-id>/tests:/tests:ro" \
  -v "$RUN_LOG:/logs/verifier" \
  ai-infra-bench/<task-id>:base \
  bash /tests/test.sh
```

预期 reward 必须为 `0`，而且失败点必须是目标行为，不是 import、缺包、权限或下载失败。

### 9.4 Oracle

在相同镜像、硬件和 verifier 下应用 Oracle：

```bash
bash /solution/solve.sh
bash /tests/test.sh
```

预期 reward 必须为 `1`。Oracle 运行时挂载 solution；solution、Head 文件和 hidden tests
不能固化在 Agent 镜像中。

### 9.5 完整性检查

- Harbor SDK 能加载 `task.toml`；
- `harbor run -p <task> -a nop --print-config` 成功；
- Agent 用户非 root、工作树可写；
- `git apply --check` 在 Agent 用户下通过；
- candidate source/native 路径正确；
- Git 只有 synthetic commit、没有 remote；
- runtime route 为空、代理环境为空；
- image history 没有凭据；
- `tests/` 和 `solution/` 不在 image；
- Base `0`、Oracle `1` 的原始日志保留。

## 10. 证据与文档

每项 `validation/docker-build.md` 至少记录：

- Source、Base、Oracle 和 ancestry；
- 原始 Issue 环境与本地环境差异；
- 基础镜像 tag、digest、Torch/CUDA/Python；
- source/archive/tree/native manifests；
- pull、build、rebuild 时间和 image ID/size；
- Base/Oracle 命令、退出码和关键输出；
- GPU UUID/capability、网络与非 root 检查；
- 数据/模型是否使用以及大小；
- 复现层级：完整症状、生产路径或 causal primitive；
- 未覆盖的模型、拓扑、consumer 和硬件；
- 所有已知 donor、ABI 和 verifier trust 风险。

文档不能把以下概念混在一起：

- image 能 build；
- runtime smoke 通过；
- 原症状已复现；
- Oracle 映射稳定；
- verifier 已完成；
- Task 可正式发布。

## 11. 状态判定

当前构造流程只使用三个面向交付的状态：

### `validated`

Harbor 结构完成，环境可构建，真实目标行为 Base=0/Oracle=1，关键错误解法被拒绝，范围
和残余风险已记录。

### `environment-only`

环境和部分生产路径可运行，但缺少稳定 Oracle、完整 verifier、关键 consumer、exact
native 或足够的原始问题覆盖。可以继续补强，但不能作为正式评分任务。

### `blocked`

硬件、模型容量、问题定义、Base/Oracle 或上游状态使任务无法诚实构造。保留审计证据，
不要制造 substitute smoke。

## 12. 发布前检查清单

### Source 与范围

- [ ] Issue/PR/cutoff 已冻结；
- [ ] Base 是 Oracle 祖先；
- [ ] squash/merge mapping 正确；
- [ ] 行为边界唯一；
- [ ] 缩小范围仍执行同一因果路径；
- [ ] 未覆盖范围已写明。

### Docker 与依赖

- [ ] 官方 base image 使用 digest；
- [ ] exact source SHA/tree 已校验；
- [ ] 不使用 `VLLM_TARGET_DEVICE=empty`；
- [ ] candidate Python/native 从工作树加载；
- [ ] synthetic Git、无 remote、Agent 非 root 可写；
- [ ] native donor 有白名单和 hash；
- [ ] runtime offline；
- [ ] image/history 无代理凭据；
- [ ] 模型/数据 revision 与 digest 已锁。

### Tests 与 Harbor

- [ ] `task.toml` 可被当前 Harbor 解析；
- [ ] Docker build context 自洽；
- [ ] `solution/`、`tests/` 未进入 Agent image；
- [ ] Base reward 为 0；
- [ ] Oracle reward 为 1；
- [ ] no-op 和至少一个常见错误解法失败；
- [ ] verifier 执行真实生产行为；
- [ ] 性能题 correctness 先于 timing；
- [ ] 原始日志和 residual risk 已归档。

只有全部适用项通过后，Task 才能从 `environment-only` 升级为 `validated`。
