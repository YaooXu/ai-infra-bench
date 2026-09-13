# PR #56 离线开发环境修复

题目保留，本轮离线开发环境修复验证通过。新镜像的 pytest、适用 pre-commit hooks、选定评分回归以及正式 Harbor Oracle/提前退出对照均符合预期。10/10 个维度可评分，20/20；结论限定于下述代表性场景和既有评分安全边界，镜像尚未发布到 registry。

本轮以 ff71f0d1547b825f3d115835e78a347cf3e60fc8 为父提交，任务版本更新为 1.1.1。工作目录 `/tmp/ai-infra-pr56-hardening`，分支 `codex/runner-selection-permissions`。Git 本地身份为 `yaoxu <csxuyao@qq.com>`。原始题面、Oracle、verifier 和控制补丁均未改变；精确最终文件哈希见 e2e-evidence.json。此前全矩阵证据移至 history/ff71f0d，不能把历史镜像结果写成本轮重跑结果。

| # | 检查维度 | 得分 / 状态 | 关键依据或边界 | 下一步 |
| --- | --- | --- | --- | --- |
| 1 | 题目真实吗、清楚吗？ | 2 | 三段开发者请求未变，无工作目录提示或手动断行 | 保持当前契约 |
| 2 | 是否独立于原 PR？ | 2 | 仍按行为定义自动选择、覆盖和启动 | 不要求复刻 Oracle |
| 3 | 环境能解题吗？ | 2 | 新镜像 agent 禁网下 20 个上游测试、14 个适用 hooks 通过 | 维护工具锁 |
| 4 | 题面与测试双向对齐吗？ | 2 | 题面及 21 场景 verifier 未变，沿用 semantic-boundary.md 映射 | 不把工具 smoke 加进评分 |
| 5 | 测到了真正的执行过程吗？ | 2 | 新镜像仍执行真实配置、init_device 和 runner 构造 | 不宣称完整生成测试 |
| 6 | 不同正确实现能通过吗？ | 2 | 当前镜像的独立替代实现获 1，全部 21 场景完成 | 保留正例 |
| 7 | 错误实现能被准确拒绝吗？ | 2 | Base、缺失 prompt-embeds 判断、伪造 checkpoint、两类提前退出均拒绝 | 保留实际失败原因 |
| 8 | Oracle 本身可靠吗？ | 2 | 当前 Harbor Oracle 完成 21 场景获 1；既有独立挑战见历史证据 | 本轮未重复全部历史挑战，运行时版本保持一致 |
| 9 | 评分结果可信吗？ | 2 | 当前 Harbor 两类提前退出均 0、无 trial error，伪造认证数据也失败 | 不扩展为任意原生攻击隔离声明 |
| 10 | 验收能复现、交付说清楚了吗？ | 2 | Docker、工具锁、禁网 smoke、控制和 Harbor 日志/哈希已保存 | Git 推送后由发布流程处理 registry |

修复针对实际 rollout 暴露的问题：按上游 AGENTS 创建的隔离 venv 没有 pytest 和运行时包，pre-commit 即便安装也会首次联网下载 hooks。Dockerfile 现在创建 agent 可写、继承固定运行时的 `.venv`，安装带哈希的测试工具，并预装 Base 配置的所有 hook 环境。相关 Python 工具依赖、hook Git revisions、Node/Go archives 和 npm shrinkwrap 都有版本锁；不修改上游 `.pre-commit-config.yaml`。Go 的正常命令路径也已提供，避免 login shell 改变 PATH 后导致 pre-commit 换用未缓存的环境。ShellCheck 预装，避免本地 shell hook 临时下载。

工具属于 cutoff 豁免的基础设施。运行时仍来自同一 digest-pinned vLLM 镜像，Base、九个原生扩展及两个生成文件的绑定不变。禁网 smoke 比对 venv 与系统同名包的版本，确认没有升级运行时依赖，且 vLLM 和原生扩展都从可编辑工作区导入。新镜像 ID 为 `sha256:772e7f217078cee581097b23a4ea0458727f49a5a91afad996fa666a7c8acd2f`，canonical local tag 为 `ai-infra-bench/vllm-runner-v2-selection:base-c7560af42487-v1.1.1`。

禁网 smoke 以 agent 用户运行，不挂载任务隐藏 tests 或 solution：`pytest tests/config/test_config_utils.py -q` 通过 20 项，耗时 28.46 秒；pre-commit 通过 14 个适用 hook，11 个不匹配文件类型的 hook 正常跳过。另用通用未定义名称样例确认 Ruff 实际返回 F821 和退出码 1。完整脚本和日志在 evidence/offline-tools-20260913.tar.gz；执行器 exit 0 和最后的 smoke marker 均核对。

本轮评分对照使用单张 A100/容器、4 CPU、16 GB，离线执行原始 `bash /tests/test.sh`。两个本地控制容器并发使用 GPU 4，Harbor 使用 GPU 3；单次 task 的资源需求仍为一张 GPU。以下耗时包含本地容器设置和验证，不能当作隔离硬件的性能基准。

| 本地控制 | 预期 reward | 实际 reward | 秒 |
| --- | --- | --- | --- |
| base | 0 | 0 | 95.45 |
| alternative | 1 | 1 | 143.99 |
| forged-checkpoints | 0 | 0 | 8.54 |
| missing-prompt-embeds | 0 | 0 | 91.79 |

Base 因自动 Qwen3 和重复自动启动仍选 V1 被拒，达到真实目标路径，没有用 import 失败代替行为失败。缺失 prompt-embeds 判断的控制因错误选择 V2 被拒；伪造 checkpoint 因认证失败且未完成所需检查被拒。正例完成 21 个场景；失败组之后未执行的场景不记为通过。

正式 Harbor 使用 0.22.0，以及仅处理 GPU 分配和共享内存的本地 Docker adapter；没有改评分逻辑，没有调用付费模型。Oracle 直接使用原 solution；提前退出试验在隔离 task 副本中只用对应控制补丁替换 solution/oracle.patch，让真实 artifact transfer 和 separate verifier 路径处理控制。每次 trial 的冻结输入哈希及输入 checksum 保留，执行后核对未变化。

| Harbor trial | reward | trial error | 输入 checksum |
| --- | --- | --- | --- |
| oracle | 1.0 | 无 | `d23760038e8cc806c8a78eededee3d9949e6436494f697fa8eeef8ca6c9fdbf0` |
| os-exit | 0.0 | 无 | `99e52500b6100afad1743c0c260f383209ffae0fea952ef44a6995c131df6215` |
| system-exit | 0.0 | 无 | `2e9b25fee8deb429a8523923071c79367eb3b04047a8be38d3f25c5a514056a2` |

以上 task checksum 对应执行前快照。最后只更新报告、镜像验证状态和证据索引，因此最终目录 checksum 会不同；评分代码、控制、题面和 Docker 构建输入未在执行后修改。没有重新运行此前的付费 Codex 作答，其原始零分和轨迹保持不变。

限制：离线支持正常 Python 开发和已缓存的 Base hooks，不代表整个上游测试集的所有模型/数据都已下载。修改依赖或重新生成 requirements 可以仍然需要外部索引。评分继续覆盖 runner 选择与启动，不执行权重加载和完整 token 生成；认证机制不是任意 Python/native 篡改的完整沙箱。完整 18 组历史矩阵及独立挑战仍可追溯，本轮只重跑与环境变更相关的代表控制。镜像未推送 registry，Git 提交与推送状态以交付消息为准。
