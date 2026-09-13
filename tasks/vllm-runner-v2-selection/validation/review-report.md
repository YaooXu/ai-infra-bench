# PR #56：pooling 兼容性补测与修复

题目保留，1.1.2 的 pooling 漏测已修复，完整控制矩阵和最终 Harbor Oracle 均通过预期验证。10/10 维度可评分，19/20；没有已知未修复的本轮行为阻塞，registry 发布仍待处理。以下结论仅覆盖代表性配置和已验证的评分边界，不声称穷尽所有 V2 特性组合。

父提交 a688bee189d2b7935d39535f6db6c635def5eb4b；工作区 `/tmp/ai-infra-pr56-hardening`，分支 `codex/runner-selection-permissions`。题面与环境不变，task 升为 1.1.2，继续使用 1.1.1 的同一镜像 digest。Git 身份为 `yaoxu <csxuyao@qq.com>`。此前环境报告保存在 history/pre-pooling-a688bee，原始 agent 分数保持不变。

| # | 维度 | 分数 / 状态 | 依据或边界 | 下一步 |
|---|---|---|---|---|
| 1 | 真实清楚 | 2 | 三段专业开发者 query 不变 | 保持简洁 |
| 2 | 独立于历史 PR | 2 | 根据 Base 行为纠正 Oracle | 不豁免参考解 |
| 3 | 环境可解 | 2 | 已验证镜像不变；三份真实 rollout 使用离线 pytest/hooks | 无需重建 |
| 4 | 双向对齐 | 2 | 27 项映射到既有选择和兼容性要求 | 保留代表性范围说明 |
| 5 | 真实执行路径 | 2 | 配置、CUDA/NCCL、Worker 和 runner 构造 | 不宣称权重加载或生成 |
| 6 | 正确替代实现 | 2 | 六个不同写法的正例及 Codex r2 通过 | 保留实现自由 |
| 7 | 错误解拒绝 | 2 | Base、旧 Oracle、r1/r3 和所有负例按预期失败 | 保留失败原因 |
| 8 | Oracle 独立验证 | 2 | 从源码推导的 pooling 挑战及最终 27 项验证 | 按同一契约验证 |
| 9 | 评分可信 | 2 | 伪造报告/认证、提前退出对照仍拒绝；Harbor 两类退出也验证 | 非任意 native 攻击沙箱 |
| 10 | 复现与交付 | 1 | 输入哈希、日志、最终 Harbor 和补丁等价性证据齐全 | registry 发布待完成 |

Gate 1、2 沿用未改变的契约和环境证据；Gate 3 本轮复查并验证。目标边界是配置和 override → 真实 ModelConfig/VllmConfig → Worker.init_device → 实际 runner 或启动诊断。只有无关的 ElasticEP orchestration 被替代，不加载权重，不执行完整 embedding 或文本生成。

原 verifier 对显式 V2 的 pooling 支持范围缺少检查：旧 Oracle 和 Codex r1/r3 会让 MEAN、CLS、关闭归一化的配置启动 V2。Base 的 V2 PoolingRunner 只返回归一化 LAST embeddings，题面已有“显式选择不支持配置应报错”的要求，因此新增六个场景没有扩大功能范围。加入三项负例，以及 normalized LAST→V2、自动 MEAN→V1、强制 V1 MEAN→V1 三项对照。测试只检查实际启动行为，不依赖私有 helper、异常类型或文案。

修复 Oracle 和六个正例的 pooling 判断，旧 Oracle 保留为 missing-pooling-check.patch。所有正例完整完成 27 项；负例可能在失败组后停止，未执行的项目不记为通过。详细失败条件和完成记录见 pooling-evidence.json。

| 本地重放 | 预期 | 实际 | PASS 项数 | 秒 |
|---|---|---|---|---|
| oracle | 1 | 1 | 27 | 152.3 |
| renamed-worker-field | 1 | 1 | 27 | 157.22 |
| system-exit-zero-bypass | 0 | 0 | 0 | 6.68 |
| missing-pooling-check | 0 | 0 | 24 | 151.62 |
| base | 0 | 0 | 10 | 94.96 |
| delayed-startup-validation | 1 | 1 | 27 | 143.5 |
| os-exit-zero-bypass | 0 | 0 | 0 | 6.98 |
| r1 | 0 | 0 | 24 | 152.27 |
| alternative-agent-implementation | 1 | 1 | 27 | 150.97 |
| wrong-runner-construction | 0 | 0 | 10 | 94.47 |
| forged-observations | 0 | 0 | 0 | 9.08 |
| r2 | 1 | 1 | 27 | 144.73 |
| boolean-accessor-alternative | 1 | 1 | 27 | 152.65 |
| missing-raw-logits-check | 0 | 0 | 10 | 91.45 |
| forged-checkpoints | 0 | 0 | 0 | 7.87 |
| r3 | 0 | 0 | 24 | 147.65 |
| cached-startup-env | 1 | 1 | 27 | 150.83 |
| missing-prompt-embeds-check | 0 | 0 | 11 | 94.77 |
| unfixed-boolean-alternative | 0 | 0 | 9 | 97.76 |
| clear-error-wording | 1 | 1 | 27 | 152.29 |
| incomplete-agent-implementation | 0 | 0 | 10 | 95.01 |
| forged-callback | 0 | 0 | 0 | 10.24 |

旧 forged-observations 反例仍读取过时的 stdin 格式，首次因 JSONDecodeError 失败，不能作为防伪造证据。本轮修正它，使其打印当前格式的完整成功报告并 os._exit(0)。重新经本地 test.sh 和正式 Harbor 验证，评分父进程均因零条认证完成记录拒绝，reward=0；不再依赖格式错误。该控制的更新不改变题面、评分器或 Oracle，已有 Oracle 验证仍对应同一执行内容。

本地矩阵使用冻结快照，最后仅清理了八个 patch 的空白上下文格式。对每个格式化后的 patch 都在干净 Base 上重新应用，并证明生成的完整 git diff SHA-256 与已运行版本相同；证明见 pooling-evidence.json。最终 Harbor Oracle 另以格式化后的实际 task 重跑成功，不用格式等价性替代正式入口验证。

最终 Harbor 使用单卡 GPU adapter，只指定设备及共享内存，不改评分逻辑。最终 Oracle reward=1、无 trial error、27 项全部完成；另一次 Oracle 与 SystemExit(0)、os._exit(0) 对照的正式 Harbor 结果也符合预期。输入 checksum 在最终证据里；更新证据和报告会改变 task 目录 checksum，不改变执行过的题面、测试或解法。

本轮没有新调用 Codex/DS，只重放已有最终解法。Codex r2 在当前完整 suite 通过，r1/r3 因 pooling 缺陷失败；这不重写它们历史 21/21 的分数，也不应当作为新的独立模型通过率。环境的所有依赖或平台组合不在此次穷举范围，镜像 registry 发布尚未完成。Git commit/push 状态以交付消息及远端 HEAD 为准。
