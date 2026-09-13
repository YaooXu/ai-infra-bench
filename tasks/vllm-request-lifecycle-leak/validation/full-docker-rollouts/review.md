| 评审项 | 分数 | 关键依据与问题 | 下一步 |
| --- | --- | --- | --- |
| 1. 题目真实吗、清楚吗 | 2 | 以长期服务的请求保留问题提出需求；明确正常结束、取消、流式等待及最终结束、缓存正确性，无私有 helper 要求 | 保留当前题面 |
| 2. 是否独立于原 PR | 2 | 依据生命周期与缓存行为评分；本次 B 保留有效 hash，仅重算失效部分，与 Oracle 全量重算不同 | 不将 Oracle 的组织方式加入题面 |
| 3. 环境能解题吗 | 1 | 最新 Dockerfile 完整构建、离线非 root pytest 可用，真实 Codex B 解出；部分上游测试所需 HF 配置未缓存，ruff 未安装 | 后续可预装常用配置；不宣称完整上游测试可离线运行 |
| 4. 题面与测试双向对齐吗 | 2 | 释放、保留、初始/追加/流式缓存、不同 token/media 的隔离及禁止显式 GC 均对应明确要求；跨完整块边界是缓存正确性的合理推论 | 保留现有行为契约与测试 |
| 5. 测到了真正的执行过程吗 | 2 | 正式核验真实 Request、Scheduler、EncoderCacheManager、KVCacheManager，正常结束走 schedule/update_from_output；父进程独立读 9 次 OS RSS | 本结论限 Python 生命周期，不声称跑过完整 Qwen GPU 服务 |
| 6. 不同正确实现能通过吗 | 2 | Oracle、替代生命周期实现、重命名 helper 对照通过；B 的部分 hash 保留策略也通过正式评分及独立多轮检查 | 保留语义替代实现与独立检查 |
| 7. 错误实现能被准确拒绝吗 | 2 | 11 项对照均符合预期；未完整修复、错误 hash、强制 GC、只处理外部结束、提前退出及伪造报告均被拒；A 真实失败重现 | 保留反例及具体失败原因 |
| 8. Oracle 本身可靠吗 | 2 | 新镜像正式 Harbor Oracle 为 1；独立三轮缓存与取消检查通过；Oracle 与候选接受相同正式评分 | 后续有范围内新反例时继续挑战 |
| 9. 评分结果可信吗 | 1 | root 父进程独立采样 RSS，已拒绝已知报告/退出绕过；Python 行为观察器仍与候选同进程 | 不声称能防御同时模拟内存活动和报告的任意恶意代码 |
| 10. 验收能复现、交付说清楚了吗 | 1 | 两份完整最终状态、原始评分、重放、输入哈希和镜像 ID 完整；本地完整构建通过，尚未验证远端镜像发布/拉取 | 合并/发布流程应另核对远端可取得性 |

评分进度：10/10，17/20。当前已验证范围内未发现阻断性的 task/verifier 缺陷；上述环境便利性、安全边界和远端发布缺口仍保留，不能用总分抵消。结论是可以保留该任务及本轮原始评分，不是证明 task 在所有实现和环境下完全无问题。

本轮按用户要求使用完整最新 Dockerfile，并行执行两次真实 Codex。结果一份 0、一份 1，均无 Harbor 错误；各自完整最终文件在新镜像重放后仍为 0、1。11 项对照全部符合预期，正式 Harbor Oracle 为 1。没有修改候选后覆盖原始得分，也没有额外抽样到成功为止。

验证模式为 Oracle-based。任务 v1.3.1，源提交 `232d709046f473e515db88ee73f4cc93dece0183`；执行前冻结到 `/data/pr55-final-two-rollouts/task`。仅 task.toml 的运行镜像 tag/ID 替换为本次完整构建结果，Dockerfile、题面、verifier、Oracle 与源提交相同。执行输入哈希见 pre-run.json；事后全量校验见 final-integrity.json。报告写回时同步运行镜像信息，README 与镜像 manifest 仅更新说明，不改变已执行的行为代码。

镜像：`ai-infra-bench/vllm-request-lifecycle-leak:base-e94ec597334d-pytest-full`，ID `sha256:7f6d77e4040061600c355d38a9c16f775405dc721473f1a2c39594396009cd51`。本次从完整 environment/Dockerfile 构建，源码从 GitHub 获取准确 Base `e94ec597334d9a3e9b0d04bc17152e2747c83d51`，tree 为 `bfdf97989c2997f550a44ebc42ad8aa5582d67a7`。非 root 离线 smoke 实际通过 1 个 Request 测试、收集 87 个 scheduler 测试；收集不等于执行通过。测试依赖预装 pytest 8.4.1、pytest-asyncio 1.1.0、tblib 3.1.0。没有 GPU 或模型下载需求来验证该 Python 所有权边界。

两次均使用 Codex CLI 0.153.4，gpt-6-astra，medium，Harbor 0.22.0。后端不可变模型 revision 未记录。每次 4 CPU、16 GiB RAM，agent 用户非 root；两个 trial 并发，整组约 7 分 29 秒。Agent Step 只计 ATIF source=agent，工具数计顶层编排调用，不把一次编排内的多个命令当成多个模型 turn。

| Attempt | 原始 reward / 重放 | 总耗时 / agent / verifier | Agent Step / 工具 | 最终改动 | 原因与独立结论 |
| --- | --- | --- | --- | --- | --- |
| A：task__WScpthh | 0 / 0 | 386.42 / 330.00 / 31.10 秒 | 24 / 23 | 4 tracked，+174/-12；生产 2、测试 2；untracked 0 | 移除自引用但未修流式完整块 hash 失效；独立检查应复用 12、实际 8 |
| B：task__iK5A7YN | 1 / 1 | 447.99 / 395.30 / 31.50 秒 | 27 / 26 | 5 tracked，+184/-19；生产 3、测试 2；untracked 0 | 移除自引用，丢弃失效 hash，处理多媒体追加；独立三轮复用 12/16/20 且取消释放通过 |

两个 trial 的 Harbor exception 均为空。正式评分不是 pytest 计数协议，未记录单项 passed/total/errors/skips，不能编造这些数值：A 跑过前置生命周期和 initial/append 检查，stream 同前缀复用断言失败，后续 RSS 阶段未到达；B 完成全部行为检查及 9 个 OS 采样点。A 不能因前置弱引用检查通过就被记为整个内存批次验证通过。

已按 rollout-review 审查两份轨迹的记录输入、工具调用/返回、最终 patch 和最终文件。原工具的长输出部分已有截断，不能宣称逐字看到了未记录的输出。审查材料中未观察到读取答案或绕过评分的行为。完整 repo archive 包含 tracked、untracked、ignored 文件，SHA256SUMS 验证通过，正式传递的 vllm 文件与归档逐文件一致；无 untracked 文件不等于没有 ignored 文件。重放直接还原完整 archive，而非只重建 patch。轨迹中的 /tmp 测试日志不在 repo 归档范围内，其已返回的摘要仍保留在轨迹中。

A 发现并修掉 `partial(block_hasher, self)` 的循环引用，增加 GC 关闭条件下的释放测试，最终本地 17 个 Request/streaming 与 48 个 prefix-cache 测试通过。它的流式测试使用较大 block，删除的 token 没让已有完整块失效，因此未抓住边界。正式 scorer 的 `stream: same prefix not reused` 与重放一致。独立探针通过真实 add/schedule/update 生成两步输出，继续输入时丢弃最后 EOS，期望整个保留前缀可复用；第一轮 `(expected=12, actual=8)` 确认是候选缺陷，不是报告格式或环境错误。

B 先修循环引用，再用完整重算结果比较增量 hash。其最初测试错误地替换内部 token 列表，破坏只读视图，随后改为原位更新；fixture 缺少 is_encoder_decoder=False 也只在测试侧修正。发现流式问题后，它删除被丢弃 token 涉及的 hash 后缀，保留其他有效 hash；同时避免追加多媒体输入时只检查最后一个输入。最终 focused 命令为 `pytest -q tests/v1/test_request.py tests/v1/streaming_input/test_scheduler_streaming.py tests/v1/core/test_kv_cache_utils.py -k 'request or hash' --tb=short`，38 passed、24 deselected。较早完整三个文件的尝试有 8 个 HF 配置缺失失败，不能宣称整个 suite 通过。它自己的部分生命周期单测直接调用私有方法、手工设状态，证据较弱；正式 verifier 的真实生命周期与独立重放补充了这一点。B 与 Oracle 不同：Oracle 流式更新全量重算 hash，B 只重算受影响部分；不能把 B 对增量多媒体扫描的改动误报为 Oracle 同样存在的 bug。

独立探针在两次模型运行前已冻结，沿用上一轮已存在场景，非新盲测集。它复用 verifier 的对象构造 helper，但自行定义三轮事件序列及期望 token/缓存长度；不依赖 Oracle 私有表示。Oracle 与语义替代解在同一新镜像也通过该探针。A 失败、B 成功与正式评分一致；此次没有修改 scorer，也未将诊断探针临时升格为新的评分门槛。小样本 1/2 不能估计模型总体通过率。

防伪对照的成功结论限定在实际复现的退出/报告绕过。外部 RSS 由父进程从 OS 读取，不能仅靠 stdout 打印完成，但仍不是任意恶意 Python 的安全沙箱。题面从 Qwen2.5-VL 服务症状切入；这里验证的是决定请求与多模态 payload 保留的真实 Python 生命周期，不是完整模型推理、HTTP 服务或长时间线上稳定性。

原始证据目录为 `/data/pr55-final-two-rollouts`：build.log、environment-smoke.log、setup-identity.json、pre-run.json、job-two.json、jobs/codex-two、jobs/oracle、control-matrix、replays、independent-*.log、两份 ATIF readable 与原生 codex.txt。本目录提交精简结果、正式输出、独立日志和最终 patch；大型完整归档与轨迹留在上述原始证据目录，未复制到 agent 镜像。

基础设施记录也保留：启动前两个 capture/network smoke 因宿主 Docker Compose 不可用及修复脚本的权限步骤失败，均未调用模型；安装校验过的 Compose v2.39.4 到本轮独立 DOCKER_CONFIG 后，第三次 smoke 通过，随后才启动两次付费运行。宿主适配器只配置模型网关与保存最终状态，正式评分逻辑未改；模型网关只允许 OpenAI 相关端点，HF/pip 请求被拒符合离线策略。环境及工具限制没有隐藏，也未计入模型的两次正式尝试。

复现完整构建：在任务 worktree 执行 `docker build --network host --build-arg HTTPS_PROXY=http://127.0.0.1:7890 --build-arg HTTP_PROXY=http://127.0.0.1:7890 -t ai-infra-bench/vllm-request-lifecycle-leak:base-e94ec597334d-pytest-full tasks/vllm-request-lifecycle-leak/environment`。代理是本机联网配置，其他机器应使用自己的网络。矩阵入口为 `python3 tasks/vllm-request-lifecycle-leak/validation/run-local-matrix.py --help` 所列参数；本次实际配置与输出保存在原始证据中。完整状态重放为 `python3 /data/pr55-final-two-rollouts/replay.py task__WScpthh` 或 `task__iK5A7YN`，依赖保留的归档与本地镜像。正式 Harbor Oracle 与两次 Codex 的有效配置保存在 jobs 下。Harbor 本地适配需要本轮 Compose 配置与受限模型网关，不是 task 所需的额外答案或业务依赖。
