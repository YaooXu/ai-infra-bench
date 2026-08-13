# RQ1：vLLM 的真实维护 workload 到底是什么？

数据截止：**2026-07-31 23:59:59 UTC**

## 结论摘要

vLLM 面临的核心问题不是 issue 数量失控，而是 **PR 输入速度与可见 review/merge capacity 之间的缺口持续扩大**。

2026 年 1–7 月相对 2025 年月均：

| 月均指标 | 2025 | 2026 年 1–7 月 | 变化 |
|---|---:|---:|---:|
| 新增 issue | 578.2 | 609.9 | +5.5% |
| 新增 PR | 1,068.6 | 2,102.6 | **+96.8%** |
| merged PR | 720.1 | 974.4 | +35.3% |
| 活跃 May-18 roster reviewer | 54.3 | 58.0 | +6.9% |
| reviewer-days | 560.6 | 697.6 | +24.4% |
| submitted review | 2,316.7 | 2,727.3 | +17.7% |
| inline review comment | 1,949.3 | 2,003.4 | +2.8% |
| 每个新增 PR 的 submitted review | 2.17 | 1.35 | **−37.7%** |
| 每个新增 PR 的 inline comment | 1.83 | 1.01 | **−44.8%** |

新增 PR 几乎翻倍，但 merge、reviewer 数、review submissions 和 reviewer-days 都没有同比例增长。到 7 月，单月新增 2,722 个 PR、merge 1,134 个，只有 52 名 roster reviewer 有 review 活动；每名活跃 reviewer 对应 52.3 个新增 PR。

这个缺口已经转化为队列：

- open PR 从 2025 年底的 1,320 个增至 2026-07-31 的 **4,194 个**，增长 217.7%；
- open issue 同期从 1,791 个增至 **2,055 个**，增长 14.7%；
- 4,194 个 open PR 中，3,405 个（81.2%）没有 submitted roster review，3,013 个（71.8%）仍有 outstanding review request；
- 2,055 个 open issue 中，1,570 个（76.4%）没有可观察到的 roster response，只有 165 个（8.0%）有当前 assignee。

因此，RQ1 对 benchmark 的核心约束是：任务不能只代表 merged feature，也必须代表 bug diagnosis、异构硬件、review-heavy integration、open/closed-unmerged work、缺少 verifier 的任务和依赖 specialist judgement 的工作。

![Activity and backlog](assets/rq1/activity_and_backlog.png)

## 数据与估计对象

本研究使用 Simon Mo 提供的 [*vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran)*](https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1) 作为基础，并将 GitHub 数据补充到 2026-07-31。合并数据库和校验信息发布在 [`vllm-github-data-2026-07-31`](https://github.com/ai-infra-bench/ai-infra-bench/releases/tag/vllm-github-data-2026-07-31)。

分析总体包括：

- 49,925 个 issue/PR artifact；
- 16,990 个 issue；
- 32,935 个 PR；
- 205,998 条 issue/PR conversation comment；
- 131,473 个 submitted review；
- 122,491 条 inline review comment；
- 77,682 个 PR–commit association；
- 19,416 个 default-branch commit。

观察窗口为：

1. 项目启动至 2024 年底；
2. 2025 全年；
3. 2026 年 1–7 月七个完整月份。

所有当前状态和队列指标定义在 2026-07-31。7/14/30/90/180 天 outcome 只使用在 cutoff 前已经获得相应观察时间的 cohort。PR 从首次 ready-for-review 开始计时；从未 ready 的 draft 不进入 response-time risk set。Response 指 artifact author 之外的可观察响应。

`repo_collaborator` 仍是 2026-05-18 的权限快照。因此报告使用 **May-18 snapshot roster**，不把它称为 7 月 roster，也不把当前权限倒推为历史 maintainer 身份。Any-human response 是主要 estimand；roster response 是维护者 capacity 的 sensitivity definition。

## 数据质量与可复现性

合并数据库的 19 个 release validation 全部通过。分析器还执行以下检查：

- canonical artifact、PR、comment、review、inline comment、commit 和 file 数量逐表对账；
- GitHub PR database ID 显式映射到对应 issue database ID；
- delta-refreshed artifact 使用 canonical timeline，其他历史 artifact 使用原 Fivetran history，避免重复计算；
- 所有分析事件均不晚于 cutoff；
- artifact、PR、comment、review 和 inline comment ID 无重复；
- 生成结果重复运行得到完全相同的 `summary.json`、19 张图和 57 张聚合 CSV 表。

保留而未静默删除的源数据异常包括：

- 443 个 closed artifact 没有 close-history row，月末状态使用 `closed_at` 回退；
- current state 与最后一个可观察 close/reopen event 有 2 个不一致，cutoff materialized state 优先；
- 2 个 submitted review 缺少 event time；
- 21 条 inline comment 无法映射到 canonical PR，不进入 PR-level 分析；
- 88 个 default-branch commit 没有关联 PR。

GitHub 的通用 artifact state 还会把 279 个已有 merge event 的 PR 表示为 `CLOSED`；反过来，有 168 个 PR 只有 cutoff materialized merge time、没有保留下来的 merge event。分析以两种 cutoff-consistent signal 的并集定义 merge，并将这 168 个缺失的 merge actor 保持为 unknown，不作推断。

1,799 个 artifact 的文本在 cutoff 后又被观察到更新，1,227 个 open PR 的 file list 不能证明在 cutoff 后没有继续变化。排除这些记录后：issue intent 最大变化 0.75 个百分点、PR work type 0.28、hardware 0.28、topic 1.00。分类结论对这项限制稳健。

## Demand、throughput 与 backlog

PR demand 的增长远大于 issue demand。2026 年 1–7 月月均 2,102.6 个 PR，比 2025 年高 96.8%；月均 merge 只高 35.3%。这不是一个短期尖峰：3 月新增 2,342 个，6 月 2,545 个，7 月 2,722 个。

PR backlog 的月末轨迹为：

| 月末 | Open PR | Open issue |
|---|---:|---:|
| 2025-12 | 1,320 | 1,791 |
| 2026-03 | 2,230 | 1,773 |
| 2026-04 | 2,779 | 1,937 |
| 2026-05 | 3,125 | 1,993 |
| 2026-06 | 3,472 | 1,970 |
| 2026-07 | **4,194** | **2,055** |

Issue backlog 相对稳定但没有消失；PR backlog 则在七个月内增加 2,874 个。2026 cohort 共 14,718 个 PR，其中截止时 6,619 个 merged、4,048 个 closed-unmerged、4,051 个仍 open。按 competing-risk 估计，30 天内 48.3% merge、18.1% closed-unmerged；90 天内 51.8% merge、23.8% closed-unmerged。

## Cutoff 时维护者实际看到的队列

### Open issue

2,055 个 open issue 中：

- 1,136 个（55.3%）是 bug/correctness；
- 682 个（33.2%）没有任何 non-author human response；
- 1,570 个（76.4%）没有 May-18 roster response；
- 798 个（38.8%）超过 90 天；
- 519 个（25.3%）同时超过 90 天且没有 roster response；
- 165 个（8.0%）有当前 assignee。

### Open PR

4,194 个 open PR 中：

- 1,528 个（36.4%）是 bug/correctness；
- 757 个（18.0%）仍是 draft；
- 3,139 个（74.8%）没有 roster response；
- 3,405 个（81.2%）没有 submitted roster review；
- 3,013 个（71.8%）仍有 outstanding review request；
- 1,783 个（42.5%）带 rebase/conflict signal；
- 245 个（5.8%）带 stale signal。

这些数不代表每个 open item 都应该被处理，也不把正确关闭或拒绝的 PR 视为浪费；它们描述的是 maintainer 必须 triage、review、redirect、close 或合并的公开工作面。

![Current queues](assets/rq1/current_queues.png)

## Responsiveness 与 review latency

固定 7 天 response rate：

| Artifact / responder | Launch–2024 | 2025 | 2026 Jan–Jul |
|---|---:|---:|---:|
| Issue：any non-author human | 65.0% | 59.1% | **53.3%** |
| Issue：May-18 roster | 46.1% | 40.0% | **23.0%** |
| PR：any non-author human | 82.8% | 82.6% | **63.5%** |
| PR：May-18 roster | 79.9% | 80.1% | **57.6%** |
| PR：submitted roster review | 72.1% | 73.8% | **50.8%** |

2026 cohort 的 30 天率为：issue any-human 61.0%、issue roster 27.5%、PR any-human 72.0%、PR roster 66.6%、正式 submitted roster review 59.4%。增加等待时间能够恢复一部分 response，但没有消除早期 review gap。

作者角色对 PR response 影响很大。在具有 7 天观察期的 2026 PR 中：

- external-human PR 的 any-human / roster response 为 57.8% / 50.8%；
- roster-authored PR 为 81.7% / 79.2%。

因此，所有 PR 混合的 response rate 会掩盖外部贡献者经历。Benchmark 必须把 author role 当作 sampling 和 reporting stratum。

![Response within seven days](assets/rq1/response_within_7_days.png)

## Issue workload 与处置结果

2026 年 1–7 月的 4,269 个 issue 中：

| Issue intent | 数量 | 占比 |
|---|---:|---:|
| Bug/correctness | 2,475 | 58.0% |
| Feature/model/backend request | 534 | 12.5% |
| Other/tracking | 302 | 7.1% |
| CI/infrastructure | 291 | 6.8% |
| Design/RFC | 282 | 6.6% |
| Usage/configuration | 141 | 3.3% |
| Performance | 129 | 3.0% |

Bug/correctness 占比从 2025 年的 55.2% 升至 58.0%。Usage/configuration 从 13.0% 降至 3.3%，而 CI/infrastructure 和 Design/RFC 明显上升；这可能同时反映模板、标签和社区使用方式变化，不应解释成底层需求的纯因果变化。

在具有 90 天观察期的 2026 issue 中，当前标为 completed 且在 90 天内关闭的比例差异很大：CI/infrastructure 86.0%、installation/build 52.2%、bug 45.0%、feature/model/backend 28.9%、performance 22.4%、design/RFC 20.6%、usage/configuration 20.8%。这些是 disposition 信号，不是“问题被工程性解决”的无偏估计；2026 close event 中 47.2% 由 bot 执行。

## PR 类型：社区到底每天在做什么

2026 年 1–7 月的 14,718 个 PR：

| PR work type | 数量 | 占比 | 2025 占比 |
|---|---:|---:|---:|
| Bug/correctness | 4,982 | **33.8%** | 22.8% |
| Other/unclear | 2,194 | 14.9% | 16.7% |
| CI/build/release | 2,100 | 14.3% | 16.6% |
| Documentation/API/UX | 1,984 | 13.5% | 20.1% |
| Feature/capability | 1,917 | 13.0% | 13.7% |
| Performance/efficiency | 895 | 6.1% | 5.5% |
| Refactor/maintainability | 486 | 3.3% | 3.6% |
| Test/evaluation | 125 | 0.8% | 0.5% |

最明确的结构变化是 bug/correctness 占比上升 11 个百分点。60.1% 的 2026 PR 分类来自明确 title tag/current label，25.0% 来自 deterministic lexical heuristic，14.9% 仍 unresolved。因此这些类型适合做 source-frame strata，但 benchmark 最终题目仍需人工编码。

不同类型的 90-day outcome 和 review 不同：bug PR 90 天内 merge 48.1%，CI/build 67.0%，feature 49.2%，performance 52.6%，refactor 74.7%。不能把所有 PR 用一个 success definition 排名。

![Workload mix](assets/rq1/workload_mix.png)

## 谁在实现，谁在 review 和 merge

2026 年 1–7 月 14,643 个 human-authored PR 中：

- external human：10,993 个（75.1%），3,401 名作者；
- May-18 snapshot write+：2,896 个（19.8%），56 名作者；
- May-18 snapshot triage-only：754 个（5.1%），19 名作者。

外部社区提供了绝大多数实现 intake，但集成结果差异很大。在获得 90 天观察期的 PR 中，external-human PR 42.5% 在 90 天内 merge，write+ author 为 83.8%，triage-only 为 80.8%。External PR 有 46.9% 获得至少一次 roster review，write+ PR 为 76.8%。这不能直接解释为 patch quality：任务选择、专业领域、作者历史、reviewer familiarity 和组织优先级都混杂在 author role 中。

Merge gatekeeping 高度集中于有写权限的人：6,445 个 actor 可观测的 2026 human-authored merge 中，96.1% 由 May-18 write+ actor 执行，3.8% 由 triage-only actor 执行，0.1% 由其他 actor 执行；另有 168 个 merge actor 缺失且未被插补。Top-five merge actor 的占比依 work type 从 bug 的 43.8% 到 refactor 的 60.2%。

当前 103 人 roster 在 2026 可观察行为中：

- 72 人同时有 engineering 和 gatekeeping 行为；
- 3 人只有 PR authorship；
- 8 人只有 gatekeeping；
- 20 人没有这里定义的公开行为。

最后一组不等于“没有工作”：private discussion、security、Slack、release、CI babysitting、vendor coordination 和其他 repository 均不可见。

![Engineering and review ownership](assets/rq1/engineering_and_review_ownership.png)

## Review capacity 与集中度

2026 年 1–7 月有 77 名 roster member 提交 19,091 个 non-author review。Top five 占 35.0%，10 人完成一半，23 人完成 80%，Gini 为 0.664。相对 May-18 分析，review population 从 75 增至 77，top-five share 从 39.7% 降至 35.0%；review 稍微变宽，但远不足以追上 PR intake。

按 action 的 top-five share：submitted review 35.0%、inline comment 36.2%、PR conversation comment 34.8%、issue conversation comment 37.2%、label change 38.9%、close/reopen 41.3%、merge 44.4%。

专业领域仍比总体更集中：top-five reviewer 完成 multimodal/audio review 的 61.6%、frontend/API 的 55.5%、ROCm 的 52.7%、XPU 的 51.3%、quantization 的 49.8%、MoE 的 49.6%。这些是重叠的 heuristic topic signals，不是 expertise credential，但能定位 benchmark 中 verifier 和 expert review 可能稀缺的区域。

Open 和 closed-unmerged work 也消耗了大量 review。2026 cohort 中：

- merged PR 获得 81.4% 的 roster submitted review 和 74.1% 的 inline comment；
- closed-unmerged PR 获得 8.9% 的 review 和 11.5% 的 inline comment；
- open PR 已获得 9.7% 的 review 和 14.3% 的 inline comment。

所以只统计 merged PR 会漏掉 18.6% 的 submitted review 和 25.9% 的 inline review workload。

![Review capacity](assets/rq1/review_capacity.png)

## 外部贡献者 lifecycle

2026 年 1–7 月有 3,401 名 external author 提交 10,993 个 PR，其中 2,807 人是首次在数据中观察到提交 PR。

按这七个月内的频率：

| External author frequency | 作者 | PR | PR 占比 |
|---|---:|---:|---:|
| 1 个 PR | 1,896 | 1,896 | 17.2% |
| 2–4 个 PR | 984 | 2,548 | 23.2% |
| 5+ 个 PR | 521 | 6,549 | **59.6%** |

广泛 onboarding 和高产 repeat contributors 同时存在。首次 PR 获得 roster review 的比例为 34.0%，第 2–5 个为 43.0%，第 6 个以上为 56.9%；90 天内 merge 分别为 27.9%、38.8% 和 53.8%。

在有完整 90 天观察期的 2026 first-time external authors 中，42.0% 在 90 天内再次提交；这个 retention 估计只衡量公开 PR return，不衡量其他参与方式。

![External contributor lifecycle](assets/rq1/external_contributor_lifecycle.png)

## 推理系统、异构硬件与专业 workload

2026 PR 的主要 multi-label topic signals：

| Topic | PR | 2026 占比 | 2025 占比 |
|---|---:|---:|---:|
| Distributed and parallelism | 5,418 | **36.8%** | 29.1% |
| Attention and kernels | 4,025 | 27.3% | 19.8% |
| V1 engine and model runner | 3,692 | 25.1% | 17.3% |
| Model support | 2,704 | 18.4% | 17.5% |
| Frontend, serving, APIs | 2,493 | 16.9% | 18.9% |
| KV cache/connectors/offload | 2,148 | 14.6% | 8.8% |
| Quantization/low precision | 2,133 | 14.5% | 10.9% |
| MoE/expert parallelism | 1,761 | 12.0% | 9.0% |
| Speculative decoding | 1,505 | 10.2% | 7.8% |

这表明 benchmark 的“AI inference engineering”主体不是简单添加模型配置，而是分布式执行、attention/kernel、V1 runtime、KV cache、quantization、MoE、speculative decoding 和 serving integration 的组合。

硬件信号同样扩大：

| Hardware signal | 2026 PR | 2026 占比 | 2025 占比 |
|---|---:|---:|---:|
| NVIDIA/CUDA | 2,505 | 17.0% | 10.7% |
| AMD/ROCm | 2,432 | 16.5% | 10.8% |
| CPU | 1,099 | 7.5% | 5.0% |
| Cross-backend | 1,000 | 6.8% | 4.7% |
| Intel/XPU | 879 | 6.0% | 2.3% |
| TPU | 148 | 1.0% | 6.2% |
| Ascend/NPU | 42 | 0.3% | 0.1% |
| MLU | 0 | 0.0% | 0.0% |

ROCm 与 CUDA 已具有相近的可观察 PR 规模，XPU 增长明显。Ascend/NPU 和 MLU 在 vLLM source frame 中仍太少，不能按代表性 sampling 获得大权重；如果 benchmark 纳入它们，应标为 maintainer-nominated heterogeneous stress track，或补充更相关的 repository 数据。

![Engineering topics](assets/rq1/engineering_topics.png)

![Subsystems and hardware](assets/rq1/subsystems_and_hardware.png)

## 对 benchmark source frame 的直接含义

截至 07-31，共有 6,613 个 merged、human-authored、具有 commit data 的 2026 PR 可作为代表性 source frame：

- 45.0% touch test；
- 39.0% 有 hardware signal；
- 5.9% 属于 performance intent；
- 9.6% 是 large change；
- 14.5% 是 review-intensive；
- 2.9% 是 docs-only。

Test touched 只是“存在可见 verifier signal”，不表示测试足以验证 benchmark query。按 work type，merged source frame 中 test-touched 比例为：performance 37.4%、bug 40.2%、other 41.9%、feature 44.9%、refactor 47.3%、docs/API 48.2%、CI/build 52.4%。Performance 和 bug 恰好是 benchmark 很重要但 verifier coverage 较弱的类别。

硬件 source frame 的 test-touched 比例为 ROCm 58.0%、CUDA 52.8%、CPU 54.5%、XPU 47.9%、cross-backend 72.9%。Ascend 的 18 个 source-frame PR 虽有 94.4% touch test，但样本极小且 88.9% 是 large change，不能据此认为环境容易构建。

建议 76 个 representative tasks 至少分层覆盖：

1. Bug/correctness，并区分 runtime、kernel、distributed、API 和 model support；
2. Feature/capability 与 model/backend integration；
3. Performance/efficiency，保留连续 reward；
4. CI/build/release 与跨平台 breakage；
5. Refactor/maintainability 和 architecture migration；
6. Test/evaluation 与 verifier construction；
7. CUDA、ROCm、XPU、CPU、cross-backend；
8. Distributed、attention/kernel、V1、KV cache、quantization、MoE 和 speculative decoding；
9. Open/closed-unmerged、review-intensive 或 maintainer-nominated tasks，补足 merged-only source frame 的盲点。

Sampling 权重应来自人工编码后的 eligible frame，而不是直接把 heuristic share 当最终配额。24 个 memorable tasks 应单独报告，不与 probability-sampled representative track 混为一个总体 pass rate。

![Benchmark task signals](assets/rq1/benchmark_task_signals.png)

![Verifier signals](assets/rq1/verifier_signals.png)

## 与 May-18 结论相比，什么改变了

新增 5–7 月后，旧报告的中心结论不仅成立，而且更强：

- 2026 月均 PR intake 从当时估计的 1,836.3 上调为 2,102.6；
- 月均 merge 只从 909.0 上调到 974.4；
- active roster reviewer 从 60.3 下调到七个月平均 58.0；
- 每个新增 PR 的 submitted review 从 1.55 降到 1.35；
- cutoff open PR 从 3,037 增到 4,194；
- issue roster 7-day response 从 27.2% 降到 23.0%；
- PR roster response 从 62.2% 降到 57.6%；
- submitted roster review 从 55.0% 降到 50.8%。

需要修正的一点是：review gatekeeping 并没有变得更集中。Top-five review share 从 39.7% 降到 35.0%，active reviewers 从 75 增到 77。更准确的表述是：**review population 略有扩展，但 capacity 增长远慢于 demand，因此单个 PR 能获得的可见 review 密度显著下降。**

## 限制

- May-18 roster 不是 July roster，也不是历史 membership table。
- GitHub 数据不包含 Slack、private security、vendor coordination、release planning 和本地/未提交工作。
- Event count、active day 和 review count 不能转换为工时。
- Title/label/path taxonomy 是 deterministic exploratory classification，不是人工 gold coding。
- 删除于抓取前的内容无法恢复；cutoff 后编辑的文本没有历史版本。
- Commit timestamp 不能揭示 commit 首次 push 到 open PR 的时间。
- PR-level file coverage 对 delta-refreshed PR 最完整；更老 PR 使用原 snapshot 的 per-commit file data。
- Review submission 与 inline comment 不能完整表示 review thread resolution、同步设计讨论或 CI babysitting。
- Outcome 是 observational association，不是 author role、hardware 或 work type 的因果效应。

## RQ1 最终回答

vLLM 的现实维护 workload 是一个 **高增长、社区驱动、专业集成受限** 的系统：PR demand 在 2026 年接近翻倍，三分之二以上涉及推理系统内部和异构硬件信号，外部作者贡献约四分之三的 human PR，但 review、merge 和专业 gatekeeping 仍依赖规模增长很慢的可见维护者群体。

对 AI inference benchmark 而言，真正要测的不是 agent 能否完成普通 repository edit，而是它能否在这类 workload 中完成 diagnosis、实现、测试、性能验证、硬件适配和 review-driven revision。以当前 source frame 衡量，benchmark 必须同时覆盖可自动验证的任务和需要合成 verifier、真实 accelerator 或专家 judgement 的任务；否则无法回答“LLM 能解决多少真实 AI inference engineering workload”。
