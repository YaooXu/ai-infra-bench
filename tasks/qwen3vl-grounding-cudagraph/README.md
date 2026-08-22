# qwen3vl-grounding-cudagraph —— 题目交付说明

Harbor 任务: 修复 Qwen3-VL 在 vLLM 默认编译路径下的 grounding 坐标偏差
(source: vllm-project/vllm#29595)

## 目录结构

```text
tasks/qwen3vl-grounding-cudagraph/
├── task.toml              # Harbor 元数据与资源规格 (2x H20-3e, 2 GPU, TP=2)
├── instruction.md         # 给 agent 的题目说明 (不泄露解法)
├── environment/
│   ├── Dockerfile         # 基于官方 vllm 0.11.1 镜像, editable 源码安装
│   └── lock/README.md     # 依赖说明 (由基础镜像提供, 无独立 lock)
├── solution/
│   └── solve.sh           # 占位 (参考解法未开发)
└── tests/
    ├── test.sh            # verifier 评测脚本 (起服务 -> 比对 bbox)
    ├── required/case_wikipedia/   # 公开用例 (test_image.png + expected_bbox)
    └── heldout/README.md          # 留出测试说明 (待补充新图)
```

## 构建与验证状态

| 项 | 状态 |
|---|---|
| 基础镜像 | `vllm/vllm-openai:v0.11.1` (torch 2.9.0 + triton 3.5.0, 命中 bug 组合) |
| 源码安装镜像 | `tj-vllm-0.11.1:latest` (35.8GB) 已构建, editable 安装已验证 |
| 模型 | 从 HF (`Qwen/Qwen3-VL-30B-A3B-Thinking`) 下载到容器内 `/models/Qwen3-VL-30B-A3B-Thinking` |
| Bug 复现 | ✅ wikipedia: default `[962,27,988,45]` vs eager `[957,35,988,52]`, 差 8px |
| Bug 复现 | ✅ signin: default `[800,19,882,61]` vs eager `[796,19,879,71]`, 差 10px |
| required 用例 | case_wikipedia + case_signin (均含 eager 参考, threshold 3px) |
| heldout 用例 | 待补充 (需 curator 在 eager 模式下生成参考) |
| 参考解法 | 待开发 (上游无修复 PR, 需自行设计) |

## 模型获取机制

Dockerfile 在**构建阶段**（有网络）从 Hugging Face 下载模型到预设目录：

```dockerfile
ENV MODEL_DIR=/models/Qwen3-VL-30B-A3B-Thinking
RUN huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Thinking --local-dir $MODEL_DIR
```

- 标准源: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking (58G, 13 shard)
- 国内加速: 构建时设 `HF_ENDPOINT=https://hf-mirror.com`
- 模型打进镜像后 agent 离线可用（运行环境 `no-network`）
- **镜像体积注意**: 58G 模型 + 35.8G vllm 基础 ≈ 94G+，构建与分发成本高；
  如改用 Harbor checkpoint 单独挂载（推荐，可避免镜像膨胀），调整 `task.toml` 的
  `checkpoint_digests` 为完整模型目录 digest 即可，Dockerfile 去掉下载步骤

## 环境构建上下文

**Base 镜像**（Dockerfile 的 FROM）是 vLLM 官方公开镜像，任何人都可获取：

```bash
# Docker Hub (vLLM 官方): https://hub.docker.com/r/vllm/vllm-openai
docker pull vllm/vllm-openai:v0.11.1@sha256:d5b12dfb74d605615f8b29ebafaa52294c118bcac7bc9e941785c4108fdb913a
# 国内网络可用镜像加速源, 如:
#   docker pull docker.m.daocloud.io/vllm/vllm-openai:v0.11.1
#   (拉取后 docker tag docker.m.daocloud.io/vllm/vllm-openai:v0.11.1 vllm/vllm-openai:v0.11.1)
```

tag 已锁定 digest，保证不同机器构建出的环境一致。

**vLLM 源码**：Dockerfile 直接 `git clone --branch v0.11.1` 从上游
(vllm-project/vllm) 获取，不随本仓库分发。源码 tag 锁定 `v0.11.1`
(commit `4393684`)，与 issue #29595 复现版本一致。

Dockerfile 需要的 `grounding/` 资产已随本仓库发布 (见 `environment/grounding/`):
`test_image.png`, `query_bbox.py`, `eager_correct.json`, `eager_correct.png`。
构建时无需额外准备, 直接:

```bash
cd environment/
docker build -t tj-vllm-0.11.1 .
```

> 注: 首次构建会完整编译 vLLM CUDA 扩展 (~30-40 分钟)。
> 重新 build 时 Docker layer 缓存命中, 不会重复编译。

## 待办 (发布前)

1. 实际构建验证 git clone 方案的 Dockerfile（含模型下载；预计 40-60 分钟，需确认构建机可访问 HF）
2. 补充 heldout 用例 (第三张图 + eager 参考)
3. 开发参考解法 solve.sh, 并验证 base/参考/no-op/错误解四态
4. 更新 checkpoint_digests (完整模型目录 sha256) 与 image_digest (最终镜像)
5. 确认模型获取方式: 打进镜像 (当前 Dockerfile) 或 Harbor checkpoint 挂载 (推荐, 避免 94G+ 镜像)
