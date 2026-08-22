# 修复 Qwen3-VL grounding（视觉定位）坐标偏差

## 背景

vLLM `>= 0.11.1` 在**默认编译路径**（torch.compile + CUDA Graph）下，Qwen3-VL 系列模型
（例如 `Qwen3-VL-30B-A3B-Thinking`）输出的视觉定位（grounding）bounding-box 坐标存在
**系统性偏差**，在 Hopper 系列 GPU（H100 / H20 / H200）上可稳定复现。

社区已知 `--enforce-eager` 可以绕开该问题，但会显著降低推理吞吐，且这不是可接受的
长期解法。问题出在默认编译路径（语言模型部分的 Inductor 编译），与视觉编码器无关。

## 任务目标

修改 vLLM 源码，使 Qwen3-VL 在**默认编译模式**下（即**不**启用 `--enforce-eager`）
输出的 grounding bbox 坐标恢复正常。不得通过降低精度、关闭优化等配置手段绕开问题
（例如全局禁用 torch.compile 或关闭 CUDA Graph 不属于本题期望的修复）。

## 环境说明

- **vLLM 源码**：已以 editable 方式安装在 `/workspace/vllm-src`（pip install -e，
  即源码改动即时生效，无需重新安装）。基于 commit `4393684`（git describe 为 v0.11.1）。
- **模型**：`Qwen3-VL-30B-A3B-Thinking`（30B MoE, 13 shard）位于 `/models/`。
- **测试资产**：`/workspace/grounding/` 下包含：
  - `test_image.png` —— 测试截图（Wikipedia 页面）
  - `query_bbox.py` —— 提交 grounding 请求的脚本（OpenAI 兼容 API）
  - `eager_correct.json` / `eager_correct.png` —— eager 模式下的**正确参考结果**（bbox: `[957, 35, 988, 52]`）
  - `results/` —— 预生成的参考结果

## 复现步骤

1. 启动服务（**默认编译模式**，注意不要加 `--enforce-eager`）：

   ```bash
   cd /workspace/vllm-src
   CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=fork \
     python3 -m vllm.entrypoints.openai.api_server \
       --model /models/Qwen3-VL-30B-A3B-Thinking \
       --served-model-name qwen-vl \
       --tensor-parallel-size 2 --pipeline-parallel-size 1 \
       --reasoning_parser deepseek_r1 --no-enable-prefix-caching \
       --port 8000 --max-model-len 8192
   ```

   等待 `http://localhost:8000/health` 返回 200（首次启动含 CUDA Graph 捕获，需数分钟）。

2. 提交 grounding 请求：

   ```bash
   cd /workspace/grounding
   python3 query_bbox.py test_image.png "Locate the button that is used to log in" \
     --api-base http://localhost:8000/v1 --model qwen-vl \
     --seed 1 --max-tokens 1024
   ```

3. 对比参考：默认模式下输出 `[962, 27, 988, 45]`（有偏差），eager 参考为 `[957, 35, 988, 52]`。
   最大坐标差 8px（> 3px 阈值），即复现了 bug。

## 评测方式

- Verifier 会以**默认编译模式**启动你修改后的源码，对测试图片提交 grounding 请求，
  将检测到的 bbox 与预存参考对比。
- 判据：bbox 四个坐标与参考的最大差异 ≤ **3px**（并检测到正确的目标框）。
- 评测将使用**多张测试图片 / 多个 prompt**（含本仓库 required 之外的 held-out 用例），
  请确保修复是通用的，而非针对单个样本。

## 提示

- 聚焦于 vLLM 源码本身（`/workspace/vllm-src`），重点排查语言模型前向路径与
  Inductor / CUDA Graph 编译相关代码。
- 修改后可直接重启服务验证（editable 安装，无需重新 build）。
