# Qwen3-VL grounding 坐标偏差修复 —— agent 环境镜像
#
# 以官方 vllm 0.11.1 镜像(cu129, torch 2.9.0 + triton 3.5.0, 命中 bug 组合)为基础,
# 源码从上游 git clone + checkout v0.11.1 tag, 以 editable 方式安装到
# /workspace/vllm-src, 便于 agent 修改源码后直接生效。
#
# 构建步骤:
#   1. 拉取官方 base 镜像 (Docker Hub 公开镜像, 已锁定 digest 保证可复现):
#        docker pull vllm/vllm-openai:v0.11.1@sha256:d5b12dfb74d605615f8b29ebafaa52294c118bcac7bc9e941785c4108fdb913a
#      镜像来源: https://hub.docker.com/r/vllm/vllm-openai (vLLM 官方)
#      (国内网络可配置镜像加速器, 如 docker.m.daocloud.io/vllm/vllm-openai:v0.11.1,
#       拉取后需重新打上 vllm/vllm-openai:v0.11.1 tag)
#   2. git clone vLLM 源码需要访问 github.com (构建阶段有网络):
#        docker build -t tj-vllm-0.11.1 .
#      若构建机需代理访问 GitHub, 通过 --build-arg HTTP(S)_PROXY 传入。
#   3. 模型 (Qwen3-VL-30B-A3B-Thinking, 58G) 由 Harbor checkpoint 挂载到
#      /models/Qwen3-VL-30B-A3B-Thinking (运行时不打进镜像, 见 task.toml checkpoint_digests)。
#
# 注1: tag v0.11.1 对应官方构建 (torch 2.9.0 + triton 3.5.0 + cu129),
#      即 issue #29595 中可复现 bug 的精确依赖组合。
# 注2: git clone 的源码没有预编译产物, 首次构建需完整编译 CUDA 扩展
#      (~30-40 分钟, 依赖基础镜像内的 CUDA 工具链, 无需外部网络);
#      模型由外部挂载, 不在镜像内。
FROM vllm/vllm-openai:v0.11.1@sha256:d5b12dfb74d605615f8b29ebafaa52294c118bcac7bc9e941785c4108fdb913a

# 构建代理配置 (可选; 不设置则走直连; 仅 git clone 需要)
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ENV HTTP_PROXY=$HTTP_PROXY \
    HTTPS_PROXY=$HTTPS_PROXY

# 卸载 wheel 版 vllm (源码安装替代, 便于 agent 修改源码)
RUN pip uninstall -y vllm && rm -rf /usr/local/lib/python3.12/dist-packages/vllm

# 从上游 clone vLLM v0.11.1 源码 (commit 4393684 = git tag v0.11.1)
RUN git clone --branch v0.11.1 --depth 1 \
      https://github.com/vllm-project/vllm.git /workspace/vllm-src

# 复制 grounding 测试资产 (test_image.png, query_bbox.py, eager 参考结果)
COPY grounding /workspace/grounding

# 预设模型目录位置 (instruction.md / tests/test.sh 均引用此路径)
# 模型由 Harbor checkpoint 挂载到此处, 不随镜像分发
ENV MODEL_DIR=/models/Qwen3-VL-30B-A3B-Thinking

# editable 源码安装:
#   --no-build-isolation : 使用镜像内已有 torch 2.9.0 等, 不联网重拉
#   --no-deps            : 不重新解析 python 依赖
#   (首次构建会编译 CUDA 扩展; 之后重新 build 时 Docker layer 缓存命中,
#    pip install 不会再触发重编译)
RUN pip install -e /workspace/vllm-src --no-build-isolation --no-deps --no-cache-dir

WORKDIR /workspace/vllm-src

# 校验安装
RUN python3 -c "import vllm; print('vllm', vllm.__version__, 'at', vllm.__file__)"
