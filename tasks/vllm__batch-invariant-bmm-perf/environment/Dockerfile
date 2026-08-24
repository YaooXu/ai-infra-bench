# Baseline environment for the vLLM batch-invariant BMM optimization task.
# Built from base commit b07555d26f4c7ad9a2d1ec45428a9d4287db612c (2025-11-24).
#
# 设计要点（重要）：
# 官方镜像 vllm/vllm-openai:v0.11.0 在 dist-packages 里提供了**已编译**的 vLLM
# （含 C 扩展 vllm._C.abi3.so），这是运行时不可或缺的——直接把 base_commit 的
# 整棵源码树塞进 PYTHONPATH 会导致 `import vllm` 走未编译源码树，触发
# `ModuleNotFoundError: No module named 'vllm._C'`。
#
# 但 batch_invariant.py 是 base_commit 才引入的新文件，0.11.0 的 dist-packages
# 里并没有它。经核查，base 版 batch_invariant.py 依赖的所有 vllm 符号
# （vllm.envs / vllm.logger / vllm.platforms.current_platform /
#  vllm.triton_utils.{tl,triton} / vllm.utils.torch_utils.is_torch_equal_or_newer）
# 在已编译的 0.11.0 里全部存在。
#
# 因此策略是：保留 dist-packages 的已编译 vLLM 作为运行时，仅把 base 版
# batch_invariant.py 落到源码树 /app/vllm，并用软链把 dist-packages 中的对应
# 路径指向它。这样：
#   - `import vllm` 用已编译包（有 _C）；
#   - `vllm.model_executor.layers.batch_invariant` 这一个模块解析到 base 版文件；
#   - agent 按 instruction 编辑 /app/vllm/.../batch_invariant.py，改的就是运行时
#     真正加载的文件（软链同一 inode）；verifier import 到的也是它。
FROM vllm/vllm-openai:v0.11.0

WORKDIR /app

# Fetch vLLM source at the frozen base commit (build-time network).
RUN git clone https://github.com/vllm-project/vllm.git /app/vllm && \
    cd /app/vllm && \
    git checkout b07555d26f4c7ad9a2d1ec45428a9d4287db612c

# Anti-leak: strip history, leave a single synthetic "base snapshot" commit so
# the agent cannot recover the reference change from git log.
RUN cd /app/vllm && \
    rm -rf .git && \
    git init -q && \
    git add -A && \
    git -c user.email=bench@local -c user.name=bench commit -qm "base snapshot"

# 把 dist-packages 中 batch_invariant 的路径软链到源码树的 base 版文件。
# dist-packages 的 layers/ 目录本来没有该文件（0.11.0 未引入），这里新建软链。
RUN SP="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')" && \
    SRC=/app/vllm/vllm/model_executor/layers/batch_invariant.py && \
    test -f "$SRC" && \
    ln -sf "$SRC" "$SP/model_executor/layers/batch_invariant.py" && \
    echo "linked $SP/model_executor/layers/batch_invariant.py -> $SRC"

# 模块路径兼容 shim：base_commit(2025-11-24) 的 vLLM 已把若干工具函数从
# vllm/utils/__init__.py 拆分到 vllm/utils/torch_utils.py，而运行时用的已编译
# 0.11.0 里还没有这个子模块（相关函数仍在 vllm.utils 顶层）。base 版
# batch_invariant.py 里有 `from vllm.utils.torch_utils import is_torch_equal_or_newer`，
# 故补一个轻量 shim 模块把该名字从 vllm.utils 重导出，桥接这一次 API 迁移。
# 这不改变任何 bmm 计算逻辑，仅修复模块路径 skew。
RUN SP="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')" && \
    printf '%s\n' \
      '# Compatibility shim: re-export symbols that moved to this module in a' \
      '# later vLLM revision but still live in vllm.utils in this compiled 0.11.0.' \
      'from vllm.utils import is_torch_equal_or_newer  # noqa: F401' \
      > "$SP/utils/torch_utils.py" && \
    echo "wrote shim $SP/utils/torch_utils.py"

# Sanity check at build time: the module must import against the compiled vLLM.
RUN python3 -c "from vllm.model_executor.layers.batch_invariant import bmm_batch_invariant; print('import OK:', bmm_batch_invariant)"

# 重置基础镜像的 ENTRYPOINT。vllm/vllm-openai 的 ENTRYPOINT 会启动 OpenAI API
# server；而 harbor 的 build compose 用 `command: ["sh","-c","sleep infinity"]`
# 做 keepalive 保持容器存活以便 exec agent/verifier。若不清空 ENTRYPOINT，容器
# 实际会执行 `python3 -m vllm...api_server sh -c "sleep infinity"`，keepalive 失效、
# 容器立刻退出，导致 agent 被 SIGKILL(exit 137) 且 verifier 无法产出 reward。
ENTRYPOINT []

# 提供一个恒成功的轻量 healthcheck。harbor 用 `docker compose up --detach --wait`
# 等待容器健康：若沿用基础镜像里探测 vLLM server 端口的健康检查（本任务不起
# server）会永远不健康而超时；而完全移除（HEALTHCHECK NONE）又会让 `--wait` 报
# "container has no healthcheck configured"。这里用 `CMD true` 让容器立刻判定健康。
HEALTHCHECK --interval=5s --timeout=3s --retries=1 --start-period=1s CMD true

WORKDIR /app/vllm
