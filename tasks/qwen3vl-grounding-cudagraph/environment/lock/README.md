# lock/ —— 离线依赖锁定目录

agent 环境的 Python 依赖由基础镜像 `vllm/vllm-openai:v0.11.1` 提供
(torch 2.9.0+cu129 / triton 3.5.0 / flashinfer 0.5.2 / pillow / requests 等),
且 vllm 采用 editable 源码安装 (`--no-deps --no-build-isolation`),
因此无独立 `requirements.lock`。

如后续需要固化依赖, 在此生成:
```bash
pip freeze > requirements.lock
```
