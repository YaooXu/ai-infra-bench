# vLLM Harbor all-in-one Dockerfile template

Each vLLM task keeps its own self-contained `environment/Dockerfile`. The
generated files install the same CPU-native Python environment, Rust toolchain,
and test tooling; their only template-level difference is the pinned base
commit read from `task.toml`.

Generate one or more task Dockerfiles from the repository root:

```bash
python3 templates/vllm-harbor-all-in-one/generate.py \
  tasks/vllm-tool-argument-union \
  tasks/vllm-kv-admission-thrashing
```

Verify that checked-in Dockerfiles still match the template and task metadata:

```bash
python3 templates/vllm-harbor-all-in-one/generate.py --check tasks/vllm-*
```

Edit the template, not a generated Dockerfile. The generated Dockerfiles do not
refer back to this directory and can therefore be distributed with an
individual Harbor task.
