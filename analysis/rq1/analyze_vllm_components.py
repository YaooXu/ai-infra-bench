"""Derive a benchmark-oriented vLLM component taxonomy.

The analysis triangulates three sources available at the July 31 cutoff:

1. the checked-out vLLM source tree;
2. maintainer-authored Buildkite test-area definitions; and
3. changed paths in 2026 pull requests from the merged GitHub database.

The resulting component labels are multi-label. Work type, hardware backend,
capability/topic, and verification surface are intentionally separate axes.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pathspec
import yaml
from matplotlib import pyplot as plt


CUTOFF = "2026-07-31T23:59:59Z"
CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
PRODUCTION_PREFIXES = ("vllm/", "csrc/", "rust/")


@dataclass(frozen=True)
class Component:
    name: str
    layer: str
    description: str
    patterns: tuple[str, ...]


# Order is also the deterministic tie-break order for assigning a primary
# component. Specific components precede broad/shared ones.
COMPONENTS = (
    Component(
        "Scheduler and core runtime",
        "Runtime control plane",
        "Request scheduling, batch formation, block coordination, and core state machines.",
        (
            r"^vllm/v1/core/",
            r"^tests/v1/core/",
            r"^tests/v1/test_async_scheduling\.py$",
            r"^tests/v1/e2e/general/test_async_scheduling\.py$",
        ),
    ),
    Component(
        "Engine lifecycle",
        "Runtime control plane",
        "Synchronous/asynchronous engine APIs, engine processes, lifecycle, and IPC.",
        (
            r"^vllm/(v1/engine|engine)/",
            r"^tests/(v1/engine|engine)/",
            r"^tests/v1/test_tensor_ipc_queue\.py$",
            r"^vllm/sequence\.py$",
        ),
    ),
    Component(
        "Worker and model runner",
        "Execution data plane",
        "Device workers, model runners, execution state, and GPU/CPU execution loops.",
        (
            r"^vllm/(v1/worker|worker)/",
            r"^tests/(v1/worker|worker)/",
            r"model_runner",
            r"^tests/v1/cudagraph/",
        ),
    ),
    Component(
        "KV cache and data movement",
        "Execution data plane",
        "KV allocation, caching, connectors, offload, and prefill/decode data transfer.",
        (
            r"^vllm/(device_allocator|v1/kv_offload|v1/simple_kv_offload)/",
            r"^vllm/distributed/(kv_transfer|ec_transfer|weight_transfer)/",
            r"^tests/.*/(kv_|kv-|kv/)",
            r"^tests/v1/(kv_connector|kv_offload|ec_connector)/",
            r"kv_cache",
            r"cache_manager",
            r"block_pool",
            r"block_table",
        ),
    ),
    Component(
        "Distributed execution",
        "Execution data plane",
        "Executors, collectives, tensor/data/pipeline parallelism, and multi-node coordination.",
        (
            r"^vllm/(distributed|v1/executor|executor|ray)/",
            r"^tests/(distributed|v1/distributed|v1/executor)/",
            r"^tests/.*/parallel",
            r"^tests/ray/",
        ),
    ),
    Component(
        "Serving APIs and frontend",
        "Serving and request interface",
        "OpenAI/Anthropic-compatible servers, CLI, MCP, pooling, scale-out, and Rust frontend.",
        (
            r"^vllm/entrypoints/",
            r"^tests/entrypoints/",
            r"^rust/",
            r"^tests/rust_frontend/",
        ),
    ),
    Component(
        "Input, multimodal, and rendering",
        "Model and data semantics",
        "Input preprocessing, multimodal data, tokenization, chat rendering, parsers, and tools.",
        (
            r"^vllm/(inputs|multimodal|parser|reasoning|renderers|tokenizers|tool_parsers|transformers_utils)/",
            r"^vllm/v1/structured_output/",
            r"^tests/(inputs|multimodal|parser|reasoning|renderers|tokenizers|tool_parsers|transformers_utils|structured_output)/",
            r"^tests/v1/structured_output/",
        ),
    ),
    Component(
        "Model implementations and registry",
        "Model and data semantics",
        "Model architectures, registry integration, configs, processors, and model-specific behavior.",
        (
            r"^vllm/(model_executor/models|models)/",
            r"^tests/models/",
            r"^tests/model_executor/models/",
        ),
    ),
    Component(
        "Model loading and execution layers",
        "Model and data semantics",
        "Weights, loaders, generic neural-network layers, warmup, offloading, and execution utilities.",
        (
            r"^vllm/model_executor/(model_loader|layers|offloader|warmup)/",
            r"^vllm/model_executor/(parameter|pooling|utils|weight_utils)",
            r"^tests/model_executor/(model_loader|layers|weight|test_weight|test_model_loader)",
            r"^tests/model_executor/",
        ),
    ),
    Component(
        "Attention",
        "Kernels and compilation",
        "Attention backends, attention layers, paged attention, and flash-attention integration.",
        (
            r"^vllm/(v1/attention|model_executor/layers/attention|vllm_flash_attn)/",
            r"^csrc/(attention|libtorch_stable/attention)/",
            r"^tests/(attention|v1/attention)/",
            r"attention_backend",
            r"attention_layer",
        ),
    ),
    Component(
        "Kernels and custom operators",
        "Kernels and compilation",
        "Native/Triton/Helion operators and low-level operator bindings not owned by a narrower domain.",
        (
            r"^csrc/",
            r"^vllm/(kernels|cute_utils|triton_utils|_custom_ops\.py|_aiter_ops\.py)",
            r"^tests/(kernels|ops)/",
            r"^tests/model_executor/test_custom_op",
        ),
    ),
    Component(
        "Quantization and low precision",
        "Kernels and compilation",
        "Quantization configurations, low-precision layers, formats, and quantized kernels.",
        (
            r"^vllm/model_executor/layers/quantization/",
            r"^csrc/(quantization|libtorch_stable/quantization)/",
            r"^tests/(quantization|model_executor/layers/quantization)/",
            r"quantiz",
            r"fp8",
            r"fp4",
        ),
    ),
    Component(
        "MoE and expert parallelism",
        "Execution data plane",
        "Mixture-of-experts layers and kernels, expert parallelism, load balancing, and routing.",
        (
            r"^vllm/model_executor/layers/fused_moe/",
            r"^csrc/(moe|libtorch_stable/moe)/",
            r"^vllm/distributed/(elastic_ep|eplb)/",
            r"^tests/.*(moe|expert_parallel|eplb)",
        ),
    ),
    Component(
        "Compilation, graphs, and IR",
        "Kernels and compilation",
        "torch.compile integration, graph capture, compilation passes, and vLLM IR.",
        (
            r"^vllm/(compilation|ir)/",
            r"^tests/(compile|compilation|v1/cudagraph|ir)/",
            r"cudagraph",
            r"torch_compile",
        ),
    ),
    Component(
        "Sampling",
        "Execution data plane",
        "Logits processing, sampling algorithms, and output selection.",
        (
            r"^vllm/v1/sample/",
            r"^vllm/sampling_params\.py$",
            r"^vllm/logits_process\.py$",
            r"^tests/(samplers|sampling|v1/sample|v1/logits_processors)/",
        ),
    ),
    Component(
        "Speculative decoding",
        "Execution data plane",
        "Draft-model, EAGLE, Medusa, MTP, and speculative decoding orchestration.",
        (
            r"^vllm/v1/spec_decode/",
            r"^tests/(spec_decode|v1/spec_decode)/",
            r"spec_decode",
        ),
    ),
    Component(
        "LoRA and adapters",
        "Model and data semantics",
        "LoRA execution, adapter management, resolver plugins, and related kernels.",
        (
            r"^vllm/lora/",
            r"^tests/lora/",
            r"lora_resolver",
            r"prompt_adapter",
        ),
    ),
    Component(
        "Platform and device abstraction",
        "Platform and operations",
        "Runtime selection and behavioral abstraction for CUDA, ROCm, CPU, XPU, TPU, and plugins.",
        (
            r"^vllm/platforms/",
            r"^tests/platforms/",
            r"^vllm/plugins/",
            r"^tests/plugins/",
        ),
    ),
    Component(
        "Reliability and observability",
        "Platform and operations",
        "Fault tolerance, health, metrics, tracing, profiling, logging, and usage telemetry.",
        (
            r"^vllm/(v1/fault_tolerance|v1/metrics|fault_tolerance|profiler|tracing|logging_utils|usage)/",
            r"^tests/(fault_tolerance|metrics|profiler|tracing|logging_utils|v1/fault_tolerance|v1/metrics)/",
            r"^vllm/(logger|logging)\.py$",
            r"^tests/(test_logger|test_logging)",
        ),
    ),
    Component(
        "Configuration and shared infrastructure",
        "Platform and operations",
        "Configuration, environment settings, shared utilities, public parameters, and common infrastructure.",
        (
            r"^vllm/(config|utils)/",
            r"^vllm/(envs|forward_context|outputs|pooling_params)\.py$",
            r"^tests/(config|utils)/",
            r"^tests/test_(config|envs|utils)",
        ),
    ),
)

# Narrow domains win ties against the broader directory that contains their
# implementation. This matters for paths such as
# ``vllm/model_executor/layers/quantization/...`` which intentionally receive
# both a broad and a narrow multi-label assignment.
PRIMARY_PRIORITY = (
    "Scheduler and core runtime",
    "Engine lifecycle",
    "Worker and model runner",
    "KV cache and data movement",
    "Attention",
    "Quantization and low precision",
    "MoE and expert parallelism",
    "Compilation, graphs, and IR",
    "Sampling",
    "Speculative decoding",
    "LoRA and adapters",
    "Distributed execution",
    "Serving APIs and frontend",
    "Input, multimodal, and rendering",
    "Model implementations and registry",
    "Model loading and execution layers",
    "Platform and device abstraction",
    "Reliability and observability",
    "Configuration and shared infrastructure",
    "Kernels and custom operators",
)


SUPPORT_RULES = {
    "Tests and verification": (r"^tests?/",),
    "Build, packaging, and CI": (
        r"^(\.buildkite|\.github|cmake|docker|requirements)/",
        r"^(CMakeLists\.txt|setup\.py|pyproject\.toml)$",
    ),
    "Documentation": (r"^docs/", r"\.(md|rst)$"),
    "Examples": (r"^examples/",),
    "Benchmarks and evaluation": (r"^(benchmarks?|evals?)/", r"^vllm/benchmarks/"),
}


BACKEND_RULES = {
    "CUDA/NVIDIA": (r"cuda", r"nvidia", r"cutlass", r"triton"),
    "ROCm/AMD": (r"rocm", r"amd", r"aiter"),
    "CPU": (r"(^|/)cpu([_/.-]|$)",),
    "XPU/Intel": (r"xpu", r"intel"),
    "TPU": (r"(^|[/_.-])tpu([/_.-]|$)",),
}


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, path, flags=re.IGNORECASE) for pattern in patterns)


def component_names(path: str) -> list[str]:
    return [
        component.name for component in COMPONENTS if matches(path, component.patterns)
    ]


def component_names_for_pattern(pattern: str) -> list[str]:
    """Map a CODEOWNERS directory/glob pattern onto component path rules."""
    normalized = pattern.lstrip("/")
    probes = (normalized, normalized.rstrip("/") + "/__codeowners_probe__.py")
    names = {name for probe in probes for name in component_names(probe)}
    return [component.name for component in COMPONENTS if component.name in names]


def support_names(path: str) -> list[str]:
    return [name for name, patterns in SUPPORT_RULES.items() if matches(path, patterns)]


def backend_names(path: str) -> list[str]:
    return [name for name, patterns in BACKEND_RULES.items() if matches(path, patterns)]


def git_output(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def load_pr_files(database: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(database)
    prs = pd.read_sql_query(
        """
        SELECT p.database_id AS pull_request_id, p.artifact_id,
               a.number, a.title, a.created_at, p.source_layer,
               p.files_cutoff_stable
        FROM canonical_pull_request p
        JOIN canonical_artifact a ON a.database_id=p.artifact_id
        WHERE datetime(a.created_at) >= datetime('2026-01-01T00:00:00Z')
          AND datetime(a.created_at) <= datetime('2026-07-31T23:59:59Z')
        """,
        conn,
    )
    base = pd.read_sql_query(
        """
        SELECT DISTINCT pc.pull_request_id, cf.filename
        FROM canonical_pull_request_commit pc
        JOIN commit_file cf ON cf.commit_sha=pc.commit_sha
        JOIN canonical_pull_request p ON p.database_id=pc.pull_request_id
        JOIN canonical_artifact a ON a.database_id=p.artifact_id
        WHERE p.source_layer='base'
          AND datetime(a.created_at) >= datetime('2026-01-01T00:00:00Z')
          AND datetime(a.created_at) <= datetime('2026-07-31T23:59:59Z')
        """,
        conn,
    )
    delta = pd.read_sql_query(
        """
        SELECT DISTINCT f.pull_request_id, f.path AS filename
        FROM canonical_pull_request_file f
        JOIN canonical_pull_request p ON p.database_id=f.pull_request_id
        JOIN canonical_artifact a ON a.database_id=p.artifact_id
        WHERE p.source_layer='delta'
          AND datetime(a.created_at) >= datetime('2026-01-01T00:00:00Z')
          AND datetime(a.created_at) <= datetime('2026-07-31T23:59:59Z')
        """,
        conn,
    )
    conn.close()
    files = pd.concat([base, delta], ignore_index=True).drop_duplicates()
    return prs, files


def source_inventory(source: Path) -> pd.DataFrame:
    rows = []
    for relative in git_output(source, "ls-files").splitlines():
        path = source / relative
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        code = suffix in CODE_EXTENSIONS
        lines = 0
        if code:
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    lines = sum(1 for _ in handle)
            except OSError:
                lines = 0
        rows.append(
            {
                "filename": relative,
                "code_file": code,
                "production_file": relative.startswith(PRODUCTION_PREFIXES),
                "lines": lines,
                "components": component_names(relative),
                "support_surfaces": support_names(relative),
                "backends": backend_names(relative),
            }
        )
    return pd.DataFrame(rows)


def ci_inventory(source: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((source / ".buildkite/test_areas").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        group = payload.get("group", path.stem)
        for step in payload.get("steps", []):
            dependencies = step.get("source_file_dependencies", []) or []
            components = sorted(
                {
                    name
                    for dependency in dependencies
                    for name in component_names(str(dependency))
                }
            )
            mirrors = step.get("mirror", {}) or {}
            rows.append(
                {
                    "test_area_file": path.name,
                    "group": group,
                    "step": step.get("label"),
                    "key": step.get("key"),
                    "device": step.get("device"),
                    "num_devices": step.get("num_devices", 1),
                    "optional": bool(step.get("optional", False)),
                    "dependency_count": len(dependencies),
                    "components": "; ".join(components),
                    "mirrors": "; ".join(sorted(mirrors)),
                }
            )
    return pd.DataFrame(rows)


def hardware_ci_inventory(source: Path) -> pd.DataFrame:
    rows = []
    root = source / ".buildkite/hardware_tests"
    for path in sorted(root.rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        group = payload.get("group", path.stem)
        for step in payload.get("steps", []):
            dependencies = step.get("source_file_dependencies", []) or []
            rows.append(
                {
                    "hardware_file": str(path.relative_to(root)),
                    "group": group,
                    "step": step.get("label"),
                    "key": step.get("key"),
                    "device": step.get("device"),
                    "num_devices": step.get("num_devices", 1),
                    "optional": bool(step.get("optional", False)),
                    "soft_fail": bool(step.get("soft_fail", False)),
                    "dependency_count": len(dependencies),
                    "components": "; ".join(
                        sorted(
                            {
                                name
                                for dependency in dependencies
                                for name in component_names(str(dependency))
                            }
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def codeowners_inventory(source: Path) -> tuple[pd.DataFrame, pathspec.PathSpec]:
    codeowners = source / ".github/CODEOWNERS"
    rows = []
    patterns = []
    section = "Unsectioned"
    for raw_line in codeowners.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            # Only short heading-like comments are retained. Explanatory
            # comments do not change the active section.
            if (
                text
                and len(text) <= 45
                and not text.endswith((".", ":"))
                and "http" not in text
                and not text.lower().startswith(("see ", "for ", "so "))
            ):
                section = text
            continue
        fields = line.split()
        pattern, owners = fields[0], fields[1:]
        patterns.append(pattern)
        normalized = pattern.lstrip("/")
        rows.append(
            {
                "section": section,
                "pattern": pattern,
                "owner_count": len(owners),
                "components": "; ".join(component_names_for_pattern(pattern)),
                "backends": "; ".join(backend_names(normalized)),
            }
        )
    return pd.DataFrame(rows), pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def analyze_prs(
    prs: pd.DataFrame, files: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapped = files.copy()
    mapped["components"] = mapped["filename"].map(component_names)
    mapped["support_surfaces"] = mapped["filename"].map(support_names)
    mapped["backends"] = mapped["filename"].map(backend_names)

    scores: dict[int, Counter[str]] = defaultdict(Counter)
    support_by_pr: dict[int, set[str]] = defaultdict(set)
    backend_by_pr: dict[int, set[str]] = defaultdict(set)
    files_by_pr: Counter[int] = Counter()
    for row in mapped.itertuples(index=False):
        files_by_pr[row.pull_request_id] += 1
        weight = 1 if str(row.filename).startswith("tests/") else 2
        for name in row.components:
            scores[row.pull_request_id][name] += weight
        support_by_pr[row.pull_request_id].update(row.support_surfaces)
        backend_by_pr[row.pull_request_id].update(row.backends)

    priority = {name: index for index, name in enumerate(PRIMARY_PRIORITY)}
    pr_rows = []
    for row in prs.itertuples(index=False):
        component_score = scores[row.pull_request_id]
        ordered = sorted(
            component_score, key=lambda name: (-component_score[name], priority[name])
        )
        pr_rows.append(
            {
                "pull_request_id": row.pull_request_id,
                "number": row.number,
                "created_at": row.created_at,
                "source_layer": row.source_layer,
                "files_cutoff_stable": row.files_cutoff_stable,
                "file_count_observed": files_by_pr[row.pull_request_id],
                "components": ordered,
                "primary_component": ordered[0] if ordered else None,
                "support_surfaces": sorted(support_by_pr[row.pull_request_id]),
                "backends": sorted(backend_by_pr[row.pull_request_id]),
            }
        )
    pr_map = pd.DataFrame(pr_rows)
    observed = pr_map[pr_map["file_count_observed"] > 0]
    denominator = len(observed)

    prevalence = []
    for component in COMPONENTS:
        any_count = int(
            observed["components"].map(lambda values: component.name in values).sum()
        )
        primary_count = int(observed["primary_component"].eq(component.name).sum())
        prevalence.append(
            {
                "layer": component.layer,
                "component": component.name,
                "any_prs": any_count,
                "any_share_of_prs_with_files": any_count / denominator
                if denominator
                else None,
                "primary_prs": primary_count,
                "primary_share_of_prs_with_files": primary_count / denominator
                if denominator
                else None,
            }
        )
    prevalence_frame = pd.DataFrame(prevalence).sort_values("any_prs", ascending=False)

    overlap = []
    for left in COMPONENTS:
        left_set = set(
            observed.loc[
                observed["components"].map(lambda values: left.name in values),
                "pull_request_id",
            ]
        )
        for right in COMPONENTS:
            right_set = set(
                observed.loc[
                    observed["components"].map(lambda values: right.name in values),
                    "pull_request_id",
                ]
            )
            overlap.append(
                {
                    "left": left.name,
                    "right": right.name,
                    "prs": len(left_set & right_set),
                    "jaccard": len(left_set & right_set) / len(left_set | right_set)
                    if left_set | right_set
                    else None,
                }
            )
    overlap_frame = pd.DataFrame(overlap)

    support_rows = []
    for name in SUPPORT_RULES:
        count = int(
            observed["support_surfaces"].map(lambda values: name in values).sum()
        )
        support_rows.append(
            {
                "surface": name,
                "prs": count,
                "share_of_prs_with_files": count / denominator if denominator else None,
            }
        )
    support_frame = pd.DataFrame(support_rows).sort_values("prs", ascending=False)
    return pr_map, prevalence_frame, overlap_frame, support_frame


def validate_taxonomy() -> None:
    expected = {
        "vllm/v1/core/sched/scheduler.py": {"Scheduler and core runtime"},
        "vllm/v1/engine/core.py": {"Engine lifecycle"},
        "vllm/v1/worker/gpu/model_runner.py": {"Worker and model runner"},
        "vllm/model_executor/layers/quantization/fp8.py": {
            "Model loading and execution layers",
            "Quantization and low precision",
        },
        "vllm/model_executor/layers/attention/mla_attention.py": {
            "Model loading and execution layers",
            "Attention",
        },
        "vllm/entrypoints/openai/api_server.py": {"Serving APIs and frontend"},
        "tests/v1/kv_connector/test_basic.py": {"KV cache and data movement"},
    }
    for path, required in expected.items():
        actual = set(component_names(path))
        if not required <= actual:
            raise AssertionError(
                f"Taxonomy regression for {path}: expected {required}, got {actual}"
            )
    if backend_names("vllm/v1/structured_output/backend.py"):
        raise AssertionError("The substring 'output' must not be classified as TPU")
    if set(PRIMARY_PRIORITY) != {component.name for component in COMPONENTS}:
        raise AssertionError(
            "PRIMARY_PRIORITY and COMPONENTS must contain the same names"
        )


def render_prevalence(prevalence: pd.DataFrame, output: Path) -> None:
    frame = prevalence.sort_values("any_share_of_prs_with_files", ascending=True).copy()
    y = list(range(len(frame)))
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    ax.barh(
        y,
        frame["any_share_of_prs_with_files"] * 100,
        color="#9ecae1",
        label="Touched anywhere (multi-label)",
    )
    ax.scatter(
        frame["primary_share_of_prs_with_files"] * 100,
        y,
        color="#08519c",
        s=30,
        label="Primary component",
        zorder=3,
    )
    ax.set_yticks(y, frame["component"])
    ax.set_xlabel("Share of 2026 PRs with observed changed files (%)")
    ax.set_title("vLLM engineering components observed in pull requests, Jan–Jul 2026")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def source_component_summary(source_files: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component in COMPONENTS:
        selected = source_files[
            source_files["components"].map(lambda values: component.name in values)
        ]
        production = selected[selected["production_file"]]
        tests = selected[selected["filename"].str.startswith("tests/")]
        rows.append(
            {
                "layer": component.layer,
                "component": component.name,
                "tracked_files": len(selected),
                "code_files": int(selected["code_file"].sum()),
                "code_lines": int(selected["lines"].sum()),
                "production_files": len(production),
                "production_code_files": int(production["code_file"].sum()),
                "production_code_lines": int(production["lines"].sum()),
                "test_files": len(tests),
                "test_code_lines": int(tests["lines"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("production_code_lines", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("data/raw/vllm-source-2026-07-31")
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/derived/vllm_github_2026-07-31.sqlite"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/rq1/outputs/components")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("analysis/rq1/component_summary.json")
    )
    parser.add_argument("--figures", type=Path, default=Path("docs/assets/rq1"))
    args = parser.parse_args()

    validate_taxonomy()
    args.output.mkdir(parents=True, exist_ok=True)
    commit = git_output(args.source, "rev-parse", "HEAD")
    committed_at = git_output(args.source, "show", "-s", "--format=%cI", "HEAD")
    source_files = source_inventory(args.source)
    ci_steps = ci_inventory(args.source)
    hardware_steps = hardware_ci_inventory(args.source)
    codeowner_entries, codeowner_spec = codeowners_inventory(args.source)
    prs, files = load_pr_files(args.database)
    pr_map, prevalence, overlap, support = analyze_prs(prs, files)
    source_summary = source_component_summary(source_files)

    source_summary.to_csv(args.output / "source_component_inventory.csv", index=False)
    ci_steps.to_csv(args.output / "buildkite_test_areas.csv", index=False)
    hardware_steps.to_csv(args.output / "buildkite_hardware_tests.csv", index=False)
    codeowner_entries.to_csv(args.output / "codeowners_inventory.csv", index=False)
    prevalence.to_csv(args.output / "pr_component_prevalence_2026.csv", index=False)
    overlap.to_csv(args.output / "pr_component_overlap_2026.csv", index=False)
    support.to_csv(args.output / "pr_support_surfaces_2026.csv", index=False)
    pr_map.assign(
        components=pr_map["components"].map(lambda values: "; ".join(values)),
        support_surfaces=pr_map["support_surfaces"].map(
            lambda values: "; ".join(values)
        ),
        backends=pr_map["backends"].map(lambda values: "; ".join(values)),
    ).to_csv(args.output / "pr_component_assignments_2026.csv", index=False)
    mapped_files = files.assign(components=files["filename"].map(component_names))
    file_pr_counts = (
        mapped_files.groupby("filename")["pull_request_id"].nunique().rename("prs")
    )
    component_by_file = (
        mapped_files[["filename", "components"]]
        .drop_duplicates("filename")
        .explode("components")
        .dropna(subset=["components"])
        .merge(file_pr_counts, left_on="filename", right_index=True)
        .rename(columns={"components": "component"})
    )
    top_paths = (
        component_by_file.sort_values(
            ["component", "prs", "filename"], ascending=[True, False, True]
        )
        .groupby("component", as_index=False)
        .head(15)
    )
    top_paths.to_csv(args.output / "component_top_paths_2026.csv", index=False)
    render_prevalence(prevalence, args.figures / "vllm_component_prevalence_2026.png")

    observed = pr_map[pr_map["file_count_observed"] > 0]
    classified = observed[observed["primary_component"].notna()]
    stable_observed = observed[observed["files_cutoff_stable"].eq(1)]
    stable_classified = stable_observed[stable_observed["primary_component"].notna()]
    component_counts = observed["components"].map(len)
    test_area_component_counts = (
        ci_steps["components"]
        .fillna("")
        .map(lambda value: len([item for item in value.split("; ") if item]))
    )
    group_counts = ci_steps.groupby("group").size().sort_values(ascending=False)
    layer_counts = (
        prevalence.groupby("layer", as_index=False)[["any_prs", "primary_prs"]]
        .sum()
        .sort_values("any_prs", ascending=False)
    )
    backend_counts = []
    for backend in BACKEND_RULES:
        count = int(observed["backends"].map(lambda values: backend in values).sum())
        backend_counts.append(
            {
                "backend": backend,
                "prs": count,
                "share_of_prs_with_files": count / len(observed)
                if len(observed)
                else None,
            }
        )
    unclassified = observed[observed["primary_component"].isna()]
    source_codeowned = source_files["filename"].map(codeowner_spec.match_file)
    production = source_files["production_file"]
    file_codeowned = files.assign(
        codeowned=files["filename"].map(codeowner_spec.match_file)
    )
    codeowned_pr_ids = set(
        file_codeowned.loc[file_codeowned["codeowned"], "pull_request_id"]
    )
    observed_pr_ids = set(observed["pull_request_id"])
    owner_component_entries = Counter()
    for values in codeowner_entries["components"].fillna(""):
        for value in values.split("; "):
            if value:
                owner_component_entries[value] += 1
    non_diagonal_overlap = overlap[overlap["left"] < overlap["right"]].sort_values(
        "prs", ascending=False
    )

    summary = {
        "cutoff": CUTOFF,
        "source_commit": commit,
        "source_commit_time": committed_at,
        "method": {
            "evidence": [
                "vLLM source tree",
                "vLLM CODEOWNERS careful-review paths",
                "Buildkite test areas and hardware jobs",
                "2026 pull-request changed paths",
            ],
            "component_labels_are_multilabel": True,
            "primary_component_rule": "Highest weighted changed-file score; source file=2, test file=1; taxonomy order breaks ties.",
            "separate_axes": [
                "work type",
                "component",
                "capability/topic",
                "hardware backend",
                "verification surface",
            ],
        },
        "taxonomy": [
            {
                "name": component.name,
                "layer": component.layer,
                "description": component.description,
            }
            for component in COMPONENTS
        ],
        "source": {
            "tracked_files": len(source_files),
            "code_files": int(source_files["code_file"].sum()),
            "code_lines": int(source_files["lines"].sum()),
            "production_files": int(source_files["production_file"].sum()),
            "production_code_lines": int(
                source_files.loc[source_files["production_file"], "lines"].sum()
            ),
            "component_summary": source_summary.to_dict(orient="records"),
        },
        "buildkite": {
            "area_files": int(ci_steps["test_area_file"].nunique()),
            "groups": int(ci_steps["group"].nunique()),
            "steps": len(ci_steps),
            "steps_with_component_mapping": int((test_area_component_counts > 0).sum()),
            "steps_spanning_multiple_components": int(
                (test_area_component_counts > 1).sum()
            ),
            "largest_groups": group_counts.head(12).to_dict(),
        },
        "hardware_ci": {
            "files": int(hardware_steps["hardware_file"].nunique()),
            "steps": len(hardware_steps),
            "optional_steps": int(hardware_steps["optional"].sum()),
            "soft_fail_steps": int(hardware_steps["soft_fail"].sum()),
            "devices": hardware_steps["device"].dropna().value_counts().to_dict(),
        },
        "codeowners": {
            "entries": len(codeowner_entries),
            "ownerless_entries": int(codeowner_entries["owner_count"].eq(0).sum()),
            "entries_with_component_mapping": int(
                codeowner_entries["components"].fillna("").ne("").sum()
            ),
            "production_files_covered": int((source_codeowned & production).sum()),
            "production_file_coverage": float(
                (source_codeowned & production).sum() / production.sum()
            ),
            "prs_with_files_touching_codeowned_path": len(
                codeowned_pr_ids & observed_pr_ids
            ),
            "share_of_prs_with_files_touching_codeowned_path": len(
                codeowned_pr_ids & observed_pr_ids
            )
            / len(observed_pr_ids),
            "component_entry_counts": dict(owner_component_entries.most_common()),
            "interpretation": "CODEOWNERS explicitly describes careful-review surfaces, not an exhaustive architecture taxonomy.",
        },
        "pull_requests_2026": {
            "total": len(pr_map),
            "with_observed_files": len(observed),
            "with_component": len(classified),
            "component_coverage_of_prs_with_files": len(classified) / len(observed)
            if len(observed)
            else None,
            "component_coverage_of_all_prs": len(classified) / len(pr_map)
            if len(pr_map)
            else None,
            "stable_with_observed_files": len(stable_observed),
            "stable_with_component": len(stable_classified),
            "stable_component_coverage": len(stable_classified) / len(stable_observed)
            if len(stable_observed)
            else None,
            "unclassified_with_files": len(unclassified),
            "unclassified_support_only": int(
                unclassified["support_surfaces"].map(bool).sum()
            ),
            "median_components_per_classified_pr": float(
                component_counts[component_counts > 0].median()
            )
            if len(classified)
            else None,
            "multi_component_prs": int((component_counts > 1).sum()),
            "multi_component_share_of_classified": float(
                (component_counts > 1).sum() / len(classified)
            )
            if len(classified)
            else None,
            "prevalence": prevalence.to_dict(orient="records"),
            "layer_totals_note": "Any-PR totals overlap; primary-PR totals do not.",
            "layer_totals": layer_counts.to_dict(orient="records"),
            "support_surfaces": support.to_dict(orient="records"),
            "hardware_backends_from_paths": sorted(
                backend_counts, key=lambda row: row["prs"], reverse=True
            ),
            "largest_component_overlaps": non_diagonal_overlap.head(20).to_dict(
                orient="records"
            ),
        },
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["pull_requests_2026"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
