from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RESULTS_HEADER = (
    "timestamp\tbranch\tcommit\tmodel\tcase\ttps\tscore\tstatus\tdescription"
)
LOCAL_BENCHMARK_SCRIPT = Path(__file__).with_name("speculative_benchmark.py")
GPU_IDLE_UTILIZATION_THRESHOLD = 10


@dataclass
class ExperimentSpec:
    identifier: str
    case: str
    model: str
    description: str
    gpus: str
    tensor_parallel_size: int = 1
    num_prompts: int = 8
    max_tokens: int = 128
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    draft_model: str | None = None
    num_speculative_tokens: int | None = None
    enforce_eager: bool = False
    speculative_config: dict[str, Any] | None = None
    # General optimization flags
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None


@dataclass
class ExperimentRecord:
    timestamp: str
    branch: str
    commit: str
    model: str
    case: str
    tps: float
    score: float
    status: str
    description: str


@dataclass
class GPUStatus:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_gpu: int
    active_compute_processes: int = 0


def ensure_results_tsv(results_path: str | Path) -> None:
    path = Path(results_path)
    if not path.exists():
        path.write_text(RESULTS_HEADER + "\n", encoding="utf-8")


def append_result(results_path: str | Path, record: ExperimentRecord) -> None:
    path = Path(results_path)
    ensure_results_tsv(path)
    row = (
        f"{record.timestamp}\t{record.branch}\t{record.commit}\t{record.model}\t"
        f"{record.case}\t{record.tps:.2f}\t{record.score:.2f}\t"
        f"{record.status}\t{record.description}\n"
    )
    path.write_text(path.read_text(encoding="utf-8") + row, encoding="utf-8")


def compute_score(tps: float, baseline_tps: float) -> float:
    if baseline_tps <= 0:
        return 0.0
    return tps / baseline_tps


def parse_gpu_indices(gpus: str) -> list[int]:
    return [int(token.strip()) for token in gpus.split(",") if token.strip()]


def build_benchmark_command(
    spec: ExperimentSpec,
    output_json: str | Path,
) -> list[str]:
    command = [
        "python3",
        str(LOCAL_BENCHMARK_SCRIPT),
        "--identifier",
        spec.identifier,
        "--case",
        spec.case,
        "--model",
        spec.model,
        "--num-prompts",
        str(spec.num_prompts),
        "--max-tokens",
        str(spec.max_tokens),
        "--max-model-len",
        str(spec.max_model_len),
        "--gpu-memory-utilization",
        str(spec.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(spec.tensor_parallel_size),
        "--output-json",
        str(output_json),
    ]
    if spec.draft_model:
        command.extend(["--draft-model", spec.draft_model])
    if spec.num_speculative_tokens is not None:
        command.extend(["--num-speculative-tokens", str(spec.num_speculative_tokens)])
    if spec.enforce_eager:
        command.append("--enforce-eager")
    if spec.speculative_config:
        command.extend(
            [
                "--speculative-config-json",
                json.dumps(spec.speculative_config, sort_keys=True),
            ]
        )
    if spec.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if spec.enable_chunked_prefill:
        command.append("--enable-chunked-prefill")
    if spec.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", str(spec.max_num_batched_tokens)])
    if spec.max_num_seqs is not None:
        command.extend(["--max-num-seqs", str(spec.max_num_seqs)])
    return command


def default_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            identifier="qwen3-8b-direct-tp2",
            case="direct",
            model="/data/models/Qwen3-8B",
            description="L20 direct baseline TP=2",
            gpus="0,2",
            tensor_parallel_size=2,
            num_prompts=8,
            max_tokens=128,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
        ),
        ExperimentSpec(
            identifier="qwen3-8b-suffix-tp2",
            case="suffix",
            model="/data/models/Qwen3-8B",
            description="SuffixDecoding on L20 TP=2",
            gpus="0,2",
            tensor_parallel_size=2,
            num_prompts=8,
            max_tokens=128,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
        ),
        ExperimentSpec(
            identifier="qwen3-8b-eagle3-angelslim-tp2",
            case="eagle3",
            model="/data/models/Qwen3-8B",
            description="EAGLE3 AngelSlim draft on L20 TP=2",
            gpus="0,2",
            tensor_parallel_size=2,
            num_prompts=8,
            max_tokens=128,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            draft_model="/data/models/AngelSlim-Qwen3-8B_eagle3",
            num_speculative_tokens=3,
        ),
    ]


def load_manifest(manifest_path: str | Path | None) -> list[ExperimentSpec]:
    if manifest_path is None:
        experiments = default_experiments()
    else:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Manifest must contain a JSON list of experiments.")
        experiments = [ExperimentSpec(**item) for item in data]
    if not experiments:
        raise ValueError("Manifest must contain at least one experiment.")
    return experiments


def current_git_state() -> tuple[str, str]:
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return branch, commit
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def git_status_lines() -> list[str]:
    try:
        output = subprocess.run(
            ["git", "--no-pager", "status", "--short", "--branch"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return [line for line in output.splitlines() if line]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def benchmark_runtime_error() -> str | None:
    if not LOCAL_BENCHMARK_SCRIPT.exists():
        return f"Local benchmark helper is missing: {LOCAL_BENCHMARK_SCRIPT}"
    if importlib.util.find_spec("vllm") is None:
        return "Python package 'vllm' is not importable."
    return None


def query_gpu_status() -> dict[int, GPUStatus]:
    inventory_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    status_by_index: dict[int, GPUStatus] = {}
    for row in csv.reader(inventory_output.splitlines()):
        if not row:
            continue
        index = int(row[0].strip())
        status_by_index[index] = GPUStatus(
            index=index,
            uuid=row[1].strip(),
            name=row[2].strip(),
            memory_used_mib=int(row[3].strip()),
            memory_total_mib=int(row[4].strip()),
            utilization_gpu=int(row[5].strip()),
        )

    app_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    uuid_to_index = {status.uuid: index for index, status in status_by_index.items()}
    for row in csv.reader(app_output.splitlines()):
        if len(row) < 2:
            continue
        gpu_uuid = row[0].strip()
        pid = row[1].strip()
        if not pid.isdigit():
            continue
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is not None:
            status_by_index[gpu_index].active_compute_processes += 1
    return status_by_index


def busy_gpu_indices(
    gpu_status: dict[int, GPUStatus],
    experiments: list[ExperimentSpec],
) -> list[int]:
    required_utilization_by_gpu: dict[int, float] = {}
    for spec in experiments:
        for index in parse_gpu_indices(spec.gpus):
            current = required_utilization_by_gpu.get(index, 0.0)
            required_utilization_by_gpu[index] = max(current, spec.gpu_memory_utilization)

    busy: list[int] = []
    for index, required_utilization in required_utilization_by_gpu.items():
        status = gpu_status.get(index)
        if status is None:
            continue
        free_memory_mib = status.memory_total_mib - status.memory_used_mib
        required_free_memory_mib = int(status.memory_total_mib * required_utilization)
        if (
            status.active_compute_processes > 0
            or status.utilization_gpu > GPU_IDLE_UTILIZATION_THRESHOLD
            or free_memory_mib < required_free_memory_mib
        ):
            busy.append(index)
    return busy


def run_setup_checks(
    experiments: list[ExperimentSpec],
    manifest_path: str | Path | None,
    results_path: str | Path,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    branch = ""
    commit = ""

    ensure_results_tsv(results_path)

    runtime_error = benchmark_runtime_error()
    if runtime_error is not None:
        errors.append(runtime_error)

    branch, commit = current_git_state()
    if branch == "unknown":
        warnings.append("Git not available; branch/commit tracking disabled.")
    else:
        status_lines = git_status_lines()
        if any(not line.startswith("##") for line in status_lines):
            warnings.append("Git worktree is dirty; preserving existing changes.")

    if not any(spec.case == "direct" for spec in experiments):
        errors.append("Manifest must include a direct baseline experiment.")

    for spec in experiments:
        if not Path(spec.model).exists():
            errors.append(f"Missing target model for {spec.identifier}: {spec.model}")
        if spec.draft_model and not Path(spec.draft_model).exists():
            errors.append(
                f"Missing draft model for {spec.identifier}: {spec.draft_model}"
            )

    selected = sorted({gpu for spec in experiments for gpu in parse_gpu_indices(spec.gpus)})
    gpu_summary: dict[str, dict[str, Any]] = {}
    busy: list[int] = []
    try:
        gpu_status = query_gpu_status()
        missing_gpus = [index for index in selected if index not in gpu_status]
        if missing_gpus:
            errors.append(f"Requested GPUs are unavailable: {missing_gpus}")
        busy = busy_gpu_indices(gpu_status, experiments)
        if busy:
            warnings.append(f"Requested GPUs are busy right now: {busy}")
        gpu_summary = {
            str(index): {
                "name": gpu_status[index].name,
                "memory_used_mib": gpu_status[index].memory_used_mib,
                "memory_total_mib": gpu_status[index].memory_total_mib,
                "memory_free_mib": gpu_status[index].memory_total_mib
                - gpu_status[index].memory_used_mib,
                "utilization_gpu": gpu_status[index].utilization_gpu,
                "active_compute_processes": gpu_status[index].active_compute_processes,
            }
            for index in selected
            if index in gpu_status
        }
    except Exception as exc:
        errors.append(f"Unable to query GPUs with nvidia-smi: {type(exc).__name__}: {exc}")

    return {
        "ready": not errors,
        "manifest": str(manifest_path) if manifest_path is not None else "<default>",
        "results_path": str(results_path),
        "benchmark_script": str(LOCAL_BENCHMARK_SCRIPT),
        "branch": branch,
        "commit": commit,
        "selected_gpus": selected,
        "busy_gpus": busy,
        "can_run_now": not busy,
        "gpu_status": gpu_summary,
        "warnings": warnings,
        "errors": errors,
    }


def wait_for_idle_gpus(experiments: list[ExperimentSpec], sleep_seconds: int) -> None:
    while True:
        busy = busy_gpu_indices(query_gpu_status(), experiments)
        if not busy:
            return
        print(
            f"[{datetime.now(UTC).isoformat()}] GPUs {busy} are busy; "
            f"waiting {sleep_seconds}s before retrying.",
            flush=True,
        )
        time.sleep(sleep_seconds)


def run_experiment(
    spec: ExperimentSpec,
    artifacts_dir: str | Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    output_json = Path(artifacts_dir) / f"{spec.identifier}.json"
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = spec.gpus
    command = build_benchmark_command(spec, output_json)
    subprocess.run(command, check=True, env=env, timeout=timeout_seconds)
    return json.loads(output_json.read_text(encoding="utf-8"))


def record_result(
    results_path: str | Path,
    branch: str,
    commit: str,
    spec: ExperimentSpec,
    tps: float,
    score: float,
    status: str,
    description: str | None = None,
) -> None:
    append_result(
        results_path,
        ExperimentRecord(
            timestamp=datetime.now(UTC).isoformat(),
            branch=branch,
            commit=commit,
            model=spec.model,
            case=spec.case,
            tps=tps,
            score=score,
            status=status,
            description=description or spec.description,
        ),
    )


def run_cycle(
    experiments: list[ExperimentSpec],
    results_path: str | Path,
    artifacts_dir: str | Path,
    timeout_seconds: int,
) -> None:
    ensure_results_tsv(results_path)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    branch, commit = current_git_state()
    baseline_tps = 0.0
    for spec in experiments:
        print(
            f"[{datetime.now(UTC).isoformat()}] Running {spec.identifier} ({spec.case})",
            flush=True,
        )
        try:
            result = run_experiment(spec, artifacts_dir, timeout_seconds)
            tps = float(result["tokens_per_second"])
            if spec.case == "direct" or baseline_tps <= 0:
                baseline_tps = tps
                score = 1.0 if spec.case == "direct" else compute_score(tps, baseline_tps)
                status = "keep"
            else:
                score = compute_score(tps, baseline_tps)
                status = "keep" if score > 1.0 else "discard"
            record_result(results_path, branch, commit, spec, tps, score, status)
            print(
                f"[{datetime.now(UTC).isoformat()}] Finished {spec.identifier}: "
                f"{tps:.2f} tok/s, score={score:.2f}, status={status}",
                flush=True,
            )
        except Exception as exc:  # surfaced into results.tsv as crash
            crash_description = f"{spec.description} | crash: {type(exc).__name__}: {exc}"
            record_result(
                results_path,
                branch,
                commit,
                spec,
                0.0,
                0.0,
                "crash",
                description=crash_description,
            )
            print(
                f"[{datetime.now(UTC).isoformat()}] Crash in {spec.identifier}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM throughput research loop.")
    parser.add_argument("--manifest", default="research_manifest.json",
                        help="Path to experiment manifest JSON (default: research_manifest.json)")
    parser.add_argument("--results-path", default="results.tsv")
    parser.add_argument("--artifacts-dir", default="speculative_runs")
    parser.add_argument("--timeout-seconds", type=int, default=900,
                        help="Per-experiment timeout in seconds (default: 900)")
    parser.add_argument("--sleep-seconds", type=int, default=30)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--forever", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = load_manifest(args.manifest)
    setup_summary = run_setup_checks(experiments, args.manifest, args.results_path)
    print(json.dumps(setup_summary, indent=2), flush=True)
    if not setup_summary["ready"]:
        raise SystemExit(1)
    if args.setup:
        return

    run_once = args.once or not args.forever
    if run_once:
        wait_for_idle_gpus(experiments, args.sleep_seconds)
        run_cycle(experiments, args.results_path, args.artifacts_dir, args.timeout_seconds)
        return

    while True:
        wait_for_idle_gpus(experiments, args.sleep_seconds)
        run_cycle(experiments, args.results_path, args.artifacts_dir, args.timeout_seconds)
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
