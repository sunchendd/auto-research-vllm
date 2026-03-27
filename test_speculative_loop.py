import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speculative_loop import (
    LOCAL_BENCHMARK_SCRIPT,
    ExperimentRecord,
    ExperimentSpec,
    GPUStatus,
    append_result,
    busy_gpu_indices,
    build_benchmark_command,
    compute_score,
    ensure_results_tsv,
    run_setup_checks,
)


class SpeculativeLoopTests(unittest.TestCase):
    def test_build_benchmark_command_includes_parallel_draft_and_local_helper(self) -> None:
        spec = ExperimentSpec(
            identifier="qwen3-8b-eagle3-tp2",
            case="eagle3",
            model="/data/models/Qwen3-8B",
            description="dual-5090 eagle3 baseline",
            gpus="0,1",
            tensor_parallel_size=2,
            num_prompts=2,
            max_tokens=32,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            draft_model="/data/models/AngelSlim-Qwen3-8B_eagle3",
            num_speculative_tokens=3,
            enforce_eager=True,
            speculative_config={"suffix_decoding_max_tree_depth": 8},
        )

        command = build_benchmark_command(spec, Path("/tmp/out.json"))

        self.assertEqual(command[:2], ["python3", str(LOCAL_BENCHMARK_SCRIPT)])
        self.assertIn("--draft-model", command)
        self.assertIn("/data/models/AngelSlim-Qwen3-8B_eagle3", command)
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("2", command)
        self.assertIn("--num-speculative-tokens", command)
        self.assertIn("3", command)
        self.assertIn("--enforce-eager", command)
        config_index = command.index("--speculative-config-json") + 1
        self.assertEqual(
            json.loads(command[config_index]),
            {"suffix_decoding_max_tree_depth": 8},
        )

    def test_results_tsv_header_and_row_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"

            ensure_results_tsv(results_path)
            append_result(
                results_path,
                ExperimentRecord(
                    timestamp="2026-03-13T05:00:00Z",
                    branch="autoresearch/specdec-demo",
                    commit="abc1234",
                    model="/data/models/Qwen3-8B",
                    case="suffix",
                    tps=160.14,
                    score=1.94,
                    status="keep",
                    description="suffix decoding beats direct baseline",
                ),
            )

            lines = results_path.read_text().splitlines()

        self.assertEqual(
            lines[0],
            "timestamp\tbranch\tcommit\tmodel\tcase\ttps\tscore\tstatus\tdescription",
        )
        self.assertIn("suffix", lines[1])
        self.assertIn("160.14", lines[1])
        self.assertIn("1.94", lines[1])
        self.assertTrue(lines[1].endswith("suffix decoding beats direct baseline"))

    def test_compute_score_compares_against_direct_baseline(self) -> None:
        self.assertAlmostEqual(compute_score(tps=80.93, baseline_tps=39.56), 2.05, places=2)
        self.assertEqual(compute_score(tps=39.56, baseline_tps=0.0), 0.0)

    def test_busy_gpu_indices_uses_processes_and_utilization(self) -> None:
        experiments = [
            ExperimentSpec(
                identifier="direct",
                case="direct",
                model="/data/models/Qwen3-8B",
                description="direct baseline",
                gpus="0,1,2",
                gpu_memory_utilization=0.5,
            )
        ]
        status = {
            0: GPUStatus(
                index=0,
                uuid="gpu-0",
                name="RTX 5090",
                memory_used_mib=24000,
                memory_total_mib=24576,
                utilization_gpu=0,
                active_compute_processes=0,
            ),
            1: GPUStatus(
                index=1,
                uuid="gpu-1",
                name="RTX 5090",
                memory_used_mib=100,
                memory_total_mib=24576,
                utilization_gpu=25,
                active_compute_processes=0,
            ),
            2: GPUStatus(
                index=2,
                uuid="gpu-2",
                name="RTX 5090",
                memory_used_mib=100,
                memory_total_mib=24576,
                utilization_gpu=0,
                active_compute_processes=1,
            ),
        }
        self.assertEqual(busy_gpu_indices(status, experiments), [0, 1, 2])

    def test_run_setup_checks_creates_results_and_reports_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            draft_dir = Path(tmpdir) / "draft"
            results_path = Path(tmpdir) / "results.tsv"
            model_dir.mkdir()
            draft_dir.mkdir()
            experiments = [
                ExperimentSpec(
                    identifier="direct",
                    case="direct",
                    model=str(model_dir),
                    description="direct baseline",
                    gpus="0,1",
                ),
                ExperimentSpec(
                    identifier="eagle3",
                    case="eagle3",
                    model=str(model_dir),
                    description="eagle candidate",
                    gpus="0,1",
                    draft_model=str(draft_dir),
                    num_speculative_tokens=3,
                ),
            ]

            with (
                patch("speculative_loop.benchmark_runtime_error", return_value=None),
                patch("speculative_loop.current_git_state", return_value=("master", "abc1234")),
                patch("speculative_loop.git_status_lines", return_value=["## master"]),
                patch(
                    "speculative_loop.query_gpu_status",
                    return_value={
                        0: GPUStatus(
                            index=0,
                            uuid="gpu-0",
                            name="RTX 5090",
                            memory_used_mib=0,
                            memory_total_mib=24576,
                            utilization_gpu=0,
                            active_compute_processes=0,
                        ),
                        1: GPUStatus(
                            index=1,
                            uuid="gpu-1",
                            name="RTX 5090",
                            memory_used_mib=0,
                            memory_total_mib=24576,
                            utilization_gpu=0,
                            active_compute_processes=0,
                        ),
                    },
                ),
            ):
                summary = run_setup_checks(experiments, "manifest.json", results_path)
                header = results_path.read_text().splitlines()[0]

        self.assertTrue(summary["ready"])
        self.assertTrue(summary["can_run_now"])
        self.assertEqual(summary["busy_gpus"], [])
        self.assertEqual(
            header,
            "timestamp\tbranch\tcommit\tmodel\tcase\ttps\tscore\tstatus\tdescription",
        )

    def test_run_setup_checks_requires_direct_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            experiments = [
                ExperimentSpec(
                    identifier="suffix-only",
                    case="suffix",
                    model=str(model_dir),
                    description="suffix candidate",
                    gpus="0",
                )
            ]

            with (
                patch("speculative_loop.benchmark_runtime_error", return_value=None),
                patch("speculative_loop.current_git_state", return_value=("master", "abc1234")),
                patch("speculative_loop.git_status_lines", return_value=["## master"]),
                patch(
                    "speculative_loop.query_gpu_status",
                    return_value={
                        0: GPUStatus(
                            index=0,
                            uuid="gpu-0",
                            name="RTX 5090",
                            memory_used_mib=0,
                            memory_total_mib=24576,
                            utilization_gpu=0,
                            active_compute_processes=0,
                        )
                    },
                ),
            ):
                summary = run_setup_checks(experiments, "manifest.json", Path(tmpdir) / "results.tsv")

        self.assertFalse(summary["ready"])
        self.assertIn("Manifest must include a direct baseline experiment.", summary["errors"])


if __name__ == "__main__":
    unittest.main()
