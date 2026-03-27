from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs


PROMPT_BANK = [
    "Summarize the main engineering tradeoffs of speculative decoding in two sentences.",
    "Write three concise bullet points explaining why suffix decoding can improve throughput.",
    "Describe a reproducible benchmark plan for comparing direct decoding and EAGLE-3 on local GPUs.",
    "Explain how tensor parallelism changes the setup considerations for offline vLLM benchmarking.",
    "What are the key differences between paged attention and standard attention for LLM serving?",
    "Explain how prefix caching (RadixAttention) reduces redundant KV cache computation.",
    "Describe the tradeoffs between chunked prefill and standard prefill in continuous batching.",
    "How does disaggregated prefill-decode improve LLM serving throughput and latency?",
    "What is the key idea behind H2O (Heavy Hitter Oracle) for KV cache eviction?",
    "Explain the memory bandwidth vs compute tradeoffs for LLM inference on different GPU architectures.",
    "How does FlashInfer differ from FlashAttention for serving workloads?",
    "What optimizations does vLLM's continuous batching provide over static batching?",
]


def benchmark_prompts(num_prompts: int) -> list[str]:
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive.")
    return [PROMPT_BANK[index % len(PROMPT_BANK)] for index in range(num_prompts)]


# Cases that use speculative config; all others are treated as engine-flag experiments.
_SPECULATIVE_CASES = {"suffix", "eagle3", "medusa", "lookahead", "ngram"}
# Cases that do not use speculative config but require special engine flags.
_ENGINE_FLAG_CASES = {"direct", "prefix_cache", "chunked_prefill"}


def build_speculative_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.case not in _SPECULATIVE_CASES:
        return None

    config: dict[str, Any] = {}
    if args.case == "suffix":
        config["method"] = "suffix"
    elif args.case == "eagle3":
        if not args.draft_model:
            raise ValueError("eagle3 runs require --draft-model.")
        config.update({"method": "eagle3", "model": args.draft_model})
    else:
        config["method"] = args.case
        if args.draft_model:
            config["model"] = args.draft_model

    if args.num_speculative_tokens is not None:
        config["num_speculative_tokens"] = args.num_speculative_tokens

    if args.speculative_config_json:
        config.update(json.loads(args.speculative_config_json))

    return config or None


def build_engine_args(
    args: argparse.Namespace,
    speculative_config: dict[str, Any] | None,
) -> EngineArgs:
    kwargs: dict[str, Any] = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        speculative_config=speculative_config,
        disable_log_stats=True,
        use_tqdm_on_load=False,
        performance_mode="throughput",
    )

    if args.enable_prefix_caching:
        kwargs["enable_prefix_caching"] = True

    if args.enable_chunked_prefill:
        kwargs["enable_chunked_prefill"] = True
        if args.max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs

    return EngineArgs(**kwargs)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    speculative_config = build_speculative_config(args)
    engine_args = build_engine_args(args, speculative_config)
    llm = LLM(**engine_args.__dict__)

    warmup_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=min(4, args.max_tokens),
    )
    llm.generate([PROMPT_BANK[0]], warmup_params, use_tqdm=False)

    prompts = benchmark_prompts(args.num_prompts)
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=args.max_tokens,
        )
        for _ in prompts
    ]

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = generated_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "identifier": args.identifier,
        "case": args.case,
        "model": args.model,
        "draft_model": args.draft_model,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "elapsed_time": elapsed,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "speculative_config": speculative_config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM throughput benchmark — speculative and general.")
    parser.add_argument("--identifier", required=True)
    parser.add_argument("--case", required=True,
                        help="Experiment case: direct, suffix, eagle3, prefix_cache, chunked_prefill, ...")
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    parser.add_argument("--speculative-config-json", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    # New flags for broader optimization research
    parser.add_argument("--enable-prefix-caching", action="store_true",
                        help="Enable prefix caching (RadixAttention).")
    parser.add_argument("--enable-chunked-prefill", action="store_true",
                        help="Enable chunked prefill.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Max batched tokens per iteration (used with chunked prefill).")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Max number of sequences in a batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
