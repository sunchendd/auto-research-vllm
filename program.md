# vLLM Throughput AutoResearch

This repo hosts an autonomous vLLM optimization research loop.

## Mission

Your job is to behave like an always-on research engineer for vLLM inference optimization:

1. Read recent papers on LLM serving, inference optimization, attention mechanisms, and KV cache.
2. Convert them into concrete benchmark ideas or vLLM code patches.
3. Validate on the local machine with **4x NVIDIA L20 (49GB each)**.
4. Compare all candidates against a `direct` baseline. TPS must improve to keep.
5. Record everything in `results.tsv`.
6. Identify novel ideas that may constitute patentable improvements — log in `patent_tracker.md`.
7. **Never stop unless manually interrupted.**

The primary metric is **TPS** (`tokens_per_second`). Higher is better.

---

## Research Domains

Prioritize papers across these vLLM optimization dimensions, roughly in impact order:

### Tier 1 — High Impact (try first)
- **Speculative Decoding**: suffix, EAGLE/EAGLE3, medusa, lookahead, multi-draft, async drafting
- **KV Cache Optimization**: prefix caching (RadixAttention), cross-request KV sharing, KV compression
- **Attention Kernels**: FlashAttention-3, FlashInfer, paged attention variants, sparse attention
- **Scheduling**: chunked prefill, continuous batching improvements, priority-aware scheduling

### Tier 2 — Medium Impact
- **Disaggregated Prefill/Decode (PD separation)**: mooncake-style, prefill-only instances
- **Quantization-Aware Serving**: FP8, AWQ, dynamic quant, mixed-precision KV cache
- **Memory Management**: block size tuning, KV cache pooling, CPU offloading
- **Parallelism**: tensor/pipeline/sequence parallel improvements, expert parallelism (MoE)

### Tier 3 — Emerging / Patent-Rich
- **Sparse KV Cache Eviction**: H2O, StreamingLLM, SnapKV, PyramidKV, InfiniGen
- **Asynchronous Decoding**: SwiftSpec, SSD-style async speculative, non-blocking draft
- **Token Budget / Adaptive Decoding**: dynamic token budget, confidence-based early exit
- **RAG Acceleration**: KV reuse across RAG documents, shared prefix for retrieval contexts

---

## Hardware

```
GPU 0: NVIDIA L20, 49140 MiB  (primary)
GPU 1: NVIDIA L20, 49140 MiB  (may be in use — check before assigning)
GPU 2: NVIDIA L20, 49140 MiB  (primary)
GPU 3: NVIDIA L20, 49140 MiB  (primary)
GPU 4: NVIDIA GeForce RTX 4090, 23028 MiB  (auxiliary only, do not use for main experiments)
```

**Default GPU allocation:**
- 8B models: `CUDA_VISIBLE_DEVICES=0,2` (TP=2) — use GPUs 0 and 2 by default
- 32B models: `CUDA_VISIBLE_DEVICES=0,2,3` (TP=3) or `0,1,2,3` (TP=4) if all L20s are free
- Single-GPU fast iteration: GPU 0 only

Always call `--setup` to check GPU status before running.

---

## Available Models

```
/data/models/Qwen3-8B                                # Main 8B target model
/data/models/Qwen3-32B                               # Main 32B target model
/data/models/Qwen3-0.6B                              # Tiny model for fast iteration
/data/models/AngelSlim-Qwen3-8B_eagle3               # Eagle3 draft for Qwen3-8B
/data/models/Qwen3-8B-speculator.eagle3              # Alternative eagle3 draft for Qwen3-8B
/data/models/RedHatAI-Qwen3-32B-speculator.eagle3    # Eagle3 draft for Qwen3-32B
/data/models/Qwen3-8B-DFlash-b16                     # DFlash variant (benchmark separately)
/data/models/Qwen3-Next-80B-A3B-Instruct             # Large MoE model (TP=4 required)
/data/models/amd-PARD-Qwen3-0.6B                     # Tiny model variant
```

---

## Files in Scope

Read these files before making changes:

- `README.md`
- `program.md`
- `speculative_loop.py`
- `speculative_benchmark.py`
- `research_manifest.json`
- `patent_tracker.md`

You may also inspect and modify:
- Installed vLLM source: `/usr/local/lib/python3.12/dist-packages/vllm/`
- vLLM benchmark helpers: `/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/`

## Files You May Edit

- `speculative_loop.py`
- `speculative_benchmark.py`
- `research_manifest.json`
- `speculative_candidates.json`
- `program.md`
- `patent_tracker.md`
- Any local helper scripts you create
- vLLM source code **only** when a paper requires an algorithmic patch

Do **not** modify `prepare.py` or `train.py`.

---

## Constraints

- Use only local assets and locally available Python packages.
- Do **not** rely on any external API key or hosted model endpoint.
- Prefer GPUs 0 and 2 for the continuous loop (avoids conflicting with GPU 1).
- If a method cannot be implemented faithfully, document the blocker and approximate.

---

## Results File

`results.tsv` is the score ledger. It must stay **untracked** (gitignored).

Header:
```
timestamp	branch	commit	model	case	tps	score	status	description
```

- `score` = `tps / direct_tps` from the same cycle (1.0 = baseline)
- `status`: `keep` (beats direct), `discard` (doesn't), `crash` (exception)

---

## The Research Loop

**Primary directive: implement algorithms from papers, not config knobs.**
Every new experiment must trace to a specific paper. Parameter tuning alone is insufficient.

---

### Step 1: Orient
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
python3 speculative_loop.py --setup --manifest research_manifest.json --results-path results.tsv
tail -20 results.tsv
```

---

### Step 2: Find a paper to implement

Search arxiv for recent (2023–2025) papers in these areas (in priority order):

**Tier 1 — High impact for serving**
- Speculative decoding variants: "speculative decoding 2024 2025 LLM serving"
- KV cache compression: "KV cache eviction quantization 2024"
- Attention kernels: "FlashInfer FlashAttention serving throughput"
- Draft model architecture: "eagle speculative decoding draft model"

**Tier 2 — Structural improvements**
- Disaggregated prefill/decode: "prefill decode disaggregation mooncake"
- Continuous batching improvements: "sarathi scheduling LLM"
- MoE-specific: "mixture of experts inference routing"

**vLLM 0.18.0 built-in methods worth testing (many unexplored):**
- `ngram_gpu` — GPU-accelerated n-gram (faster than CPU ngram at large batch)
- `parallel_drafting` — parallel draft generation flag in speculative_config
- `rejection_sample_method: "strict"` vs probabilistic — affects acceptance rate vs output quality
- Suffix decoding params: `suffix_decoding_max_tree_depth`, `suffix_decoding_max_spec_factor`
- MTP methods for Qwen3-Next-80B: `qwen3_next_mtp` (native multi-token prediction, no draft model)

Read: abstract + methods section. Extract the **one key algorithmic contribution**.

---

### Step 3: Classify and implement

#### Class A — `patch_module` implementation (preferred)
The algorithm can be implemented as a Python file in `implementations/`.

Create `implementations/<paper-name>.py` with this interface:
```python
def apply_patch() -> None:
    """Optional: monkey-patch vLLM before engine creation (env vars, module swaps)."""
    pass

def extra_engine_kwargs() -> dict:
    """Optional: additional EngineArgs kwargs."""
    return {}

def run_benchmark(args) -> dict | None:
    """Optional: fully replace the benchmark logic. Return None to use default."""
    return None
```

Add to manifest with `"patch_module": "implementations/<paper-name>.py"`.

**Examples already implemented:**
- `implementations/fp8_kv_cache.py` — FP8 KV quantization (KIVI paper)
- `implementations/flashinfer_backend.py` — FlashInfer attention backend
- `implementations/adaptive_spec_profiler.py` — online spec token auto-tuning
- `implementations/ngram_spec.py` — prompt-lookup n-gram speculation (REST paper)
- `implementations/h2o_kv_eviction.py` — H2O heavy hitter KV eviction (stub + full TODO)

#### Class B — vLLM source patch
The algorithm requires modifying vLLM's Python source at:
`/usr/local/lib/python3.12/dist-packages/vllm/`

Steps:
1. Identify the target file (e.g., `vllm/spec_decode/spec_decode_worker.py`)
2. Read the file, understand the data flow
3. Implement the minimal change that tests the algorithm
4. Document the patch in `implementations/<paper-name>.py` with:
   - `apply_patch()` that monkey-patches the target module at runtime
   - A comment block pointing to the modified function/class
5. Add to manifest as a patch_module experiment

**Prefer monkey-patching over editing the installed file directly.**
Monkey-patching is reversible; editing the installed file affects all experiments.

#### Class C — New benchmark script
The algorithm needs its own generation loop (e.g., speculative tree decoding, multi-draft).

Create `implementations/<paper-name>_bench.py` as a standalone script.
Add to manifest as `"patch_module": "implementations/<paper-name>_bench.py"` with a
`run_benchmark(args) -> dict` that returns the standard result format.

#### NOT ACCEPTABLE — Config-only experiments
Do not add experiments that only change `num_speculative_tokens`, `max_num_seqs`,
`enable_prefix_caching`, `batch_size`, or other existing vLLM flags without implementing
an algorithm. Those experiments were already explored and showed minimal gains.

---

### Step 4: Write the implementation

Minimum viable implementation:
1. File has a docstring with: paper title, arxiv link, key idea, why L20 is relevant
2. At least one of: `apply_patch()`, `extra_engine_kwargs()`, or `run_benchmark()`
3. A stub comment for anything not yet implemented (mark as `# TODO: full kernel implementation`)

Test the file loads without errors:
```bash
python3 -c "import importlib.util; spec=importlib.util.spec_from_file_location('p','implementations/<name>.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK')"
```

---

### Step 5: Add to manifest

```json
{
  "identifier": "qwen3-8b-<paper-name>-tp2",
  "case": "direct",
  "model": "/data/models/Qwen3-8B",
  "description": "[PaperTitle] One-line description of what this tests",
  "gpus": "0,2",
  "tensor_parallel_size": 2,
  "num_prompts": 8,
  "max_tokens": 128,
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.85,
  "patch_module": "implementations/<paper-name>.py"
}
```

Commit: `git add implementations/<paper-name>.py research_manifest.json && git commit -m "experiment: [PaperTitle] short description"`

---

### Step 6: Run and evaluate

```bash
python3 speculative_loop.py --once --manifest research_manifest.json --results-path results.tsv
```

Scoring:
- score > 1.10 → strong win, promote to "current best", also try stacking with other winners
- score 1.05–1.10 → meaningful, keep; consider why it works better on L20 than papers report
- score 1.00–1.05 → marginal; discard unless it enables future stacking
- score < 1.00 → discard; document WHY in the implementation file's docstring
- crash → read the error, fix or document the blocker, move to next paper

---

### Step 7: Patent check (every cycle)

After each cycle, ask:
1. Does this method outperform existing literature significantly **on L20 specifically**?
2. Is there a novel **combination** of techniques not described in any paper?
3. Did a failure reveal an **unexplored optimization opportunity** (negative result with mechanism)?
4. Does the method's behavior difference (L20 vs A100/H100 in papers) suggest a **hardware-specific patent**?

For YES → append to `patent_tracker.md` immediately.

---

### Step 8: Next paper selection

After evaluating results:
- If win: look for papers that **stack with** the winner (same dimension, compatible mechanism)
- If loss: understand the failure mechanism — often reveals a better paper to try next
- Remove discarded experiments from manifest to keep it focused
- Add at least one new paper experiment before the next cycle

---

## Baseline Knowledge (L20, TP=2, max_tokens=128) — updated 2026-03-30

### Qwen3-8B (batch=8, GPU 0+2)
| Case | TPS | Score | Method | Status |
|------|-----|-------|--------|--------|
| direct | 712 | 1.00 | baseline | — |
| eagle3 AngelSlim n=3 | 744 | **1.05** | speculative decoding | current best 8B |
| flashinfer direct | 724 | ~1.02 | FlashInfer attention backend | marginal, noisy |
| ngram n=3 | 630 | 0.88 | n-gram speculation | DISCARDED — wrong workload |
| ngram n=5 | 627 | 0.88 | n-gram speculation | DISCARDED |
| fp8kv direct | 718 | ~1.01 | FP8 KV quantization | neutral on 8B |
| fp8kv + eagle3 | 577 | 0.80 | FP8 KV + speculative | DISCARDED — incompatible on 8B |

**8B ceiling to beat:** 1.05x

### Qwen3-32B (batch=4, GPU 0+2, gpu_mem=0.90)
| Case | TPS | Score | Method | Status |
|------|-----|-------|--------|--------|
| direct | 95.4 | 1.00 | baseline | — |
| eagle3 RedHatAI n=3 | 124.2 | 1.30 | speculative decoding | previous best |
| fp8kv + eagle3 n=3 | 125.5 | 1.31 | FP8 KV + speculative | marginal gain |
| **adaptive spec (n=2)** | **130.7** | **1.37** | adaptive profiler | **CURRENT BEST 32B** |

**32B ceiling to beat:** 1.37x (adaptive n=2)

**Key findings from 2026-03-30 experiments:**
- Adaptive profiler correctly identifies n=2 as optimal for 32B (not n=3 as previously assumed)
- Optimal n is model-size-dependent: larger models → fewer spec tokens optimal
- FP8 KV + Eagle3 is incompatible on 8B (0.80x) but compatible on 32B (1.31x)
- N-gram speculation fails on instruction-following prompts — requires text-repetition workloads
- FlashInfer marginal on 8B direct (+1.8%) — worth testing stacked with Eagle3

**Next experiments in manifest (priority order):**
1. `qwen3-32b-fp8kv-adaptive-tp2` — FP8 KV + adaptive spec on 32B — could push past 1.37x
2. `qwen3-32b-eagle3-n2-tp2` — confirm n=2 is the cause of adaptive win (control experiment)
3. `qwen3-32b-flashinfer-eagle3-tp2` — FlashInfer + Eagle3 n=2 on 32B
4. `qwen3-8b-flashinfer-eagle3-tp2` — FlashInfer + Eagle3 on 8B (marginally positive stack)
5. `qwen3-8b-adaptive-flashinfer-tp2` — adaptive profiler under FlashInfer backend

---

## Failure Policy

- Run crashes → log `crash`, keep moving.
- Same failure 3x → document blocker in description, stop retrying.
- GPU busy → wait with `--sleep-seconds 60` or pick different GPU indices.
- Experiment >10 min → likely stuck, kill and reduce `num_prompts`/`max_tokens`.

---

## Non-stop Rule

Once the loop starts, **do not ask whether to continue**. Keep iterating until manually stopped.

---

## Starting the Loop

Setup check:
```bash
python3 speculative_loop.py --setup --manifest research_manifest.json --results-path results.tsv
```

Single cycle (debug/test):
```bash
python3 speculative_loop.py --once --manifest research_manifest.json --results-path results.tsv
```

Endless loop (production):
```bash
python3 speculative_loop.py --forever --manifest research_manifest.json --results-path results.tsv > research_loop.log 2>&1
```
