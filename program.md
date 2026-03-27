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

Follow this loop forever:

### Step 1: Orient
```bash
git status && git log --oneline -5
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
python3 speculative_loop.py --setup --manifest research_manifest.json --results-path results.tsv
```

### Step 2: Find a paper idea
Search arxiv and recent ML blogs for:
- "vLLM optimization", "LLM serving throughput", "speculative decoding 2025"
- "KV cache eviction", "attention approximation", "prefill disaggregation"
- "chunked prefill", "prefix caching", "flashinfer", "radix attention"

Read the abstract + methods section. Extract the **one key technical idea**.

Classify the idea:
- **Config-only**: can be expressed as flags to `speculative_benchmark.py` → add to manifest
- **Small code patch**: requires changes to `speculative_benchmark.py` or vLLM source
- **Approximation**: faithful implementation is blocked → find nearest proxy, document gap

### Step 3: Design and encode the experiment

For config-only ideas, add an entry to `research_manifest.json`:
```json
{
  "identifier": "qwen3-8b-<method>-tp2",
  "case": "<method>",
  "model": "/data/models/Qwen3-8B",
  "description": "<paper title / technique / what you expect>",
  "gpus": "0,2",
  "tensor_parallel_size": 2,
  "num_prompts": 8,
  "max_tokens": 128,
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.85
}
```

For code patches, implement in `speculative_benchmark.py` or vLLM source, then add to manifest.

### Step 4: Commit the experiment idea
```bash
git add research_manifest.json speculative_benchmark.py  # whichever files changed
git commit -m "experiment: <short description of paper/idea>"
```

### Step 5: Run one cycle
```bash
python3 speculative_loop.py --once --manifest research_manifest.json --results-path results.tsv
```

### Step 6: Analyze results
- Read the new rows in `results.tsv`
- score > 1.05 → meaningful improvement, keep
- score 1.0–1.05 → marginal, discard unless it enables future stacking
- score < 1.0 → discard

### Step 7: Patent check (do this every cycle)
After each cycle, evaluate:
1. Does this method outperform existing literature significantly on L20?
2. Is there a novel **combination** of techniques not described in any paper?
3. Did a failure reveal an **unexplored optimization opportunity** (negative result with insight)?
4. Does the method's behavior on L20 (vs A100/H100 in papers) suggest a hardware-specific patent?

For any YES → append to `patent_tracker.md` immediately.

### Step 8: Advance or discard
- Improved: keep commit, update baseline knowledge section in `program.md`
- Did not improve: `git revert HEAD` or manually undo manifest/code changes
- Repeat from Step 1

---

## Baseline Knowledge (L20, TP=2)

Current measured TPS on this machine:

| Model | Case | TPS | Score | Config |
|-------|------|-----|-------|--------|
| Qwen3-8B | direct | 213.58 | 1.00 | TP=2, GPU 0+2, num_prompts=2, max_tokens=32 |
| Qwen3-8B | suffix | 533.82 | 2.50 | TP=2, GPU 0+2 |
| Qwen3-8B | eagle3 (AngelSlim) | 175.18 | 0.82 | TP=2, GPU 0+2, draft=AngelSlim |

**Key observations:**
- suffix delivers 2.5x on L20 — currently the strongest method
- eagle3 underperforms direct — possible optimization target (draft model quality? token budget?)
- All measurements use num_prompts=2, max_tokens=32 — consider larger batch for throughput research

**Next experiments to try (priority order):**
1. eagle3 with `Qwen3-8B-speculator.eagle3` (different draft model) — may outperform AngelSlim
2. suffix + prefix caching combined (enable_prefix_caching=True with suffix)
3. chunked_prefill baseline — enable chunked prefill, measure throughput on longer prompts
4. Qwen3-32B direct + suffix + eagle3 (RedHatAI draft) — 32B baseline on TP=3/4
5. Increase num_prompts to 16-32 for more realistic throughput measurement

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
