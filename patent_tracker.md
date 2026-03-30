# Patent Ideas Tracker

Record novel observations, unexpected results, and optimization ideas that may be patentable.
Each entry should be actionable — describe the observation, why it's novel, and the claim.

---

## How to Add an Entry

After each experiment cycle, ask:
1. Does this method work significantly better than expected on L20?
2. Is this a novel combination of techniques not described in existing papers?
3. Does a failure reveal an unexplored optimization opportunity?
4. Does behavior on L20 (vs A100/H100 in papers) suggest hardware-specific optimization?

If yes → add an entry below.

---

## Entry Format

```markdown
### [DATE] Title of the idea

**Observation**: What happened / what was measured.
**Why novel**: What existing work doesn't cover this.
**Potential claim**: The specific technical contribution.
**Evidence**: Experiment IDs and TPS numbers from results.tsv.
**Priority**: High / Medium / Low
**Status**: Idea / Investigating / Validated / Filed
```

---

## Ideas

### [2026-03-27] Eagle3 underperforms direct on L20 TP=2

**Observation**: Eagle3 with AngelSlim draft achieves 175.18 TPS vs direct 213.58 TPS on L20 TP=2.
This is 0.82x baseline — speculative decoding is making things **worse**.

**Why novel**: Most eagle3 benchmarks are on A100/H100. L20 has different memory bandwidth
characteristics (1.6x memory BW of A100 but lower compute TFLOPs for small models).
The draft model verification overhead may dominate on L20 at low batch sizes.

**Potential claim**: A hardware-adaptive speculative decoding scheduler that dynamically
disables/enables draft-model speculation based on GPU memory bandwidth-to-compute ratio,
measured at runtime. Could patent the decision boundary metric.

**Evidence**: speculative_runs/qwen3-8b-eagle3-tp2.json (175.18 TPS) vs qwen3-8b-direct-tp2.json (213.58 TPS)

**Priority**: High
**Status**: Idea

---

### [2026-03-27] Suffix decoding 2.5x speedup on L20 — mechanism unclear

**Observation**: Suffix decoding achieves 533.82 TPS (2.5x direct) on L20 TP=2, Qwen3-8B.
Historical single-GPU result was only 1.94x on 5090. L20 shows stronger relative gain.

**Why novel**: The L20-vs-5090 difference suggests suffix decoding may be particularly
well-suited for HBM2e memory bandwidth profile of L20. If the mechanism is memory-bound
verification rather than compute-bound, this suggests a family of "bandwidth-optimized"
speculative decoders specifically tuned for server-class GPUs with high memory bandwidth.

**Potential claim**: A speculative decoding method that optimizes draft-token acceptance
for high-bandwidth/lower-compute GPU architectures (L20, A800) by increasing speculation
depth to saturate memory bandwidth.

**Evidence**: suffix 533.82 TPS on L20 vs 160.14 TPS historical on single GPU (different system).

**Priority**: Medium
**Status**: Idea

---

---

### [2026-03-27] Suffix decoding batch-size inversion on L20

**Observation**: Suffix decoding achieves 2.5x at num_prompts=2 but only 0.86x at num_prompts=8 on L20 TP=2 (Qwen3-8B). A crossover point exists between batch=2 and batch=8 where suffix decoding switches from beneficial to harmful.

**Why novel**: No published paper characterizes the batch-size crossover for suffix decoding. The mechanism is likely that at higher concurrency, the suffix prefix-tree lookup overhead and memory pressure from the suffix store dominates the draft acceptance gains. This crossover is hardware- and model-dependent.

**Potential claim**: A dynamic suffix decoding scheduler that enables/disables suffix decoding based on real-time queue depth or batch size measurement, using a learned or profiled crossover threshold. Could also patent a "suffix decoding profiler" that measures the crossover at engine startup.

**Evidence**: batch=2 → 533 TPS (2.5x), batch=8 → 614 TPS (0.86x) on same L20 TP=2 Qwen3-8B setup.

**Priority**: High
**Status**: Idea

---

### [2026-03-27] Eagle3 batch-size scalability advantage on L20

**Observation**: Eagle3 (AngelSlim draft) switches from 0.82x at batch=2 to 1.05x at batch=8 for Qwen3-8B. For Qwen3-32B, eagle3 (RedHatAI draft) achieves 1.30x at batch=4. Eagle3 improves *with* batch size while suffix degrades.

**Why novel**: The batch-size scalability difference between eagle3 and suffix is not characterized in literature. The mechanism may be that eagle3's draft verification parallelizes efficiently with larger batches (same GPU kernel), while suffix decoding's sequential prefix-tree lookup creates a bottleneck.

**Potential claim**: A serving scheduler that selects between speculative decoding methods (eagle3 vs suffix vs direct) based on batch size, dynamically switching at runtime to maximize throughput. Could be implemented as a meta-scheduler sitting above vLLM's request scheduler.

**Evidence**: eagle3 8B: 0.82x→1.05x from batch=2 to batch=8; suffix 8B: 2.5x→0.86x same transition.

**Priority**: High
**Status**: Idea

---

### [2026-03-30] Attention backend determines optimal num_speculative_tokens — adaptive profiler critical

**Observation**: The optimal num_speculative_tokens is NOT fixed — it depends on the attention
backend, not just the model size. FlashInfer shifts the optimal n for 8B Eagle3 from 3 → 1:
  - FlashAttention default + Eagle3 n=3: 744 TPS (1.05x)
  - FlashInfer + Eagle3 **fixed** n=3: 607 TPS (0.85x) — WORSE than baseline
  - FlashInfer + Eagle3 **adaptive** (selected n=1): 821 TPS (1.15x) — NEW BEST

The adaptive profiler was the only way to discover this. Without it, FlashInfer would have been
incorrectly discarded as incompatible with speculative decoding.

**Why novel**: The interaction between attention kernel implementation and speculative decoding
performance is unstudied. FlashInfer makes verification faster relative to drafting, shifting
the optimal speculation depth. This creates a 3-way dependency: (target model, draft model,
attention backend) all jointly determine the optimal strategy.

**Potential claim**: A serving system that co-optimizes attention backend selection and
speculative token count via joint profiling, rather than treating them as independent
hyperparameters.

**Evidence** (3 consecutive cycles, all consistent):
  - adaptive_flashinfer 8B: 821 TPS ± 0.1 (1.15x) — confirmed stable
  - flashinfer + fixed n=3 8B: 607–629 TPS (0.85–0.88x) — confirmed failure

**Priority**: Very High — directly enables patent claim
**Status**: Validated (3 cycles)

---

### [2026-03-30] Adaptive speculative token count outperforms fixed n on 32B

**Observation**: Adaptive profiler selected n=2 for Qwen3-32B (vs conventional n=3), achieving
130.70 TPS (1.37x) vs fixed n=3 at 124.23 TPS (1.30x). The profiling batch showed:
n=2 (112.5) > n=5 (110.4) > n=1 (108.9) > n=3 (107.6) > n=7 (104.8) TPS.
For 8B the ordering was: n=3 > n=2 > n=1 > n=5 > n=7.

**Why novel**: The optimal num_speculative_tokens is model-size-dependent and cannot be
determined from model architecture alone. Larger models have slower target-model forward
passes, shifting the optimal towards fewer speculative tokens (lower rejection cost relative
to verification cost). No existing paper characterizes the optimal n as a function of
target/draft model size ratio.

**Potential claim**: A workload-adaptive speculative decoding system that profiles n values
at engine startup to select the Pareto-optimal num_speculative_tokens for the specific
(target model, draft model, hardware, batch size) tuple. Patent covers the profiling
algorithm and the model-size-to-optimal-n relationship.

**Evidence**: 32B n=2 adaptive: 130.70 TPS (1.37x) vs 32B n=3 fixed: 124.23 TPS (1.30x).
8B: n=3 optimal as before. Implemented in implementations/adaptive_spec_profiler.py.

**Priority**: High
**Status**: Validated

---

### [2026-03-30] FP8 KV cache + Eagle3 incompatibility is model-size-dependent

**Observation**: FP8 KV cache (fp8_e5m2) combined with Eagle3 speculative decoding gives
0.80x on Qwen3-8B but +1.31x on Qwen3-32B (marginally better than Eagle3 alone 1.30x).
The 8B result suggests a vLLM-level incompatibility or significant precision degradation
when FP8 KV is combined with Eagle3 token verification on smaller models.

**Why novel**: The interaction between KV cache quantization and speculative decoding
verification is unstudied. For 8B models, Eagle3 verification may be more sensitive to KV
precision (draft model has less "headroom" for quantization error). For 32B, the model is
robust enough that FP8 KV noise doesn't disrupt verification. This creates a model-size
threshold below which KV quantization + speculation cannot be safely combined.

**Potential claim**: A serving system that selects KV cache quantization level based on the
target model size and active speculation method, disabling quantization when the expected
precision loss would degrade speculative token acceptance rate below a threshold.

**Evidence**: 8B FP8+Eagle3: 577 TPS (0.80x) vs 8B Eagle3 alone: 744 TPS (1.05x).
32B FP8+Eagle3: 125.49 TPS (1.31x) vs 32B Eagle3 alone: 124.23 TPS (1.30x).

**Priority**: High
**Status**: Idea — needs root cause analysis

---

## Validated Patents (TBD)

*None yet — promote from Ideas once independently confirmed.*

---

## Discarded Ideas

### [2026-03-30] N-gram speculation on instruction-following prompts

N-gram speculation (REST/PLD) gives 0.88–0.89x on our benchmark (independent technical
questions). Root cause: zero prompt-to-prompt text overlap, so n-gram lookup never finds
matches. N-gram is not a general inference optimization — it requires workloads with high
textual repetition (code completion, RAG with shared context, document continuation).
Not worth pursuing for general-purpose serving benchmarks.
