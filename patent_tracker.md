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

## Validated Patents (TBD)

*None yet — populate as experiments validate ideas.*

---

## Discarded Ideas

*Record why an idea was discarded to avoid re-investigating it.*
