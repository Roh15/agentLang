# Latent Agent Communication — KV Cache Passing POC

**Can same-model agent pipelines skip redundant prefill by passing KV cache directly between agents?**

Yes. This repo documents the experiments that prove it, quantify it, and characterise its limits.

---

## The Problem

In a standard multi-agent pipeline, Agent A finishes and its output is decoded to text. Agent B then re-tokenizes that entire output and runs a full prefill before generating token 1. That prefill replicates computation Agent A already performed. You're paying for it twice.

At an N-hop pipeline, this pattern wastes N-1 full prefills per call.

## The Proposed Fix

A stateful session primitive: when Agent B's inference call is chained to Agent A's completed call (same model, same session), pass the KV cache directly. Agent B prefills its system prompt only — ~41 tokens — and skips re-prefilling Agent A's output entirely. The required API change is a single `session_id` field on inference calls.

**Transfer cost is negligible. Quality is preserved. The savings scale with model size and context length.**

---

## Key Results

| Experiment | Headline |
|---|---|
| EXP-001 (3B, baseline) | **3.22x** prefill speedup, 6/6 quality, transfer cost = 0.9% of saving |
| EXP-002 (RoPE ablation) | Without position re-indexing: generation immediately broken (SyntaxError) |
| EXP-003 (scaling curve) | Speedup 1.0x at 50 tokens → **9.6x at 800 tokens** |
| EXP-004 (7B model) | **6.74x** prefill speedup, 6/6 quality, 28 MB payload |
| EXP-005 (cross-process) | 73 ms naive serialization vs 581 ms saving (12.6%); 30/30 quality |
| EXP-006 (system prompt sweep) | 15/16 grid cells ≥ 1.6x; only 1.3x at short output + long system prompt |

**Practical rule:** the mechanism pays off when `prior_context_tokens / system_prompt_tokens > 3`.

### Speedup Grows With Model Size

| Model | Standard prefill | KV prefix prefill | Speedup |
|---|---|---|---|
| 3B | 301 ms | 93 ms | **3.2x** (measured) |
| 7B | 682 ms | 101 ms | **6.7x** (measured) |
| 13B | ~1,300 ms | ~110 ms | ~12x (extrapolated) |
| 70B | ~7,000 ms | ~150 ms | ~47x (extrapolated) |

KV prefix prefill cost is nearly model-size-invariant at short sequence lengths (41 tokens barely stresses additional layers). Standard prefill is not. The case gets stronger at production model sizes.

### Transfer Cost Hierarchy

| Method | Cost (28 MB payload) | % of 581 ms saving |
|---|---|---|
| In-process clone (unified memory) | 1.4 ms | 0.2% |
| Cross-process `torch.save`/`torch.load` to `/dev/shm` | 73 ms | 12.6% |
| CUDA IPC (same GPU, different process) | ~14 µs | < 0.01% |
| NVLink (different GPUs, same node) | ~47 µs | < 0.01% |

The 73 ms cross-process number is a worst-case pessimist. A provider using CUDA IPC reduces overhead to effectively zero.

---

## How It Works

### Standard Pipeline (baseline)
```
Agent A: [sys_A | user_msg] → generate → text output
Agent B: [sys_B | text_output] → FULL PREFILL → generate
```

### KV Cache Pipeline
```
Agent A: [sys_A | user_msg] → generate → KV cache slice extracted
                              ↓
                    RoPE re-indexed to Agent B's position space
                              ↓
Agent B: [sys_B] → prefill 41 tokens → KV slice injected → generate
```

Agent B processes its short system prompt and then "sees" Agent A's full output as already-computed attention state. It never re-reads a single token of Agent A's output.

### Why RoPE Re-indexing Is Non-Negotiable

Agent A's KV tensors encode positions from its own context (e.g. positions 50–560). Agent B expects its payload at positions starting at ~41. Without correcting this mismatch, generation fails immediately — not degrades, fails. EXP-002 produced `mid = (low - 1` (truncated mid-expression, SyntaxError) on the very first token.

The fix is an inverse rotation on the K tensors before injecting them:

```python
k_raw = k_rotated * cos_old - rotate_half(k_rotated) * sin_old
k_new = k_raw * cos_new + rotate_half(k_raw) * sin_new
```

---

## Repository Layout

```
agentLang/
├── CLAUDE.md                    ← hardware, implementation notes, quick-ref
├── JOURNAL.md                   ← full experiment log with raw data and conclusions
├── config.py                    ← model path, device, agent prompts, hyperparams
├── buggy_function.py            ← test subject: binary_search with 6 planted bugs
├── test_buggy_function.py       ← 6-test quality oracle
├── benchmark.py                 ← EXP-001: prefill savings, quality, multi-hop
├── exp_rope_ablation.py         ← EXP-002
├── exp_scaling_curve.py         ← EXP-003
├── exp_7b.py                    ← EXP-004
├── exp_cross_process.py         ← EXP-005: orchestrator
├── exp_cross_process_a.py       ← EXP-005: Process A (generate + serialize)
├── exp_cross_process_b.py       ← EXP-005: Process B (deserialize + run)
├── exp_system_prompt_sweep.py   ← EXP-006: 2D speedup grid
├── pipelines/
│   ├── standard.py              ← text-handoff baseline
│   └── kv_cache.py              ← KV cache passing pipeline
└── results/
    └── *.json, *.png            ← all experiment outputs
```

---

## Hardware

All experiments ran on a **Jetson AGX Orin** (ARM64, 64 GB LPDDR5 unified memory, CUDA 11.4, JetPack R35.3.1). Clocks were locked for every timed measurement:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Model: **Qwen2.5-7B-Instruct** (bfloat16, ~15 GB). Earlier experiments used 3B.

---

## Reproducing Experiments

All experiments run inside Docker. Build once, then run any script:

```bash
# Build image (ARM64 base — Jetson only)
docker build -t latent-poc .

# Run an experiment (example: EXP-004, 7B replication)
docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e PYTHONUNBUFFERED=1 \
  -v /home/admin/models:/models \
  -v /dev/shm:/dev/shm \
  -v /home/admin/agentLang:/workspace \
  latent-poc python3 -u exp_7b.py
```

Results are written to `results/` as JSON + PNG.

---

## Conclusions

1. **The mechanism works.** Prefill savings are real, reproducible, and quality-preserving across 325 data points.
2. **Savings grow with model size.** The KV prefix cost is near-constant at short system prompts; standard prefill is not. At 70B+, the speedup likely exceeds 40x per hop.
3. **RoPE re-indexing is a hard requirement**, not an optimisation.
4. **Cross-process transfer is viable** even with naive serialization (12.6% overhead). With CUDA IPC it is essentially free.
5. **The practical limit:** if Agent B's system prompt is longer than ~33% of Agent A's output, gains drop below 2x. Most real agent patterns (analysis → repair, research → synthesis) stay well within the profitable regime.
6. **The required API primitive is minimal:** a `session_id` field that allows a provider's inference server to locate and inject a prior call's KV cache. No new model weights, no architectural changes.

See [`JOURNAL.md`](JOURNAL.md) for full methodology, raw data, and per-experiment discussion.
