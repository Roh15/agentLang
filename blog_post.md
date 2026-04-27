# Your Agent Pipeline Is Paying for Compute It Already Did

Every time Agent A finishes and hands off to Agent B, something wasteful happens. Agent B takes Agent A's output, decodes it to a string, re-tokenizes it, and runs a full forward pass over the entire sequence before generating token 1. That forward pass — the prefill — is identical to computation Agent A already performed to produce those tokens. You're paying for it twice.

This isn't a niche problem. It happens at every handoff in every multi-agent pipeline, on every inference call, at every provider. At 7B parameter scale I measured it at 682ms per hop. For a 4-hop pipeline that's over two seconds of pure waste per call, before a single new token is generated.

I built a POC to test whether this can be fixed and what the fix actually costs. The short version: it works, the savings are real and scale with model size, and the implementation requires one non-trivial technical step. Here's what I found.

---

## The Problem

When you call an LLM API, the work splits into two phases:

**Prefill:** The model processes your input sequence and builds a KV cache — a compressed representation of every token's contribution to attention across all layers. This is O(n²) in sequence length for the attention component and can't be parallelized across tokens.

**Decode:** The model generates tokens one at a time, each attending back to the KV cache built during prefill.

In a two-agent pipeline, Agent A generates 500 tokens of analysis. Agent B receives those 500 tokens as text, tokenizes them again, and runs prefill over the full sequence (500 tokens + its system prompt) before it can start generating. That prefill pass produces a KV cache that is functionally identical to the one Agent A already built internally while generating its output.

The waste is structural. Standard agent pipelines always pay it.

---

## The Fix

Instead of passing text between agents, pass the KV cache. Agent A generates with `use_cache=True`, slices the KV cache to cover only its output tokens, and hands that slice to Agent B. Agent B prefills only its system prompt — 41 tokens in my setup — then concatenates Agent A's KV slice to its own cache and starts generating directly.

The payload is a set of tensors: for a 7B model at 512 output tokens it's 28MB. On unified memory (same process) the "transfer" is a pointer. Cross-process with naive serialization it's 73ms. With proper CUDA IPC it's microseconds.

There's one technical requirement that isn't optional: **RoPE re-indexing**.

Transformer KV caches encode positional information via Rotary Position Embedding — the K tensors are rotated by an angle proportional to their position in the sequence. Agent A's output tokens are at positions 500–1000 (wherever they landed in Agent A's context). Agent B needs those tokens at positions 41–550 (after its own system prompt). If you transplant the KV slice without correcting the positions, the attention scores are wrong and generation breaks immediately — not degrades, breaks. I tested this explicitly. Without re-indexing, the model produced `mid = (low - 1` and stopped. Syntax error.

The fix is a mathematical inverse: undo the original rotation, re-apply at the new positions. Since RoPE is orthogonal, the inverse is just negating the sine term. About 20 lines of PyTorch.

---

## What I Measured

I ran everything on a Jetson AGX Orin (64GB unified memory, CUDA 11.4) with clocks locked for consistent timing. Two models: Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. Twenty timed runs per condition.

**Agent B prefill time, 512-token prior context:**

| Model | Standard pipeline | KV cache pipeline | Speedup |
|---|---|---|---|
| 3B | 301 ms | 93 ms | 3.2x |
| 7B | 682 ms | 101 ms | **6.7x** |

The KV prefix cost barely changes between 3B and 7B (93ms → 101ms) because at 41 tokens, the computation sits in the fixed-overhead floor — kernel launch and memory access patterns dominate over sequence-dependent math. The standard pipeline doesn't have this floor for longer inputs. As models get larger, the gap widens. The mechanism compounds with model size.

Quality check: both pipelines produced identical corrected functions. 6/6 pytest tests passed in every run.

**Transfer cost reality check (7B, 28MB payload):**

| Method | Cost | % of saving |
|---|---|---|
| In-process tensor reference | ~0 µs | ~0% |
| Cross-process (torch.save/load to /dev/shm) | 73 ms | 12.6% |
| CUDA IPC (same GPU, different process) | ~14 µs | <0.01% |
| NVLink (different GPUs, same node) | ~47 µs | <0.01% |

The naive cross-process approach costs 73ms but saves 581ms — still net-positive. A proper provider implementation with CUDA IPC makes the transfer cost essentially zero.

**Multi-hop scaling:**

Standard pipeline per-hop prefill on a 4-hop run: 101ms → 302ms → 345ms → 409ms. Growing every hop as context accumulates.

KV cache pipeline: 99ms → 95ms → 95ms → 97ms. Flat. Every agent only prefills its own system prompt.

---

## The Operating Envelope

The speedup isn't fixed — it depends on how much Agent A writes relative to Agent B's system prompt length. I swept a 4×4 grid to find the boundaries.

*[Insert EXP-006 heatmap here]*

The mechanism pays off across almost all realistic operating points. The only marginal case (1.3x) is 200 tokens of prior context against a 400-token system prompt — an agent that barely writes anything handed off to one with an elaborate instruction set. Most real pipelines don't look like this.

**Practical rule:** if Agent A produces at least 3x more tokens than Agent B's system prompt, you're getting ≥2x speedup. Analysis-then-repair, research-then-synthesis, extraction-then-formatting — these all comfortably exceed that ratio.

The 400-token system prompt column is the warning. Enterprise agents with highly-specified, detailed instructions see smaller gains. Providers would need to characterise their workload distribution to estimate the real-world average.

---

## What Needs to Be Built

The mechanism itself is proven. The bottleneck is infrastructure.

In a standard inference stack, each API call is stateless. Requests land on whatever node has capacity. There's no continuity of GPU memory between calls. For KV cache passing to work at provider scale, you need:

1. A `session_id` field on inference calls
2. Routing: when two calls share a session ID, send them to the same GPU instance
3. KV cache retention: hold the cache in GPU memory between calls (for some bounded TTL)
4. RoPE re-indexing in the serving layer when handing off

Items 1–3 are infrastructure decisions. Item 4 is 20 lines of math. None of this requires model retraining, architectural changes, or changes to client code beyond passing one extra field.

The analogy that keeps coming to mind: HTTP/1.0 opened a new TCP connection for every request. HTTP/1.1 added keep-alive. The web got faster for free. The underlying protocol didn't change — just the connection lifecycle management. This is the same idea applied to inference sessions.

---

## Honest Caveats

**This is prefill savings, not end-to-end latency.** Generation time dominates e2e latency and is unchanged. Saving 580ms of prefill in a pipeline where each agent generates for 30 seconds is a ~1% wall-clock improvement. The argument is cost reduction at scale, not user-visible speed.

**One model family.** Everything here is Qwen2.5 on a Jetson. The mechanism is architecture-agnostic in principle but the exact numbers are hardware and model-specific.

**Same model required.** KV caches only transfer correctly between identical model weights. A pipeline mixing Claude with GPT-4 gets nothing from this. Same-provider, same-model pipelines are the target.

**The savings are proportional to the ratio.** A pipeline where Agent B has a 400-token system prompt and Agent A writes 300 tokens barely benefits. Know your pipeline's token budget before deciding whether this is worth building.

---

## The Scale Argument

At 7B parameters, each wasted prefill hop costs 682ms. A 4-hop pipeline wastes three of them: 2,046ms per call of pure re-computation.

At 10M pipeline calls/day at that scale, that's roughly 6,000 GPU-hours/day of avoidable prefill. That's not a rounding error. That's a cost line worth a product decision.

The savings grow with model size. By the time you're at 70B+ — where most production inference actually runs — the per-hop waste is seconds, not hundreds of milliseconds.

---

## Code

The full POC, all experiment scripts, and the raw data are on GitHub at [repo link]. The research journal with methodology, raw run data, and conclusions for each of the six experiments is in `JOURNAL.md`.

The key file is `pipelines/kv_cache.py`. The RoPE re-indexing is in `reindex_kv_cache`. If you want to understand the mechanism, start there.

---

If you're building agent infrastructure and have thoughts on the session routing problem or the memory management tradeoffs, I'd like to hear them.
