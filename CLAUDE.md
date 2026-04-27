# Latent Agent Communication — POC

## Thesis

Same-model agent pipelines waste compute at every handoff. When Agent A finishes
and Agent B starts, the standard pattern decodes Agent A's output to text, then
Agent B re-tokenizes and re-prefills that entire sequence before generating token 1.
That computation is identical to what Agent A already did. We're paying for it twice.

**The proposed solution:** A stateful session primitive. When Agent B's call is
chained to Agent A's completed call (same model, same session), pass the KV cache
directly — Agent B prefills its system prompt only and skips re-prefilling Agent A's
output entirely.

**The argument:** At N-hop pipelines, standard inference wastes N-1 full prefills.
KV cache passing eliminates them. The transfer cost is negligible. Quality is
preserved. The API primitive required is a `session_id` field on inference calls.

**Status: Experiments complete.** See `JOURNAL.md` for full methodology and raw data.
See `results/` for all JSON data and plots.

---

## Hardware

**Device:** Jetson AGX Orin — JetPack R35.3.1, CUDA 11.4, ARM64  
**Memory:** 64GB LPDDR5 unified memory (CPU + GPU share the same physical pool)  
**Storage:** ~18GB free eMMC after experiments  
**Docker:** v24.0.5, base image `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`  
**Python:** 3.8.10 inside container  
**PyTorch:** 2.0.0a0+nv23.02 (Nvidia JetPack build)  
**Transformers:** 4.46.3  

**Clock locking required for all timing runs:**
```bash
sudo nvpmodel -m 0      # MAX power mode
sudo jetson_clocks      # lock clocks at max frequency
```

**No `nvidia-smi` on Tegra.** Use `tegrastats` or `torch.cuda.memory_allocated()`.

---

## Models Used

### Qwen2.5-7B-Instruct (current — on device)
- Path: `/models/Qwen2.5-7B-Instruct`
- Architecture: 28 layers, 4 KV heads, 128 head_dim, hidden_size=3584
- Size: ~15GB bfloat16
- KV cache at 512 tokens: **28MB**

### Qwen2.5-3B-Instruct (removed to free disk — 3B results in EXP-001 to EXP-003)
- Architecture: 36 layers, 2 KV heads, 128 head_dim, hidden_size=2048
- Size: ~6.2GB bfloat16
- KV cache at 512 tokens: **18MB**

**Note:** `past_key_values` on both models with transformers 4.46.3 + Python 3.8 is
a **plain tuple of (K, V) pairs**, not a `DynamicCache` object. Shape per layer:
`[batch=1, num_kv_heads, seq_len, head_dim]`. All slice/concat/reindex operations
work on this tuple format directly.

---

## Implementation Notes

**Chat template:** `tokenizer.apply_chat_template` fails on this stack (Jinja2 not
installed). Use `_format_prompt` from `pipelines/standard.py` instead:
```python
def _format_prompt(system_prompt, user_message):
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

**RoPE re-indexing is non-negotiable.** Without it, generation produces broken
syntax immediately (EXP-002). The inverse rotation formula:
```python
k_raw = k_rotated * cos_old - rotate_half(k_rotated) * sin_old
k_new = k_raw * cos_new + rotate_half(k_raw) * sin_new
```
Access rotary embedding via `model.model.layers[0].self_attn.rotary_emb`.
Signature: `rotary_emb(x, position_ids)` → `(cos, sin)` each `[batch, seq_len, head_dim]`.

**Do not use `model.generate()` for Pipeline 2.** Use the manual generation loop
in `pipelines/kv_cache.py:_greedy_generate`.

**Python 3.8 compatibility:** No `list[str]` type hints (use `List[str]` from
`typing`), no `str | None` (use `Optional[str]`), no backslashes in f-strings.

**Suppress tuple deprecation warning** — transformers 4.46 warns when passing
`past_key_values` as a tuple of tuples. Add to pipeline files:
```python
warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*")
```

**Running experiments:** All experiments run inside Docker. Always mount the
workspace and models:
```bash
docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e PYTHONUNBUFFERED=1 \
  -v /home/admin/models:/models -v /dev/shm:/dev/shm \
  -v /home/admin/agentLang:/workspace latent-poc python3 -u <script>.py
```

---

## Key Results

| Experiment | Finding |
|---|---|
| EXP-001 (3B baseline) | 3.22x prefill speedup, 6/6 quality, transfer cost 0.9% of saving |
| EXP-002 (RoPE ablation) | Without re-indexing: generation immediately broken (SyntaxError) |
| EXP-003 (scaling curve) | Speedup 1.0x at 50 tokens → 9.6x at 800 tokens; scales with context |
| EXP-004 (7B model) | 6.74x prefill speedup, 6/6 quality, 28MB payload |
| EXP-005 (cross-process) | 73ms naive serialization vs 581ms saving (12.6%); 30/30 quality |
| EXP-006 (sys prompt sweep) | 15/16 grid cells ≥1.6x; only 1.3x at 200-token output + 400-token sys prompt |

**Practical operating rule:** mechanism delivers meaningful savings when
`prior_context_tokens / system_prompt_tokens > 3`.

**Transfer cost hierarchy:**
| Method | Cost for 28MB | % of 581ms saving |
|---|---|---|
| In-process clone (Orin unified memory) | 1.4 ms | 0.2% |
| Cross-process torch.save/load to /dev/shm | 73 ms | 12.6% |
| CUDA IPC (same GPU, different process) | ~14 µs | <0.01% |
| NVLink (different GPUs, same node) | ~47 µs | <0.01% |

---

## File Structure

```
agentLang/
├── CLAUDE.md                    ← this file
├── JOURNAL.md                   ← full experiment log with raw data and conclusions
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── config.py                    ← model path, device, agent prompts, hyperparams
├── buggy_function.py            ← test subject: binary_search with 6 planted bugs
├── conftest.py                  ← pytest --repaired-path option
├── test_buggy_function.py       ← 6-test quality oracle
├── smoke_quality.py             ← quick quality check for both pipelines
├── benchmark.py                 ← EXP-001: prefill savings, quality, multi-hop
├── exp_rope_ablation.py         ← EXP-002: STRICT_ROPE_REINDEX=False
├── exp_scaling_curve.py         ← EXP-003: prefill vs sequence length
├── exp_7b.py                    ← EXP-004: 7B model replication
├── exp_cross_process.py         ← EXP-005: orchestrator for cross-process transfer
├── exp_cross_process_a.py       ← EXP-005: Process A (generate + serialize KV)
├── exp_cross_process_b.py       ← EXP-005: Process B (deserialize KV + run Agent B)
├── exp_system_prompt_sweep.py   ← EXP-006: 2D speedup grid
├── pipelines/
│   ├── __init__.py
│   ├── standard.py              ← text handoff baseline
│   └── kv_cache.py              ← KV cache passing pipeline
└── results/
    └── *.json, *.png            ← all experiment outputs
```
