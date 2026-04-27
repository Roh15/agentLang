"""
KV Cache Passing Pipeline
=========================
past_key_values on this model (Qwen2.5-3B + transformers 4.46 / Python 3.8) is a
plain tuple of length num_layers. Each element is a (K, V) pair of tensors with
shape [batch=1, num_kv_heads=2, seq_len, head_dim=128].

Rotary embedding API: rotary_emb(x, position_ids) → (cos, sin)
  x       : [batch, seq_len, head_dim] — only used for dtype/device
  position_ids: [batch, seq_len]
  cos/sin : [batch, seq_len, head_dim]
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

# Suppress "passing past_key_values as tuple of tuples is deprecated" from transformers 4.46
warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*")


# Alias for the KV cache type used by this model
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _get_rope_cos_sin(
    model,
    positions: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (cos, sin) for the given position indices.
    positions: [1, seq_len]
    Returns: cos, sin each [1, seq_len, head_dim]
    """
    rotary_emb = model.model.layers[0].self_attn.rotary_emb
    seq_len = positions.shape[1]
    dummy = torch.zeros(1, seq_len, head_dim, dtype=dtype, device=device)
    with torch.no_grad():
        cos, sin = rotary_emb(dummy, positions)
    return cos, sin


def reindex_kv_cache(
    kv: KVCache,
    old_start: int,
    new_start: int,
    model,
    dtype: torch.dtype,
    device: str,
    head_dim: int,
    strict: bool = True,
) -> KVCache:
    """
    Returns a new KV cache with K tensors re-indexed from old_start to new_start.

    K tensors have RoPE baked in at their original generation positions. When
    transplanting a KV slice into a new agent's context, we undo the original
    rotation and re-apply at the target positions. V tensors are unchanged.

    Inverse RoPE: k_raw = k_rotated * cos - rotate_half(k_rotated) * sin
    (negating sin is the exact inverse for orthogonal RoPE rotations)

    If strict=False, skips re-indexing — useful to isolate quality impact.
    """
    if not strict:
        return kv

    seq_len = kv[0][0].shape[2]

    old_pos = torch.arange(old_start, old_start + seq_len, device=device).unsqueeze(0)
    new_pos = torch.arange(new_start, new_start + seq_len, device=device).unsqueeze(0)

    cos_old, sin_old = _get_rope_cos_sin(model, old_pos, dtype, device, head_dim)
    cos_new, sin_new = _get_rope_cos_sin(model, new_pos, dtype, device, head_dim)

    # [1, 1, seq_len, head_dim] — broadcasts with K [1, 2, seq_len, 128]
    cos_old = cos_old.unsqueeze(1)
    sin_old = sin_old.unsqueeze(1)
    cos_new = cos_new.unsqueeze(1)
    sin_new = sin_new.unsqueeze(1)

    new_layers = []
    for k, v in kv:
        k_f = k.to(torch.float32)
        k_raw = k_f * cos_old.float() - _rotate_half(k_f) * sin_old.float()
        k_new = k_raw * cos_new.float() + _rotate_half(k_raw) * sin_new.float()
        new_layers.append((k_new.to(dtype), v))

    return tuple(new_layers)


def slice_kv_cache(kv: KVCache, start: int, end: int) -> KVCache:
    """Slice a KV cache along the sequence dimension."""
    return tuple(
        (k[:, :, start:end, :].clone(), v[:, :, start:end, :].clone())
        for k, v in kv
    )


def concat_kv_caches(kv_a: KVCache, kv_b: KVCache) -> KVCache:
    """Concatenate two KV caches along the sequence dimension (kv_a first)."""
    return tuple(
        (torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2))
        for (ka, va), (kb, vb) in zip(kv_a, kv_b)
    )


def measure_kv_clone_us(kv: KVCache) -> float:
    """
    Time a full deep copy of the KV cache.
    On the Orin this is near-zero (unified memory pointer).
    Reported alongside results to show the transfer overhead even in a
    distributed provider environment would be negligible.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = tuple((k.clone(), v.clone()) for k, v in kv)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6


def kv_cache_size_mb(kv: KVCache) -> float:
    return sum(k.element_size() * k.numel() + v.element_size() * v.numel() for k, v in kv) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

def _tokenize_prefix(tokenizer, system_prompt: str, device: str) -> torch.Tensor:
    """
    Tokenize everything up to and including the start of the user turn.
    Produces: <|im_start|>system\\n{sys}<|im_end|>\\n<|im_start|>user\\n
    """
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)


def _tokenize_suffix(tokenizer, device: str) -> torch.Tensor:
    """
    Tokenize the closing of the user turn + assistant start.
    Produces: <|im_end|>\\n<|im_start|>assistant\\n
    """
    text = "<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenerationSlice:
    """
    KV cache of one agent's generated tokens, normalized to positions 0..length-1.
    Stored normalized so that re-indexing to the correct offset happens once, at
    the start of the next agent's turn, not at slice creation time.
    """
    kv: KVCache
    length: int
    output_text: str


@dataclass
class KVCacheResult:
    agent_outputs: List[str]
    prefill_times_ms: List[float]     # per agent — prefix-only for agents 2..N
    kv_clone_times_us: List[float]    # per agent (0 for Agent A)
    e2e_latency_ms: float
    kv_payload_sizes_mb: List[float]  # per handoff (cumulative payload)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _greedy_generate(
    model,
    tokenizer,
    past_kv: KVCache,
    first_logits: torch.Tensor,
    start_position: int,
    device: str,
    max_new_tokens: int,
) -> Tuple[List[int], KVCache, int]:
    """
    Greedy generation loop from an existing KV cache.
    first_logits: logits from the last prefill/suffix step — sample token 0 from these.
    Returns (token_ids, final_kv, final_position).
    """
    generated = []
    current_pos = start_position

    with torch.no_grad():
        next_logits = first_logits
        for _ in range(max_new_tokens):
            next_id = int(next_logits.argmax(-1))
            generated.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            out = model(
                input_ids=torch.tensor([[next_id]], device=device),
                past_key_values=past_kv,
                position_ids=torch.tensor([[current_pos]], device=device),
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_logits = out.logits[0, -1, :]
            current_pos += 1

    return generated, past_kv, current_pos


# ---------------------------------------------------------------------------
# Per-agent runners
# ---------------------------------------------------------------------------

def _run_agent_a(
    model,
    tokenizer,
    system_prompt: str,
    initial_input: str,
    device: str,
    dtype,
    max_new_tokens: int,
    head_dim: int,
    strict_rope: bool,
) -> Tuple[str, float, "GenerationSlice"]:
    """
    Run Agent A. Uses standard text input — there is no prior KV to receive.
    Returns (output_text, prefill_ms, generation_slice normalized to pos 0..len-1).
    """
    from pipelines.standard import _format_prompt
    prompt = _format_prompt(system_prompt, initial_input)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    generated_ids, full_kv, _ = _greedy_generate(
        model, tokenizer, prefill_out.past_key_values,
        prefill_out.logits[0, -1, :], input_len, device, max_new_tokens,
    )
    gen_len = len(generated_ids)

    # Slice generation tokens out of the full KV cache
    gen_kv_raw = slice_kv_cache(full_kv, input_len, input_len + gen_len)

    # Normalize to positions 0..gen_len-1 for clean handoff
    gen_kv = reindex_kv_cache(
        gen_kv_raw, old_start=input_len, new_start=0,
        model=model, dtype=dtype, device=device, head_dim=head_dim, strict=strict_rope,
    )

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text, prefill_ms, GenerationSlice(kv=gen_kv, length=gen_len, output_text=output_text)


def _build_payload_kv(
    prior_slices: List["GenerationSlice"],
    l_prefix: int,
    model,
    dtype,
    device: str,
    head_dim: int,
    strict_rope: bool,
) -> KVCache:
    """
    Reindex all prior generation slices to positions l_prefix..l_prefix+total_len-1
    and concatenate into a single payload KV cache.
    """
    combined: Optional[KVCache] = None
    offset = 0
    for s in prior_slices:
        reindexed = reindex_kv_cache(
            s.kv, old_start=0, new_start=l_prefix + offset,
            model=model, dtype=dtype, device=device, head_dim=head_dim, strict=strict_rope,
        )
        combined = reindexed if combined is None else concat_kv_caches(combined, reindexed)
        offset += s.length
    return combined


def _run_agent_n(
    model,
    tokenizer,
    system_prompt: str,
    prior_slices: List["GenerationSlice"],
    device: str,
    dtype,
    max_new_tokens: int,
    head_dim: int,
    strict_rope: bool,
) -> Tuple[str, float, float, "GenerationSlice"]:
    """
    Run any agent after Agent A.

    Structure of the combined KV passed to the model:
      [sys_prefix | payload (all prior gen slices, reindexed) | suffix]

    The TIMED section is the prefix prefill only. The payload arrives as KV
    and is never re-prefilled — that's the saving.

    Returns (output_text, prefill_ms, clone_us, generation_slice).
    """
    prefix_ids = _tokenize_prefix(tokenizer, system_prompt, device)
    l_prefix = prefix_ids.shape[1]

    # === TIMED: prefix-only prefill (the headline metric) ===
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefix_out = model(input_ids=prefix_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    kv_prefix = prefix_out.past_key_values

    # Build and clone payload (clone_us simulates real transfer cost)
    payload_kv = _build_payload_kv(prior_slices, l_prefix, model, dtype, device, head_dim, strict_rope)
    clone_us = measure_kv_clone_us(payload_kv)

    total_prior_len = sum(s.length for s in prior_slices)

    # Stitch together: prefix + payload
    combined_kv = concat_kv_caches(kv_prefix, payload_kv)

    # Extend with suffix tokens at their correct positions
    suffix_ids = _tokenize_suffix(tokenizer, device)
    l_suffix = suffix_ids.shape[1]
    suffix_start = l_prefix + total_prior_len
    suffix_positions = torch.arange(suffix_start, suffix_start + l_suffix, device=device).unsqueeze(0)

    with torch.no_grad():
        suffix_out = model(
            input_ids=suffix_ids,
            past_key_values=combined_kv,
            position_ids=suffix_positions,
            use_cache=True,
        )

    gen_start_pos = suffix_start + l_suffix
    generated_ids, final_kv, _ = _greedy_generate(
        model, tokenizer, suffix_out.past_key_values,
        suffix_out.logits[0, -1, :], gen_start_pos, device, max_new_tokens,
    )
    gen_len = len(generated_ids)

    # Slice and normalize this agent's generation for the next hop
    total_kv_len = final_kv[0][0].shape[2]
    gen_kv_raw = slice_kv_cache(final_kv, total_kv_len - gen_len, total_kv_len)
    gen_kv = reindex_kv_cache(
        gen_kv_raw, old_start=gen_start_pos, new_start=0,
        model=model, dtype=dtype, device=device, head_dim=head_dim, strict=strict_rope,
    )

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text, prefill_ms, clone_us, GenerationSlice(kv=gen_kv, length=gen_len, output_text=output_text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_kv_cache_pipeline(
    model,
    tokenizer,
    agent_keys: List[str],
    agent_prompts: dict,
    initial_input: str,
    device: str,
    dtype,
    max_new_tokens: int,
    head_dim: int,
    strict_rope: bool = True,
) -> KVCacheResult:
    """
    N-hop KV cache passing pipeline.

    Agent A: text input, full prefill (no prior KV exists).
    Agents B..N: prefix-only prefill + KV payload from all prior agents.
    Each agent's prefill cost is O(sys_prompt_len) regardless of context depth.
    """
    t_start = time.perf_counter()

    outputs: List[str] = []
    prefill_times: List[float] = []
    clone_times: List[float] = []
    payload_sizes: List[float] = []
    slices: List[GenerationSlice] = []

    for i, key in enumerate(agent_keys):
        if i == 0:
            output_text, prefill_ms, gen_slice = _run_agent_a(
                model, tokenizer, agent_prompts[key], initial_input,
                device, dtype, max_new_tokens, head_dim, strict_rope,
            )
            clone_us = 0.0
        else:
            payload_mb = sum(kv_cache_size_mb(s.kv) for s in slices)
            payload_sizes.append(payload_mb)
            output_text, prefill_ms, clone_us, gen_slice = _run_agent_n(
                model, tokenizer, agent_prompts[key], slices,
                device, dtype, max_new_tokens, head_dim, strict_rope,
            )

        outputs.append(output_text)
        prefill_times.append(prefill_ms)
        clone_times.append(clone_us)
        slices.append(gen_slice)

    e2e_ms = (time.perf_counter() - t_start) * 1000

    return KVCacheResult(
        agent_outputs=outputs,
        prefill_times_ms=prefill_times,
        kv_clone_times_us=clone_times,
        e2e_latency_ms=e2e_ms,
        kv_payload_sizes_mb=payload_sizes,
    )
