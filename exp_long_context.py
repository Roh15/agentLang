"""
EXP-007: Long-Context Prefill Scaling (800 – 4000 tokens)
===========================================================
Extends EXP-003 into the range where O(n^2) attention cost begins to dominate.
Uses the LRU cache analysis task, which reliably drives Agent A to 1500-2500
output tokens — enough to sweep the region EXP-003 could not reach.

Two phases:
  Phase 1 — Timing sweep (synthetic padding, no generation needed).
             Measures standard prefill and KV prefix prefill at each token count.
  Phase 2 — One real end-to-end quality run at TARGET_QUALITY_LEN tokens.
             Verifies the mechanism still produces correct output at long context.

Model: Qwen2.5-7B-Instruct (3B was removed from device after EXP-003).
Records results to results/exp007_long_context_<timestamp>.json
"""
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipelines.standard import _format_prompt
from pipelines.kv_cache import _tokenize_prefix, reindex_kv_cache

MODEL_PATH = "/models/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
DTYPE = torch.bfloat16
RESULTS_DIR = Path("results")
NUM_RUNS = 10
NUM_WARMUP = 3

# Extend from where EXP-003 stopped (800 tokens)
TARGET_LENGTHS = [800, 1000, 1200, 1500, 2000, 2500, 3000, 4000]

# Token length for the single real quality run
TARGET_QUALITY_LEN = 1500

MAX_NEW_TOKENS_A = 2000   # give Agent A room to write a full analysis
MAX_NEW_TOKENS_B = 700    # repair agent output (corrected class)

# ── Agent prompts for LRU cache task ─────────────────────────────────────────

AGENT_A_PROMPT = (
    "You are a code analysis agent. The LRUCache class below contains exactly "
    "7 bugs — logic errors that cause incorrect behaviour.\n\n"
    "For EACH bug provide:\n"
    "  1. The method name and the exact line(s) affected\n"
    "  2. What the code does versus what it should do\n"
    "  3. A concrete test case that exposes the bug: the input, the wrong result "
    "produced, and the correct expected result\n"
    "  4. The corrected code for that line with an explanation of why the fix is right\n\n"
    "After all 7 bugs, write a paragraph describing which bugs compound each other "
    "(i.e. fixing one may unmask another or change the failure mode)."
)

AGENT_B_PROMPT = (
    "You are a code repair agent. Given the bug analysis above, rewrite the complete "
    "LRUCache class (including the Node class) with every identified bug fixed. "
    "Output only the corrected Python code in a single code block."
)

with open("buggy_lru_cache.py") as f:
    BUGGY_LRU_SOURCE = f.read()


# ── Helpers ───────────────────────────────────────────────────────────────────

def pad_to_length(tokenizer, base_text: str, target_len: int) -> torch.Tensor:
    """Encode base_text then pad/truncate to exactly target_len tokens."""
    ids = tokenizer.encode(base_text, add_special_tokens=False)
    filler_id = tokenizer.encode(" ", add_special_tokens=False)[0]
    if len(ids) >= target_len:
        ids = ids[:target_len]
    else:
        ids = ids + [filler_id] * (target_len - len(ids))
    return torch.tensor([ids], device=DEVICE)


def measure_prefill_ms(model, input_ids: torch.Tensor) -> float:
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def greedy_generate(model, tokenizer, input_ids: torch.Tensor,
                    past_kv, start_pos: int, max_new_tokens: int) -> str:
    """Generate greedily from existing past_kv."""
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=past_kv,
                    position_ids=torch.tensor([[start_pos]], device=DEVICE),
                    use_cache=True)
    generated = []
    pos = start_pos + 1

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            next_id = int(out.logits[0, -1, :].argmax(-1))
            generated.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            next_input = torch.tensor([[next_id]], device=DEVICE)
            out = model(input_ids=next_input,
                        past_key_values=out.past_key_values,
                        position_ids=torch.tensor([[pos]], device=DEVICE),
                        use_cache=True)
            pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True)


def run_agent_a(model, tokenizer) -> Tuple[str, int, object]:
    """Run Agent A on the LRU cache source. Returns (output, token_count, past_kv)."""
    prompt = _format_prompt(AGENT_A_PROMPT, BUGGY_LRU_SOURCE)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    print(f"  Agent A input: {input_ids.shape[1]} tokens")

    with torch.no_grad():
        torch.cuda.synchronize()
        prefill_out = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()

    # Generate Agent A's analysis
    analysis = greedy_generate(
        model, tokenizer,
        input_ids=torch.tensor([[int(prefill_out.logits[0, -1, :].argmax(-1))]],
                                device=DEVICE),
        past_kv=prefill_out.past_key_values,
        start_pos=input_ids.shape[1],
        max_new_tokens=MAX_NEW_TOKENS_A,
    )

    analysis_ids = tokenizer.encode(analysis, add_special_tokens=False)
    print(f"  Agent A output: {len(analysis_ids)} tokens")
    return analysis, len(analysis_ids), prefill_out.past_key_values


def run_agent_b_standard(model, tokenizer, analysis: str) -> Tuple[str, float]:
    """Standard pipeline Agent B: re-prefills everything. Returns (output, prefill_ms)."""
    prompt = _format_prompt(AGENT_B_PROMPT, analysis)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    output = greedy_generate(
        model, tokenizer,
        input_ids=torch.tensor([[int(prefill_out.logits[0, -1, :].argmax(-1))]],
                                device=DEVICE),
        past_kv=prefill_out.past_key_values,
        start_pos=input_ids.shape[1],
        max_new_tokens=MAX_NEW_TOKENS_B,
    )
    return output, prefill_ms


def run_agent_b_kv(model, tokenizer, agent_a_past_kv,
                   agent_a_input_len: int, agent_a_output_len: int) -> Tuple[str, float]:
    """KV pipeline Agent B: prefills own system prompt only, reuses Agent A's cache."""
    prefix_ids = _tokenize_prefix(tokenizer, AGENT_B_PROMPT, DEVICE)
    prefix_len = prefix_ids.shape[1]

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefix_out = model(input_ids=prefix_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    # Re-index Agent A's generation KV slice to Agent B's position space.
    # old_start=0 because we already sliced out only the generation tokens.
    agent_a_kv_slice = tuple(
        (k[:, :, agent_a_input_len:, :], v[:, :, agent_a_input_len:, :])
        for k, v in agent_a_past_kv
    )
    reindexed = reindex_kv_cache(
        agent_a_kv_slice,
        old_start=0,
        new_start=prefix_len,
        model=model,
        dtype=DTYPE,
        device=DEVICE,
        head_dim=128,  # Qwen2.5-7B: hidden_size(3584) / num_attention_heads(28)
    )

    # Concatenate: Agent B prefix KV + re-indexed Agent A generation KV
    combined_kv = tuple(
        (torch.cat([pk, rk], dim=2), torch.cat([pv, rv], dim=2))
        for (pk, pv), (rk, rv) in zip(prefix_out.past_key_values, reindexed)
    )

    combined_len = prefix_len + agent_a_output_len
    last_logits = prefix_out.logits[0, -1, :]

    output = greedy_generate(
        model, tokenizer,
        input_ids=torch.tensor([[int(last_logits.argmax(-1))]], device=DEVICE),
        past_kv=combined_kv,
        start_pos=combined_len,
        max_new_tokens=MAX_NEW_TOKENS_B,
    )
    return output, prefill_ms


def extract_code(text: str) -> str:
    """Extract the first ```python ... ``` block, or return raw text."""
    m = re.search(r"```(?:python)?\s*([\s\S]+?)```", text)
    return m.group(1).strip() if m else text.strip()


def score_repair(code: str) -> int:
    """Write repaired code to a temp file and run the pytest oracle. Returns pass count."""
    import subprocess, tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                     dir="/workspace") as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "test_buggy_lru_cache.py",
             f"--repaired-path={tmp_path}", "-q", "--tb=no"],
            capture_output=True, text=True, cwd="/workspace",
        )
        print(f"    pytest stdout: {result.stdout.strip()}")
        m = re.search(r"(\d+) passed", result.stdout)
        return int(m.group(1)) if m else 0
    finally:
        os.unlink(tmp_path)


# ── Phase 1: Timing sweep ─────────────────────────────────────────────────────

def phase1_timing_sweep(model, tokenizer):
    print("\n" + "=" * 60)
    print("PHASE 1: Timing Sweep (synthetic padding, 800-4000 tokens)")
    print("=" * 60)

    base_prompt = _format_prompt(AGENT_B_PROMPT, "")
    prefix_ids = _tokenize_prefix(tokenizer, AGENT_B_PROMPT, DEVICE)
    kv_prefix_len = prefix_ids.shape[1]

    # Warmup
    warmup_ids = pad_to_length(tokenizer, base_prompt, TARGET_LENGTHS[-1])
    for _ in range(NUM_WARMUP):
        measure_prefill_ms(model, warmup_ids)
    print(f"Warmup done. KV prefix length: {kv_prefix_len} tokens")

    # KV prefix baseline
    kv_runs = [measure_prefill_ms(model, prefix_ids) for _ in range(NUM_RUNS)]
    kv_mean = float(np.mean(kv_runs))
    kv_std = float(np.std(kv_runs))
    print(f"KV prefix baseline: {kv_mean:.1f} ± {kv_std:.2f} ms\n")

    print(f"{'Tokens':>8} {'Std mean (ms)':>15} {'Std std':>10} {'Speedup':>10}")
    print("-" * 47)

    sweep_results = []
    for target_len in TARGET_LENGTHS:
        input_ids = pad_to_length(tokenizer, base_prompt, target_len)
        actual_len = input_ids.shape[1]

        runs = [measure_prefill_ms(model, input_ids) for _ in range(NUM_RUNS)]
        mean_ms = float(np.mean(runs))
        std_ms = float(np.std(runs))
        speedup = mean_ms / kv_mean

        sweep_results.append({
            "target_len": target_len,
            "actual_len": actual_len,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "runs_ms": runs,
            "speedup_vs_kv": speedup,
        })
        print(f"{actual_len:>8} {mean_ms:>15.1f} {std_ms:>10.2f} {speedup:>9.1f}x")

    # Power law fit
    lengths = np.array([r["actual_len"] for r in sweep_results], dtype=float)
    means = np.array([r["mean_ms"] for r in sweep_results], dtype=float)
    log_fit = np.polyfit(np.log(lengths), np.log(means), 1)
    exponent = float(log_fit[0])
    print(f"\nPower-law fit: prefill ∝ n^{exponent:.3f}  (O(n²) → 2.0)")

    return {
        "kv_prefix_baseline": {
            "len_tokens": kv_prefix_len,
            "mean_ms": kv_mean,
            "std_ms": kv_std,
        },
        "standard_sweep": sweep_results,
        "power_law_exponent": exponent,
    }


# ── Phase 2: Real quality run ─────────────────────────────────────────────────

def phase2_quality_run(model, tokenizer):
    print("\n" + "=" * 60)
    print("PHASE 2: Real End-to-End Quality Run (LRU cache task)")
    print("=" * 60)

    print("\nRunning Agent A (LRU cache analysis)...")
    analysis, analysis_token_count, agent_a_past_kv = run_agent_a(model, tokenizer)

    print(f"\nAgent A produced {analysis_token_count} tokens of analysis.")
    if analysis_token_count < TARGET_QUALITY_LEN:
        print(f"  Note: output shorter than target {TARGET_QUALITY_LEN} tokens. "
              f"Use a more verbose prompt if longer output is needed.")

    print("\nRunning Agent B — Standard pipeline (full re-prefill)...")
    std_output, std_prefill_ms = run_agent_b_standard(model, tokenizer, analysis)
    std_code = extract_code(std_output)
    std_score = score_repair(std_code)
    print(f"  Standard prefill: {std_prefill_ms:.1f} ms | Quality: {std_score}/19")

    print("\nRunning Agent B — KV cache pipeline...")
    # Approximate Agent A input length from its prompt
    prompt_len = len(tokenizer.encode(
        _format_prompt(AGENT_A_PROMPT, BUGGY_LRU_SOURCE), add_special_tokens=False
    ))
    kv_output, kv_prefill_ms = run_agent_b_kv(
        model, tokenizer, agent_a_past_kv, prompt_len, analysis_token_count
    )
    kv_code = extract_code(kv_output)
    kv_score = score_repair(kv_code)
    speedup = std_prefill_ms / kv_prefill_ms
    print(f"  KV prefill:       {kv_prefill_ms:.1f} ms | Quality: {kv_score}/19")
    print(f"  Speedup:          {speedup:.2f}x")

    return {
        "agent_a_output_tokens": analysis_token_count,
        "standard_prefill_ms": std_prefill_ms,
        "standard_quality": f"{std_score}/19",
        "kv_prefill_ms": kv_prefill_ms,
        "kv_quality": f"{kv_score}/19",
        "speedup": speedup,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    phase1 = phase1_timing_sweep(model, tokenizer)

    # Save phase 1 immediately so a phase 2 crash doesn't lose timing data
    phase1_out = RESULTS_DIR / f"exp007_phase1_{ts}.json"
    with open(phase1_out, "w") as f:
        json.dump({"experiment": "EXP-007", "timestamp": ts,
                   "model": MODEL_PATH, "phase1_timing": phase1}, f, indent=2)
    print(f"\nPhase 1 saved to {phase1_out}")

    phase2 = phase2_quality_run(model, tokenizer)

    results = {
        "experiment": "EXP-007",
        "description": "Long-context prefill scaling (800-4000 tokens), LRU cache task",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_PATH,
        "num_runs": NUM_RUNS,
        "task": "LRU cache bug analysis and repair",
        "phase1_timing": phase1,
        "phase2_quality": phase2,
    }

    out = RESULTS_DIR / f"exp007_long_context_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    sweep = phase1["standard_sweep"]
    lengths = [r["actual_len"] for r in sweep]
    speedups = [r["speedup_vs_kv"] for r in sweep]
    for ln, sp in zip(lengths, speedups):
        print(f"  {ln:5d} tokens → {sp:.1f}x speedup")
    print(f"\n  Power-law exponent: {phase1['power_law_exponent']:.3f}")
    print(f"\n  Quality run at ~{phase2['agent_a_output_tokens']} tokens:")
    print(f"    Standard:  {phase2['standard_prefill_ms']:.0f} ms, {phase2['standard_quality']}")
    print(f"    KV cache:  {phase2['kv_prefill_ms']:.0f} ms, {phase2['kv_quality']}")
    print(f"    Speedup:   {phase2['speedup']:.2f}x")


if __name__ == "__main__":
    main()
