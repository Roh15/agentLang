"""
EXP-004: 7B Model Replication
================================
Replicates EXP-001 Experiments 1 and 2 on Qwen2.5-7B-Instruct to test
whether the prefill savings mechanism generalises to larger models and to
measure the absolute saving at production-relevant model scale.

Records results to results/exp004_7b_<timestamp>.json + plot.
"""
import json
import re
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, DTYPE, AGENT_PROMPTS, RESULTS_DIR, STRICT_ROPE_REINDEX
from pipelines.standard import run_standard_pipeline, _format_prompt
from pipelines.kv_cache import run_kv_cache_pipeline, _tokenize_prefix, _run_agent_a, kv_cache_size_mb, measure_kv_clone_us

MODEL_PATH_7B = "/models/Qwen2.5-7B-Instruct"
NUM_TIMED_RUNS = 20
NUM_WARMUP_RUNS = 1
MAX_NEW_TOKENS = 512


def get_head_dim(model) -> int:
    cfg = model.config
    return cfg.hidden_size // cfg.num_attention_heads


def measure_prefill_ms(model, input_ids):
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def run_and_test(model, tokenizer, strict_rope: bool, head_dim: int) -> dict:
    buggy_fn = Path("buggy_function.py").read_text()
    result = run_kv_cache_pipeline(
        model, tokenizer, ["A", "B"], AGENT_PROMPTS, buggy_fn,
        DEVICE, DTYPE, MAX_NEW_TOKENS, head_dim, strict_rope=strict_rope,
    )
    output = result.agent_outputs[1]
    m = re.search(r"```python\n(.*?)```", output, re.DOTALL)
    if not m:
        m = re.search(r"```\n(.*?)```", output, re.DOTALL)
    code = m.group(1).strip() if m else None

    passed, total = 0, 0
    if code:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp = f.name
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "test_buggy_function.py",
             f"--repaired-path={tmp}", "-q", "--tb=no"],
            capture_output=True, text=True,
        )
        os.unlink(tmp)
        m_pass = re.search(r"(\d+) passed", proc.stdout)
        m_fail = re.search(r"(\d+) failed", proc.stdout)
        passed = int(m_pass.group(1)) if m_pass else 0
        total = passed + (int(m_fail.group(1)) if m_fail else 0)

    return {"passed": passed, "total": total, "code_extracted": code is not None, "output": output}


def main():
    print(f"Loading {MODEL_PATH_7B} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_7B, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_7B)

    cfg = model.config
    head_dim = get_head_dim(model)
    print(f"Architecture: {cfg.num_hidden_layers} layers, {cfg.num_key_value_heads} KV heads, {head_dim} head_dim")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    buggy_fn = Path("buggy_function.py").read_text()

    # Warmup
    print("\nWarmup...")
    run_standard_pipeline(model, tokenizer, ["A", "B"], AGENT_PROMPTS, buggy_fn, DEVICE, DTYPE, 256)
    print("Warmup done.\n")

    # ---- EXP-004 Part 1: Prefill timing ----
    print("=== Part 1: Prefill Savings (n=20) ===")

    # Get Agent A output first
    std_a = run_standard_pipeline(model, tokenizer, ["A"], AGENT_PROMPTS, buggy_fn, DEVICE, DTYPE, MAX_NEW_TOKENS)
    agent_a_output = std_a.agent_outputs[0]
    print(f"Agent A output: {len(agent_a_output)} chars")

    # Standard: time Agent B full prefill
    prompt = _format_prompt(AGENT_PROMPTS["B"], agent_a_output)
    full_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    print(f"Standard input tokens: {full_input_ids.shape[1]}")

    std_runs = []
    for i in range(NUM_WARMUP_RUNS + NUM_TIMED_RUNS):
        ms = measure_prefill_ms(model, full_input_ids)
        if i >= NUM_WARMUP_RUNS:
            std_runs.append(ms)
            print(f"  std run {i - NUM_WARMUP_RUNS + 1}/{NUM_TIMED_RUNS}: {ms:.1f} ms")

    # KV: get Agent A slice, time prefix-only prefill
    _, _, gen_slice_a = _run_agent_a(
        model, tokenizer, AGENT_PROMPTS["A"], buggy_fn,
        DEVICE, DTYPE, MAX_NEW_TOKENS, head_dim, STRICT_ROPE_REINDEX,
    )
    kv_size = kv_cache_size_mb(gen_slice_a.kv)
    prefix_ids = _tokenize_prefix(tokenizer, AGENT_PROMPTS["B"], DEVICE)
    print(f"KV prefix tokens: {prefix_ids.shape[1]}")
    print(f"KV payload size: {kv_size:.1f} MB ({gen_slice_a.length} tokens)")

    kv_runs, clone_runs = [], []
    for i in range(NUM_WARMUP_RUNS + NUM_TIMED_RUNS):
        ms = measure_prefill_ms(model, prefix_ids)
        us = measure_kv_clone_us(gen_slice_a.kv)
        if i >= NUM_WARMUP_RUNS:
            kv_runs.append(ms)
            clone_runs.append(us)
            print(f"  kv run {i - NUM_WARMUP_RUNS + 1}/{NUM_TIMED_RUNS}: {ms:.1f} ms | clone {us:.0f} µs")

    speedup = float(np.mean(std_runs) / np.mean(kv_runs))
    print(f"\nSpeedup: {speedup:.2f}x")

    # ---- EXP-004 Part 2: Quality ----
    print("\n=== Part 2: Quality Preservation ===")
    std_quality = run_standard_pipeline(model, tokenizer, ["A", "B"], AGENT_PROMPTS, buggy_fn, DEVICE, DTYPE, MAX_NEW_TOKENS)
    std_output = std_quality.agent_outputs[1]
    m = re.search(r"```python\n(.*?)```", std_output, re.DOTALL)
    std_code = m.group(1).strip() if m else None
    if std_code:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(std_code)
            tmp = f.name
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "test_buggy_function.py",
             f"--repaired-path={tmp}", "-q", "--tb=no"],
            capture_output=True, text=True,
        )
        os.unlink(tmp)
        m_p = re.search(r"(\d+) passed", proc.stdout)
        std_passed = int(m_p.group(1)) if m_p else 0
        std_total = 6
    else:
        std_passed, std_total = 0, 0

    kv_quality = run_and_test(model, tokenizer, strict_rope=True, head_dim=head_dim)
    print(f"Standard: {std_passed}/{std_total} | KV: {kv_quality['passed']}/{kv_quality['total']}")

    # Save results
    results = {
        "experiment": "EXP-004",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_PATH_7B,
        "architecture": {
            "num_layers": cfg.num_hidden_layers,
            "num_kv_heads": cfg.num_key_value_heads,
            "head_dim": head_dim,
            "hidden_size": cfg.hidden_size,
        },
        "part1_prefill": {
            "std_input_tokens": full_input_ids.shape[1],
            "kv_prefix_tokens": prefix_ids.shape[1],
            "agent_a_output_tokens": gen_slice_a.length,
            "kv_payload_size_mb": kv_size,
            "standard_prefill_ms": std_runs,
            "kv_prefill_ms": kv_runs,
            "kv_clone_us": clone_runs,
            "summary": {
                "std_mean_ms": float(np.mean(std_runs)),
                "std_std_ms": float(np.std(std_runs)),
                "kv_mean_ms": float(np.mean(kv_runs)),
                "kv_std_ms": float(np.std(kv_runs)),
                "clone_mean_us": float(np.mean(clone_runs)),
                "speedup": speedup,
            },
        },
        "part2_quality": {
            "standard": {"passed": std_passed, "total": std_total},
            "kv_cache": {"passed": kv_quality["passed"], "total": kv_quality["total"]},
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"exp004_7b_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Plot comparison with 3B (load EXP-001 numbers)
    exp001_std = 301.2
    exp001_kv = 93.5
    exp004_std = float(np.mean(std_runs))
    exp004_kv = float(np.mean(kv_runs))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(2)
    w = 0.3
    ax.bar(x - w/2, [exp001_std, exp001_kv], w, label="3B model", color=["#d9534f", "#5cb85c"])
    ax.bar(x + w/2, [exp004_std, exp004_kv], w, label="7B model", color=["#a02020", "#2d7a2d"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Standard\n(full re-prefill)", "KV Cache\n(prefix only)"])
    ax.set_ylabel("Agent B Prefill Time (ms)")
    ax.set_title(
        f"Prefill Time: 3B vs 7B Model\n"
        f"3B speedup: {exp001_std/exp001_kv:.1f}x | 7B speedup: {exp004_std/exp004_kv:.1f}x"
    )
    ax.legend()
    for rect, val in zip(ax.patches, [exp001_std, exp004_std, exp001_kv, exp004_kv]):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"exp004_7b_{ts}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("SUMMARY — EXP-004 (7B)")
    print("=" * 60)
    print(f"Standard prefill:    {exp004_std:.1f} ± {float(np.std(std_runs)):.1f} ms")
    print(f"KV prefix prefill:   {exp004_kv:.1f} ± {float(np.std(kv_runs)):.1f} ms")
    print(f"Speedup:             {speedup:.2f}x")
    print(f"Transfer cost:       {float(np.mean(clone_runs)):.0f} µs for {kv_size:.0f} MB")
    print(f"Quality standard:    {std_passed}/{std_total}")
    print(f"Quality KV cache:    {kv_quality['passed']}/{kv_quality['total']}")
    print(f"Results: {out}")
    print(f"Plot:    {plot_path}")


if __name__ == "__main__":
    main()
