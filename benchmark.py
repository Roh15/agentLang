"""
Latent Agent Communication Benchmark
=====================================
Run both pipelines across three experiments and write results to results/.

Before running:
    sudo nvpmodel -m 0
    sudo jetson_clocks
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_PATH, RESULTS_DIR, DEVICE, DTYPE,
    NUM_WARMUP_RUNS, NUM_TIMED_RUNS, HOP_COUNTS,
    HOP_AGENT_SEQUENCE, AGENT_PROMPTS, STRICT_ROPE_REINDEX, HEAD_DIM,
)
from pipelines import run_standard_pipeline, run_kv_cache_pipeline


# ---------------------------------------------------------------------------
# Setup / hardware helpers
# ---------------------------------------------------------------------------

def check_clock_lock():
    """Warn if jetson_clocks hasn't been run (GR3D_FREQ will be 0 at idle)."""
    try:
        proc = subprocess.Popen(
            ["/usr/bin/tegrastats", "--interval", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        line = proc.stdout.readline()
        proc.kill()
        if "GR3D_FREQ" in line:
            print(f"[tegrastats] {line.strip()}")
    except Exception:
        pass
    print(
        "\n⚠  Reminder: Run 'sudo nvpmodel -m 0 && sudo jetson_clocks' before "
        "timing runs for reproducible results.\n"
    )


def sample_ram_mb() -> float:
    """Read current RAM usage from tegrastats."""
    try:
        proc = subprocess.Popen(
            ["/usr/bin/tegrastats", "--interval", "200"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        line = proc.stdout.readline()
        proc.kill()
        m = re.search(r"RAM (\d+)/\d+MB", line)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return float(torch.cuda.memory_allocated() // (1024 ** 2))


def load_model(model_path: str, device: str, dtype):
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------

def extract_code_block(text: str) -> Optional[str]:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if not m:
        m = re.search(r"```\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def run_pytest(code: str, test_file: str = "test_buggy_function.py") -> Tuple[int, int]:
    """Write code to a temp file and run pytest. Returns (passed, total)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file,
             f"--repaired-path={tmp_path}", "-q", "--tb=no"],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout + result.stderr
        m_pass = re.search(r"(\d+) passed", output)
        m_fail = re.search(r"(\d+) failed", output)
        passed = int(m_pass.group(1)) if m_pass else 0
        failed = int(m_fail.group(1)) if m_fail else 0
        total = passed + failed
        return passed, total
    except Exception as e:
        print(f"  pytest error: {e}")
        return 0, 0
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Experiment 1 — Prefill savings (isolated measurement)
# ---------------------------------------------------------------------------

def experiment_1(model, tokenizer, buggy_fn_text: str, device: str, dtype) -> dict:
    """
    Isolate Agent B prefill time for both pipelines.
    Standard: full input (sys_B + agent_A_output)
    KV: prefix only (sys_B system prompt tokens)
    Also measures KV clone time as simulated transfer cost.
    """
    print("\n=== Experiment 1: Prefill Savings ===")

    # First: get Agent A's output (same for both pipelines)
    agent_a_result = run_standard_pipeline(
        model, tokenizer, ["A"], AGENT_PROMPTS, buggy_fn_text,
        device, dtype, 512,
    )
    agent_a_output = agent_a_result.agent_outputs[0]
    print(f"  Agent A output length: {len(agent_a_output)} chars")

    from pipelines.standard import _format_prompt
    from pipelines.kv_cache import (
        _tokenize_prefix, measure_kv_clone_us,
        _run_agent_a, kv_cache_size_mb,
    )

    # ---- Standard pipeline: time Agent B prefill ----
    prompt = _format_prompt(AGENT_PROMPTS["B"], agent_a_output)
    full_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    std_prefill_ms = []
    print(f"  Running {NUM_WARMUP_RUNS} warmup + {NUM_TIMED_RUNS} timed runs (standard)...")
    for i in range(NUM_WARMUP_RUNS + NUM_TIMED_RUNS):
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=full_input_ids, use_cache=True)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
        if i >= NUM_WARMUP_RUNS:
            std_prefill_ms.append(ms)
            print(f"    run {i - NUM_WARMUP_RUNS + 1}/{NUM_TIMED_RUNS}: {ms:.1f} ms")

    # ---- KV pipeline: get Agent A's generation slice, then time prefix-only prefill ----
    _, _, gen_slice_a = _run_agent_a(
        model, tokenizer, AGENT_PROMPTS["A"], buggy_fn_text,
        device, dtype, 512, HEAD_DIM, STRICT_ROPE_REINDEX,
    )
    kv_size_mb = kv_cache_size_mb(gen_slice_a.kv)
    print(f"  Agent A generation KV size: {kv_size_mb:.2f} MB ({gen_slice_a.length} tokens)")

    prefix_ids = _tokenize_prefix(tokenizer, AGENT_PROMPTS["B"], device)
    kv_prefill_ms = []
    kv_clone_us = []
    print(f"  Running {NUM_WARMUP_RUNS} warmup + {NUM_TIMED_RUNS} timed runs (KV pipeline)...")
    for i in range(NUM_WARMUP_RUNS + NUM_TIMED_RUNS):
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=prefix_ids, use_cache=True)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
        clone_us = measure_kv_clone_us(gen_slice_a.kv)
        if i >= NUM_WARMUP_RUNS:
            kv_prefill_ms.append(ms)
            kv_clone_us.append(clone_us)
            print(f"    run {i - NUM_WARMUP_RUNS + 1}/{NUM_TIMED_RUNS}: prefill {ms:.1f} ms | clone {clone_us:.0f} µs")

    result = {
        "standard_prefill_ms": std_prefill_ms,
        "kv_prefill_ms": kv_prefill_ms,
        "kv_clone_us": kv_clone_us,
        "kv_payload_size_mb": kv_size_mb,
        "std_input_tokens": full_input_ids.shape[1],
        "kv_prefix_tokens": prefix_ids.shape[1],
        "agent_a_output_tokens": gen_slice_a.length,
        "summary": {
            "std_prefill_mean_ms": float(np.mean(std_prefill_ms)),
            "kv_prefill_mean_ms": float(np.mean(kv_prefill_ms)),
            "kv_clone_mean_us": float(np.mean(kv_clone_us)),
            "speedup": float(np.mean(std_prefill_ms) / np.mean(kv_prefill_ms)),
        },
    }
    print(f"\n  Speedup: {result['summary']['speedup']:.1f}x")
    print(f"  Transfer overhead: {np.mean(kv_clone_us):.0f} µs vs {np.mean(std_prefill_ms):.0f} ms prefill savings")
    return result


# ---------------------------------------------------------------------------
# Experiment 2 — Quality preservation
# ---------------------------------------------------------------------------

def experiment_2(model, tokenizer, buggy_fn_text: str, device: str, dtype) -> dict:
    """
    Run both pipelines end-to-end and check output correctness via pytest.
    """
    print("\n=== Experiment 2: Quality Preservation ===")

    std_result = run_standard_pipeline(
        model, tokenizer, ["A", "B"], AGENT_PROMPTS, buggy_fn_text,
        device, dtype, 512,
    )
    std_output = std_result.agent_outputs[1]
    std_code = extract_code_block(std_output)
    if std_code:
        std_passed, std_total = run_pytest(std_code)
    else:
        print("  WARNING: No code block found in standard pipeline Agent B output")
        std_passed, std_total = 0, 0

    kv_result = run_kv_cache_pipeline(
        model, tokenizer, ["A", "B"], AGENT_PROMPTS, buggy_fn_text,
        device, dtype, 512, HEAD_DIM, STRICT_ROPE_REINDEX,
    )
    kv_output = kv_result.agent_outputs[1]
    kv_code = extract_code_block(kv_output)
    if kv_code:
        kv_passed, kv_total = run_pytest(kv_code)
    else:
        print("  WARNING: No code block found in KV pipeline Agent B output")
        kv_passed, kv_total = 0, 0

    print(f"  Standard pipeline: {std_passed}/{std_total} tests passed")
    print(f"  KV cache pipeline: {kv_passed}/{kv_total} tests passed")

    return {
        "standard": {
            "passed": std_passed, "total": std_total,
            "output": std_output, "code_extracted": std_code is not None,
        },
        "kv_cache": {
            "passed": kv_passed, "total": kv_total,
            "output": kv_output, "code_extracted": kv_code is not None,
        },
    }


# ---------------------------------------------------------------------------
# Experiment 3 — Multi-hop scaling
# ---------------------------------------------------------------------------

def experiment_3(model, tokenizer, buggy_fn_text: str, device: str, dtype) -> dict:
    """
    Measure per-hop prefill cost for both pipelines at 2, 3, 4 hops.
    Standard pipeline: prefill grows with cumulative context.
    KV pipeline: prefill stays flat (only system prompt tokens per hop).
    """
    print("\n=== Experiment 3: Multi-Hop Scale ===")
    results = {}

    for hop_count in HOP_COUNTS:
        agent_keys = HOP_AGENT_SEQUENCE[hop_count]
        print(f"\n  {hop_count}-hop pipeline:")

        std_result = run_standard_pipeline(
            model, tokenizer, agent_keys, AGENT_PROMPTS, buggy_fn_text,
            device, dtype, 512,
        )
        kv_result = run_kv_cache_pipeline(
            model, tokenizer, agent_keys, AGENT_PROMPTS, buggy_fn_text,
            device, dtype, 512, HEAD_DIM, STRICT_ROPE_REINDEX,
        )

        for j, key in enumerate(agent_keys):
            print(
                f"    Agent {key}: std={std_result.prefill_times_ms[j]:.0f} ms | "
                f"kv={kv_result.prefill_times_ms[j]:.0f} ms"
            )

        results[str(hop_count)] = {
            "agent_keys": agent_keys,
            "standard_prefill_ms": std_result.prefill_times_ms,
            "kv_prefill_ms": kv_result.prefill_times_ms,
            "standard_e2e_ms": std_result.e2e_latency_ms,
            "kv_e2e_ms": kv_result.e2e_latency_ms,
            "kv_payload_sizes_mb": kv_result.kv_payload_sizes_mb,
        }

    return results


# ---------------------------------------------------------------------------
# Results writing and plotting
# ---------------------------------------------------------------------------

def write_results(results: dict, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"results_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {path}")
    return path


def plot_results(results: dict, out_dir: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Experiment 1 — prefill time comparison
    e1 = results["experiment_1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Standard\n(full re-prefill)", "KV Cache\n(prefix only)"]
    means = [
        np.mean(e1["standard_prefill_ms"]),
        np.mean(e1["kv_prefill_ms"]),
    ]
    stds = [
        np.std(e1["standard_prefill_ms"]),
        np.std(e1["kv_prefill_ms"]),
    ]
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=["#d9534f", "#5cb85c"], width=0.4)
    ax.set_ylabel("Agent B Prefill Time (ms)")
    ax.set_title(
        f"Prefill Time: Standard vs KV Cache Pipeline\n"
        f"Standard: {e1['std_input_tokens']} tokens | "
        f"KV prefix: {e1['kv_prefix_tokens']} tokens | "
        f"Speedup: {e1['summary']['speedup']:.1f}x"
    )
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 2, f"{mean:.0f} ms",
                ha="center", va="bottom", fontweight="bold")
    # Annotate clone overhead
    ax.annotate(
        f"Transfer overhead: {e1['summary']['kv_clone_mean_us']:.0f} µs\n"
        f"({e1['kv_payload_size_mb']:.1f} MB payload)",
        xy=(1, means[1]), xytext=(1.3, means[1] + 30),
        arrowprops=dict(arrowstyle="->"), fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"exp1_prefill_{ts}.png", dpi=150)
    plt.close(fig)

    # Plot 2: Experiment 3 — per-hop prefill cost
    e3 = results["experiment_3"]
    fig, axes = plt.subplots(1, len(HOP_COUNTS), figsize=(5 * len(HOP_COUNTS), 5), sharey=False)
    if len(HOP_COUNTS) == 1:
        axes = [axes]

    for ax, hop_count in zip(axes, HOP_COUNTS):
        data = e3[str(hop_count)]
        x = np.arange(hop_count)
        w = 0.35
        ax.bar(x - w/2, data["standard_prefill_ms"], w, label="Standard", color="#d9534f")
        ax.bar(x + w/2, data["kv_prefill_ms"], w, label="KV Cache", color="#5cb85c")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Agent {k}" for k in data["agent_keys"]])
        ax.set_ylabel("Prefill Time (ms)")
        ax.set_title(f"{hop_count}-Hop Pipeline")
        ax.legend()

    fig.suptitle(
        "Per-Hop Prefill Cost: Standard (growing context) vs KV Cache (flat)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"exp3_multihop_{ts}.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    check_clock_lock()
    buggy_fn_text = Path("buggy_function.py").read_text()
    model, tokenizer = load_model(MODEL_PATH, DEVICE, DTYPE)

    print("\n--- Warmup run (discarded) ---")
    ram_before = sample_ram_mb()
    run_standard_pipeline(
        model, tokenizer, ["A", "B"], AGENT_PROMPTS, buggy_fn_text,
        DEVICE, DTYPE, 256,
    )
    ram_after = sample_ram_mb()
    print(f"RAM delta during warmup: {ram_after - ram_before:.0f} MB")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_PATH,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "num_timed_runs": NUM_TIMED_RUNS,
            "strict_rope_reindex": STRICT_ROPE_REINDEX,
        },
        "experiment_1": experiment_1(model, tokenizer, buggy_fn_text, DEVICE, DTYPE),
        "experiment_2": experiment_2(model, tokenizer, buggy_fn_text, DEVICE, DTYPE),
        "experiment_3": experiment_3(model, tokenizer, buggy_fn_text, DEVICE, DTYPE),
    }

    write_results(results, RESULTS_DIR)
    plot_results(results, RESULTS_DIR)

    # Summary table
    e1 = results["experiment_1"]["summary"]
    e2 = results["experiment_2"]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Prefill speedup (Agent B):  {e1['speedup']:.1f}x")
    print(f"Transfer overhead:          {e1['kv_clone_mean_us']:.0f} µs")
    print(f"Standard quality:           {e2['standard']['passed']}/{e2['standard']['total']} tests")
    print(f"KV cache quality:           {e2['kv_cache']['passed']}/{e2['kv_cache']['total']} tests")
    print("=" * 60)


if __name__ == "__main__":
    main()
