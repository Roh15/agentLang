"""
EXP-003: Prefill Scaling with Sequence Length
===============================================
Measures standard pipeline Agent B prefill time at multiple input token lengths.
KV prefix prefill is also measured (expected constant at ~41 tokens regardless of
prior context length — that's the whole point).

Hypothesis: standard prefill scales ~O(n^1.5 to n^2) with sequence length.
KV prefix cost stays flat. The gap widens with longer contexts.

Records results to results/exp003_scaling_curve_<timestamp>.json + plot.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_PATH, DEVICE, DTYPE, AGENT_PROMPTS, HEAD_DIM, RESULTS_DIR
from pipelines.standard import _format_prompt
from pipelines.kv_cache import _tokenize_prefix

NUM_RUNS = 10
# Token lengths to sweep. 41 is the KV prefix baseline; we go well above it.
TARGET_LENGTHS = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800]


def pad_to_length(tokenizer, base_text: str, target_len: int, device: str) -> torch.Tensor:
    """
    Encode base_text, then pad or truncate to exactly target_len tokens.
    Uses a neutral filler token (space) to reach the target length.
    Returns input_ids of shape [1, target_len].
    """
    ids = tokenizer.encode(base_text, add_special_tokens=False)
    filler_id = tokenizer.encode(" ", add_special_tokens=False)[0]

    if len(ids) >= target_len:
        ids = ids[:target_len]
    else:
        ids = ids + [filler_id] * (target_len - len(ids))

    return torch.tensor([ids], device=device)


def measure_prefill_ms(model, input_ids: torch.Tensor) -> float:
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model loaded.\n")

    # Warmup — run a prefill at max length to warm CUDA kernels
    warmup_ids = pad_to_length(tokenizer, "warmup", TARGET_LENGTHS[-1], DEVICE)
    for _ in range(3):
        measure_prefill_ms(model, warmup_ids)
    print("Warmup done.\n")

    # KV prefix baseline — always 41 tokens regardless of context length
    prefix_ids = _tokenize_prefix(tokenizer, AGENT_PROMPTS["B"], DEVICE)
    kv_prefix_len = prefix_ids.shape[1]
    kv_prefix_runs = [measure_prefill_ms(model, prefix_ids) for _ in range(NUM_RUNS)]
    kv_prefix_mean = float(np.mean(kv_prefix_runs))
    kv_prefix_std = float(np.std(kv_prefix_runs))
    print(f"KV prefix baseline: {kv_prefix_mean:.1f} ± {kv_prefix_std:.2f} ms ({kv_prefix_len} tokens)\n")

    # Standard pipeline sweep
    print(f"{'Length':>8} {'Mean (ms)':>12} {'Std (ms)':>10} {'Speedup':>10}")
    print("-" * 45)

    std_results = []
    base_prompt = _format_prompt(AGENT_PROMPTS["B"], "")

    for target_len in TARGET_LENGTHS:
        input_ids = pad_to_length(tokenizer, base_prompt, target_len, DEVICE)
        actual_len = input_ids.shape[1]

        runs = [measure_prefill_ms(model, input_ids) for _ in range(NUM_RUNS)]
        mean_ms = float(np.mean(runs))
        std_ms = float(np.std(runs))
        speedup = mean_ms / kv_prefix_mean

        std_results.append({
            "target_len": target_len,
            "actual_len": actual_len,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "runs_ms": runs,
            "speedup_vs_kv": speedup,
        })
        print(f"{actual_len:>8} {mean_ms:>12.1f} {std_ms:>10.2f} {speedup:>9.1f}x")

    # Fit power law: prefill_ms = a * n^b
    lengths = np.array([r["actual_len"] for r in std_results], dtype=float)
    means = np.array([r["mean_ms"] for r in std_results], dtype=float)
    log_fit = np.polyfit(np.log(lengths), np.log(means), 1)
    exponent = log_fit[0]
    print(f"\nPower law fit: prefill ∝ n^{exponent:.3f}  (theoretical O(n²) → exponent=2.0)")

    # Plot
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute prefill time vs sequence length
    ax1.plot(lengths, means, "o-", color="#d9534f", label=f"Standard (fit: n^{exponent:.2f})", linewidth=2)
    ax1.axhline(kv_prefix_mean, color="#5cb85c", linestyle="--", linewidth=2,
                label=f"KV prefix (constant {kv_prefix_mean:.0f} ms, {kv_prefix_len} tokens)")
    fit_x = np.linspace(lengths[0], lengths[-1], 200)
    fit_y = np.exp(log_fit[1]) * fit_x ** exponent
    ax1.plot(fit_x, fit_y, "--", color="#d9534f", alpha=0.5, label="Power law fit")
    ax1.fill_between(lengths,
                     [r["mean_ms"] - r["std_ms"] for r in std_results],
                     [r["mean_ms"] + r["std_ms"] for r in std_results],
                     alpha=0.2, color="#d9534f")
    ax1.set_xlabel("Input token count")
    ax1.set_ylabel("Prefill time (ms)")
    ax1.set_title("Prefill Time vs Sequence Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: speedup vs sequence length
    speedups = [r["speedup_vs_kv"] for r in std_results]
    ax2.plot(lengths, speedups, "o-", color="#5bc0de", linewidth=2)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("Input token count (standard pipeline)")
    ax2.set_ylabel("Speedup (standard / KV prefix)")
    ax2.set_title(f"KV Cache Speedup vs Context Length\n(KV prefix = {kv_prefix_len} tokens = constant)")
    ax2.grid(True, alpha=0.3)
    for x, y in zip(lengths, speedups):
        ax2.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    fig.suptitle(
        f"EXP-003: Prefill Scaling — Qwen2.5-3B-Instruct | n={NUM_RUNS} runs/point",
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"exp003_scaling_{ts}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    results = {
        "experiment": "EXP-003",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_PATH,
        "num_runs": NUM_RUNS,
        "kv_prefix_baseline": {
            "len_tokens": kv_prefix_len,
            "mean_ms": kv_prefix_mean,
            "std_ms": kv_prefix_std,
        },
        "standard_sweep": std_results,
        "power_law_exponent": float(exponent),
        "theoretical_exponent": 2.0,
    }

    out = RESULTS_DIR / f"exp003_scaling_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPlot: {plot_path}")
    print(f"Data: {out}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequence length range: {int(lengths[0])} – {int(lengths[-1])} tokens")
    print(f"Standard prefill range: {means[0]:.0f} ms – {means[-1]:.0f} ms")
    print(f"KV prefix (constant):   {kv_prefix_mean:.0f} ms")
    print(f"Speedup range:          {speedups[0]:.1f}x – {speedups[-1]:.1f}x")
    print(f"Power law exponent:     {exponent:.3f}  (O(n^{exponent:.2f}))")


if __name__ == "__main__":
    main()
