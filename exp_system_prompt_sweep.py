"""
EXP-006 — System Prompt Length Sensitivity
============================================
Measures prefill speedup across a 2D grid of:
  - Prior context lengths (how much Agent A wrote): 200, 400, 600, 800 tokens
  - System prompt lengths (how long Agent B's prompt is): 41, 100, 200, 400 tokens

The speedup ratio is approximately prior_context / system_prompt. This experiment
confirms that relationship and shows the operating envelope — where the mechanism
pays off and where it doesn't.

Key insight: the 41-token row is our EXP-001/004 operating point. Real production
agents often have longer system prompts (100-400 tokens), which reduces the speedup.

Records to results/exp006_prompt_sweep_<timestamp>.json + heatmap plot.
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

from config import DTYPE, RESULTS_DIR

MODEL_PATH = "/models/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
NUM_RUNS = 10

PRIOR_CONTEXT_LENGTHS = [200, 400, 600, 800]   # tokens — Agent A's output length
SYSTEM_PROMPT_LENGTHS = [41, 100, 200, 400]     # tokens — Agent B's system prompt length


def pad_ids_to_length(tokenizer, target_len: int, device: str) -> torch.Tensor:
    """Construct a token sequence of exactly target_len tokens using neutral filler."""
    filler_id = tokenizer.encode(" the", add_special_tokens=False)[0]
    ids = [filler_id] * target_len
    return torch.tensor([ids], device=device)


def build_standard_input(tokenizer, prior_len: int, sys_len: int, device: str) -> torch.Tensor:
    """
    Construct a token sequence representing: [sys_prompt_tokens | prior_context_tokens]
    Total length = sys_len + prior_len.
    We build it directly as token ids rather than via text to get exact lengths.
    """
    filler_id = tokenizer.encode(" the", add_special_tokens=False)[0]
    ids = [filler_id] * (sys_len + prior_len)
    return torch.tensor([ids], device=device)


def build_kv_prefix(tokenizer, sys_len: int, device: str) -> torch.Tensor:
    """Construct a token sequence of exactly sys_len tokens (the KV prefix)."""
    filler_id = tokenizer.encode(" the", add_special_tokens=False)[0]
    ids = [filler_id] * sys_len
    return torch.tensor([ids], device=device)


def measure_prefill(model, input_ids: torch.Tensor, n_runs: int) -> tuple:
    runs = []
    for _ in range(n_runs):
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=input_ids, use_cache=True)
            torch.cuda.synchronize()
        runs.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(runs)), float(np.std(runs))


def main():
    print(f"Loading {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model loaded.\n")

    # Warmup at max size
    warmup_ids = build_standard_input(tokenizer, PRIOR_CONTEXT_LENGTHS[-1], SYSTEM_PROMPT_LENGTHS[-1], DEVICE)
    for _ in range(3):
        measure_prefill(model, warmup_ids, 1)
    print("Warmup done.\n")

    print("=" * 60)
    print("EXP-006: System Prompt Length Sensitivity")
    print("=" * 60)
    print(f"Grid: {len(PRIOR_CONTEXT_LENGTHS)} prior lengths × {len(SYSTEM_PROMPT_LENGTHS)} sys prompt lengths")
    print(f"Runs per cell: {NUM_RUNS}\n")

    # Results grid: [prior_len_idx][sys_len_idx]
    std_means = np.zeros((len(PRIOR_CONTEXT_LENGTHS), len(SYSTEM_PROMPT_LENGTHS)))
    std_stds = np.zeros_like(std_means)
    kv_means = np.zeros_like(std_means)
    kv_stds = np.zeros_like(std_means)
    speedups = np.zeros_like(std_means)

    header_label = "Prior\\Sys"
    header = f"{header_label:>10}" + "".join(f"{sl:>10}" for sl in SYSTEM_PROMPT_LENGTHS)
    print(f"\nSpeedup grid (standard_prefill / kv_prefix_prefill):")
    print(header)
    print("-" * (10 + 10 * len(SYSTEM_PROMPT_LENGTHS)))

    all_cells = []

    for i, prior_len in enumerate(PRIOR_CONTEXT_LENGTHS):
        row_str = f"{prior_len:>10}"
        for j, sys_len in enumerate(SYSTEM_PROMPT_LENGTHS):
            # Standard: full input (sys + prior)
            std_ids = build_standard_input(tokenizer, prior_len, sys_len, DEVICE)
            std_mean, std_std = measure_prefill(model, std_ids, NUM_RUNS)

            # KV prefix: sys only
            kv_ids = build_kv_prefix(tokenizer, sys_len, DEVICE)
            kv_mean, kv_std = measure_prefill(model, kv_ids, NUM_RUNS)

            speedup = std_mean / kv_mean

            std_means[i, j] = std_mean
            std_stds[i, j] = std_std
            kv_means[i, j] = kv_mean
            kv_stds[i, j] = kv_std
            speedups[i, j] = speedup

            row_str += f"{speedup:>9.1f}x"

            all_cells.append({
                "prior_len": prior_len,
                "sys_len": sys_len,
                "std_mean_ms": std_mean,
                "std_std_ms": std_std,
                "kv_mean_ms": kv_mean,
                "kv_std_ms": kv_std,
                "speedup": speedup,
            })

        print(row_str)

    # Heatmap plot
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: speedup heatmap
    ax = axes[0]
    im = ax.imshow(speedups, cmap="RdYlGn", vmin=1.0, vmax=speedups.max(), aspect="auto")
    ax.set_xticks(range(len(SYSTEM_PROMPT_LENGTHS)))
    ax.set_xticklabels([f"{s}" for s in SYSTEM_PROMPT_LENGTHS])
    ax.set_yticks(range(len(PRIOR_CONTEXT_LENGTHS)))
    ax.set_yticklabels([f"{p}" for p in PRIOR_CONTEXT_LENGTHS])
    ax.set_xlabel("System prompt length (tokens) — Agent B prefix cost")
    ax.set_ylabel("Prior context length (tokens) — Agent A output")
    ax.set_title("Speedup: Standard / KV Prefix Prefill")
    plt.colorbar(im, ax=ax, label="Speedup (x)")
    for i in range(len(PRIOR_CONTEXT_LENGTHS)):
        for j in range(len(SYSTEM_PROMPT_LENGTHS)):
            ax.text(j, i, f"{speedups[i, j]:.1f}x",
                    ha="center", va="center", fontweight="bold", fontsize=11,
                    color="black" if speedups[i, j] < speedups.max() * 0.7 else "white")

    # Right: absolute prefill times
    ax2 = axes[1]
    x = np.arange(len(PRIOR_CONTEXT_LENGTHS))
    w = 0.18
    colors_std = ["#d9534f", "#c0392b", "#a93226", "#922b21"]
    colors_kv = ["#5cb85c", "#27ae60", "#1e8449", "#196f3d"]

    for j, (sys_len, cs, ck) in enumerate(zip(SYSTEM_PROMPT_LENGTHS, colors_std, colors_kv)):
        offset = (j - 1.5) * w
        ax2.bar(x + offset - w*0.5, std_means[:, j], w, label=f"Std sys={sys_len}", color=cs, alpha=0.85)
        ax2.bar(x + offset + w*0.5, kv_means[:, j], w, label=f"KV sys={sys_len}", color=ck, alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p} tok" for p in PRIOR_CONTEXT_LENGTHS])
    ax2.set_xlabel("Prior context length (Agent A output)")
    ax2.set_ylabel("Prefill time (ms)")
    ax2.set_title("Absolute Prefill Times")
    ax2.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"EXP-006: System Prompt Length Sensitivity — Qwen2.5-7B | n={NUM_RUNS} runs/cell",
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"exp006_prompt_sweep_{ts}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    results = {
        "experiment": "EXP-006",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_PATH,
        "num_runs": NUM_RUNS,
        "prior_context_lengths": PRIOR_CONTEXT_LENGTHS,
        "system_prompt_lengths": SYSTEM_PROMPT_LENGTHS,
        "cells": all_cells,
        "speedup_grid": speedups.tolist(),
        "std_prefill_grid_ms": std_means.tolist(),
        "kv_prefill_grid_ms": kv_means.tolist(),
    }

    out = RESULTS_DIR / f"exp006_prompt_sweep_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPlot: {plot_path}")
    print(f"Data: {out}")

    # Highlight the break-even point
    print("\nBreak-even analysis (speedup < 1.5x = marginal benefit):")
    for cell in all_cells:
        if cell["speedup"] < 1.5:
            print(f"  prior={cell['prior_len']}, sys={cell['sys_len']}: {cell['speedup']:.2f}x — MARGINAL")


if __name__ == "__main__":
    main()
