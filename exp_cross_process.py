"""
EXP-005 — Cross-Process KV Transfer
=====================================
Orchestrates two separate Python subprocesses to simulate a cross-process
KV cache handoff, as would occur between separate inference workers in a
provider deployment.

Process A: loads model, runs Agent A, serializes KV slice to /dev/shm
Process B: loads model independently, deserializes KV from /dev/shm, runs Agent B

Measures serialize + deserialize time as total inter-process transfer cost.
Compares against in-process clone time from EXP-004 (1.445 ms).
Runs quality check to confirm mechanism works across process boundaries.

Records to results/exp005_cross_process_<timestamp>.json
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
KV_PAYLOAD_PATH = "/dev/shm/kv_a_output.pt"
N_RUNS = 5  # full cross-process runs (each loads model twice — slow but real)

# Reference numbers from EXP-004 for comparison
EXP004_INPROCESS_CLONE_MS = 1.445
EXP004_SPEEDUP = 6.74
EXP004_STD_PREFILL_MS = 682.4


def run_subprocess(script: str) -> dict:
    """Run a script as a separate process, return its stdout JSON."""
    result = subprocess.run(
        [sys.executable, "-u", script],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  STDERR from {script}:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"{script} exited with code {result.returncode}")

    # Last non-empty line should be the JSON result
    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    return json.loads(lines[-1])


def main():
    print("=" * 60)
    print("EXP-005: Cross-Process KV Transfer")
    print("=" * 60)
    print(f"Running {N_RUNS} full cross-process round trips.")
    print("Each run loads the model twice (once per process) — this is slow by design.\n")

    serialize_times = []
    deserialize_times = []
    transfer_totals = []
    quality_results = []

    for i in range(N_RUNS):
        print(f"--- Run {i+1}/{N_RUNS} ---")

        # Process A
        print("  Process A: loading model, generating KV...")
        t_a_start = time.perf_counter()
        result_a = run_subprocess("/workspace/exp_cross_process_a.py")
        t_a_end = time.perf_counter()

        serialize_ms = result_a["serialize_ms"]
        payload_mb = result_a["payload_size_mb"]
        print(f"  Process A done in {t_a_end - t_a_start:.1f}s | serialize: {serialize_ms:.2f} ms | payload: {payload_mb:.1f} MB")

        # Process B
        print("  Process B: loading model, deserializing KV, generating...")
        t_b_start = time.perf_counter()
        result_b = run_subprocess("/workspace/exp_cross_process_b.py")
        t_b_end = time.perf_counter()

        deserialize_ms = result_b["deserialize_ms"]
        passed = result_b["passed"]
        total = result_b["total"]
        print(f"  Process B done in {t_b_end - t_b_start:.1f}s | deserialize: {deserialize_ms:.2f} ms | quality: {passed}/{total}")

        transfer_total = serialize_ms + deserialize_ms
        print(f"  Total transfer cost: {transfer_total:.2f} ms")

        serialize_times.append(serialize_ms)
        deserialize_times.append(deserialize_ms)
        transfer_totals.append(transfer_total)
        quality_results.append({"passed": passed, "total": total})

        # Clean up payload between runs
        if os.path.exists(KV_PAYLOAD_PATH):
            os.remove(KV_PAYLOAD_PATH)

    all_passed = sum(r["passed"] for r in quality_results)
    all_total = sum(r["total"] for r in quality_results)

    results = {
        "experiment": "EXP-005",
        "timestamp": datetime.now().isoformat(),
        "model": "/models/Qwen2.5-7B-Instruct",
        "n_runs": N_RUNS,
        "serialize_ms": serialize_times,
        "deserialize_ms": deserialize_times,
        "transfer_total_ms": transfer_totals,
        "quality": quality_results,
        "summary": {
            "serialize_mean_ms": float(np.mean(serialize_times)),
            "serialize_std_ms": float(np.std(serialize_times)),
            "deserialize_mean_ms": float(np.mean(deserialize_times)),
            "deserialize_std_ms": float(np.std(deserialize_times)),
            "transfer_total_mean_ms": float(np.mean(transfer_totals)),
            "transfer_total_std_ms": float(np.std(transfer_totals)),
            "quality_passed": all_passed,
            "quality_total": all_total,
            "inprocess_clone_ms_ref": EXP004_INPROCESS_CLONE_MS,
            "overhead_vs_inprocess": float(np.mean(transfer_totals)) / EXP004_INPROCESS_CLONE_MS,
            "transfer_as_pct_of_saving": float(np.mean(transfer_totals)) / (EXP004_STD_PREFILL_MS - EXP004_STD_PREFILL_MS / EXP004_SPEEDUP) * 100,
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"exp005_cross_process_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print("\n" + "=" * 60)
    print("SUMMARY — EXP-005")
    print("=" * 60)
    print(f"Serialize (torch.save to /dev/shm):   {s['serialize_mean_ms']:.2f} ± {s['serialize_std_ms']:.2f} ms")
    print(f"Deserialize (torch.load from /dev/shm): {s['deserialize_mean_ms']:.2f} ± {s['deserialize_std_ms']:.2f} ms")
    print(f"Total cross-process transfer:          {s['transfer_total_mean_ms']:.2f} ± {s['transfer_total_std_ms']:.2f} ms")
    print(f"In-process clone (EXP-004 ref):        {EXP004_INPROCESS_CLONE_MS:.3f} ms")
    print(f"Cross-process overhead multiplier:     {s['overhead_vs_inprocess']:.1f}x vs in-process")
    print(f"Transfer as % of prefill saving:       {s['transfer_as_pct_of_saving']:.1f}%")
    print(f"Quality across {N_RUNS} runs:          {all_passed}/{all_total} tests passed")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
