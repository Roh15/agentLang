"""
EXP-002: RoPE Re-indexing Ablation
====================================
Runs the 2-hop KV pipeline with STRICT_ROPE_REINDEX=False (no position correction)
and compares output quality against the full re-indexing result from EXP-001.

Records results to results/exp002_rope_ablation_<timestamp>.json
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_PATH, DEVICE, DTYPE, AGENT_PROMPTS, HEAD_DIM, RESULTS_DIR
from pipelines.kv_cache import run_kv_cache_pipeline
from pipelines.standard import run_standard_pipeline


def run_and_test(model, tokenizer, strict_rope: bool, label: str) -> dict:
    buggy_fn = Path("buggy_function.py").read_text()

    result = run_kv_cache_pipeline(
        model, tokenizer, ["A", "B"], AGENT_PROMPTS, buggy_fn,
        DEVICE, DTYPE, 512, HEAD_DIM, strict_rope=strict_rope,
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
             f"--repaired-path={tmp}", "-v", "--tb=short"],
            capture_output=True, text=True,
        )
        os.unlink(tmp)
        m_pass = re.search(r"(\d+) passed", proc.stdout)
        m_fail = re.search(r"(\d+) failed", proc.stdout)
        passed = int(m_pass.group(1)) if m_pass else 0
        failed = int(m_fail.group(1)) if m_fail else 0
        total = passed + failed
        print(f"\n{label}:\n{proc.stdout[-2000:]}")
    else:
        print(f"\n{label}: No code block extracted. Output:\n{output[:500]}")

    return {
        "strict_rope": strict_rope,
        "passed": passed,
        "total": total,
        "code_extracted": code is not None,
        "output": output,
    }


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model loaded.")

    # Warmup
    buggy_fn = Path("buggy_function.py").read_text()
    run_kv_cache_pipeline(
        model, tokenizer, ["A", "B"], AGENT_PROMPTS, buggy_fn,
        DEVICE, DTYPE, 256, HEAD_DIM, strict_rope=True,
    )
    print("Warmup done.\n")

    print("=" * 60)
    print("EXP-002: RoPE Re-indexing Ablation")
    print("=" * 60)

    with_rope = run_and_test(model, tokenizer, strict_rope=True, label="With RoPE re-indexing (STRICT=True)")
    without_rope = run_and_test(model, tokenizer, strict_rope=False, label="Without RoPE re-indexing (STRICT=False)")

    results = {
        "experiment": "EXP-002",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_PATH,
        "with_rope_reindex": with_rope,
        "without_rope_reindex": without_rope,
        "conclusion": (
            "RoPE re-indexing is NECESSARY"
            if with_rope["passed"] > without_rope["passed"]
            else "RoPE re-indexing makes NO DIFFERENCE for this task"
            if with_rope["passed"] == without_rope["passed"]
            else "UNEXPECTED: without re-indexing is better (investigate)"
        ),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"exp002_rope_ablation_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"With RoPE re-indexing:    {with_rope['passed']}/{with_rope['total']} tests passed")
    print(f"Without RoPE re-indexing: {without_rope['passed']}/{without_rope['total']} tests passed")
    print(f"Conclusion: {results['conclusion']}")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
