"""Quick quality check: run KV pipeline, extract code, run pytest."""
import os
import re
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import AGENT_PROMPTS, HEAD_DIM, STRICT_ROPE_REINDEX, DEVICE, DTYPE
from pipelines.kv_cache import run_kv_cache_pipeline
from pipelines.standard import run_standard_pipeline

model = AutoModelForCausalLM.from_pretrained(
    "/models/Qwen2.5-3B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("/models/Qwen2.5-3B-Instruct")

buggy_fn = open("buggy_function.py").read()

for label, pipeline_fn, kwargs in [
    ("Standard", run_standard_pipeline, {}),
    ("KV Cache", run_kv_cache_pipeline, {"head_dim": HEAD_DIM, "strict_rope": STRICT_ROPE_REINDEX}),
]:
    result = pipeline_fn(
        model, tokenizer, ["A", "B"], AGENT_PROMPTS, buggy_fn, DEVICE, DTYPE, 512, **kwargs
    )
    output = result.agent_outputs[1]
    m = re.search(r"```python\n(.*?)```", output, re.DOTALL)
    if not m:
        m = re.search(r"```\n(.*?)```", output, re.DOTALL)
    code = m.group(1).strip() if m else None

    if not code:
        print(f"{label}: No code block found. Output snippet: {output[:300]}")
        continue

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_buggy_function.py",
         f"--repaired-path={tmp}", "-v", "--tb=short"],
        capture_output=True, text=True,
    )
    os.unlink(tmp)
    print(f"\n=== {label} Pipeline ===")
    print(proc.stdout[-3000:])
    if proc.returncode not in (0, 1):
        print("STDERR:", proc.stderr[-500:])
