"""
EXP-005 — Cross-Process KV Transfer: Agent B
=============================================
Loads the KV cache serialized by exp_cross_process_a.py from /dev/shm,
runs Agent B, checks quality. Called by exp_cross_process.py.

Outputs (to stdout, one JSON line):
  {"deserialize_ms": float, "passed": int, "total": int,
   "code_extracted": bool, "output": str}
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace")
from config import DTYPE, AGENT_PROMPTS, STRICT_ROPE_REINDEX
from pipelines.kv_cache import (
    _tokenize_prefix, _tokenize_suffix, reindex_kv_cache,
    concat_kv_caches, kv_cache_size_mb,
    _greedy_generate, GenerationSlice,
)

MODEL_PATH = "/models/Qwen2.5-7B-Instruct"
KV_PAYLOAD_PATH = "/dev/shm/kv_a_output.pt"
DEVICE = "cuda"


def get_head_dim(model) -> int:
    return model.config.hidden_size // model.config.num_attention_heads


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    head_dim = get_head_dim(model)

    # Deserialize KV cache from shared memory
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    payload = torch.load(KV_PAYLOAD_PATH, map_location=DEVICE)
    torch.cuda.synchronize()
    deserialize_ms = (time.perf_counter() - t0) * 1000

    gen_slice = GenerationSlice(
        kv=payload["kv"],
        length=payload["length"],
        output_text="",
    )

    # Run Agent B with deserialized KV — same logic as _run_agent_n
    prefix_ids = _tokenize_prefix(tokenizer, AGENT_PROMPTS["B"], DEVICE)
    l_prefix = prefix_ids.shape[1]

    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True)

    kv_prefix = prefix_out.past_key_values

    # Reindex and build payload
    reindexed = reindex_kv_cache(
        gen_slice.kv, old_start=0, new_start=l_prefix,
        model=model, dtype=DTYPE, device=DEVICE, head_dim=head_dim,
        strict=STRICT_ROPE_REINDEX,
    )
    combined_kv = concat_kv_caches(kv_prefix, reindexed)

    suffix_ids = _tokenize_suffix(tokenizer, DEVICE)
    l_suffix = suffix_ids.shape[1]
    suffix_start = l_prefix + gen_slice.length
    suffix_positions = torch.arange(suffix_start, suffix_start + l_suffix, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        suffix_out = model(
            input_ids=suffix_ids,
            past_key_values=combined_kv,
            position_ids=suffix_positions,
            use_cache=True,
        )

    gen_start_pos = suffix_start + l_suffix
    generated_ids, _, _ = _greedy_generate(
        model, tokenizer, suffix_out.past_key_values,
        suffix_out.logits[0, -1, :], gen_start_pos, DEVICE, 512,
    )
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Quality check
    m = re.search(r"```python\n(.*?)```", output_text, re.DOTALL)
    if not m:
        m = re.search(r"```\n(.*?)```", output_text, re.DOTALL)
    code = m.group(1).strip() if m else None

    passed, total = 0, 0
    if code:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp = f.name
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "/workspace/test_buggy_function.py",
             f"--repaired-path={tmp}", "-q", "--tb=no"],
            capture_output=True, text=True,
        )
        os.unlink(tmp)
        mp = re.search(r"(\d+) passed", proc.stdout)
        mf = re.search(r"(\d+) failed", proc.stdout)
        passed = int(mp.group(1)) if mp else 0
        total = passed + (int(mf.group(1)) if mf else 0)

    result = {
        "deserialize_ms": deserialize_ms,
        "passed": passed,
        "total": total,
        "code_extracted": code is not None,
        "output": output_text,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
