"""
EXP-005 — Cross-Process KV Transfer: Agent A
=============================================
Runs Agent A, slices the generation KV cache, serializes it to /dev/shm,
and writes a metadata file. Called by exp_cross_process.py.

Outputs (to stdout, one JSON line):
  {"serialize_ms": float, "payload_size_mb": float, "gen_len": int,
   "input_len": int, "agent_a_output": str}
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace")
from config import DTYPE, AGENT_PROMPTS
from pipelines.kv_cache import _run_agent_a, kv_cache_size_mb

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

    buggy_fn = open("/workspace/buggy_function.py").read()

    output_text, prefill_ms, gen_slice = _run_agent_a(
        model, tokenizer, AGENT_PROMPTS["A"], buggy_fn,
        DEVICE, DTYPE, 512, head_dim, strict_rope=True,
    )

    # Serialize KV cache to shared memory
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.save({"kv": gen_slice.kv, "length": gen_slice.length}, KV_PAYLOAD_PATH)
    torch.cuda.synchronize()
    serialize_ms = (time.perf_counter() - t0) * 1000

    payload_mb = kv_cache_size_mb(gen_slice.kv)

    result = {
        "serialize_ms": serialize_ms,
        "payload_size_mb": payload_mb,
        "gen_len": gen_slice.length,
        "agent_a_output": output_text,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
