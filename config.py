from pathlib import Path
import torch

# Paths
MODEL_PATH = "/models/Qwen2.5-3B-Instruct"
RESULTS_DIR = Path("results")

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Qwen2.5-3B-Instruct architecture (confirmed from config.json)
NUM_LAYERS = 36
NUM_ATTENTION_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128  # hidden_size(2048) / num_attention_heads(16)

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE = 0  # greedy throughout

# Benchmark
NUM_WARMUP_RUNS = 1
NUM_TIMED_RUNS = 20
HOP_COUNTS = [2, 3, 4]

# RoPE re-indexing (set False to skip for debugging — will likely degrade quality)
STRICT_ROPE_REINDEX = True

# Agent system prompts
AGENT_PROMPTS = {
    "A": (
        "You are a code analysis agent. Find every bug in the following Python function. "
        "List each bug with its line number, what is wrong, and why it causes incorrect behavior."
    ),
    "B": (
        "You are a code repair agent. Given a bug analysis, rewrite the function with every "
        "identified bug fixed. Output only the corrected Python function in a single code block."
    ),
    "C": (
        "You are a documentation agent. Add a complete Google-style docstring to the following "
        "Python function. Output only the function with the docstring added."
    ),
    "D": (
        "You are a technical writer. Write one concise paragraph explaining what the following "
        "Python function does, its parameters, and what it returns."
    ),
}

HOP_AGENT_SEQUENCE = {
    2: ["A", "B"],
    3: ["A", "B", "C"],
    4: ["A", "B", "C", "D"],
}
