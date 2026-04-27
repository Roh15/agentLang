import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*")

import torch


@dataclass
class StandardResult:
    agent_outputs: List[str]
    prefill_times_ms: List[float]   # per agent (index 0 = Agent A)
    e2e_latency_ms: float


def _format_prompt(system_prompt: str, user_message: str) -> str:
    """Qwen2.5-Instruct chat format (no Jinja2 required)."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# Keep as a thin wrapper so benchmark.py can import it
def _build_messages(system_prompt: str, user_message: str) -> Tuple[str, str]:
    return system_prompt, user_message


def _run_agent(
    model,
    tokenizer,
    system_prompt: str,
    user_message: str,
    device: str,
    dtype,
    max_new_tokens: int,
) -> Tuple[str, float]:
    """
    Run one agent. Returns (output_text, prefill_ms).

    Prefill is timed as a standalone forward pass (use_cache=True). Generation
    then continues from the returned past_key_values. This gives a clean prefill
    measurement uncontaminated by generation time.
    """
    prompt = _format_prompt(system_prompt, user_message)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Timed prefill pass
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    past_kv = prefill_out.past_key_values
    next_token_logits = prefill_out.logits[0, -1, :]

    # Greedy generation from existing cache
    generated_ids = []
    current_pos = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_id = int(next_token_logits.argmax(-1))
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break

            next_input = torch.tensor([[next_id]], device=device)
            position_ids = torch.tensor([[current_pos]], device=device)
            out = model(
                input_ids=next_input,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_token_logits = out.logits[0, -1, :]
            current_pos += 1

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text, prefill_ms


def run_standard_pipeline(
    model,
    tokenizer,
    agent_keys: List[str],
    agent_prompts: dict,
    initial_input: str,
    device: str,
    dtype,
    max_new_tokens: int,
) -> StandardResult:
    """
    Run an N-hop standard (text handoff) pipeline.

    Each agent's user message is the accumulated text of all prior agents'
    outputs, so prefill cost grows with each hop. This is the realistic
    multi-agent pattern and makes the per-hop prefill growth visible.
    """
    t_start = time.perf_counter()

    outputs: List[str] = []
    prefill_times: List[float] = []

    for i, key in enumerate(agent_keys):
        system_prompt = agent_prompts[key]

        if i == 0:
            user_message = initial_input
        else:
            # Cumulative context: all prior outputs concatenated
            user_message = "\n\n".join(outputs)

        output_text, prefill_ms = _run_agent(
            model, tokenizer, system_prompt, user_message,
            device, dtype, max_new_tokens,
        )
        outputs.append(output_text)
        prefill_times.append(prefill_ms)

    e2e_ms = (time.perf_counter() - t_start) * 1000

    return StandardResult(
        agent_outputs=outputs,
        prefill_times_ms=prefill_times,
        e2e_latency_ms=e2e_ms,
    )
