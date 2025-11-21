import contextlib
import csv
import time
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import TextGenerationPipeline

from bench_runner.tasks import TaskExample


@dataclass
class InferenceResult:
    output_text: str
    latency_s: float
    tokens_generated: int
    tokens_per_s: float
    peak_mem_gb: float


def maybe_enable_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return True
    return False


@contextlib.contextmanager
def timed() -> Iterable[float]:
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def count_new_tokens(output_ids: torch.Tensor, input_length: int, full_text: bool) -> int:
    """
    Count generated tokens.
    If pipeline returns only the generated continuation (return_full_text=False), just use its length.
    If full text is returned, subtract the input length to get new tokens.
    """
    total = int(output_ids.numel())
    if not full_text:
        return total
    return max(total - input_length, 0)


def extract_last_number(text: str) -> str:
    """
    Extract the last integer/float-like number string from text.
    Supports comma-separated thousands (e.g., 1,200.5).
    """
    matches = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    if not matches:
        return ""
    raw = matches[-1]
    return raw.replace(",", "")


def numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def run_example(
    pipe: TextGenerationPipeline,
    tokenizer,
    example: TaskExample,
    max_new_tokens: int = 64,
    use_chat_template: bool = False,
    use_cache: bool = True,
) -> InferenceResult:
    peak_enabled = maybe_enable_peak_memory()
    prompt_text = example.prompt
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids

    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with timed() as elapsed:
            generation = pipe(
                prompt_text,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                use_cache=use_cache,
            )[0]
            output = generation["generated_text"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    output_ids = tokenizer(output, return_tensors="pt").input_ids
    tokens_generated = count_new_tokens(output_ids, input_ids.shape[-1], full_text=False)
    latency_s = elapsed()
    tokens_per_s = tokens_generated / latency_s if latency_s > 0 else 0.0

    peak_mem_gb = 0.0
    if peak_enabled:
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        peak_mem_gb = round(peak_mem, 3)

    return InferenceResult(
        output_text=output.strip(),
        latency_s=latency_s,
        tokens_generated=tokens_generated,
        tokens_per_s=tokens_per_s,
        peak_mem_gb=peak_mem_gb,
    )


def save_results_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
