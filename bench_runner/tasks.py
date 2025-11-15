import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TaskExample:
    name: str
    prompt: str
    reference: Optional[str] = None  # expected answer for quick inspection


# Minimal prompts to exercise the pipeline without external datasets.
BUILTIN_TASKS = {
    "classification-sst2": [
        TaskExample(
            name="sst2-pos",
            prompt="Review: This movie was surprisingly thoughtful and moving.\nIs the sentiment positive or negative?",
            reference="positive",
        ),
        TaskExample(
            name="sst2-neg",
            prompt="Review: The plot was messy and the acting felt wooden.\nIs the sentiment positive or negative?",
            reference="negative",
        ),
    ],
    "reasoning-gsm8k": [
        TaskExample(
            name="gsm8k-easy",
            prompt=(
                "Solve step by step: If Jenny has 3 apples and buys 4 more, "
                "then gives 2 to her friend, how many apples does she have now?"
            ),
            reference="5",
        ),
    ],
    "summarization-xsum": [
        TaskExample(
            name="xsum-style",
            prompt=(
                "Summarize in one sentence:\n"
                "A new community garden opened downtown, offering free plots to residents "
                "and hosting weekly workshops on composting and native plants."
            ),
            reference="One-line summary about the new community garden.",
        ),
    ],
}


def list_tasks() -> List[str]:
    return sorted(BUILTIN_TASKS)


def load_examples_from_file(path: Path) -> List[TaskExample]:
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")

    ext = path.suffix.lower()
    examples: List[TaskExample] = []

    if ext in {".jsonl", ".json"}:
        with path.open() as f:
            if ext == ".jsonl":
                rows = [json.loads(line) for line in f if line.strip()]
            else:
                rows = json.load(f)
        for idx, row in enumerate(rows):
            prompt = row.get("prompt")
            reference = row.get("reference")
            if not prompt:
                continue
            name = row.get("id") or row.get("name") or f"sample-{idx}"
            examples.append(TaskExample(name=name, prompt=prompt, reference=reference))
    elif ext == ".csv":
        with path.open() as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                prompt = row.get("prompt")
                reference = row.get("reference")
                if not prompt:
                    continue
                name = row.get("id") or row.get("name") or f"sample-{idx}"
                examples.append(TaskExample(name=name, prompt=prompt, reference=reference))
    else:
        raise SystemExit("Unsupported data format. Use .jsonl, .json, or .csv with prompt/reference fields.")

    if not examples:
        raise SystemExit("No valid examples found in data file.")
    return examples


def slice_examples(examples: List[TaskExample], max_samples: int, seed: Optional[int]) -> List[TaskExample]:
    if max_samples and max_samples > 0 and len(examples) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(examples)
        return examples[:max_samples]
    return examples


def load_examples(task_key: str, data_path: Optional[Path], max_samples: int, shuffle_seed: Optional[int]) -> List[TaskExample]:
    if data_path:
        examples = load_examples_from_file(data_path)
    else:
        examples = BUILTIN_TASKS[task_key]
    return slice_examples(examples, max_samples, shuffle_seed)

