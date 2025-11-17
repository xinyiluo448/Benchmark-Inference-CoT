"""
Convert GSM8K to jsonl with prompt/reference for bench_framework.

Usage examples:
python3 data_prep/prepare_gsm8k.py --split train --max-samples 200 --shuffle-seed 42 --out gsm8k_train_200.jsonl
python3 data_prep/prepare_gsm8k.py --split test --out gsm8k_test.jsonl
"""

import argparse
import json
import random

from datasets import load_dataset


def extract_final_answer(answer: str) -> str:
    # GSM8K answers often end with "#### 42"
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def build_prompt(question: str) -> str:
    return f"Solve step by step: {question}"


def main():
    parser = argparse.ArgumentParser(description="Convert GSM8K to jsonl for bench_framework.")
    parser.add_argument("--split", default="train", help="Dataset split: train/test")
    parser.add_argument("--out", required=True, help="Output jsonl path")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample cap (0 for all)")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Shuffle seed before capping")
    args = parser.parse_args()

    ds = load_dataset("gsm8k", "main", split=args.split)
    rows = list(ds)
    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(rows)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    with open(args.out, "w") as f:
        for i, row in enumerate(rows):
            prompt = build_prompt(row["question"])
            reference = extract_final_answer(row["answer"])
            rec = {"id": f"{args.split}-{i}", "prompt": prompt, "reference": reference}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
