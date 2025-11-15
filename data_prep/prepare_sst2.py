"""
从 HuggingFace Datasets 下载 SST-2，并转存为当前基准脚本可用的 jsonl（字段：id/prompt/reference）。

用法示例：
python3 data_prep/prepare_sst2.py --split validation --out sst2_dev.jsonl
python3 data_prep/prepare_sst2.py --split train --max-samples 50000 --shuffle-seed 42 --out sst2_train_sample.jsonl
"""

import argparse
import json
import random

from datasets import load_dataset


def build_prompt(text: str) -> str:
    return f"Review: {text}\nIs the sentiment positive or negative?"


def main():
    parser = argparse.ArgumentParser(description="Convert SST-2 to jsonl for bench_framework.")
    parser.add_argument("--split", default="validation", help="Dataset split: train/validation/test")
    parser.add_argument("--out", required=True, help="Output jsonl path")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample cap (0 for all)")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Shuffle seed before capping")
    args = parser.parse_args()

    ds = load_dataset("glue", "sst2", split=args.split)
    rows = list(ds)
    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(rows)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    with open(args.out, "w") as f:
        for i, row in enumerate(rows):
            label = "positive" if row["label"] == 1 else "negative"
            prompt = build_prompt(row["sentence"])
            rec = {"id": f"{args.split}-{i}", "prompt": prompt, "reference": label}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
