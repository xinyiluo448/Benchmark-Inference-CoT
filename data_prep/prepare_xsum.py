"""
Convert XSum to jsonl with prompt/reference for bench_framework.

Usage examples:
python3 data_prep/prepare_xsum.py --split validation --max-samples 200 --shuffle-seed 42 --out xsum_val_200.jsonl
python3 data_prep/prepare_xsum.py --split test --out xsum_test.jsonl
"""

import argparse
import json
import random

from datasets import Dataset, load_dataset


def build_prompt(document: str) -> str:
    return f"Summarize in one sentence:\n{document}"


def main():
    parser = argparse.ArgumentParser(description="Convert XSum to jsonl for bench_framework.")
    parser.add_argument("--split", default="validation", help="Dataset split: train/validation/test")
    parser.add_argument("--out", required=True, help="Output jsonl path")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample cap (0 for all)")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Shuffle seed before capping")
    # Accept unknown args silently so shell wrappers can pass extra flags to downstream steps.
    args, _ = parser.parse_known_args()

    try:
        ds: Dataset = load_dataset("xsum", split=args.split)
    except RuntimeError as e:
        raise SystemExit(
            "Failed to load XSum. With datasets>=3, script-based datasets are blocked. "
            "Workarounds:\n"
            "1) Install datasets<3: pip install 'datasets<3'\n"
            "2) Use a parquet-converted XSum repo and pass its name to load_dataset\n"
            f"Original error: {e}"
        ) from e
    rows = list(ds)
    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(rows)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    with open(args.out, "w") as f:
        for i, row in enumerate(rows):
            prompt = build_prompt(row["document"])
            reference = row["summary"]
            rec = {"id": f"{args.split}-{i}", "prompt": prompt, "reference": reference}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
