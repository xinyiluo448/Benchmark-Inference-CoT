"""
CLI 入口：单次运行一组样本，打印结果并可保存 CSV。
"""

import argparse
from pathlib import Path

from bench_runner.model import load_model_and_tokenizer
from bench_runner.runner import run_example, save_results_csv
from bench_runner.tasks import list_tasks, load_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM inference profiling scaffold")
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model id (default: mistralai/Mistral-7B-Instruct-v0.2)",
    )
    parser.add_argument(
        "--task", choices=list_tasks(), default="classification-sst2", help="内置任务键（或用 --data 覆盖）"
    )
    parser.add_argument("--data", type=Path, help="jsonl/json/csv，包含 prompt 和可选 reference。")
    parser.add_argument("--max-samples", type=int, default=0, help="抽样数量上限（0 表示全部）。")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="抽样前的 shuffle 种子。")
    parser.add_argument("--save-csv", type=Path, help="保存结果的 CSV 路径。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--load-4bit", action="store_true", help="使用 bitsandbytes 4bit（需要已安装）。")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading model {args.model} (4-bit={args.load_4bit})...")
    pipe, tokenizer = load_model_and_tokenizer(args.model, load_4bit=args.load_4bit)

    examples = load_examples(args.task, args.data, args.max_samples, args.shuffle_seed)

    print(f"Running task: {args.task} | examples: {len(examples)}")
    results_for_csv = []
    for example in examples:
        print(f"\nExample: {example.name}")
        result = run_example(pipe, tokenizer, example, max_new_tokens=args.max_new_tokens)
        print(f"Prompt: {example.prompt}")
        print(f"Output: {result.output_text}")
        if example.reference:
            print(f"Reference (for inspection): {example.reference}")
        print(
            f"Latency: {result.latency_s:.3f}s | New tokens: {result.tokens_generated} | "
            f"Tokens/s: {result.tokens_per_s:.2f} | Peak GPU (GB): {result.peak_mem_gb}"
        )
        results_for_csv.append(
            {
                "name": example.name,
                "prompt": example.prompt,
                "output": result.output_text,
                "reference": example.reference or "",
                "latency_s": f"{result.latency_s:.3f}",
                "tokens_generated": result.tokens_generated,
                "tokens_per_s": f"{result.tokens_per_s:.2f}",
                "peak_mem_gb": result.peak_mem_gb,
            }
        )

    if args.save_csv:
        save_results_csv(args.save_csv, results_for_csv)
        print(f"\nSaved results to {args.save_csv}")


if __name__ == "__main__":
    main()
