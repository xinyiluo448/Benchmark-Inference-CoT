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
    parser.add_argument("--save-outputs", action="store_true", help="在 CSV 中同时保存 prompt/output/reference。")
    parser.add_argument("--eval-rouge", action="store_true", help="计算 ROUGE-1/2/L（需安装 rouge-score，需有输出与参考）")
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
    total_latency = 0.0
    total_tokens = 0
    total_peak_mem = 0.0
    correct = 0
    evaluated = 0
    rouge_pairs = []  # (reference, output)
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
        is_correct = None
        if example.reference is not None:
            ref_raw = example.reference.strip()
            pred_raw = result.output_text.strip()
            ref_norm = ref_raw.lower()
            pred_norm = pred_raw.lower()
            if args.eval_rouge:
                rouge_pairs.append((ref_raw, pred_raw))

            # GSM8K: 尝试提取数字答案；否则回退精确匹配。
            if args.task == "reasoning-gsm8k":
                from bench_runner.runner import extract_last_number, numeric_equal

                ref_num = extract_last_number(ref_raw)
                pred_num = extract_last_number(pred_raw)
                if ref_num and pred_num and numeric_equal(ref_num, pred_num):
                    is_correct = 1
                else:
                    is_correct = int(pred_norm == ref_norm)
            else:
                is_correct = int(pred_norm == ref_norm)

            correct += is_correct
            evaluated += 1

        total_latency += result.latency_s
        total_tokens += result.tokens_generated
        total_peak_mem += result.peak_mem_gb
        row = {
            "name": example.name,
            "latency_s": f"{result.latency_s:.3f}",
            "tokens_generated": result.tokens_generated,
            "tokens_per_s": f"{result.tokens_per_s:.2f}",
            "peak_mem_gb": result.peak_mem_gb,
            "reference": example.reference or "",
            "correct": "" if is_correct is None else is_correct,
        }
        if args.save_outputs:
            row["prompt"] = example.prompt
            row["output"] = result.output_text
        results_for_csv.append(row)

    # 汇总信息
    n = len(examples)
    avg_latency = total_latency / n if n else 0.0
    avg_tokens_per_s = (total_tokens / total_latency) if total_latency > 0 else 0.0
    avg_peak_mem = total_peak_mem / n if n else 0.0
    acc = (correct / evaluated) if evaluated else None

    print(
        "\nSummary: "
        f"samples={n}, "
        f"avg_latency={avg_latency:.3f}s, "
        f"avg_tokens_per_s={avg_tokens_per_s:.2f}, "
        f"avg_peak_mem_gb={avg_peak_mem:.3f}, "
        f"accuracy={(acc * 100):.2f}% " if acc is not None else "accuracy=N/A "
    )

    if args.eval_rouge:
        if not rouge_pairs:
            print("ROUGE: no reference/output pairs to evaluate.")
        else:
            try:
                from rouge_score import rouge_scorer
            except Exception as e:  # pragma: no cover - optional dependency
                print(f"ROUGE: rouge_score not available ({e}). Install via `pip install rouge-score`.")
            else:
                scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
                totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
                for ref, pred in rouge_pairs:
                    scores = scorer.score(ref, pred)
                    for k in totals:
                        totals[k] += scores[k].fmeasure
                m = len(rouge_pairs)
                print(
                    f"ROUGE avg (F1): "
                    f"R1={totals['rouge1']/m:.4f}, "
                    f"R2={totals['rouge2']/m:.4f}, "
                    f"RL={totals['rougeLsum']/m:.4f} over {m} pairs"
                )

    if args.save_csv:
        save_results_csv(args.save_csv, results_for_csv)
        print(f"\nSaved results to {args.save_csv}")


if __name__ == "__main__":
    main()
