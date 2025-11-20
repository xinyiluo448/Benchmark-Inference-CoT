#!/usr/bin/env python3
"""
Run a matrix of models x tasks, save per-run CSVs and an aggregated summary CSV.

Supports:
- Built-in tasks (classification-sst2 / reasoning-gsm8k / summarization-xsum) with optional --data-map.
- External task configs (json/yaml) via --task-configs.

Example:
python scripts/run_matrix.py \
  --models mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Llama-2-7b-chat-hf \
  --tasks classification-sst2 reasoning-gsm8k \
  --data-map data_map.json \
  --out-dir staging/matrix_runs \
  --max-new-tokens 128 \
  --eval-rouge \
  --save-outputs

data_map.json example:
{
  "classification-sst2": "staging/sst2_dev.jsonl",
  "reasoning-gsm8k": "staging/gsm8k_train_100.jsonl",
  "summarization-xsum": "staging/xsum_validation_100.jsonl"
}
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from bench_runner.model import load_model_and_tokenizer
from bench_runner.runner import run_example, save_results_csv
from bench_runner.tasks import load_examples, load_task_config
from bench_runner.metrics import evaluate, DEFAULT_TASK_METRIC


def load_data_map(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    with path.open() as f:
        return json.load(f)


def maybe_load_rouge(eval_rouge: bool):
    if not eval_rouge:
        return None
    try:
        from rouge_score import rouge_scorer
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"ROUGE disabled: rouge_score not available ({e}). Install via `pip install rouge-score`.")
        return None
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)


def run_task(
    model_id: str,
    task_name: str,
    metric: str,
    examples,
    out_csv: Path,
    max_new_tokens: int,
    load_4bit: bool,
    bits: int,
    dtype: str,
    eval_rouge: bool,
    save_outputs: bool,
    use_chat_template: bool,
):
    target_bits = bits if bits is not None else (4 if load_4bit else None)
    pipe, tokenizer = load_model_and_tokenizer(model_id, load_4bit=False, bits=target_bits, dtype=dtype)
    model_device = getattr(pipe.model, "device", None)
    results_for_csv = []
    total_latency = 0.0
    total_tokens = 0
    total_peak_mem = 0.0
    correct = 0
    evaluated = 0
    total_generated_tokens = 0
    rouge_pairs: List[Tuple[str, str]] = []
    for example in examples:
        result = run_example(
            pipe,
            tokenizer,
            example,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
        )
        ref_raw = (example.reference or "").strip()
        pred_raw = result.output_text.strip()

        if eval_rouge and ref_raw:
            rouge_pairs.append((ref_raw, pred_raw))

        if ref_raw and metric != "rouge":
            eval_res = evaluate(metric, pred_raw, ref_raw)
            if eval_res.correct is not None:
                correct += eval_res.correct
                evaluated += 1

        total_latency += result.latency_s
        total_tokens += result.tokens_generated
        total_peak_mem += result.peak_mem_gb
        total_generated_tokens += result.tokens_generated

        row = {
            "name": example.name,
            "latency_s": f"{result.latency_s:.3f}",
            "tokens_generated": result.tokens_generated,
            "tokens_per_s": f"{result.tokens_per_s:.2f}",
            "peak_mem_gb": result.peak_mem_gb,
            "reference": example.reference or "",
        }
        if save_outputs:
            row["prompt"] = example.prompt
            row["output"] = result.output_text
        results_for_csv.append(row)

    # Cleanup to release GPU memory before next task/model
    del pipe, tokenizer
    try:
        import torch
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if out_csv:
        save_results_csv(out_csv, results_for_csv)

    n = len(examples)
    avg_latency = total_latency / n if n else 0.0
    avg_tokens_per_s = (total_tokens / total_latency) if total_latency > 0 else 0.0
    avg_peak_mem = total_peak_mem / n if n else 0.0
    avg_generated_tokens = total_generated_tokens / n if n else 0.0
    acc = (correct / evaluated) if evaluated else None

    rouge_scores = None
    if eval_rouge:
        scorer = maybe_load_rouge(True)
        if scorer and rouge_pairs:
            totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
            for ref, pred in rouge_pairs:
                scores = scorer.score(ref, pred)
                for k in totals:
                    totals[k] += scores[k].fmeasure
            m = len(rouge_pairs)
            rouge_scores = {k: totals[k] / m for k in totals}

    return {
        "model": model_id,
        "task": task_name,
        "samples": n,
        "avg_latency_s": round(avg_latency, 3),
        "avg_tokens_per_s": round(avg_tokens_per_s, 2),
        "avg_peak_mem_gb": round(avg_peak_mem, 3),
        "avg_generated_tokens": round(avg_generated_tokens, 2),
        "accuracy": None if acc is None else round(acc, 4),
        "rouge1": None if not rouge_scores else round(rouge_scores["rouge1"], 4),
        "rouge2": None if not rouge_scores else round(rouge_scores["rouge2"], 4),
        "rougeL": None if not rouge_scores else round(rouge_scores["rougeLsum"], 4),
        "csv_path": str(out_csv),
    }


def save_summary(path: Path, rows: List[dict]):
    if not rows:
        return
    import csv

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run model-task matrix and aggregate results.")
    p.add_argument("--models", nargs="+", help="List of HF model ids")
    p.add_argument("--tasks", nargs="+", default=["classification-sst2", "reasoning-gsm8k", "summarization-xsum"], help="Built-in task keys")
    p.add_argument("--task-configs", nargs="*", type=Path, help="Custom task config files (json/yaml)")
    p.add_argument("--data-map", type=Path, help="JSON mapping task key -> data path for built-ins")
    p.add_argument("--out-dir", type=Path, default=Path("staging/matrix_runs"), help="Output directory")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--bits", type=int, choices=[4, 8, 16], help="Quantization bits (4/8); 16 = no quantization")
    p.add_argument("--bits-list", nargs="*", help="List of quantization bits to sweep, e.g., 4 8 16")
    p.add_argument("--dtype", default="float16", help="Weight dtype (float16/bfloat16/float32)")
    p.add_argument("--dtype-list", nargs="*", help="List of dtypes to sweep, e.g., float16 bfloat16")
    p.add_argument("--model-configs", type=Path, help="JSON list of {model,bits,dtype,use_chat_template} to run, bypassing bits/dtype sweep")
    p.add_argument("--eval-rouge", action="store_true", help="Compute ROUGE for tasks with metric=rouge (requires rouge-score)")
    p.add_argument("--save-outputs", action="store_true", help="Store prompt/output in per-run CSVs")
    p.add_argument("--max-samples", type=int, default=0, help="Sample cap for built-in tasks")
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--use-chat-template", action="store_true", help="Apply tokenizer.chat_template for chat models")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.models and not args.model_configs:
        raise SystemExit("Provide --models or --model-configs.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_map = load_data_map(args.data_map)

    task_entries = []
    # task configs
    if args.task_configs:
        for cfg_path in args.task_configs:
            task_name, examples, metric = load_task_config(cfg_path)
            task_entries.append((task_name, examples, metric))
    # built-in tasks (can be mixed with configs)
    for t in args.tasks:
        if args.task_configs and t not in DEFAULT_TASK_METRIC:
            continue
        examples = load_examples(t, Path(data_map[t]) if t in data_map else None, args.max_samples, args.shuffle_seed)
        metric = DEFAULT_TASK_METRIC.get(t, "none")
        task_entries.append((t, examples, metric))

    summary_rows = []

    def normalize_bits_list(bits_list):
        if not bits_list:
            return []
        normalized = []
        for b in bits_list:
            try:
                val = int(b)
                normalized.append(val if val != 16 else None)
            except ValueError:
                if str(b).lower() in {"none", "fp16", "float16"}:
                    normalized.append(None)
        return normalized

    # Model configs override bits/dtype sweep
    model_entries = []
    if args.model_configs:
        cfg = json.loads(args.model_configs.read_text())
        for item in cfg:
            model_entries.append(
                {
                    "model": item["model"],
                    "bits": item.get("bits"),
                    "dtype": item.get("dtype", args.dtype),
                    "use_chat_template": item.get("use_chat_template", args.use_chat_template),
                }
            )
    else:
        bits_candidates = normalize_bits_list(args.bits_list)
        dtype_candidates = args.dtype_list if args.dtype_list else [args.dtype]
        if not bits_candidates:
            bits_candidates = [args.bits if args.bits is not None else (None if not args.load_4bit else 4)]
        for model in args.models:
            for bits_setting in bits_candidates:
                for dtype_setting in dtype_candidates:
                    model_entries.append(
                        {
                            "model": model,
                            "bits": bits_setting,
                            "dtype": dtype_setting,
                            "use_chat_template": args.use_chat_template,
                        }
                    )

    for entry in model_entries:
        model = entry["model"]
        bits_setting = entry.get("bits")
        dtype_setting = entry.get("dtype", args.dtype)
        use_chat = entry.get("use_chat_template", args.use_chat_template)
        bits_label = "none" if bits_setting is None else str(bits_setting)
        for task_name, examples, metric in task_entries:
            csv_path = args.out_dir / f"{task_name.replace('/', '_')}_{Path(model).name}_bits{bits_label}_run.csv"
            print(
                f"\n== Running model={model} task={task_name} bits={bits_label} dtype={dtype_setting} samples={len(examples)} ==")
            summary = run_task(
                model_id=model,
                task_name=task_name,
                metric=metric,
                examples=examples,
                out_csv=csv_path,
                max_new_tokens=args.max_new_tokens,
                load_4bit=False,  # use bits_setting for explicit control
                bits=bits_setting,
                dtype=dtype_setting,
                eval_rouge=args.eval_rouge if metric == "rouge" else False,
                save_outputs=args.save_outputs,
                use_chat_template=use_chat,
            )
            summary["bits"] = bits_label
            summary["dtype"] = dtype_setting
            summary_rows.append(summary)
            print(
                f"Summary: latency={summary['avg_latency_s']}s, t/s={summary['avg_tokens_per_s']}, "
                f"avg_tokens={summary['avg_generated_tokens']}, "
                f"peak_mem={summary['avg_peak_mem_gb']} GB, acc={summary['accuracy']}, "
                f"rouge1={summary['rouge1']}, rouge2={summary['rouge2']}, rougeL={summary['rougeL']}"
            )
            print(f"Saved detail CSV: {csv_path}")

    summary_csv = args.out_dir / "summary.csv"
    save_summary(summary_csv, summary_rows)
    print(f"\nAggregated summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
