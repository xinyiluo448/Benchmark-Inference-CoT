#!/usr/bin/env python3
"""Aggregate per-run CSVs from run_matrix into a single summary."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List

from bench_runner.metrics import evaluate

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None


def load_config_map(config_path: Path | None) -> Dict[Tuple[str, str], dict]:
    mapping: Dict[Tuple[str, str], dict] = {}
    if not config_path:
        return mapping
    data = json.loads(config_path.read_text())
    for entry in data:
        model_name = Path(entry["model"]).name
        bits = entry.get("bits")
        bits_label = "none" if bits is None else str(bits)
        mapping[(model_name, bits_label)] = {
            "dtype": entry.get("dtype", "float16"),
            "use_chat_template": entry.get("use_chat_template"),
            "use_kv_cache": entry.get("use_kv_cache"),
        }
    return mapping


def parse_filename(filepath: Path):
    name = filepath.stem  # remove .csv
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filepath.name}")
    task = parts[0]
    model = parts[1]
    bits_label = "unknown"
    kv_state = "unknown"
    for part in parts[2:]:
        if part.startswith("bits"):
            bits_label = part[len("bits") :]
        elif part.startswith("kv"):
            kv_state = part[len("kv") :]
    return task, model, bits_label, kv_state


def summarize_csv(path: Path, compute_rouge: bool, metric: str):
    count = 0
    sum_latency = 0.0
    total_tokens = 0.0
    sum_peak = 0.0
    peak_samples = 0
    correct = 0
    evaluated = 0

    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rouge_pairs: List[Tuple[str, str]] = []

    for row in rows:
        count += 1
        lat = float(row.get("latency_s", 0) or 0)
        sum_latency += lat
        toks = float(row.get("tokens_generated", 0) or 0)
        total_tokens += toks
        peak_val = row.get("peak_mem_gb")
        if peak_val not in (None, ""):
            sum_peak += float(peak_val)
            peak_samples += 1
        ref = row.get("reference") or ""
        pred = row.get("output") or ""
        if metric != "rouge" and ref:
            eval_res = evaluate(metric, pred, ref)
            if eval_res.correct is not None:
                evaluated += 1
                correct += eval_res.correct
        if compute_rouge and row.get("reference") and row.get("output"):
            rouge_pairs.append((row["reference"], row["output"]))

    avg_latency = sum_latency / count if count else 0.0
    avg_generated = total_tokens / count if count else 0.0
    avg_tokens_per_s = total_tokens / sum_latency if sum_latency > 0 else 0.0
    avg_peak = sum_peak / peak_samples if peak_samples else 0.0
    accuracy = (correct / evaluated) if evaluated else None

    rouge_vals = {"rouge1": None, "rouge2": None, "rougeL": None}
    if compute_rouge and rouge_pairs and rouge_scorer:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
        for ref, pred in rouge_pairs:
            scores = scorer.score(ref, pred)
            for k in totals:
                totals[k] += scores[k].fmeasure
        m = len(rouge_pairs)
        rouge_vals = {
            "rouge1": round(totals["rouge1"] / m, 4),
            "rouge2": round(totals["rouge2"] / m, 4),
            "rougeL": round(totals["rougeLsum"] / m, 4),
        }

    return {
        "samples": count,
        "avg_latency_s": round(avg_latency, 3),
        "avg_tokens_per_s": round(avg_tokens_per_s, 2),
        "avg_peak_mem_gb": round(avg_peak, 3),
        "avg_generated_tokens": round(avg_generated, 2),
        "accuracy": None if accuracy is None else round(accuracy, 4),
        "rouge1": rouge_vals["rouge1"],
        "rouge2": rouge_vals["rouge2"],
        "rougeL": rouge_vals["rougeL"],
    }


def main():
    parser = argparse.ArgumentParser(description="Gather per-run CSVs into a summary table")
    parser.add_argument("--runs-dir", type=Path, default=Path("staging/matrix_runs"))
    parser.add_argument("--output", type=Path, default=Path("staging/matrix_runs/summary.csv"))
    parser.add_argument("--model-configs", type=Path, help="Optional model_configs.json to enrich metadata")
    parser.add_argument("--tasks", nargs="+", help="Tasks (e.g., classification-sst2 reasoning-gsm8k summarization-xsum)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Annotate summary with max_new_tokens")
    parser.add_argument("--eval-rouge", action="store_true", help="Recompute ROUGE if reference/output stored")
    args = parser.parse_args()

    config_map = load_config_map(args.model_configs)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    csv_files: List[Tuple[Path, str]] = []
    if args.model_configs:
        data = json.loads(args.model_configs.read_text())
        for entry in data:
            model_id = entry["model"]
            model_name = Path(model_id).name
            bits = entry.get("bits")
            bits_label = "none" if bits is None else str(bits)
            kv_state = "on" if entry.get("use_kv_cache", True) else "off"
            for task in args.tasks:
                pattern = f"{task}_{model_name}_bits{bits_label}_kv{kv_state}_run.csv"
                matches = [p for p in args.runs_dir.glob(pattern) if p.is_file()]
                if not matches:
                    print(f"Warning: no file for model={model_name} task={task} bits={bits_label} kv={kv_state}")
                for task_path in matches:
                    csv_files.append((task_path, kv_state))
    else:
        csv_files = [(p, "unknown") for p in sorted(args.runs_dir.glob("*_run.csv")) if p.is_file()]

    if not csv_files:
        raise SystemExit("No *_run.csv files matched the model configs.")

    if args.eval_rouge and rouge_scorer is None:
        print("Warning: rouge-score not installed; skipping ROUGE computation")

    for csv_path, kv_state in csv_files:
        task, model, bits_label, kv = parse_filename(csv_path)
        metric = "rouge" if task.startswith("summarization") else "exact"
        if task.startswith("reasoning"):
            metric = "numeric"
        summary = summarize_csv(csv_path, compute_rouge=args.eval_rouge and rouge_scorer is not None, metric=metric)
        config_meta = config_map.get((model, bits_label), {})
        row = {
            "model": model,
            "task": task,
            "samples": summary["samples"],
            "avg_latency_s": summary["avg_latency_s"],
            "avg_tokens_per_s": summary["avg_tokens_per_s"],
            "avg_peak_mem_gb": summary["avg_peak_mem_gb"],
            "avg_generated_tokens": summary["avg_generated_tokens"],
            "accuracy": summary["accuracy"],
            "rouge1": summary["rouge1"] if summary["rouge1"] is not None else "",
            "rouge2": summary["rouge2"] if summary["rouge2"] is not None else "",
            "rougeL": summary["rougeL"] if summary["rougeL"] is not None else "",
            "csv_path": str(csv_path),
            "bits": bits_label,
            "dtype": config_meta.get("dtype", ""),
            "use_kv_cache": kv,
            "use_chat_template": config_meta.get("use_chat_template", ""),
            "max_new_tokens": args.max_new_tokens if args.max_new_tokens is not None else "",
        }
        rows.append(row)

    fieldnames = [
        "model",
        "task",
        "samples",
        "avg_latency_s",
        "avg_tokens_per_s",
        "avg_peak_mem_gb",
        "avg_generated_tokens",
        "accuracy",
        "rouge1",
        "rouge2",
        "rougeL",
        "csv_path",
        "bits",
        "dtype",
        "use_kv_cache",
        "use_chat_template",
        "max_new_tokens",
    ]

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
