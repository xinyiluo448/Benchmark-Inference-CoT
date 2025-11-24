#!/usr/bin/env python3
"""
Visualize summary metrics per task for Qwen comparison experiments.

Generates a figure per task with bars for tokens/s, latency, and peak GPU memory,
grouped by model size + quantization label.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_model_size_b(model_name: str) -> float | None:
    match = re.search(r"([\d\.]+)\s*B", model_name, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def make_quant_label(row: pd.Series) -> str:
    if "gptq" in row["model"].lower():
        return "GPTQ-Int8"
    bits = row.get("bits")
    if bits in ("none", "", None):
        return "FP/Native"
    return f"{bits}-bit"


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "avg_tokens_per_s",
        "avg_latency_s",
        "avg_peak_mem_gb",
        "avg_generated_tokens",
        "accuracy",
        "rouge1",
        "rouge2",
        "rougeL",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model_size_b"] = df["model"].apply(parse_model_size_b)
    df["quant_label"] = df.apply(make_quant_label, axis=1)
    df["label"] = df.apply(
        lambda r: f"{r['model_size_b'] or '?'}B-{r['quant_label']}",
        axis=1,
    )
    return df


def plot_task(df: pd.DataFrame, task: str, out_dir: Path):
    metrics = [
        ("avg_tokens_per_s", "Tokens / s"),
        ("avg_peak_mem_gb", "Peak GPU (GB)"),
    ]
    task_df = df[df["task"] == task].copy()
    palette = plt.cm.tab20.colors
    quant_order = {"FP/Native": 0, "16-bit": 0, "8-bit": 1, "4-bit": 2, "GPTQ-Int8": 3}
    task_df["quant_order"] = task_df["quant_label"].map(lambda x: quant_order.get(x, 4))
    task_df = task_df.sort_values(["quant_order", "model_size_b", "model"])
    if task_df.empty:
        return

    labels = task_df["label"].tolist()
    x = range(len(task_df))
    colors = [palette[i % len(palette)] for i in x]

    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(5 * (len(metrics) + 1), 4), constrained_layout=True)
    palette = plt.cm.tab20.colors

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        ax.bar(x, task_df[metric], color=colors)
        ax.set_title(f"{task}: {title}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(title)
    metric_col = "accuracy"
    metric_title = "Accuracy"
    if task.startswith("summarization"):
        metric_col = "rougeL"
        metric_title = "ROUGE-L"
    ax = axes[-1]
    ax.bar(x, task_df[metric_col], color=colors)
    ax.set_title(f"{task}: {metric_title}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric_title)

    out_file = out_dir / f"{task}_metrics.png"
    fig.suptitle(f"{task} Metrics by Model/Quantization", fontsize=14)
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {out_file}")


def plot_relations(df: pd.DataFrame, out_dir: Path):
    task_metrics = [
        ("classification-sst2", "accuracy"),
        ("reasoning-gsm8k", "accuracy"),
        ("summarization-xsum", "rougeL"),
    ]

    fig, axes = plt.subplots(1, len(task_metrics), figsize=(5 * len(task_metrics), 4), constrained_layout=True)
    for ax, (task, metric) in zip(axes, task_metrics):
        sub = df[(df["task"] == task)].dropna(subset=["avg_tokens_per_s", metric]).copy()
        if sub.empty:
            ax.set_title(f"{task} (no data)")
            continue
        colors = plt.cm.tab20.colors
        for idx, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(row["avg_tokens_per_s"], row[metric], color=colors[idx % len(colors)])
            ax.text(row["avg_tokens_per_s"], row[metric], row["label"], fontsize=7)
        ylabel = "Accuracy" if metric == "accuracy" else "ROUGE-L"
        ax.set_xlabel("Tokens/s")
        ax.set_ylabel(ylabel)
        ax.set_title(task)
    fig.suptitle("Tokens/s vs Quality")
    fig.savefig(out_dir / "scatter_tokens_vs_quality.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(task_metrics), figsize=(5 * len(task_metrics), 4), constrained_layout=True)
    for ax, (task, metric) in zip(axes, task_metrics):
        sub = df[(df["task"] == task)].dropna(subset=[metric, "avg_peak_mem_gb"]).copy()
        if sub.empty:
            ax.set_title(f"{task} (no data)")
            continue
        colors = plt.cm.tab20.colors
        for idx, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(row["avg_peak_mem_gb"], row[metric], color=colors[idx % len(colors)])
            ax.text(row["avg_peak_mem_gb"], row[metric], row["label"], fontsize=7)
        ylabel = "Accuracy" if metric == "accuracy" else "ROUGE-L"
        ax.set_xlabel("Peak GPU (GB)")
        ax.set_ylabel(ylabel)
        ax.set_title(task)
    fig.suptitle("Quality vs Peak GPU")
    fig.savefig(out_dir / "scatter_quality_vs_peak.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize summary CSV for Qwen compare experiments")
    parser.add_argument("--summary", type=Path, default=Path("result/Qwen_compare/summary2.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("result/Qwen_compare/plots"))
    args = parser.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"Summary CSV not found: {args.summary}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.summary)
    tasks = sorted(df["task"].unique())
    for task in tasks:
        plot_task(df, task, args.out_dir)
    plot_relations(df, args.out_dir)


if __name__ == "__main__":
    main()
