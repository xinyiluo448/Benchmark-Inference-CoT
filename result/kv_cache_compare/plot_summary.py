#!/usr/bin/env python3
"""Visualize KV cache strategy comparison."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["avg_tokens_per_s", "avg_latency_s", "avg_peak_mem_gb", "avg_generated_tokens", "accuracy", "rouge1", "rouge2", "rougeL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_task(df: pd.DataFrame, task: str, out_dir: Path):
    task_df = df[df["task"] == task].copy()
    if task_df.empty:
        return
    metrics = [
        ("avg_tokens_per_s", "Tokens/s"),
        ("avg_peak_mem_gb", "Peak GPU (GB)"),
    ]
    task_df = task_df.sort_values(["kv_strategy", "model"])
    labels = task_df.apply(lambda r: f"{r['model']}\n{r['kv_strategy']}", axis=1)
    x = range(len(task_df))
    palette = plt.cm.tab20.colors
    colors = [palette[i % len(palette)] for i in x]
    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(5 * (len(metrics) + 1), 4), constrained_layout=True)
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
    fig.savefig(out_dir / f"{task}_kv_metrics.png", dpi=200)
    plt.close(fig)


def plot_tokens_vs_quality(df: pd.DataFrame, out_dir: Path):
    metrics = {
        "classification-sst2": ("accuracy", "Accuracy"),
        "reasoning-gsm8k": ("accuracy", "Accuracy"),
        "summarization-xsum": ("rougeL", "ROUGE-L"),
    }
    for task, (metric_col, ylabel) in metrics.items():
        sub = df[(df["task"] == task)].dropna(subset=["avg_tokens_per_s", metric_col]).copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = plt.cm.tab20.colors
        for idx, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(row["avg_tokens_per_s"], row[metric_col], color=colors[idx % len(colors)])
            ax.text(row["avg_tokens_per_s"], row[metric_col], f"{row['model']}\n{row['kv_strategy']}", fontsize=7)
        ax.set_xlabel("Tokens/s")
        ax.set_ylabel(ylabel)
        ax.set_title(task)
        fig.savefig(out_dir / f"scatter_tokens_vs_{metric_col}_{task}.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot KV cache comparison summary")
    ap.add_argument("--summary", type=Path, default=Path("result/kv_cache_compare/summary.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("result/kv_cache_compare/plots"))
    args = ap.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"Summary not found: {args.summary}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.summary)
    tasks = sorted(df["task"].unique())
    for task in tasks:
        plot_task(df, task, args.out_dir)
    plot_tokens_vs_quality(df, args.out_dir)


if __name__ == "__main__":
    main()
