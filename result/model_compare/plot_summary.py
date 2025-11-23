#!/usr/bin/env python3
"""Visualize model comparison summary (similar to Qwen plots)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    return df


def plot_per_task(df: pd.DataFrame, out_dir: Path):
    metrics = [
        ("avg_tokens_per_s", "Tokens/s"),
        ("avg_peak_mem_gb", "Peak GPU (GB)"),
    ]
    tasks = sorted(df["task"].unique())
    palette = plt.cm.tab20.colors
    for task in tasks:
        task_df = df[df["task"] == task].copy()
        if task_df.empty:
            continue
        task_df = task_df.sort_values(["model"])
        labels = task_df["model"].tolist()
        x = range(len(task_df))
        fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(5 * (len(metrics) + 1), 4), constrained_layout=True)
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx]
            ax.bar(x, task_df[metric], color=[palette[i % len(palette)] for i in x])
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
        ax.bar(x, task_df[metric_col], color=[palette[(i + len(metrics)) % len(palette)] for i in x])
        ax.set_title(f"{task}: {metric_title}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.savefig(out_dir / f"{task}_metrics.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot model comparison summary")
    ap.add_argument("--summary", type=Path, default=Path("result/model_compare/summary2.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("result/model_compare/plots"))
    args = ap.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"Summary not found: {args.summary}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.summary)
    plot_per_task(df, args.out_dir)


if __name__ == "__main__":
    main()
