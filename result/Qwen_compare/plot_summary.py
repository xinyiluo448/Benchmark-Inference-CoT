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
        ("avg_latency_s", "Latency (s)"),
        ("avg_peak_mem_gb", "Peak GPU (GB)"),
    ]
    task_df = df[df["task"] == task].copy()
    task_df = task_df.sort_values(["model_size_b", "quant_label", "model"])
    if task_df.empty:
        return

    labels = task_df["label"].tolist()
    x = range(len(task_df))

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), constrained_layout=True)
    palette = plt.cm.tab20.colors

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        ax.bar(x, task_df[metric], color=[palette[i % len(palette)] for i in x])
        ax.set_title(f"{task}: {title}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(title)

    legend_items = {}
    for label, color in zip(labels, [palette[i % len(palette)] for i in x]):
        quant = label.split("-", 1)[-1]
        legend_items.setdefault(quant, color)
    axes[-1].legend(
        [plt.Line2D([0], [0], color=c, lw=10) for c in legend_items.values()],
        legend_items.keys(),
        title="Quantization",
        loc="best",
    )

    out_file = out_dir / f"{task}_metrics.png"
    fig.suptitle(f"{task} Metrics by Model/Quantization", fontsize=14)
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {out_file}")


def plot_relations(df: pd.DataFrame, out_dir: Path):
    scatter_pairs = [
        ("avg_tokens_per_s", "accuracy", "tokens_vs_accuracy"),
        ("avg_tokens_per_s", "rougeL", "tokens_vs_rougeL"),
    ]

    tasks = sorted(df["task"].unique())
    for x_metric, y_metric, tag in scatter_pairs:
        fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), constrained_layout=True)
        if len(tasks) == 1:
            axes = [axes]
        for ax, task in zip(axes, tasks):
            sub = df[(df["task"] == task)].copy()
            sub = sub.dropna(subset=[x_metric, y_metric])
            if sub.empty:
                ax.set_title(f"{task} (no data)")
                continue
            colors = [plt.cm.tab20(i % 20) for i in range(len(sub))]
            ax.scatter(sub[x_metric], sub[y_metric], c=colors)
            for (_, row), color in zip(sub.iterrows(), colors):
                ax.text(row[x_metric], row[y_metric], row["label"], fontsize=7, color=color)
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric)
            ax.set_title(task)
        fig.suptitle(f"{x_metric} vs {y_metric}")
        out_file = out_dir / f"scatter_{tag}.png"
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        print(f"Saved figure: {out_file}")


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
