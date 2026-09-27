#!/usr/bin/env python3
"""Create publication-ready figures for the Hisemble component ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ORDER = ["full", "no_diffusion", "no_dynamic_sparsification", "dense_initial"]
LABELS = {
    "full": "Full Hisemble",
    "no_diffusion": "No cross-diffusion",
    "no_dynamic_sparsification": r"No dynamic $S_k$",
    "dense_initial": "Dense initial affinity",
}
COLORS = {
    "full": "#4C78A8",
    "no_diffusion": "#F58518",
    "no_dynamic_sparsification": "#E45756",
    "dense_initial": "#7A5195",
}
DATASET_LABELS = {
    "FCA_fetal_33184180_spleen": "FCA spleen",
    "FCA_fetal_33184180_cerebellum": "FCA cerebellum",
    "mca_Cerebellum_62216": "MCA cerebellum",
    "mca_LargeIntestineA_62816": "MCA large intestine",
}


def load_json_directory(path: Path) -> pd.DataFrame:
    rows = []
    for json_path in sorted(path.glob("*.json")):
        with json_path.open() as handle:
            row = json.load(handle)
        row["source_file"] = json_path.name
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No JSON result files found in {path}")
    frame = pd.DataFrame(rows)
    frame["variant_label"] = frame["variant"].map(LABELS)
    frame["dataset_label"] = frame["dataset"].map(DATASET_LABELS).fillna(frame["dataset"])
    if {
        "graph_construction_cpu_seconds",
        "fusion_cpu_seconds",
    }.issubset(frame.columns):
        frame["analysis_cpu_seconds"] = (
            frame["graph_construction_cpu_seconds"] + frame["fusion_cpu_seconds"]
        )
    else:
        frame["analysis_cpu_seconds"] = np.nan
    return frame


def style_axis(axis, panel: str) -> None:
    axis.text(
        -0.03,
        1.04,
        panel,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    axis.spines[["top", "right"]].set_visible(False)


def component_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    variants = [variant for variant in ORDER[:3] if variant in set(frame["variant"])]
    datasets = [
        DATASET_LABELS[name]
        for name in DATASET_LABELS
        if name in set(frame["dataset"])
    ]
    palette = [COLORS[variant] for variant in variants]
    order = [LABELS[variant] for variant in variants]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    panels = [
        ("ARI", "Adjusted Rand index", False),
        ("NMI", "Normalized mutual information", False),
        ("maximum_total_nnz", "Maximum nonzero entries across views", True),
        ("peak_memory_mb", "Peak memory (MB)", True),
    ]
    for panel, axis, (metric, ylabel, log_scale) in zip("ABCD", axes.flat, panels):
        sns.barplot(
            data=frame,
            x="dataset_label",
            y=metric,
            hue="variant_label",
            order=datasets,
            hue_order=order,
            palette=palette,
            errorbar=None,
            edgecolor="black",
            linewidth=0.45,
            ax=axis,
        )
        if log_scale:
            axis.set_yscale("log")
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=22)
        if metric in {"ARI", "NMI"}:
            axis.set_ylim(0, 1.02)
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
        style_axis(axis, panel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "component_ablation_multidataset.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "component_ablation_multidataset.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def scaling_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    variants = [variant for variant in ORDER if variant in set(frame["variant"])]
    sample_sizes = sorted(frame["n_cells"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    panels = [
        ("analysis_cpu_seconds", "Graph construction + fusion CPU time (s)", True),
        ("peak_memory_mb", "Peak memory (MB)", False),
        ("maximum_total_nnz", "Maximum nonzero entries across views", True),
    ]
    for panel, axis, (metric, ylabel, log_scale) in zip("ABC", axes, panels):
        for variant in variants:
            subset = frame[frame["variant"] == variant].sort_values("n_cells")
            axis.plot(
                subset["n_cells"],
                subset[metric],
                marker="o",
                markersize=5,
                linewidth=1.8,
                color=COLORS[variant],
                label=LABELS[variant],
            )
        if log_scale:
            axis.set_yscale("log")
        axis.set_xticks(sample_sizes)
        axis.set_xticklabels([f"{value:,}" for value in sample_sizes], rotation=30)
        axis.set_xlabel("Number of cells")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="major", linewidth=0.4, alpha=0.35)
        style_axis(axis, panel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(variants),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "component_scaling_downsampling.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "component_scaling_downsampling.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multidataset-dir", type=Path, required=True)
    parser.add_argument("--scaling-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)

    component = load_json_directory(args.multidataset_dir)
    scaling = load_json_directory(args.scaling_dir)
    component.sort_values(["dataset", "variant"]).to_csv(
        args.output_dir / "component_ablation_multidataset.csv", index=False
    )
    scaling.sort_values(["n_cells", "variant"]).to_csv(
        args.output_dir / "component_scaling_downsampling.csv", index=False
    )
    component_figure(component, args.output_dir)
    scaling_figure(scaling, args.output_dir)


if __name__ == "__main__":
    main()
