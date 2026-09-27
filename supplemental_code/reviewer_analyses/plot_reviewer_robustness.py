#!/usr/bin/env python3
"""Create response-letter figures for seed stability and degraded-view analyses."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_LABELS = {
    "FCA_fetal_33184180_spleen": "FCA spleen",
    "FCA_fetal_33184180_cerebellum": "FCA cerebellum",
    "mca_Cerebellum_62216": "MCA cerebellum",
    "mca_LargeIntestineA_62816": "MCA large intestine",
}
DATASET_COLORS = {
    "FCA_fetal_33184180_spleen": "#4C78A8",
    "FCA_fetal_33184180_cerebellum": "#F58518",
    "mca_Cerebellum_62216": "#54A24B",
    "mca_LargeIntestineA_62816": "#B279A2",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
    axis.tick_params(labelsize=9)


def plot_seed_stability(input_dir: Path, output_dir: Path) -> None:
    per_seed = read_rows(input_dir / "seed_stability_per_seed.csv")
    pairwise = read_rows(input_dir / "seed_stability_pairwise_partitions.csv")
    datasets = [dataset for dataset in DATASET_LABELS if any(r["dataset"] == dataset for r in per_seed)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [
        ("ARI", per_seed, "ARI"),
        ("NMI", per_seed, "NMI"),
        ("Pairwise partition ARI", pairwise, "partition_ARI"),
    ]
    for panel_index, (title, rows, key) in enumerate(panels):
        values = [
            [float(row[key]) for row in rows if row["dataset"] == dataset]
            for dataset in datasets
        ]
        box = axes[panel_index].boxplot(
            values,
            patch_artist=True,
            widths=0.62,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.55},
        )
        for patch, dataset in zip(box["boxes"], datasets):
            patch.set_facecolor(DATASET_COLORS[dataset])
            patch.set_alpha(0.85)
        for x_value, dataset, dataset_values in zip(
            range(1, len(datasets) + 1), datasets, values
        ):
            offsets = np.linspace(-0.09, 0.09, len(dataset_values))
            axes[panel_index].scatter(
                x_value + offsets,
                dataset_values,
                s=18,
                facecolors="white",
                edgecolors=DATASET_COLORS[dataset],
                linewidths=0.9,
                alpha=0.8,
                zorder=3,
            )
        axes[panel_index].set_title(title, fontsize=12, fontweight="bold")
        axes[panel_index].set_xticks(
            range(1, len(datasets) + 1),
            [DATASET_LABELS[dataset] for dataset in datasets],
            rotation=28,
            ha="right",
        )
        axes[panel_index].set_ylim(0, 1.02)
        axes[panel_index].text(
            -0.12,
            1.03,
            chr(ord("A") + panel_index),
            transform=axes[panel_index].transAxes,
            fontsize=14,
            fontweight="bold",
        )
        style_axis(axes[panel_index])

    fig.tight_layout(w_pad=2.0)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"seed_stability.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def plot_degraded_view(input_dir: Path, output_dir: Path) -> None:
    rows = read_rows(input_dir / "degraded_view_results.csv")
    datasets = [dataset for dataset in DATASET_LABELS if any(r["dataset"] == dataset for r in rows)]
    x_positions = np.arange(6)
    x_labels = ["Clean", "Noise\n0.5", "Noise\n1.0", "Noise\n2.0", "Row\npermuted", "Omit\nLSI"]

    fig, axes = plt.subplots(2, len(datasets), figsize=(3.2 * len(datasets), 6.0), sharex=True)
    if len(datasets) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, dataset in enumerate(datasets):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for row_index, metric in enumerate(("ARI", "NMI")):
            values_by_x: list[list[float]] = []
            values_by_x.append([float(r[metric]) for r in dataset_rows if r["condition"] == "clean"])
            for level in (0.5, 1.0, 2.0):
                values_by_x.append(
                    [
                        float(r[metric])
                        for r in dataset_rows
                        if r["condition"] == "gaussian_noise"
                        and abs(float(r["noise_level"]) - level) < 1e-9
                    ]
                )
            values_by_x.append(
                [float(r[metric]) for r in dataset_rows if r["condition"] == "row_permuted"]
            )
            values_by_x.append(
                [float(r[metric]) for r in dataset_rows if r["condition"] == "omit_lsi"]
            )
            means, deviations = zip(*(aggregate(values) for values in values_by_x))
            axis = axes[row_index, column]
            axis.errorbar(
                x_positions,
                means,
                yerr=deviations,
                color=DATASET_COLORS[dataset],
                marker="o",
                linewidth=1.8,
                capsize=3,
                zorder=3,
            )
            for x_value, values in zip(x_positions, values_by_x):
                if len(values) > 1:
                    offsets = np.linspace(-0.08, 0.08, len(values))
                    axis.scatter(
                        x_value + offsets,
                        values,
                        s=14,
                        color=DATASET_COLORS[dataset],
                        alpha=0.35,
                        zorder=2,
                    )
            axis.set_ylim(0, 1.02)
            axis.set_xticks(x_positions, x_labels)
            if column == 0:
                axis.set_ylabel(metric, fontsize=11, fontweight="bold")
            if row_index == 0:
                axis.set_title(DATASET_LABELS[dataset], fontsize=11, fontweight="bold")
            panel = row_index * len(datasets) + column
            axis.text(
                -0.13,
                1.03,
                chr(ord("A") + panel),
                transform=axis.transAxes,
                fontsize=13,
                fontweight="bold",
            )
            style_axis(axis)

    fig.tight_layout(w_pad=1.4, h_pad=1.4)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"degraded_view_robustness.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_seed_stability(args.input_root / "seed_stability", args.output_dir)
    plot_degraded_view(args.input_root / "degraded_view", args.output_dir)


if __name__ == "__main__":
    main()
