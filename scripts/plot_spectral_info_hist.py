"""
Plot spectral_information_score histograms from two parquet outputs produced by process_spectral_library.py.

Usage:
    python plot_spectral_info_hist.py /path/to/first.parquet /path/to/second.parquet --bins 60 --out /tmp/out.png

Produces a single figure with two subplots:
- Top: positive mode (ion_mode == 'P')
- Bottom: negative mode (ion_mode == 'N')

Each subplot shows both files overlaid with the same exact bin edges.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
#import argparse
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import polars as pl
import matplotlib.pyplot as plt
import os


@dataclass
class PlotConfig:
    input_files: List[Path]
    input_labels: List[str]
    output_file: Path
    bins: int = 50
    score_range: Tuple[float, float] = (0.0, 5.0)  # uniform binning range applied to both files
    figsize: Tuple[int, int] = (10, 8)
    color_first: str = "blue"
    color_second: str = "red"
    # visual options: border color and separators between bins
    border_color: str = "black"
    border_linewidth: float = 0.8
    # vertical separators are disabled by default; keep only black borders around bars themselves
    show_bin_separators: bool = False
    separator_linewidth: float = 0.6
    separator_alpha: float = 0.6


def read_scores(parquet_path: Path, ion_mode: str) -> NDArray[np.float64]:
    """
    Read 'spectral_information_score' values for a given ion_mode.
    Reasons to use Polars: memory-safety and consistent DataFrame processing across project.

    Returns:
      np.ndarray(shape=(n_scores,)) of float values. If no rows match, returns an empty array.
    """
    assert parquet_path.exists() and parquet_path.is_file(), f"File not found: {parquet_path}"
    df = pl.read_parquet(parquet_path)
    assert "spectral_information_score" in df.columns, "Column 'spectral_information_score' missing from parquet"
    assert "ion_mode" in df.columns, "Column 'ion_mode' missing from parquet"

    sel = (
        df.filter(pl.col("ion_mode") == ion_mode)
          .select(pl.col("spectral_information_score"))
          .drop_nulls()
    )
    if sel.is_empty():
        return np.array([], dtype=np.float64)
    return sel.to_series().to_numpy()


def plot_pair_histograms(cfg: PlotConfig) -> None:
    # Expect exactly two inputs — fail fast to respect repository rules
    assert len(cfg.input_files) == 2 and len(cfg.input_labels) == 2, "Provide exactly two input files and two labels in the source config"

    file1, file2 = cfg.input_files
    label1, label2 = cfg.input_labels

    # shared histogram binning range [min, max]; we want deterministic identical edges
    min_edge, max_edge = cfg.score_range
    assert min_edge < max_edge, "score_range min must be lower than max"
    edges = np.linspace(min_edge, max_edge, cfg.bins + 1)

    ion_modes = [("P", "Positive mode (P)"), ("N", "Negative mode (N)")]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=cfg.figsize, tight_layout=True)

    # pre-load arrays for both files and both modes to get totals and overflow statistics
    for ax, (mode_key, mode_label) in zip(axes, ion_modes):
        arr1 = read_scores(file1, mode_key)
        arr2 = read_scores(file2, mode_key)

        # Print summary statistics on over-limit values (outside the chosen range)
        for arr, label in ((arr1, label1), (arr2, label2)):
            total = arr.size
            # "over" means strictly greater than the upper limit; report count and share
            n_over = int(np.count_nonzero(arr > max_edge))
            pct_over = 100.0 * n_over / total if total > 0 else 0.0
            print(f"{label} - mode={mode_key}: total={total}, over {max_edge} = {n_over} ({pct_over:.2f}%)")

        # also print combined summary across the two files
        combined_total = arr1.size + arr2.size
        combined_over = int(np.count_nonzero(arr1 > max_edge) + np.count_nonzero(arr2 > max_edge))
        combined_pct = 100.0 * combined_over / combined_total if combined_total > 0 else 0.0
        print(f"Combined - mode={mode_key}: total={combined_total}, over {max_edge} = {combined_over} ({combined_pct:.2f}%)")

        # If both empty, annotate the subplot and continue
        if arr1.size == 0 and arr2.size == 0:
            ax.text(0.5, 0.5, "No data for this ion mode", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(mode_label)
            continue

        # Use numpy histogram to produce counts per bin using the identical edges for both files.
        counts1, _ = np.histogram(arr1, bins=edges) if arr1.size > 0 else (np.zeros(len(edges)-1, dtype=int), edges)
        counts2, _ = np.histogram(arr2, bins=edges) if arr2.size > 0 else (np.zeros(len(edges)-1, dtype=int), edges)

        # Stacked bars: bottom segment is counts1, top segment is counts2.
        # Use left-aligned bars so bins exactly match the histogram edges.
        width = edges[1] - edges[0]
        left_edges = edges[:-1]

        # Draw the stacked bars with black borders so each bar is outlined.
        ax.bar(
            left_edges,
            counts1,
            width=width,
            align="edge",
            color=cfg.color_first,
            edgecolor=cfg.border_color,
            linewidth=cfg.border_linewidth,
            alpha=0.9,
            label=label1,
            zorder=2,
        )
        ax.bar(
            left_edges,
            counts2,
            width=width,
            align="edge",
            bottom=counts1,
            color=cfg.color_second,
            edgecolor=cfg.border_color,
            linewidth=cfg.border_linewidth,
            alpha=0.9,
            label=label2,
            zorder=1,
        )

        # Ensure the y-limits leave room for separators and labels:
        max_bin_height = int((counts1 + counts2).max()) if (counts1.size > 0) else 1
        ax.set_ylim(0, max(5, int(max_bin_height * 1.05)))

        # Add vertical separators between bins if requested — draws a thin black line at each edge.
        # No vertical separators: user requested only border lines around bars, keep bar outlines via edgecolor
        # (The explicit axvline calls that draw vertical lines between bins were removed.)

        ax.set_xlim(min_edge, max_edge)
        ax.set_title(mode_label)
        ax.set_xlabel("spectral_information_score")
        ax.set_ylabel("count")
        ax.legend()

    fig.suptitle(f"Spectral Information Score Comparison — Shared Range {min_edge}-{max_edge}", y=0.98)
    plt.savefig(cfg.output_file, dpi=200)
    print(f"Wrote histogram figure to {cfg.output_file}")


if __name__ == "__main__":
    # Core change: input files and labels are defined in source code instead of CLI
    # Why: makes it explicit and repeatable for quick analysis scripts in the repository.
    # Edit these two Paths and labels for your comparisons:
    SOURCE_FIRST = Path("/home/analytit_admin/Data/msp_for_Yonathan/NIST.parquet")
    SOURCE_SECOND = Path("/home/analytit_admin/Data/msp_for_Yonathan/fraghub.parquet")
    LABEL_FIRST = "NIST"
    LABEL_SECOND = "Fraghub"

    # Fail fast: files must exist
    assert SOURCE_FIRST.exists() and SOURCE_FIRST.is_file(), f"First file not found: {SOURCE_FIRST}"
    assert SOURCE_SECOND.exists() and SOURCE_SECOND.is_file(), f"Second file not found: {SOURCE_SECOND}"

    cfg = PlotConfig(
        input_files=[SOURCE_FIRST, SOURCE_SECOND],
        input_labels=[LABEL_FIRST, LABEL_SECOND],
        output_file=Path("spectral_info_hist_range0-5.png"),
        bins=20,
        score_range=(0.0, 5.0),
    )
    plot_pair_histograms(cfg)