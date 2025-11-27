import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import numpy.typing as npt
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from pathlib import Path
    from typing import List, Tuple
    return List, Path, Tuple, dataclass, np, npt, pl, plt


@app.cell
def _(List, Path, Tuple, dataclass, np, npt, pl, plt):
    @dataclass
    class PlotConfig:
        """
        Configuration for plotting spectral information distributions.

        parquet_paths: list of parquet files to read (one per library).
        labels: corresponding labels for the datasets (in same order).
        output_path: path where the figure will be saved.
        bins: number of histogram bins.
        range: value range for the histogram (min, max).
        """
        parquet_paths: List[Path]
        labels: List[str]
        output_path: Path
        bins: int = 50
        range: Tuple[float, float] = (0.0, 3.0)


    def read_library_scores_and_inchikeys(path: Path) -> pl.DataFrame:
        """
        Read spectral_information_score and base_inchikey from a parquet file, returning a polars DataFrame.
        Fail fast if file is missing or expected columns are absent.

        Returns:
            pl.DataFrame containing columns ['spectral_information_score', 'base_inchikey'].
        """
        # Why: Fail early to avoid silent issues downstream with missing data columns or paths.
        assert path.exists(), f"Expected parquet file at {path} but it does not exist."
        df = pl.read_parquet(path, columns=["spectral_information_score", "base_inchikey"])
        assert "spectral_information_score" in df.columns, (
            f"File {path} is missing required column 'spectral_information_score'."
        )
        assert "base_inchikey" in df.columns, (
            f"File {path} is missing required column 'base_inchikey'."
        )
        # Why: Drop null spectral_information_score values to ensure accurate statistics and plotting.
        df = df.filter(pl.col("spectral_information_score").is_not_null())
        return df


    def compute_molecule_max_info(df: pl.DataFrame) -> np.ndarray:
        """
        Compute the maximum spectral_information_score per molecule (group by base_inchikey)
        and return a 1D numpy array of maxima.

        Args:
          df: Polars DataFrame with at least 'base_inchikey' and 'spectral_information_score' columns.

        Returns:
          1D numpy array of floats shape=(n_molecules,)
        """
        # Why: Use polars groupby for performance on potentially large dataframes.
        grouped = df.group_by("base_inchikey").agg(
            max_info=pl.col("spectral_information_score").max()
        )
        # Why: Ensure only valid maxima are considered and return as numpy array for plotting via matplotlib.
        return grouped.select(pl.col("max_info").cast(pl.Float64)).to_numpy().ravel()

    def plot_distributions(config: PlotConfig) -> None:
        """
        Produce a 2xN figure where each column corresponds to a single library and contains:
        - Top: per-spectrum distribution (density)
        - Bottom: per-molecule max distribution (density)

        This layout provides one graph per spectral library containing both spectra and molecules.
        """
        assert len(config.parquet_paths) == len(config.labels), "parquet_paths and labels must have same length"

        # Why: Read all libraries and compute stats for each.
        stats = []

        for path, label in zip(config.parquet_paths, config.labels):
            df = read_library_scores_and_inchikeys(path)
            n_spectra = df.height
            molecule_max_info_arr = compute_molecule_max_info(df)
            n_molecules = int(molecule_max_info_arr.size)

            spectrum_info_arr = df.select(pl.col("spectral_information_score").cast(pl.Float64)).to_numpy().ravel()
            assert spectrum_info_arr.size > 0, f"No non-null 'spectral_information_score' values found in {path}"

            stats.append({
                "label": label,
                "path": path,
                "n_spectra": int(n_spectra),
                "n_molecules": n_molecules,
                "spectrum_info_arr": spectrum_info_arr,
                "molecule_max_info_arr": molecule_max_info_arr,
            })

        # Prepare subplots: 2 rows and N columns (one per library)
        n_libs = len(stats)
        fig_width = max(5 * n_libs, 6)  # Provide reasonable width scaling with number of libraries.
        fig, axes = plt.subplots(2, n_libs, figsize=(fig_width, 8), squeeze=False, sharex=False)

        bins = config.bins
        hist_range = config.range
        bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)

        # Use a colormap to support arbitrary number of libraries (avoids hard-coded color list).
        cmap = plt.get_cmap("tab10")

        for idx, stat in enumerate(stats):
            color = cmap(idx % cmap.N)
            label = stat["label"]
            spec_arr: npt.NDArray[np.float64] = stat["spectrum_info_arr"]
            mol_arr: npt.NDArray[np.float64] = stat["molecule_max_info_arr"]

            ax_top = axes[0, idx]
            ax_bot = axes[1, idx]

            # Per-spectrum histogram (density)
            ax_top.hist(
                spec_arr,
                bins=bin_edges,
                range=hist_range,
                density=True,   # Why: Use density scaling instead of absolute counts
                alpha=0.6,
                color=color,
                edgecolor="black",
                label=f"{label} — Spectra={stat['n_spectra']}"
            )

            # Mean/median lines for spectra (safe for empty array)
            if spec_arr.size > 0:
                mean_spec = float(np.nanmean(spec_arr))
                median_spec = float(np.nanmedian(spec_arr))
                ax_top.axvline(mean_spec, color=color, linestyle="--", linewidth=1.25,
                               label=f"{label} mean={mean_spec:.2f}")
                ax_top.axvline(median_spec, color=color, linestyle=":", linewidth=1.25,
                               label=f"{label} median={median_spec:.2f}")

            # Per-molecule histogram (density)
            ax_bot.hist(
                mol_arr,
                bins=bin_edges,
                range=hist_range,
                density=True,
                alpha=0.6,
                color=color,
                edgecolor="black",
                label=f"{label} — Molecules={stat['n_molecules']}"
            )

            # Mean/median lines for molecule maxima (safe for empty array)
            if mol_arr.size > 0:
                mean_mol = float(np.nanmean(mol_arr))
                median_mol = float(np.nanmedian(mol_arr))
                ax_bot.axvline(mean_mol, color=color, linestyle="--", linewidth=1.25,
                               label=f"{label} mean={mean_mol:.2f}")
                ax_bot.axvline(median_mol, color=color, linestyle=":", linewidth=1.25,
                               label=f"{label} median={median_mol:.2f}")

            # Axis labels and styles for this column
            ax_top.set_title(f"{label}")
            ax_top.set_ylabel("Density (spectra)")
            ax_top.grid(alpha=0.25)

            ax_bot.set_ylabel("Density (molecules)")
            ax_bot.set_xlabel("Spectral Information Score")
            ax_bot.grid(alpha=0.25)

            # Ensure consistent x limits across rows and the requested histogram range.
            ax_top.set_xlim(hist_range)
            ax_bot.set_xlim(hist_range)

            # Legends
            ax_top.legend(loc="upper right", fontsize="small", framealpha=0.85)
            ax_bot.legend(loc="upper right", fontsize="small", framealpha=0.85)

        fig.suptitle("Spectral Information per Library (Top: per-spectrum density, Bottom: per-molecule max density)", fontsize=12)

        # Ensure output dir exists and save
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(str(config.output_path), dpi=400)
        plt.close(fig)
    return PlotConfig, plot_distributions


@app.cell
def _(Path, PlotConfig, plot_distributions):

    PARQUET_PATHS = [
        Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"),
        Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet"),
    ]
    LABELS = ["NIST23", "Fraghub"]
    OUTPUT_PNG = Path("/home/analytit_admin/dev/HRMS_utils/experiments/spectral_information/spectral_information_distribution.png")

    config = PlotConfig(
        parquet_paths=PARQUET_PATHS,
        labels=LABELS,
        output_path=OUTPUT_PNG,
        bins=30,
        range=(0.0, 3.0),
    )

    plot_distributions(config)
    return


if __name__ == "__main__":
    app.run()
