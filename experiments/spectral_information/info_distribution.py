import marimo

__generated_with = "0.18.2"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import numpy.typing as npt
    import matplotlib.pyplot as plt
    plt.style.use('default')

    from dataclasses import dataclass
    from pathlib import Path
    from typing import List, Tuple
    return List, Path, Tuple, dataclass, np, npt, pl, plt


@app.cell
def _(Path, pl):
    input_PARQUET_PATHS = [
        Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"),
        Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet"),
    ]
    lfs = []
    for input_parquet_path in input_PARQUET_PATHS:
        lfs.append(pl.scan_parquet(input_parquet_path))
    pl.union(lfs).sink_parquet(
        "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/combined_spectral_lib.parquet",
        engine="streaming",
    )
    return


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
        add_title:  bool = False


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
        Produce a 1xN figure where each column corresponds to a single library and contains:
        - Overlayed: per-spectrum distribution (density) and per-molecule max distribution (density)

        This layout provides one graph per spectral library containing both spectra and molecule maxima,
        plotted on the same axis and scaled consistently to the requested histogram range.
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
            print(f"Library '{label}': {n_spectra} spectra, {n_molecules} unique molecules.")
            print(f"Per spectra, the mean is {np.mean(spectrum_info_arr):.4f} and the median is {np.median(spectrum_info_arr):.4f}.")
            print(f"Per molecule max, the mean is {np.mean(molecule_max_info_arr):.4f} and the median is {np.median(molecule_max_info_arr):.4f}.")
            stats.append({
                "label": label,
                "path": path,
                "n_spectra": int(n_spectra),
                "n_molecules": n_molecules,
                "spectrum_info_arr": spectrum_info_arr,
                "molecule_max_info_arr": molecule_max_info_arr,
            })

        # Prepare subplots: 1 row and N columns (one per library) to overlay both distributions on the same axis.
        n_libs = len(stats)
        fig_width = max(5 * n_libs, 6)  # Provide reasonable width scaling with number of libraries.
        fig, axes = plt.subplots(1, n_libs, figsize=(fig_width, 5), squeeze=False, sharex=False, facecolor="white")

        bins = config.bins
        hist_range = config.range
        bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)

        # Use a colorblind-friendly palette (Wong palette) for clarity across viewers with color-vision deficiencies.
        # Kept only the two chosen colors (blue, orange) per request; commented out alternatives.
        colorblind_palette = [
            "#0072B2",  # blue (per-spectrum)
            "#D55E00",  # orange (per-molecule maxima)
            # "#009E73",  # green
            # "#CC79A7",  # pink/purple
            # "#F0E442",  # yellow
            # "#56B4E9",  # light blue
        ]

        # Keep consistent colors across libraries: blue for spectra, orange for maxima.
        spec_color = colorblind_palette[0]
        mol_color = colorblind_palette[1]

        for idx, stat in enumerate(stats):
            label = stat["label"]
            spec_arr: npt.NDArray[np.float64] = stat["spectrum_info_arr"]
            mol_arr: npt.NDArray[np.float64] = stat["molecule_max_info_arr"]

            ax = axes[0, idx]

            # Per-spectrum histogram (density)
            # Why: Use density scaling so both distributions are comparable within the same y-axis scale.
            if config.add_title:
                ax.set_title(f"{label}")
            ax.hist(
                spec_arr,
                bins=bin_edges,
                range=hist_range,
                density=True,
                alpha=0.6,
                color=spec_color,
                edgecolor="black",
            )

            # Per-molecule histogram (density) on the same axis (overlay).
            if mol_arr.size > 0:
                ax.hist(
                    mol_arr,
                    bins=bin_edges,
                    range=hist_range,
                    density=True,
                    alpha=0.6,
                    color=mol_color,
                    edgecolor="black",
                )

            # Axis labels and styles for this column
            # Title commented out to avoid top text; keep simple column label by uncommenting if needed.
            # ax.set_title(f"{label}")
            ax.set_ylabel("Density")
            ax.set_xlabel("Spectral Information Score")
            ax.grid(alpha=0.25)

            # Ensure consistent x limits across columns and the requested histogram range.
            ax.set_xlim(hist_range)

            # No legend (as requested): do not add one.
            # No mean/median lines: omitted intentionally.

        # Removed global suptitle to satisfy "remove the top text" request.

        # Ensure output dir exists and save
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(config.output_path), dpi=400, facecolor="white", transparent=False)
        plt.close(fig)
    return PlotConfig, plot_distributions


@app.cell
def _(Path, PlotConfig, plot_distributions):

    # PARQUET_PATHS = [
    #     Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"),
    #     Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet"),
    # ]
    # LABELS = ["NIST23", "Fraghub"]
    PARQUET_PATHS = [
        # Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/combined_spectral_lib.parquet"),
            Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"),
        Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet"),
    ]
    LABELS = [
        # "Combined",
     "NIST23", "Fraghub"]
    OUTPUT_PNG = Path("spectral_information_distribution_nist_and_fraghub.png")

    config = PlotConfig(
        parquet_paths=PARQUET_PATHS,
        labels=LABELS,
        output_path=OUTPUT_PNG,
        bins=50,
        range=(0.0, 5.0),
        add_title=True,
    )

    plot_distributions(config)
    return


if __name__ == "__main__":
    app.run()
