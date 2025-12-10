import marimo

__generated_with = "0.18.4"
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
    from typing import List, Tuple, Dict
    return Dict, List, Path, Tuple, dataclass, np, npt, pl, plt


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


@app.cell
def _(Dict, List, Path, Tuple, dataclass, np, npt, pl, plt):
    @dataclass
    class MoleculeSpec:
        """
        Specification for a single molecule to plot.

        Attributes:
          name: str - human readable label shown in the legend.
          base_inchikey: str - base inchikey used to filter spectra.
          color: str - color code used to draw the line (e.g. '#FF0000').
        """
        name: str
        base_inchikey: str
        color: str


    @dataclass
    class MoleculePlotConfig:
        """
        Configuration for plotting informativity vs collision energy for a set of molecules.

        Attributes:
          parquet_path: Path - parquet file with at least columns base_inchikey, collision_energy column, and info column.
          molecules: List[MoleculeSpec] - list of molecules to include.
          output_path: Path - path to write resulting PNG.
          collision_energy_column: str - column name containing collision energy (default 'collision_energy_ev').
          info_column: str - column containing informativity (default 'spectral_information_score').
          add_title: bool - whether to set per-plot title.
          marker: str - default marker used at points.
        """
        parquet_path: Path
        molecules: List[MoleculeSpec]
        output_path: Path
        collision_energy_column: str = "collision_energy_ev"
        info_column: str = "spectral_information_score"
        add_title: bool = True
        marker: str = "o"


    def plot_informativity_vs_collision_energy(
        config: MoleculePlotConfig,
    ) -> Dict[str, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """
        Plot informativity (y axis, typically 'spectral_information_score') vs collision energy (x axis in eV)
        for a list of molecules specified by their base_inchikey. Each molecule's line uses the color
        defined in its MoleculeSpec. The function preserves the row order from the parquet file while
        connecting points (so points are not sorted by energy and lines remain contiguous).

        Returns:
          A dict keyed by molecule name, with tuples (x_array, y_array) of plotted values. This makes the
          function testable and allows callers to verify the arrays used for plotting.

        Fails fast if:
          - parquet_path does not exist
          - required columns are missing
          - a defined molecule has no matching rows in the file
        """
        # Why: fail early on missing resources to avoid silent downstream errors
        assert config.parquet_path.exists(), f"Expected parquet file at {config.parquet_path} but it does not exist."

        # Only read the necessary columns to keep memory usage low
        required_cols = {"base_inchikey", config.collision_energy_column, config.info_column,"precursor_type"}
        # Read as polars DataFrame and fail if columns missing
        df = pl.read_parquet(config.parquet_path, columns=list(required_cols))
        missing = required_cols.difference(set(df.columns))
        assert not missing, (
            f"File {config.parquet_path} is missing the required columns: {sorted(list(missing))}. "
            "Required columns are: 'base_inchikey', collision_energy_column, and info_column."
        )

        # Prepare plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor="white")

        plotted_data: Dict[str, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}

        for mol in config.molecules:
            # Why: keep original row order; pl.filter doesn't reorder rows
            sub = df.filter(pl.col("base_inchikey") == mol.base_inchikey)
            # Drop rows with missing values; plotting None values leads to broken lines
            sub = sub.filter(
                pl.col(config.collision_energy_column).is_not_null() &
                pl.col(config.info_column).is_not_null(),
                pl.col("precursor_type").eq("[M+H]+"),
            ).sort(pl.col(config.collision_energy_column))

            # Fail fast if user selected a molecule that doesn't exist in dataset
            assert sub.height > 0, (
                f"No rows found for molecule {mol.name} (base_inchikey {mol.base_inchikey}) in {config.parquet_path}. "
                f"Check the base_inchikey and confirm column '{config.collision_energy_column}' and '{config.info_column}' exist."
            )

            x = sub.select(pl.col(config.collision_energy_column).cast(pl.Float64)).to_numpy().ravel()
            y = sub.select(pl.col(config.info_column).cast(pl.Float64)).to_numpy().ravel()

            # Keep the contiguous line order as in the parquet rows
            ax.plot(
                x,
                y,
                color=mol.color,
                marker=config.marker,
                linewidth=1.5,
                label=mol.name,
                linestyle='-',
            )

            # Save for tests/inspection - we cast explicitly to numpy arrays of floats
            plotted_data[mol.name] = (x, y)

        ax.set_xlabel("Collision energy (eV)")
        ax.set_ylabel("Informativity (spectral information score)")
        ax.grid(alpha=0.25)
        if config.add_title:
            ax.set_title("Informativity vs Collision Energy")
        ax.legend(frameon=False)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(config.output_path), dpi=400, facecolor="white", transparent=False)
        plt.close(fig)

        return plotted_data

    molecules = [
            # MoleculeSpec(name="Amphetamine", base_inchikey="KWTSXDURSIMDCE", color="#FF0000"),  
            # MoleculeSpec(name="Methmphetamine", base_inchikey="MYWUZJCMWCOHBA", color="#FF0000"), 
            # MoleculeSpec(name="MDMA", base_inchikey="SHXWCVYOXRDMCX", color="#FF0000"), 
            MoleculeSpec(name="Cocaine", base_inchikey="ZPUCINDJVBIVPJ", color="#0000FF"),
            # MoleculeSpec(name="Clonazepam", base_inchikey="DGBIGWXXNGSACT", color="#0000FF"),
            # MoleculeSpec(name="Fentanyl", base_inchikey="PJMPHNIQZUBGLI", color="#D55E00"),
            MoleculeSpec(name="Lidocaine", base_inchikey="NNJVILVZKWQKPM", color="#000000ff"),
            MoleculeSpec(name="Warfarin", base_inchikey="PJVWKTKQMONHTI", color="#FF0000"), 


        ]


    cfg = MoleculePlotConfig(
        parquet_path=Path("/home/analytit_admin/Data/spectral_libs/NIST_hr_msms/NIST_hr_msms.parquet"),
        molecules=molecules,
        output_path=Path("informativity_vs_collision_energy_nist.png"),
        collision_energy_column="collision_energy_ev",
        info_column="spectral_information_score",
        add_title=False,
        marker='x',
    )

    plotted = plot_informativity_vs_collision_energy(cfg)



    return


if __name__ == "__main__":
    app.run()
