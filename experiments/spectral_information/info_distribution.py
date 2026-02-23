import marimo
# from pandas.util.version import PrePostDevType

__generated_with = "0.18.3"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.typing as npt
    import polars as pl

    plt.style.use("default")

    from dataclasses import dataclass, replace
    from pathlib import Path
    from typing import Dict, List, Literal, Optional, Tuple

    return (
        Dict,
        List,
        Literal,
        Optional,
        Path,
        Tuple,
        dataclass,
        np,
        npt,
        pl,
        plt,
        replace,
    )


@app.cell
def _(Path, pl):
    input_PARQUET_PATHS = [
        Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"),
        Path(
            "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet"
        ),
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
        add_title: bool = False

    def read_library_scores_and_inchikeys(path: Path) -> pl.DataFrame:
        """
        Read spectral_information_score and base_inchikey from a parquet file, returning a polars DataFrame.
        Fail fast if file is missing or expected columns are absent.

        Returns:
            pl.DataFrame containing columns ['spectral_information_score', 'base_inchikey'].
        """
        # Why: Fail early to avoid silent issues downstream with missing data columns or paths.
        assert path.exists(), f"Expected parquet file at {path} but it does not exist."
        df = pl.read_parquet(
            path, columns=["spectral_information_score", "base_inchikey"]
        )
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
        assert len(config.parquet_paths) == len(config.labels), (
            "parquet_paths and labels must have same length"
        )

        # Why: Read all libraries and compute stats for each.
        stats = []

        for path, label in zip(config.parquet_paths, config.labels):
            df = read_library_scores_and_inchikeys(path)
            n_spectra = df.height
            molecule_max_info_arr = compute_molecule_max_info(df)
            n_molecules = int(molecule_max_info_arr.size)

            spectrum_info_arr = (
                df.select(pl.col("spectral_information_score").cast(pl.Float64))
                .to_numpy()
                .ravel()
            )
            assert spectrum_info_arr.size > 0, (
                f"No non-null 'spectral_information_score' values found in {path}"
            )
            print(
                f"Library '{label}': {n_spectra} spectra, {n_molecules} unique molecules."
            )
            print(
                f"Per spectra, the mean is {np.mean(spectrum_info_arr):.4f} and the median is {np.median(spectrum_info_arr):.4f}."
            )
            print(
                f"Per molecule max, the mean is {np.mean(molecule_max_info_arr):.4f} and the median is {np.median(molecule_max_info_arr):.4f}."
            )
            stats.append(
                {
                    "label": label,
                    "path": path,
                    "n_spectra": int(n_spectra),
                    "n_molecules": n_molecules,
                    "spectrum_info_arr": spectrum_info_arr,
                    "molecule_max_info_arr": molecule_max_info_arr,
                }
            )

        # Prepare subplots: 1 row and N columns (one per library) to overlay both distributions on the same axis.
        n_libs = len(stats)
        fig_width = max(
            5 * n_libs, 6
        )  # Provide reasonable width scaling with number of libraries.
        fig, axes = plt.subplots(
            1,
            n_libs,
            figsize=(fig_width, 5),
            squeeze=False,
            sharex=False,
            facecolor="white",
        )

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
        fig.savefig(
            str(config.output_path), dpi=400, facecolor="white", transparent=False
        )
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
        Path(
            "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet"
        ),
    ]
    LABELS = [
        # "Combined",
        "NIST23",
        "Fraghub",
    ]
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
        assert config.parquet_path.exists(), (
            f"Expected parquet file at {config.parquet_path} but it does not exist."
        )

        # Only read the necessary columns to keep memory usage low
        required_cols = {
            "base_inchikey",
            config.collision_energy_column,
            config.info_column,
            "precursor_type",
        }
        # Read as polars DataFrame and fail if columns missing
        df = pl.read_parquet(config.parquet_path, columns=list(required_cols))
        missing = required_cols.difference(set(df.columns))
        assert not missing, (
            f"File {config.parquet_path} is missing the required columns: {sorted(list(missing))}. "
            "Required columns are: 'base_inchikey', collision_energy_column, and info_column."
        )

        # Prepare plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor="white")

        plotted_data: Dict[
            str, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = {}

        for mol in config.molecules:
            # Why: keep original row order; pl.filter doesn't reorder rows
            sub = df.filter(pl.col("base_inchikey") == mol.base_inchikey)
            # Drop rows with missing values; plotting None values leads to broken lines
            sub = sub.filter(
                pl.col(config.collision_energy_column).is_not_null()
                & pl.col(config.info_column).is_not_null(),
                pl.col("precursor_type").eq("[M+H]+"),
            ).sort(pl.col(config.collision_energy_column))

            # Fail fast if user selected a molecule that doesn't exist in dataset
            assert sub.height > 0, (
                f"No rows found for molecule {mol.name} (base_inchikey {mol.base_inchikey}) in {config.parquet_path}. "
                f"Check the base_inchikey and confirm column '{config.collision_energy_column}' and '{config.info_column}' exist."
            )

            x = (
                sub.select(pl.col(config.collision_energy_column).cast(pl.Float64))
                .to_numpy()
                .ravel()
            )
            y = (
                sub.select(pl.col(config.info_column).cast(pl.Float64))
                .to_numpy()
                .ravel()
            )

            # Keep the contiguous line order as in the parquet rows
            ax.plot(
                x,
                y,
                color=mol.color,
                marker=config.marker,
                linewidth=1.5,
                label=mol.name,
                linestyle="-",
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
        fig.savefig(
            str(config.output_path), dpi=400, facecolor="white", transparent=False
        )
        plt.close(fig)

        return plotted_data

    molecules = [
        # MoleculeSpec(name="Amphetamine", base_inchikey="KWTSXDURSIMDCE", color="#FF0000"),
        # MoleculeSpec(name="Methmphetamine", base_inchikey="MYWUZJCMWCOHBA", color="#FF0000"),
        # MoleculeSpec(name="MDMA", base_inchikey="SHXWCVYOXRDMCX", color="#FF0000"),
        MoleculeSpec(name="Cocaine", base_inchikey="ZPUCINDJVBIVPJ", color="#0000FF"),
        # MoleculeSpec(name="Clonazepam", base_inchikey="DGBIGWXXNGSACT", color="#0000FF"),
        # MoleculeSpec(name="Fentanyl", base_inchikey="PJMPHNIQZUBGLI", color="#D55E00"),
        MoleculeSpec(
            name="Lidocaine", base_inchikey="NNJVILVZKWQKPM", color="#000000ff"
        ),
        MoleculeSpec(name="Warfarin", base_inchikey="PJVWKTKQMONHTI", color="#FF0000"),
    ]

    cfg = MoleculePlotConfig(
        parquet_path=Path(
            "/home/analytit_admin/Data/spectral_libs/NIST_hr_msms/NIST_hr_msms.parquet"
        ),
        molecules=molecules,
        output_path=Path("informativity_vs_collision_energy_nist.png"),
        collision_energy_column="collision_energy_ev",
        info_column="spectral_information_score",
        add_title=False,
        marker="x",
    )

    plotted = plot_informativity_vs_collision_energy(cfg)
    return


@app.cell
def _(
    Literal,
    Optional,
    Path,
    Tuple,
    dataclass,
    pl,
    plt,
    replace,
    x_array,
    y_array,
):
    @dataclass
    class InformativityVsMassConfig:
        """
        Configuration for plotting informativity vs a chosen x-axis metric for maximum per-molecule spectra.

        New unified fields:
          x_bin_width: bin width for chosen x-axis metric (applies to mass or any other metric).
          x_range: optional (min, max) bounds to apply to the x-axis (applies to the chosen metric).
          bonds_column is no longer needed; 'bonds' metric uses the precursor_formula_array dot-product.
        """

        parquet_path: Path
        output_path: Path
        mass_column: str = "precursor_mz"
        x_bin_width: float = 1.0
        x_range: Optional[Tuple[float, float]] = None
        color: str = "#D55E00"  # Colorblind-friendly orange
        figsize: Tuple[float, float] = (8, 5)

        # New plotting options
        plot_kind: Literal["line", "scatter"] = "line"
        scatter_marker: str = "o"
        scatter_alpha: float = 0.8
        scatter_size: float = 8.0
        show_mean_line: bool = True

        # New: select x-axis metric
        x_metric: Literal["mass", "heavy_atoms", "bonds", "bonds_sqrt"] = "mass"

        # Why: convenience helper to create a modified copy of the config with fail-fast validation.
        def copy(self, **changes) -> "InformativityVsMassConfig":
            """
            Return a new InformativityVsMassConfig with provided fields overwritten.
            Fail fast if unknown field names are supplied.
            """
            valid_fields = set(self.__dataclass_fields__.keys())
            unknown = set(changes) - valid_fields
            assert not unknown, (
                f"Unknown fields for InformativityVsMassConfig.copy(): {sorted(list(unknown))}"
            )
            return replace(self, **changes)

    def compute_molecule_max_info_with_metric(
        df: pl.DataFrame, metric_expr: pl.Expr, metric_name: str
    ) -> pl.DataFrame:
        """
        Compute, per molecule (base_inchikey), the row with the maximal 'spectral_information_score'
        and return a DataFrame with the maximal score and the corresponding metric value.

        The metric is computed using a Polars expression (metric_expr) and aliased to metric_name.
        This keeps all column derivation in Polars expressions and avoids materializing Python loops.

        Returns:
          pl.DataFrame with columns ['max_info', metric_name] cast to Float64.

        Fail fast if required columns are missing (base_inchikey, spectral_information_score).
        """
        required = {"base_inchikey", "spectral_information_score"}
        missing = required.difference(set(df.columns))
        assert not missing, (
            f"DataFrame is missing required columns: {sorted(list(missing))}"
        )

        # Compute metric using the provided Polars expression and keep only required columns.
        df_with_metric = df.with_columns(metric_expr.alias(metric_name))

        # Drop rows with missing score or missing metric value to avoid wrong maxima selection
        df_sub = df_with_metric.select(
            ["base_inchikey", "spectral_information_score", metric_name]
        ).filter(
            pl.col("spectral_information_score").is_not_null()
            & pl.col(metric_name).is_not_null()
        )

        assert df_sub.height > 0, (
            f"No valid rows found with non-null 'spectral_information_score' and '{metric_name}'."
        )

        # Sort descending by score and keep the first row per base_inchikey -> will be the row with max score
        best_per_mol = df_sub.sort(
            "spectral_information_score", descending=True
        ).unique(subset=["base_inchikey"], keep="first")

        return best_per_mol.select(
            pl.col("spectral_information_score").alias("max_info").cast(pl.Float64),
            pl.col(metric_name).alias("x_val").cast(pl.Float64),
        )

    def plot_informativity_vs_metric(config: InformativityVsMassConfig) -> None:
        """
        Plot either the binned mean ± std line (plot_kind == 'line') or a scatter of per-molecule maxima
        (plot_kind == 'scatter') for a chosen x-axis metric.

        The plotting bounds and bin width are defined by config.x_range and config.x_bin_width and are
        applied irrespective of the selected metric.
        """
        # Fail early on missing resources and invalid params
        assert config.parquet_path.exists(), (
            f"Expected parquet file at {config.parquet_path} but it does not exist."
        )

        # Decide which base columns we need depending on metric (fail fast if missing)
        required_cols = [
            "spectral_information_score",
            "base_inchikey",
            "precursor_formula_array",
            config.mass_column,
        ]
        lf = pl.scan_parquet(config.parquet_path).select(required_cols)
        df = lf.collect()
        print(f"Read {df.height} rows from {config.parquet_path}.")

        for col in required_cols:
            assert col in df.columns, (
                f"File {config.parquet_path} is missing required column '{col}'."
            )

        # Drop null scores
        df = df.filter(pl.col("spectral_information_score").is_not_null())
        assert df.height > 0, (
            f"No non-null 'spectral_information_score' values found in {config.parquet_path}."
        )

        # Choose metric using match statement; build a Polars expression for metric (metric_expr)
        match config.x_metric:
            case "mass":
                metric_expr = pl.col(config.mass_column).cast(pl.Float64)
                metric_label = "Precursor mass [Da]"
            case "heavy_atoms":
                # Compute (precursor_formula_array - precursor_formula_array.arr.get(0)).arr.sum()
                assert "precursor_formula_array" in df.columns, (
                    "Requested x_metric 'heavy_atoms' but 'precursor_formula_array' column is missing."
                )
                metric_expr = (
                    pl.col("precursor_formula_array").arr.sum()
                    - pl.col("precursor_formula_array").arr.get(0)
                ).cast(pl.Float64)
                metric_label = "Number heavy atoms"
            case "bonds":
                # Bonds metric: dot-product against coefficients vector using Polars expressions
                coeffs = [-0.5, 2.0, 1.5, 1.0, 0.5, 0.5, 1.5, 1.0, 0.5, 0.5, 0.5, 0.5]
                assert "precursor_formula_array" in df.columns, (
                    "Requested x_metric 'bonds' but 'precursor_formula_array' column is missing."
                )
                # Build expression: sum(arr.get(i) * coeffs[i] for i in range(len(coeffs)))
                coeff_expr = sum(
                    (pl.col("precursor_formula_array").arr.get(i) * coeffs[i])
                    for i in range(len(coeffs))
                )
                metric_expr = coeff_expr.cast(pl.Float64)
                metric_label = "Bonds metric"
            case "bonds_sqrt":
                coeffs = [-0.5, 2.0, 1.5, 1.0, 0.5, 0.5, 1.5, 1.0, 0.5, 0.5, 0.5, 0.5]
                assert "precursor_formula_array" in df.columns, (
                    "Requested x_metric 'bonds_sqrt' but 'precursor_formula_array' column is missing."
                )
                # Build expression: sum(arr.get(i) * coeffs[i] for i in range(len(coeffs)))
                coeff_expr = sum(
                    (pl.col("precursor_formula_array").arr.get(i) * coeffs[i])
                    for i in range(len(coeffs))
                )
                metric_expr = coeff_expr.cast(pl.Float64).sqrt()
                metric_label = "Bonds_sqrt metric"
        # Compute one row per molecule (the row with max informativity and its x metric)
        max_rows_df = compute_molecule_max_info_with_metric(
            df, metric_expr, "x_metric_temp"
        )

        # Optionally apply x range bounds to the per-molecule maxima (metric-agnostic)
        if config.x_range is not None:
            x_min, x_max = config.x_range
            max_rows_df = max_rows_df.filter(
                pl.col("x_val").is_between(x_min, x_max, closed="both")
            )
            assert max_rows_df.height > 0, (
                f"No per-molecule maxima remain after applying x_range {config.x_range}."
            )

        # spearman correlation using polars
        spearman_corr = max_rows_df.select(
            pl.corr("x_val", "max_info", method="spearman")
        ).item()
        print(
            f"Spearman correlation between {config.x_metric} and max informativity: {spearman_corr:.4f}"
        )

        # Use the unified bin width
        bin_width = config.x_bin_width

        # For line plots we compute binned means and stds on the per-molecule maxima
        max_binned = max_rows_df.with_columns(
            ((pl.col("x_val") / bin_width).round() * bin_width).alias("x_bin")
        )

        agg = (
            max_binned.group_by("x_bin")
            .agg(
                mean_info=pl.col("max_info").mean(),
                std_info=pl.col("max_info").std(),
                count=pl.len(),
            )
            .sort("x_bin")
        )

        x_bins = agg.get_column("x_bin").to_numpy()
        mean_info = agg.get_column("mean_info").to_numpy()
        std_info = agg.get_column("std_info").to_numpy()

        # Fail fast if nothing to plot
        if config.plot_kind == "line":
            assert x_bins.size > 0, (
                "No x-binned data found after aggregation; check input dataframe and bin width."
            )
        else:
            assert x_array.size > 0, "No per-molecule maxima found to plot as scatter."

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=config.figsize, facecolor="white")

        if config.plot_kind == "line":
            ax.plot(
                x_bins,
                mean_info,
                color=config.color,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label="Mean informativity (per-molecule maxima)",
            )
            fill_lower = mean_info - std_info
            fill_upper = mean_info + std_info
            ax.fill_between(
                x_bins,
                fill_lower,
                fill_upper,
                color=config.color,
                alpha=0.2,
                label="±1 Standard Deviation",
            )
        else:
            ax.scatter(
                x_array,
                y_array,
                s=config.scatter_size,
                marker=config.scatter_marker,
                alpha=config.scatter_alpha,
                color=config.color,
                label="Per-molecule maxima",
            )
            # Optionally overlay mean ± std line computed on bins
            if config.show_mean_line and x_bins.size > 0:
                ax.plot(
                    x_bins,
                    mean_info,
                    color=config.color,
                    linewidth=1.25,
                    linestyle="-",
                    alpha=0.9,
                    label="Binned mean (±1 std)",
                )
                ax.fill_between(
                    x_bins,
                    mean_info - std_info,
                    mean_info + std_info,
                    color=config.color,
                    alpha=0.15,
                )

        ax.set_xlabel(metric_label)
        ax.set_ylabel("Maximal Spectral Informativiness per molecule")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

        # Apply x limits if provided
        if config.x_range is not None:
            ax.set_xlim(config.x_range)

        # Ensure output dir exists and save
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(
            str(config.output_path), dpi=400, facecolor="white", transparent=False
        )
        plt.close(fig)
        print(f"Plot successfully saved to {config.output_path}")

    bonds_config = InformativityVsMassConfig(
        parquet_path=Path(
            "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/combined_spectral_lib.parquet"
        ),
        output_path=Path("informativity_vs_bonds.png"),
        mass_column="precursor_mz",
        x_bin_width=2.0,
        x_range=(0.0, 90.0),  # Set bounds here if desired, e.g. (0.0, 900.0)
        plot_kind="line",  # Change to "scatter" to plot individual points
        x_metric="bonds",  # Choose "mass", "heavy_atoms", or "bonds"
    )
    plot_informativity_vs_metric(bonds_config)

    bonds_sqrt_config = bonds_config.copy(
        output_path=Path("informativity_vs_bonds_sqrt.png"),
        x_metric="bonds_sqrt",
        x_bin_width=0.5,
        x_range=(0.0, 15.0),
    )
    plot_informativity_vs_metric(bonds_sqrt_config)

    mass_config = bonds_config.copy(
        output_path=Path("informativity_vs_mass.png"),
        x_metric="mass",
        x_bin_width=20.0,
        x_range=(0.0, 850.0),
    )
    plot_informativity_vs_metric(mass_config)

    heavy_atoms_config = bonds_config.copy(
        output_path=Path("informativity_vs_heavy_atoms.png"),
        x_metric="heavy_atoms",
        x_bin_width=1.0,
        x_range=(0.0, 60.0),
    )
    plot_informativity_vs_metric(heavy_atoms_config)
    return


@app.cell
def _(Path, Tuple, dataclass, np, pl, plt):
    @dataclass
    class StackedBarConfig:
        """
        Configuration for stacked bar chart comparing NIST and FragHub distributions.

        parquet_paths: list of parquet files (NIST, FragHub order expected).
        labels: labels for each dataset.
        output_path: where to save the figure.
        bins: number of histogram bins.
        range: value range for the histogram.
        top_percentile: percentile threshold for 'most informative' molecules (e.g., 90 means top 10%).
        """

        parquet_paths: list
        labels: list
        output_path: Path
        bins: int = 50
        range: Tuple[float, float] = (0.0, 5.0)
        top_percentile: float = 90.0

    def plot_stacked_bar_distributions(config: StackedBarConfig) -> None:
        """
        Create a 2-subplot figure with stacked bar charts:
        - Top: distribution of all molecules (using their most informative spectrum)
        - Bottom: distribution of all spectra (stacked NIST + FragHub)
        """
        assert len(config.parquet_paths) == len(config.labels) == 2, (
            "Expected exactly 2 parquet paths and labels for NIST and FragHub"
        )

        bin_edges = np.linspace(config.range[0], config.range[1], config.bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = bin_edges[1] - bin_edges[0]

        all_mol_maxes = []
        all_spectra_scores = []

        # Why: Use polars for all data processing and reuse compute_molecule_max_info.
        for path, label in zip(config.parquet_paths, config.labels):
            assert path.exists(), f"Parquet file not found: {path}"
            df = pl.read_parquet(
                path, columns=["spectral_information_score", "base_inchikey"]
            )
            df = df.filter(pl.col("spectral_information_score").is_not_null())

            # Top: Most informative spectrum per molecule
            mol_max = compute_molecule_max_info(df)
            all_mol_maxes.append(mol_max)

            # Bottom: All spectra
            spec_scores = (
                df.select(pl.col("spectral_information_score").cast(pl.Float64))
                .to_numpy()
                .ravel()
            )
            all_spectra_scores.append(spec_scores)

            print(
                f"{label}: {len(mol_max)} molecules, {len(spec_scores)} total spectra."
            )

        mol_counts_0, _ = np.histogram(all_mol_maxes[0], bins=bin_edges)
        mol_counts_1, _ = np.histogram(all_mol_maxes[1], bins=bin_edges)

        spec_counts_0, _ = np.histogram(all_spectra_scores[0], bins=bin_edges)
        spec_counts_1, _ = np.histogram(all_spectra_scores[1], bins=bin_edges)

        colors = ["#0072B2", "#D55E00"]

        fig, axes = plt.subplots(
            2, 1, figsize=(8, 8), sharex=True, facecolor="white"
        )

        # Top: Most informative spectrum per molecule
        axes[0].bar(
            bin_centers,
            mol_counts_0,
            width=bar_width,
            color=colors[0],
            label=config.labels[0],
            edgecolor="black",
            linewidth=0.5,
        )
        axes[0].bar(
            bin_centers,
            mol_counts_1,
            width=bar_width,
            bottom=mol_counts_0,
            color=colors[1],
            label=config.labels[1],
            edgecolor="black",
            linewidth=0.5,
        )
        axes[0].set_ylabel("Molecule Count")
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=0.25)

        # Bottom: All spectra
        axes[1].bar(
            bin_centers,
            spec_counts_0,
            width=bar_width,
            color=colors[0],
            label=config.labels[0],
            edgecolor="black",
            linewidth=0.5,
        )
        axes[1].bar(
            bin_centers,
            spec_counts_1,
            width=bar_width,
            bottom=spec_counts_0,
            color=colors[1],
            label=config.labels[1],
            edgecolor="black",
            linewidth=0.5,
        )
        axes[1].set_xlabel("Spectral Information Score")
        axes[1].set_ylabel("Spectrum Count")
        axes[1].legend(frameon=False)
        axes[1].grid(alpha=0.25)

        axes[1].set_xlim(config.range)

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(
            str(config.output_path), dpi=400, facecolor="white", transparent=False
        )
        plt.close(fig)
        print(f"Stacked bar chart saved to {config.output_path}")

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(
            str(config.output_path), dpi=400, facecolor="white", transparent=False
        )
        plt.close(fig)
        print(f"Stacked bar chart saved to {config.output_path}")

    stacked_config = StackedBarConfig(
        parquet_paths=[
            Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"),
            Path(
                "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet"
            ),
        ],
        labels=["NIST23", "Fraghub"],
        output_path=Path("stacked_bar_distribution_nist_fraghub.png"),
        bins=50,
        range=(0.0, 5.0),
        top_percentile=90.0,
    )
    plot_stacked_bar_distributions(stacked_config)
    return


if __name__ == "__main__":
    app.run()
