import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from pathlib import Path
    
    # Domain imports from hrms_utils
    import hrms_utils
    return (
        Path,
        dataclass,
        hrms_utils,
        np,
        pl,
        plt,
    )


@app.cell
def _(Path, dataclass):
    @dataclass
    class ExperimentConfig:
        """
        Configuration for EAD vs CID comparison.
        """
        msp_path: Path
        output_plot_path: Path
        
        # Column names in the MSP dataframe
        fragmentation_col: str = "fragmentation_method"
        ion_mode_col: str = "ion_mode"
        compound_col: str = "base_inchikey"
        
        # Labels to search for in fragmentation_col
        cid_label: str = "CID"
        ead_label: str = "EAD"

    return ExperimentConfig


@app.cell
def _(ExperimentConfig, hrms_utils, pl):
    def load_and_annotate_data(config: ExperimentConfig) -> pl.DataFrame:
        """
        Reads MSP, cleans spectra, annotates formulas, and calculates base scores.
        """
        # Why: Fail fast if resource is missing
        assert config.msp_path.exists(), f"MSP file not found at {config.msp_path}"

        # 1. Read MSP
        # Why: Use hrms_utils for standardized parsing
        df = hrms_utils.formats.read_MSPEC_file(config.msp_path)

        # add info about fragmentation method and polarity based on file name
        if "CID" in config.msp_path.name.upper():
            df = df.with_columns(
                pl.lit("CID").alias(config.fragmentation_col)
            )
        elif "EAD" in config.msp_path.name.upper():
            df = df.with_columns(
                pl.lit("EAD").alias(config.fragmentation_col)
            )
        else:
            raise ValueError("Fragmentation method (CID/EAD) not found in MSP file name.")
        if "POS" in config.msp_path.name.upper():
            df = df.with_columns(
                pl.lit("POS").alias(config.ion_mode_col)
            )
        elif "NEG" in config.msp_path.name.upper():
            df = df.with_columns(
                pl.lit("NEG").alias(config.ion_mode_col)
            )
        else:
            raise ValueError("Ion mode (POS/NEG) not found in MSP file name.")
        return df

    def get_best_spectra_per_method(
        df: pl.DataFrame, 
        config: ExperimentConfig
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Splits data into CID and EAD, then selects the single most informative 
        spectrum per compound and ion mode.
        """
        # Filter for CID and EAD
        # Why: Use case-insensitive matching for robustness
        df_cid = df.filter(
            pl.col(config.fragmentation_col).eq("CID")
        )
        df_ead = df.filter(
            pl.col(config.fragmentation_col).eq("EAD")
        )

        # Helper to pick best
        def pick_best(d: pl.DataFrame) -> pl.DataFrame:
            # Sort by score descending, then take first for each group
            return d.sort(by="spectral_information_score", descending=True).group_by(
                [config.compound_col, config.ion_mode_col]
            ).first()

        best_cid = pick_best(df_cid)
        best_ead = pick_best(df_ead)

        return best_cid, best_ead

    return get_best_spectra_per_method, load_and_annotate_data


@app.cell
def _(ExperimentConfig, hrms_utils, pl):
    def compute_combined_scores(
        best_cid: pl.DataFrame, 
        best_ead: pl.DataFrame, 
        config: ExperimentConfig
    ) -> pl.DataFrame:
        """
        Joins CID and EAD data, combines fragment formulas, and computes combined score.
        """
        # Join on compound and ion mode
        # Why: Inner join ensures we only compare compounds that have BOTH methods available
        joined = best_cid.join(
            best_ead,
            on=[config.compound_col, config.ion_mode_col],
            suffix="_ead"
        )

        # Columns containing the annotated formulas (list of strings/structs)
        # Assuming column name is 'fragment_formula_array' based on hrms_utils conventions
        formula_col = "cleaned_fragment_formulas"
        
        assert formula_col in joined.columns, f"Column {formula_col} missing from dataframe"

        # Combine the lists of formulas
        # Why: We want the union of unique formulas found in either method to represent the "combined" spectrum
        joined = joined.with_columns(
            combined_formulas=pl.col(formula_col).list.set_union(pl.col(f"{formula_col}_ead"))
        ).drop(formula_col).rename(
            {"combined_formulas": formula_col}
        ).with_columns(
            pl.struct(
            pl.col("precursor_formula_array").alias("precursor_formula"), 
            pl.col(formula_col).alias("fragment_formulas")
        ).spectral_info.spectral_info_score(ignore_hydrogens=True).alias("combined_score")
        ).rename({
            "spectral_information_score": "cid_score",
            "spectral_information_score_ead": "ead_score"
        })

        return joined

    return compute_combined_scores


@app.cell
def _(
    ExperimentConfig,
    Path,
    compute_combined_scores,
    get_best_spectra_per_method,
    load_and_annotate_data,
):
    # Execution
    
    # Update path to your actual MSP file
    MSP_PATH = Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/EAD_CID_comparison.msp")
    OUTPUT_PLOT = Path("/home/ser/dev/HRMS_utils/experiments/spectral_information/ead_vs_cid_plot.png")

    config = ExperimentConfig(
        msp_path=MSP_PATH,
        output_plot_path=OUTPUT_PLOT
    )

    # Placeholder check for development if file doesn't exist yet
    if not config.msp_path.exists():
        print(f"Skipping execution: {config.msp_path} does not exist.")
        df_final = None
    else:
        # 1. Load
        df_all = load_and_annotate_data(config)
        
        # 2. Select Best
        best_cid, best_ead = get_best_spectra_per_method(df_all, config)
        
        # 3. Combine
        df_final = compute_combined_scores(best_cid, best_ead, config)
        
        print(f"Processed {df_final.height} compounds with both CID and EAD.")
        print(df_final.select(["base_inchikey", "cid_score", "ead_score", "combined_score"]).head())

    return MSP_PATH, OUTPUT_PLOT, config, df_all, df_final, best_cid, best_ead


@app.cell
def _(df_final, np, plt):
    if df_final is not None and df_final.height > 0:
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        cid_scores = df_final["cid_score"].to_numpy()
        ead_scores = df_final["ead_score"].to_numpy()
        combined_scores = df_final["combined_score"].to_numpy()
        
        # Scatter: Max(CID, EAD) vs Combined
        max_individual = np.maximum(cid_scores, ead_scores)
        
        ax.scatter(max_individual, combined_scores, alpha=0.6, c='purple', label='Combined vs Best Individual')
        
        # Reference line y=x
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label="y=x (No Gain)")
        
        ax.set_xlabel("Max(CID Score, EAD Score)")
        ax.set_ylabel("Combined Spectrum Score")
        ax.set_title("Information Gain from Combining CID and EAD")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    return ax, cid_scores, combined_scores, ead_scores, fig, lims, max_individual

if __name__ == "__main__":
    app.run()
