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
        
        Why: Separate MSP files for each fragmentation method allows for cleaner separation
        of data sources and better reflects typical experimental data organization.
        """
        cid_msp_path: Path
        ead_msp_path: Path
        output_plot_path: Path
        
        # Column names in the MSP dataframe
        fragmentation_col: str = "fragmentation_method"
        ion_mode_col: str = "ion_mode"
        compound_col: str = "base_inchikey"
        
        # Labels to assign for fragmentation methods
        cid_label: str = "CID"
        ead_label: str = "EAD"

    return ExperimentConfig


@app.cell
def _(ExperimentConfig, hrms_utils, pl):
    def load_and_annotate_data(config: ExperimentConfig) -> pl.DataFrame:
        """
        Reads both MSP files, cleans spectra, and annotates with fragmentation method.
        
        Why: Reading separate files allows for independent validation and clearer
        error messages if one file is missing or malformed.
        """
        # Why: Fail fast if resources are missing
        assert config.cid_msp_path.exists(), f"CID MSP file not found at {config.cid_msp_path}"
        assert config.ead_msp_path.exists(), f"EAD MSP file not found at {config.ead_msp_path}"

        # Read CID MSP
        df_cid = hrms_utils.formats.read_MSPEC_file(config.cid_msp_path)
        df_cid = df_cid.with_columns(
            pl.lit(config.cid_label).alias(config.fragmentation_col)
        )
        
        # Detect ion mode from CID filename
        if "POS" in config.cid_msp_path.name.upper():
            df_cid = df_cid.with_columns(pl.lit("POS").alias(config.ion_mode_col))
        elif "NEG" in config.cid_msp_path.name.upper():
            df_cid = df_cid.with_columns(pl.lit("NEG").alias(config.ion_mode_col))
        else:
            raise ValueError(f"Ion mode (POS/NEG) not found in CID MSP file name: {config.cid_msp_path.name}")

        # Read EAD MSP
        df_ead = hrms_utils.formats.read_MSPEC_file(config.ead_msp_path)
        df_ead = df_ead.with_columns(
            pl.lit(config.ead_label).alias(config.fragmentation_col)
        )
        
        # Detect ion mode from EAD filename
        if "POS" in config.ead_msp_path.name.upper():
            df_ead = df_ead.with_columns(pl.lit("POS").alias(config.ion_mode_col))
        elif "NEG" in config.ead_msp_path.name.upper():
            df_ead = df_ead.with_columns(pl.lit("NEG").alias(config.ion_mode_col))
        else:
            raise ValueError(f"Ion mode (POS/NEG) not found in EAD MSP file name: {config.ead_msp_path.name}")

        # Why: Concatenate both dataframes to create unified dataset for downstream processing
        df = pl.concat([df_cid, df_ead], how="vertical")
        
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
        df_cid = df.filter(
            pl.col(config.fragmentation_col).eq(config.cid_label)
        )
        df_ead = df.filter(
            pl.col(config.fragmentation_col).eq(config.ead_label)
        )

        # Helper to pick best
        def pick_best(d: pl.DataFrame) -> pl.DataFrame:
            # Why: Sort by score descending to get most informative spectrum first
            return d.sort(by="spectral_information_score", descending=True).group_by(
                [config.compound_col, config.ion_mode_col]
            ).first()

        best_cid = pick_best(df_cid)
        best_ead = pick_best(df_ead)

        return best_cid, best_ead

    return get_best_spectra_per_method, load_and_annotate_data


@app.cell
def _(ExperimentConfig, pl):
    def compute_combined_scores(
        best_cid: pl.DataFrame, 
        best_ead: pl.DataFrame, 
        config: ExperimentConfig
    ) -> pl.DataFrame:
        """
        Joins CID and EAD data, combines fragment formulas, and computes combined score.
        """
        # Why: Inner join ensures we only compare compounds that have BOTH methods available
        joined = best_cid.join(
            best_ead,
            on=[config.compound_col, config.ion_mode_col],
            suffix="_ead"
        )

        # Columns containing the annotated formulas
        formula_col = "cleaned_fragment_formulas"
        
        assert formula_col in joined.columns, f"Column {formula_col} missing from dataframe. Available columns: {joined.columns}"

        # Why: Union of unique formulas from both methods represents combined information
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
    
    # Why: Update paths to point to separate CID and EAD MSP files
    CID_MSP_PATH = Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/CID_spectra.msp")
    EAD_MSP_PATH = Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/EAD_spectra.msp")
    OUTPUT_PLOT = Path("/home/ser/dev/HRMS_utils/experiments/spectral_information/ead_vs_cid_plot.png")

    config = ExperimentConfig(
        cid_msp_path=CID_MSP_PATH,
        ead_msp_path=EAD_MSP_PATH,
        output_plot_path=OUTPUT_PLOT
    )

    # Why: Check both files exist before attempting to process
    if not config.cid_msp_path.exists():
        print(f"Skipping execution: {config.cid_msp_path} does not exist.")
        df_final = None
    elif not config.ead_msp_path.exists():
        print(f"Skipping execution: {config.ead_msp_path} does not exist.")
        df_final = None
    else:
        # 1. Load both files
        df_all = load_and_annotate_data(config)
        
        # 2. Select Best
        best_cid, best_ead = get_best_spectra_per_method(df_all, config)
        
        # 3. Combine
        df_final = compute_combined_scores(best_cid, best_ead, config)
        
        print(f"Processed {df_final.height} compounds with both CID and EAD.")
        print(df_final.select(["base_inchikey", "cid_score", "ead_score", "combined_score"]).head())

    return CID_MSP_PATH, EAD_MSP_PATH, OUTPUT_PLOT, config, df_all, df_final, best_cid, best_ead


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
