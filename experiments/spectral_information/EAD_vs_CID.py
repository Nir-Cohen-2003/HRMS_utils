import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from pathlib import Path
    from typing import List, Tuple

    # Domain imports from hrms_utils
    import hrms_utils
    return List, Path, Tuple, dataclass, hrms_utils, np, pl, plt


@app.cell
def _(List, Path, dataclass):
    @dataclass
    class ExperimentConfig:
        """
        Configuration allowing an explicit list of MSP files for both fragmentation
        methods and polarities (CID/EAD x POS/NEG).

        Why: Accepting a list makes it explicit that we expect one file per (method, polarity).
        """
        # Expect exactly 4 MSP files: CID_POS, CID_NEG, EAD_POS, EAD_NEG
        msp_paths: List[Path]
        output_plot_path: Path

        # Column names in the MSP dataframe
        fragmentation_col: str = "fragmentation_method"
        ion_mode_col: str = "ion_mode"
        compound_col: str = "base_inchikey"

        # Labels to assign for fragmentation methods
        cid_label: str = "CID"
        ead_label: str = "EAD"
    return (ExperimentConfig,)


@app.cell
def _(ExperimentConfig, List, Tuple, hrms_utils, pl):
    def load_and_annotate_data(config: ExperimentConfig) -> pl.DataFrame:
        """
        Reads multiple MSP files and annotates each with fragmentation and polarity
        derived from the filename. Returns a concatenated Polars DataFrame.

        Why: Fail fast on missing files and keep metadata explicit per-spectrum.
        """
        assert len(config.msp_paths) == 4, (
            f"Expected 4 MSP files (CID/EAD x POS/NEG), got {len(config.msp_paths)}"
        )

        frames: List[pl.DataFrame] = []
        for p in config.msp_paths:
            assert p.exists(), f"MSP file not found at {p}"
            df_local = hrms_utils.formats.read_MSPEC_file(p)
            fname = p.name.upper()

            # Fragmentation detection
            if "CID" in fname:
                frag_label = config.cid_label
            elif "EAD" in fname:
                frag_label = config.ead_label
            else:
                raise ValueError(f"Fragmentation method (CID/EAD) not found in file name: {p.name}")

            df_local = df_local.with_columns(pl.lit(frag_label).alias(config.fragmentation_col))

            # Polarity detection (standardizing to 'POS'/'NEG')
            if "POS" in fname:
                ion_mode = "POS"
            elif "NEG" in fname:
                ion_mode = "NEG"
            else:
                raise ValueError(f"Ion mode (POS/NEG) not found in file name: {p.name}")

            df_local = df_local.with_columns(pl.lit(ion_mode).alias(config.ion_mode_col))
            frames.append(df_local)

        df = pl.concat(frames, how="vertical")
        return df

    def get_best_spectra_per_method(
        df: pl.DataFrame,
        config: ExperimentConfig
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Splits data into CID and EAD, then selects the single most informative
        spectrum per compound and ion mode.
        """
        df_cid = df.filter(pl.col(config.fragmentation_col).eq(config.cid_label))
        df_ead = df.filter(pl.col(config.fragmentation_col).eq(config.ead_label))

        def pick_best(d: pl.DataFrame) -> pl.DataFrame:
            # Why: Sort descending to put highest score first then take the first per group
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
        # Inner join: only keep compounds present in both methods for the same polarity
        joined = best_cid.join(
            best_ead,
            on=[config.compound_col, config.ion_mode_col],
            suffix="_ead"
        )

        formula_col = "cleaned_fragment_formulas"
        # Assert presence of both formula columns prior to union operation
        assert formula_col in joined.columns and f"{formula_col}_ead" in joined.columns, (
            f"Both {formula_col} and {formula_col}_ead are required for union. "
            f"Available: {joined.columns}"
        )

        # Why: Combine union of unique formulas from both methods
        joined = joined.with_columns(
            pl.col(formula_col).list.set_union(pl.col(f"{formula_col}_ead")).alias(formula_col)
        ).drop(f"{formula_col}_ead")

        # Compute combined spectral score using the combined fragment formulas and a chosen precursor
        joined = joined.with_columns(
            pl.struct([
                pl.col("precursor_formula_array").alias("precursor_formula"),
                pl.col(formula_col).alias("fragment_formulas"),
            ]).spectral_info.spectral_info_score(ignore_hydrogens=True).alias("combined_score")
        ).rename({
            "spectral_information_score": "cid_score",
            "spectral_information_score_ead": "ead_score"
        })

        return joined
    return (compute_combined_scores,)


@app.cell
def _(
    ExperimentConfig,
    Path,
    compute_combined_scores,
    get_best_spectra_per_method,
    load_and_annotate_data,
):
    # Execution

    # Why: Provide explicit MSP paths for each (method, polarity). Update these as needed
    CID_POS_MSP = Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/CID_spectra_POS.msp")
    CID_NEG_MSP = Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/CID_spectra_NEG.msp")
    EAD_POS_MSP = Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/EAD_spectra_POS.msp")
    EAD_NEG_MSP = Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/EAD_spectra_NEG.msp")
    OUTPUT_PLOT = Path("/home/ser/dev/HRMS_utils/experiments/spectral_information/ead_vs_cid_plot.png")

    config = ExperimentConfig(
        msp_paths=[CID_POS_MSP, CID_NEG_MSP, EAD_POS_MSP, EAD_NEG_MSP],
        output_plot_path=OUTPUT_PLOT
    )

    # Validate all files exist before running
    missing = [str(p) for p in config.msp_paths if not p.exists()]
    if missing:
        print("Skipping execution. Missing MSP files:\n" + "\n".join(missing))
        df_final = None
    else:
        # 1. Load and annotate all MSPs (frag method & polarity)
        df_all = load_and_annotate_data(config)

        # 2. Select Best per method/polarity/compound (one row each)
        best_cid, best_ead = get_best_spectra_per_method(df_all, config)

        # 3. Combine and compute combined scores
        df_final = compute_combined_scores(best_cid, best_ead, config)

        print(f"Processed {df_final.height} compounds with both CID and EAD.")
        print(df_final.select(["base_inchikey", "cid_score", "ead_score", "combined_score"]).head())
    return (df_final,)


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
    return


if __name__ == "__main__":
    app.run()
