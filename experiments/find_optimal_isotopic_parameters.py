from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from hrms_utils.formats import get_chromatogram
from hrms_utils.formula_annotation.element_table import ELEMENT_INDEX, ELEMENTS
from hrms_utils.formula_annotation.utils import formula_to_array

# Constants
CARBON_INDEX = ELEMENT_INDEX["C"]
SULFUR_INDEX = ELEMENT_INDEX["S"]
CHLORINE_INDEX = ELEMENT_INDEX["Cl"]
BROMINE_INDEX = ELEMENT_INDEX["Br"]


@dataclass
class Config:
    # File I/O (defaults intentionally empty; set in __main__)
    chromatogram_dir: Path | None = None
    chromatogram_glob: str = "**/*.mdpeak"
    spectral_library_paths: tuple[Path, ...] = ()
    output_file: Path = Path("experiments/isotopic_parameters_results.txt")

    # What sources are used to construct the calibration dataset
    use_library_search_for_ground_truth: bool = True
    use_mass_list_for_ground_truth: bool = False

    # Ground-truth list mode: CSV path containing at least the columns:
    #   - "Molecular Formula"
    #   - "Monoisotopic Mass"
    # Extra columns are allowed and ignored.
    ground_truth_csv_path: Path | None = None
    ground_truth_ms1_tolerance_ppm: float = 5.0

    # Search Parameters (library-search mode)
    precursor_tolerance_ppm: float = 5.0
    dot_product_threshold: float = 0.9
    ms2_tolerance_ppm: float = 10.0

    # Information Score Parameters
    # Why: We filter by info score before searching to ensure we only search high-quality spectra
    # that are well-explained by the candidate formula.
    info_score_threshold: float = 1.0
    fragment_cleaning_tolerance_ppm: float = 5.0
    min_dbe: float = -0.5
    max_dbe: float = 30.0

    # Optimization Parameters
    ms1_mass_tolerance_ppm: float = 5.0
    isotopic_mass_tolerance_ppm: float = 2.0
    minimum_isotopic_peak_intensity: float = 1e4
    target_success_rate: float = 0.99
    mass_accuracy_threshold_da: float = 200.0


@dataclass
class IsotopicToleranceModel:
    """
    Calibrates expected isotopic-peak intensity bounds from an observed isotopic-peak intensity.

    Contract:
      - You provide `observed_isotopic_peak_intensity` (input).
      - The model yields bounds on `expected_isotopic_peak_intensity` (output):
            expected_lower(observed) <= expected_true <= expected_upper(observed)
      - Bounds are tuned to include approximately `target_success_rate` of calibration points,
        while being as tight as possible under the chosen parametric form.

    Why:
      - Inference time: you only observe intensities.
      - You still want bounds on the *true/expected* intensity (and thus on element count via
        expected = k(element_count) * I0), so we fit an inverse calibration: expected = h(observed).
      - Detector inefficiency at low intensities often introduces a directional bias; therefore we
        fit *asymmetric* bounds (separate lower/upper quantiles).
      - Measurement uncertainty is heteroscedastic; therefore we optionally model band width as a
        function of observed intensity.
    """

    # Central mapping: expected ≈ alpha * (observed + offset) ** beta
    alpha: float
    beta: float
    offset: float

    # Asymmetric multiplicative bounds in log-space:
    #   expected_lower = expected_hat * exp(-band_lower_width)
    #   expected_upper = expected_hat * exp(+band_upper_width)
    band_lower_width: float
    band_upper_width: float

    # Optional heteroscedastic widths (if returned/used): w(observed) = a * (observed + offset) ** b
    # These are written for inspection/plotting; inference can use either constant or heteroscedastic widths.
    hetero_lower_a: float
    hetero_lower_b: float
    hetero_upper_a: float
    hetero_upper_b: float

    success_rate: float


def fit_isotopic_tolerance_parameters(
    library_hits: pl.DataFrame,
    output_path: Path,
    ms1_mass_tolerance_ppm: float = 5.0,
    isotopic_mass_tolerance_ppm: float = 3.0,
    minimum_isotopic_peak_intensity: float = 5e4,
    target_success_rate: float = 0.99,
    mass_accuracy_threshold_da: float = 200.0,
    precursor_mz_column: str = "Precursor_mz_MSDIAL",
    ms1_mz_column: str = "ms1_isotopes_m/z",
    ms1_intensity_column: str = "ms1_isotopes_intensity",
    formula_array_column: str = "precursor_formula_array",
    true_carbon_count_column: str = "true_carbon_count",
    true_sulfur_count_column: str = "true_sulfur_count",
    true_chlorine_count_column: str = "true_chlorine_count",
    true_bromine_count_column: str = "true_bromine_count",
) -> IsotopicToleranceModel:
    """
    Fit an inverse calibration model to bound expected isotopic-peak intensity from observed isotopic-peak intensity.

    This has been generalized beyond 13C to also include:
      - 34S (+2 peak)
      - 37Cl (+2 peak)
      - 81Br (+2 peak)

    Goal:
      - Learn a mapping expected = h(observed) and tight bounds such that
        at least `target_success_rate` of calibration points satisfy:
            expected_lower(observed) <= expected_true <= expected_upper(observed)

    Why this direction:
      - At inference time you only have observed isotopic-peak intensity.
      - You can then convert expected bounds into element-count bounds via:
            expected = k(n_element) * I0_observed  where k(n)=n*p_heavy/p_light
        so:
            n_lower = expected_lower / (I0_observed * p_heavy/p_light)
            n_upper = expected_upper / (I0_observed * p_heavy/p_light)

    Fits reported and plotted:
      1) Central mapping (monotone):
            expected_hat = alpha * (observed + offset) ** beta
      2) Asymmetric *constant* bounds (multiplicative in log-space, captures bias):
            expected_lower = expected_hat * exp(-wL_const)
            expected_upper = expected_hat * exp(+wU_const)
         where (wL_const, wU_const) are chosen to minimize (wL_const + wU_const) while
         achieving target coverage.
      3) Asymmetric *heteroscedastic* bounds (tighter at high intensity, wider at low):
            expected_lower = expected_hat * exp(-wL(observed))
            expected_upper = expected_hat * exp(+wU(observed))
         with widths:
            wL(observed) = aL * (observed + offset) ** bL
            wU(observed) = aU * (observed + offset) ** bU
         and parameters fit to empirical per-bin quantile residuals.
    """
    # Why: Extract true carbon count and filter for valid data.
    # Ground-truth can come either from a formula array (library search) or from an explicit
    # integer carbon count column (mass-list CSV mode).
    if true_carbon_count_column in library_hits.columns:
        data = library_hits.with_columns(
            pl.col(true_carbon_count_column).cast(pl.Int64).alias("_true_carbon_count"),
            pl.col(true_sulfur_count_column).cast(pl.Int64).alias("_true_sulfur_count"),
            pl.col(true_chlorine_count_column)
            .cast(pl.Int64)
            .alias("_true_chlorine_count"),
            pl.col(true_bromine_count_column)
            .cast(pl.Int64)
            .alias("_true_bromine_count"),
        ).filter(
            pl.col("_true_carbon_count").is_not_null(),
            pl.col("_true_carbon_count") > 0,
            pl.col(ms1_mz_column).is_not_null(),
            pl.col(ms1_intensity_column).is_not_null(),
        )
    else:
        data = library_hits.with_columns(
            pl.col(formula_array_column)
            .arr.get(CARBON_INDEX)
            .alias("_true_carbon_count"),
            pl.col(formula_array_column)
            .arr.get(SULFUR_INDEX)
            .alias("_true_sulfur_count"),
            pl.col(formula_array_column)
            .arr.get(CHLORINE_INDEX)
            .alias("_true_chlorine_count"),
            pl.col(formula_array_column)
            .arr.get(BROMINE_INDEX)
            .alias("_true_bromine_count"),
        ).filter(
            pl.col("_true_carbon_count").is_not_null(),
            pl.col("_true_carbon_count") > 0,
            pl.col(ms1_mz_column).is_not_null(),
            pl.col(ms1_intensity_column).is_not_null(),
        )

    assert data.height > 0, (
        "No valid compounds with carbon count and MS1 isotope data found"
    )

    # Why: Build a small set of diagnostic isotopic peaks, mirroring the Rust logic in
    # `deduce_isotopic_pattern_inner`:
    #   1) Find the best precursor peak *in the MS1 spectrum* within ms1 tolerance.
    #   2) For each target element, use the element-table isotope mass shift (mass_diff) to compute
    #      the expected isotopic peak m/z = precursor_ms1_mz + mass_diff.
    #   3) Find the peak in the MS1 spectrum within isotopic tolerance and use its max intensity.
    #
    # NOTE: Expected isotopic peak intensity is approximated as:
    #   expected = n_element * (p_heavy / p_light) * I0
    #
    # This keeps the original inverse-calibration contract but ensures peak matching uses the same
    # m/z anchoring as the Rust implementation (use precursor m/z from MS1, not the "true" precursor).
    isotopic_targets: list[dict[str, object]] = []

    # Carbon: 12C/13C is typically stored as [12C, 13C] with mass_differences[0] ~= +1.003355
    carbon_info = ELEMENTS[ELEMENT_INDEX["C"]]
    c_iso_dist = carbon_info.isotopic_distribution
    assert c_iso_dist is not None, (
        "Carbon isotopic distribution is missing in the element table; "
        "cannot compute expected 13C intensity. Ensure the element table is initialized correctly."
    )
    isotopic_targets.append(
        {
            "label": "13C (M+1)",
            "element_index": CARBON_INDEX,
            "prob_light": float(c_iso_dist.abundances[0]),
            "prob_heavy": float(c_iso_dist.abundances[1]),
            "mass_diff": float(c_iso_dist.mass_differences[0]),
        }
    )

    # Sulfur: use the exact element-table +2 mass shift (34S relative to 32S).
    #
    # Why: `element_table.py` defines an exact sulfur isotopic distribution:
    #   mass_differences=(1.995796,), abundances=(0.9493, 0.0429)
    # so we should use that directly instead of heuristics.
    sulfur_info = ELEMENTS[ELEMENT_INDEX["S"]]
    s_iso_dist = sulfur_info.isotopic_distribution
    assert s_iso_dist is not None, (
        "Sulfur isotopic distribution is missing in the element table; "
        "cannot compute expected sulfur isotopic peak intensity. Ensure the element table is initialized correctly."
    )

    s_mass_diffs = np.asarray(s_iso_dist.mass_differences, dtype=float)
    s_abundances = np.asarray(s_iso_dist.abundances, dtype=float)

    assert s_abundances.size >= 2, (
        f"Sulfur isotopic distribution must contain at least two isotopes (light + heavy). "
        f"Got abundances.size={int(s_abundances.size)}."
    )
    assert s_mass_diffs.size >= 1, (
        f"Sulfur isotopic distribution must contain at least one mass difference entry. "
        f"Got mass_differences.size={int(s_mass_diffs.size)}."
    )
    assert s_abundances.size == s_mass_diffs.size + 1, (
        "Sulfur isotopic distribution must satisfy: len(abundances) == len(mass_differences) + 1. "
        f"Got abundances.size={int(s_abundances.size)}, mass_differences.size={int(s_mass_diffs.size)}."
    )

    # Element table contract for S in this repo: exactly one heavy isotope (34S) with a single mass_diff entry.
    # Fail fast if that ever changes, since downstream interpretations would need review.
    assert s_mass_diffs.size == 1, (
        "Expected sulfur isotopic distribution to provide exactly one mass difference (32S->34S). "
        f"Got mass_differences={s_iso_dist.mass_differences}"
    )

    isotopic_targets.append(
        {
            "label": "34S (M+2)",
            "element_index": SULFUR_INDEX,
            "prob_light": float(s_abundances[0]),
            "prob_heavy": float(s_abundances[1]),
            "mass_diff": float(s_mass_diffs[0]),
        }
    )

    # Chlorine: use the +-2 isotope (37Cl) relative to 35Cl
    chlorine_info = ELEMENTS[ELEMENT_INDEX["Cl"]]
    cl_iso_dist = chlorine_info.isotopic_distribution
    assert cl_iso_dist is not None, (
        "Chlorine isotopic distribution is missing in the element table; "
        "cannot compute expected 37Cl intensity. Ensure the element table is initialized correctly."
    )
    isotopic_targets.append(
        {
            "label": "37Cl (M+2)",
            "element_index": CHLORINE_INDEX,
            "prob_light": float(cl_iso_dist.abundances[0]),
            "prob_heavy": float(cl_iso_dist.abundances[1]),
            "mass_diff": float(cl_iso_dist.mass_differences[0]),
        }
    )

    # Double chlorine: special M+4 peak from two 37Cl substitutions.
    #
    # Why: Mirror the Rust `deduce_isotopic_pattern_inner` "Cl second peak (M+4)" logic by
    # introducing this as an additional pair source. For the mass shift, use 2× the chlorine
    # mass_diff.
    #
    # NOTE: Expected intensity for M+4 due to two 37Cl is not `n * p_heavy/p_light * I0`.
    # The correct expected value uses a combinatorial factor for choosing 2 Cl atoms:
    #    expected(M+4) = C(n, 2) * (p_heavy^2 / p_light^2) * I0
    isotopic_targets.append(
        {
            "label": "37Cl2 (M+4)",
            "element_index": CHLORINE_INDEX,
            "prob_light": float(cl_iso_dist.abundances[0]),
            "prob_heavy": float(cl_iso_dist.abundances[1]),
            "mass_diff": float(cl_iso_dist.mass_differences[0]) * 2.0,
            "secondary_pair_source": "double_chlorine",
        }
    )

    # Bromine: use the +-2 isotope (81Br) relative to 79Br
    bromine_info = ELEMENTS[ELEMENT_INDEX["Br"]]
    br_iso_dist = bromine_info.isotopic_distribution
    assert br_iso_dist is not None, (
        "Bromine isotopic distribution is missing in the element table; "
        "cannot compute expected 81Br intensity. Ensure the element table is initialized correctly."
    )
    isotopic_targets.append(
        {
            "label": "81Br (M+2)",
            "element_index": BROMINE_INDEX,
            "prob_light": float(br_iso_dist.abundances[0]),
            "prob_heavy": float(br_iso_dist.abundances[1]),
            "mass_diff": float(br_iso_dist.mass_differences[0]),
        }
    )

    # Why: Build paired calibration data:
    #   - observed_isotopic_peak_intensity: what you have at inference (input)
    #   - expected_isotopic_peak_intensity: computed from known formula element count + observed precursor (output)
    observed_isotopic_peak_intensities: list[float] = []
    expected_isotopic_peak_intensities: list[float] = []
    per_element_pair_counts: Counter[str] = Counter()

    for row in data.iter_rows(named=True):
        precursor_mz = row[precursor_mz_column]
        ms1_mzs = np.atleast_1d(np.array(row[ms1_mz_column]))
        ms1_intensities = np.atleast_1d(np.array(row[ms1_intensity_column]))

        ms1_absolute_tolerance = (
            max(precursor_mz, mass_accuracy_threshold_da)
            * ms1_mass_tolerance_ppm
            * 1e-6
        )
        isotopic_absolute_tolerance = (
            max(precursor_mz, mass_accuracy_threshold_da)
            * isotopic_mass_tolerance_ppm
            * 1e-6
        )

        # Mirror Rust: find precursor peak in MS1 within ms1 tolerance, then use that MS1 m/z for isotope peaks.
        precursor_idx = np.where(
            np.isclose(ms1_mzs, precursor_mz, atol=ms1_absolute_tolerance, rtol=0.0)
        )[0]
        if len(precursor_idx) == 0:
            continue

        best_precursor_local = precursor_idx[ms1_intensities[precursor_idx].argmax()]
        precursor_ms1_mz = float(ms1_mzs[best_precursor_local])
        precursor_ms1_intensity = float(ms1_intensities[best_precursor_local])

        # Why: Generate calibration pairs across multiple elements and their diagnostic isotopic peaks.
        # We always scale by the monoisotopic peak intensity (I0) observed at the precursor MS1 m/z.
        for target in isotopic_targets:
            element_index = int(target["element_index"])  # type: ignore[arg-type]
            element_label = str(target["label"])  # type: ignore[arg-type]

            if element_index == CARBON_INDEX:
                element_count = int(row["_true_carbon_count"])
            elif element_index == SULFUR_INDEX:
                element_count = int(row["_true_sulfur_count"] or 0)
            elif element_index == CHLORINE_INDEX:
                element_count = int(row["_true_chlorine_count"] or 0)
            elif element_index == BROMINE_INDEX:
                element_count = int(row["_true_bromine_count"] or 0)
            else:
                element_count = 0

            if element_count <= 0:
                continue

            prob_light = float(target["prob_light"])  # type: ignore[arg-type]
            prob_heavy = float(target["prob_heavy"])  # type: ignore[arg-type]
            mass_diff = float(target["mass_diff"])  # type: ignore[arg-type]

            # Special-case: double chlorine M+4 expected intensity uses a combinatorial factor.
            # Why: Two 37Cl substitutions contribute at M+4 (approximately 2× mass shift) with:
            #   expected = C(n, 2) * (p_heavy^2 / p_light^2) * I0
            if (
                str(target.get("secondary_pair_source", "")) == "double_chlorine"
                and element_index == CHLORINE_INDEX
            ):
                if element_count < 2:
                    continue

                expected_isotopic_peak_intensity = (
                    (element_count * (element_count - 1) / 2.0)
                    * ((prob_heavy**2) / (prob_light**2))
                    * precursor_ms1_intensity
                )
            else:
                expected_isotopic_peak_intensity = (
                    element_count * prob_heavy * precursor_ms1_intensity
                ) / prob_light

            # Mirror Rust: compute target isotopic peak m/z from precursor_ms1_mz and lookup within isotopic tolerance.
            isotopic_peak_mz = precursor_ms1_mz + mass_diff
            isotopic_peaks_idx = np.where(
                np.isclose(
                    ms1_mzs,
                    isotopic_peak_mz,
                    atol=isotopic_absolute_tolerance,
                    rtol=0.0,
                )
            )[0]
            observed_isotopic_peak_intensity = (
                float(ms1_intensities[isotopic_peaks_idx].max())
                if len(isotopic_peaks_idx) > 0
                else 0.0
            )

            # Why: We need a strictly positive observed intensity for log-space calibration.
            # If observed is 0 (peak missing), this procedure cannot bound expected from observed alone
            # without explicitly modeling censoring/detection; we skip these points to keep the contract exact.
            if observed_isotopic_peak_intensity <= 0.0:
                continue

            # Why: Filter on the observed isotopic-peak intensity (input-side), so the fitted band can
            # expand at low intensity rather than excluding those points by expected-intensity filtering.
            if observed_isotopic_peak_intensity < minimum_isotopic_peak_intensity:
                continue

            observed_isotopic_peak_intensities.append(observed_isotopic_peak_intensity)
            expected_isotopic_peak_intensities.append(expected_isotopic_peak_intensity)
            per_element_pair_counts[element_label] += 1

    assert len(observed_isotopic_peak_intensities) > 0, (
        "No valid (observed_isotopic_peak, expected_isotopic_peak) calibration pairs could be formed. "
        "Check MS1 isotope data validity and/or lower `minimum_isotopic_peak_intensity`."
    )

    observed_arr = np.asarray(observed_isotopic_peak_intensities, dtype=float)
    expected_arr = np.asarray(expected_isotopic_peak_intensities, dtype=float)

    # -----------------------------
    # Fit inverse calibration: expected = alpha * (observed + offset) ** beta
    # Fit in log-space to emphasize multiplicative structure and preserve positivity.
    # -----------------------------
    def expected_from_observed_model(
        observed: np.ndarray, alpha: float, beta: float, offset: float
    ) -> np.ndarray:
        return alpha * np.power(observed + offset, beta)

    # Why: offset must keep (observed + offset) positive; we optimize in unconstrained space by squaring.
    def log_expected_from_observed_model_offset_sq(
        observed: np.ndarray, log_alpha: float, beta: float, offset_sq: float
    ) -> np.ndarray:
        offset = offset_sq**2
        return log_alpha + beta * np.log(observed + offset)

    log_expected = np.log(expected_arr)

    # Initial guesses: assume roughly proportional with slight inefficiency (beta ~ 1)
    log_alpha0 = float(np.median(np.log(expected_arr) - np.log(observed_arr)))
    beta0 = 1.0
    offset_sq0 = 0.0

    popt, _ = curve_fit(
        log_expected_from_observed_model_offset_sq,
        observed_arr,
        log_expected,
        p0=[log_alpha0, beta0, offset_sq0],
        maxfev=20000,
    )
    log_alpha_hat, beta_hat, offset_sq_hat = popt
    alpha_hat = float(np.exp(log_alpha_hat))
    offset_hat = float(offset_sq_hat**2)

    expected_hat = expected_from_observed_model(
        observed_arr, alpha_hat, beta_hat, offset_hat
    )
    assert np.all(expected_hat > 0.0), (
        "Expected_hat must be strictly positive for log-space residuals."
    )

    # -----------------------------
    # Residuals in log-space: r = log(expected_true) - log(expected_hat)
    # -----------------------------
    log_residual = np.log(expected_arr) - np.log(expected_hat)

    # -----------------------------
    # (1) Asymmetric constant bounds with minimal total width.
    # Find (wL_const, wU_const) minimizing wL_const + wU_const s.t. coverage >= target.
    #
    # Coverage condition:  -wL_const <= r <= wU_const
    # Equivalent to choose a window of (wL+wU) on the residual axis covering target fraction.
    # We compute the tightest such window by sliding over sorted residuals.
    # -----------------------------
    sorted_r = np.sort(log_residual)
    n = int(sorted_r.size)
    assert n >= 5, f"Need at least 5 calibration pairs, got {n}."

    k = int(np.ceil(target_success_rate * n))
    k = max(1, min(k, n))

    best_width = float("inf")
    best_low = float(sorted_r[0])
    best_high = float(sorted_r[-1])

    # Why: smallest interval that contains k points in sorted residuals.
    for i in range(0, n - k + 1):
        low = float(sorted_r[i])
        high = float(sorted_r[i + k - 1])
        width = high - low
        if width < best_width:
            best_width = width
            best_low = low
            best_high = high

    # Convert window [best_low, best_high] into asymmetric widths around 0.
    # We enforce bounds of the form  -wL <= r <= wU, with minimal (wL+wU) => wL=-best_low if best_low<0 else 0, wU=best_high if best_high>0 else 0.
    wL_const = max(0.0, -best_low)
    wU_const = max(0.0, best_high)

    coverage_const = float(
        np.mean((log_residual >= -wL_const) & (log_residual <= wU_const))
    )

    # -----------------------------
    # (2) Heteroscedastic widths: fit per-bin lower/upper quantiles of residuals vs observed intensity.
    # We bin in log(observed) and estimate:
    #   q_low(I):  lower quantile residual  (typically negative)
    #   q_high(I): upper quantile residual  (typically positive)
    #
    # Then define widths:
    #   wL(I) = max(0, -q_low(I)),  wU(I) = max(0, q_high(I))
    #
    # We fit monotone-ish power-law in intensity space:
    #   w(I) = a * (I + offset) ** b
    # with a>=0 enforced by a_sq**2.
    # -----------------------------
    n_samples = int(observed_arr.size)
    min_samples_per_bin = 8
    n_bins = min(25, max(3, n_samples // min_samples_per_bin))

    observed_min = max(float(np.min(observed_arr)), 1.0)
    observed_max = float(np.max(observed_arr))
    bins = np.logspace(np.log10(observed_min), np.log10(observed_max), n_bins + 1)

    bin_centers: list[float] = []
    bin_wL: list[float] = []
    bin_wU: list[float] = []

    # Central quantile window around the median for target coverage:
    # q_low = (1-target)/2, q_high = 1 - (1-target)/2
    q_low = (1.0 - target_success_rate) / 2.0
    q_high = 1.0 - q_low

    for i in range(n_bins):
        mask = (observed_arr >= bins[i]) & (observed_arr < bins[i + 1])
        if int(np.sum(mask)) < min_samples_per_bin:
            continue

        center = float(np.median(observed_arr[mask]))
        r_bin = log_residual[mask]

        low_q = float(np.quantile(r_bin, q_low))
        high_q = float(np.quantile(r_bin, q_high))

        bin_centers.append(center)
        bin_wL.append(max(0.0, -low_q))
        bin_wU.append(max(0.0, high_q))

    if len(bin_centers) < 3:
        # Fallback to constant widths if there isn't enough data for heteroscedastic fitting
        hetero_lower_a = 0.0
        hetero_lower_b = 0.0
        hetero_upper_a = 0.0
        hetero_upper_b = 0.0

        def wL_hetero(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, wL_const, dtype=float)

        def wU_hetero(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, wU_const, dtype=float)

        coverage_hetero = coverage_const
    else:
        centers = np.asarray(bin_centers, dtype=float)
        wL_data = np.asarray(bin_wL, dtype=float)
        wU_data = np.asarray(bin_wU, dtype=float)

        def width_power_offset_sq(
            observed: np.ndarray, a_sq: float, b: float
        ) -> np.ndarray:
            a = a_sq**2
            return a * np.power(observed + offset_hat, b)

        # Initial guesses: widths decrease with intensity -> b likely negative
        p0_L = [np.sqrt(max(float(np.median(wL_data)), 1e-6)), -0.25]
        p0_U = [np.sqrt(max(float(np.median(wU_data)), 1e-6)), -0.25]

        popt_L, _ = curve_fit(
            width_power_offset_sq, centers, wL_data, p0=p0_L, maxfev=20000
        )
        popt_U, _ = curve_fit(
            width_power_offset_sq, centers, wU_data, p0=p0_U, maxfev=20000
        )

        aL_sq_hat, bL_hat = popt_L
        aU_sq_hat, bU_hat = popt_U

        hetero_lower_a = float(aL_sq_hat**2)
        hetero_lower_b = float(bL_hat)
        hetero_upper_a = float(aU_sq_hat**2)
        hetero_upper_b = float(bU_hat)

        def wL_hetero(x: np.ndarray) -> np.ndarray:
            return hetero_lower_a * np.power(x + offset_hat, hetero_lower_b)

        def wU_hetero(x: np.ndarray) -> np.ndarray:
            return hetero_upper_a * np.power(x + offset_hat, hetero_upper_b)

        wL_vals = wL_hetero(observed_arr)
        wU_vals = wU_hetero(observed_arr)
        coverage_hetero = float(
            np.mean((log_residual >= -wL_vals) & (log_residual <= wU_vals))
        )

        # If the heteroscedastic fit under-covers, widen both sides by a single global factor
        # (keeps the functional form but targets the same success-rate principle as the constant window).
        if coverage_hetero < target_success_rate:
            scale_min = 1.0
            scale_max = 128.0  # wide enough to guarantee convergence in practical cases
            best_scale = None

            # Why: binary search for the minimal multiplicative widening that achieves target coverage.
            for _ in range(30):
                scale_mid = 0.5 * (scale_min + scale_max)
                wL_scaled = scale_mid * wL_vals
                wU_scaled = scale_mid * wU_vals
                cov_mid = float(
                    np.mean((log_residual >= -wL_scaled) & (log_residual <= wU_scaled))
                )
                if cov_mid >= target_success_rate:
                    best_scale = scale_mid
                    scale_max = scale_mid
                else:
                    scale_min = scale_mid

            if best_scale is None:
                # Fail fast instead of silently returning an under-covering heteroscedastic model.
                # The constant model is guaranteed to hit the target by construction (up to rounding k).
                best_scale = scale_max

            hetero_scale = float(best_scale)

            def wL_hetero(x: np.ndarray) -> np.ndarray:
                return (
                    hetero_scale
                    * hetero_lower_a
                    * np.power(x + offset_hat, hetero_lower_b)
                )

            def wU_hetero(x: np.ndarray) -> np.ndarray:
                return (
                    hetero_scale
                    * hetero_upper_a
                    * np.power(x + offset_hat, hetero_upper_b)
                )

            wL_vals = wL_hetero(observed_arr)
            wU_vals = wU_hetero(observed_arr)
            coverage_hetero = float(
                np.mean((log_residual >= -wL_vals) & (log_residual <= wU_vals))
            )

    # -----------------------------
    # Visualization: expected vs observed with multiple fits/bounds
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        observed_arr,
        expected_arr,
        alpha=0.35,
        s=12,
        color="gray",
        label="Calibration pairs (observed isotopic peak, expected isotopic peak)",
    )

    x_min = max(float(np.min(observed_arr)), 1.0)
    x_max = float(np.max(observed_arr))
    x_range = np.logspace(np.log10(x_min), np.log10(x_max), 250)

    y_center = expected_from_observed_model(x_range, alpha_hat, beta_hat, offset_hat)

    # Asymmetric constant bounds
    y_lower_const = y_center * np.exp(-wL_const)
    y_upper_const = y_center * np.exp(+wU_const)

    # Heteroscedastic bounds
    wL_x = wL_hetero(x_range)
    wU_x = wU_hetero(x_range)
    y_lower_hetero = y_center * np.exp(-wL_x)
    y_upper_hetero = y_center * np.exp(+wU_x)

    ax.plot(
        x_range,
        y_center,
        color="black",
        linewidth=2.2,
        label="Center: expected_hat(observed)",
    )

    ax.plot(
        x_range,
        y_lower_const,
        color="blue",
        linestyle="--",
        linewidth=1.8,
        label=f"Const bounds (asym), cov={coverage_const:.2%}",
    )
    ax.plot(x_range, y_upper_const, color="blue", linestyle="--", linewidth=1.8)

    ax.plot(
        x_range,
        y_lower_hetero,
        color="green",
        linestyle="--",
        linewidth=1.8,
        label=f"Hetero bounds (asym), cov={coverage_hetero:.2%}",
    )
    ax.plot(x_range, y_upper_hetero, color="green", linestyle="--", linewidth=1.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed isotopic peak intensity (input)")
    ax.set_ylabel("Expected isotopic peak intensity (output, from formula)")
    ax.set_title(
        "Inverse isotopic calibration: expected isotopic-peak bounds from observed isotopic peak\n"
        f"Target coverage={target_success_rate:.1%} | const={coverage_const:.1%} | hetero={coverage_hetero:.1%}"
    )
    ax.legend()
    plt.tight_layout()

    plot_path = output_path.with_suffix(".png")
    plt.savefig(plot_path)
    plt.close()

    with open(output_path, "a") as f:
        f.write("\nInverse calibration for deduce_isotopic_pattern:\n")
        f.write("  expected_hat(observed) = alpha * (observed + offset) ** beta\n")
        f.write(
            "  NOTE: Calibration pairs include isotopic peaks for 13C (M+1), 34S (M+2), 37Cl (M+2), 37Cl2 (M+4), 81Br (M+2).\n"
        )
        f.write("  Calibration pair counts by isotopic peak:\n")
        f.write(
            f"    - 13C (M+1):  {int(per_element_pair_counts.get('13C (M+1)', 0))}\n"
        )
        f.write(
            f"    - 34S (M+2):  {int(per_element_pair_counts.get('34S (M+2)', 0))}\n"
        )
        f.write(
            f"    - 37Cl (M+2): {int(per_element_pair_counts.get('37Cl (M+2)', 0))}\n"
        )
        f.write(
            f"    - 37Cl2 (M+4): {int(per_element_pair_counts.get('37Cl2 (M+4)', 0))}\n"
        )
        f.write(
            f"    - 81Br (M+2): {int(per_element_pair_counts.get('81Br (M+2)', 0))}\n"
        )
        f.write(f"  alpha: {alpha_hat:.6g}\n")
        f.write(f"  beta: {beta_hat:.6g}\n")
        f.write(f"  offset: {offset_hat:.6g}\n")

        f.write("\nAsymmetric constant bounds (tightest window on residuals):\n")
        f.write("  expected_lower = expected_hat * exp(-wL_const)\n")
        f.write("  expected_upper = expected_hat * exp(+wU_const)\n")
        f.write(f"  wL_const: {wL_const:.6g}\n")
        f.write(f"  wU_const: {wU_const:.6g}\n")
        f.write(f"  coverage_const: {coverage_const:.2%}\n")

        f.write(
            "\nAsymmetric heteroscedastic bounds (fit to per-bin quantile residuals):\n"
        )
        f.write("  expected_lower = expected_hat * exp(-wL(observed))\n")
        f.write("  expected_upper = expected_hat * exp(+wU(observed))\n")
        f.write("  wL(observed) = aL * (observed + offset) ** bL\n")
        f.write("  wU(observed) = aU * (observed + offset) ** bU\n")
        f.write(f"  aL: {hetero_lower_a:.6g}\n")
        f.write(f"  bL: {hetero_lower_b:.6g}\n")
        f.write(f"  aU: {hetero_upper_a:.6g}\n")
        f.write(f"  bU: {hetero_upper_b:.6g}\n")
        f.write(f"  coverage_hetero: {coverage_hetero:.2%}\n")

        f.write(f"  Plot saved to: {plot_path}\n")

    # Prefer heteroscedastic coverage if it meets target; we still return both parameters for inspection.
    chosen_success_rate = (
        coverage_hetero if coverage_hetero >= target_success_rate else coverage_const
    )

    return IsotopicToleranceModel(
        alpha=alpha_hat,
        beta=beta_hat,
        offset=offset_hat,
        band_lower_width=wL_const,
        band_upper_width=wU_const,
        hetero_lower_a=hetero_lower_a,
        hetero_lower_b=hetero_lower_b,
        hetero_upper_a=hetero_upper_a,
        hetero_upper_b=hetero_upper_b,
        success_rate=chosen_success_rate,
    )


def main(config: Config):
    assert config.chromatogram_dir is not None, (
        "config.chromatogram_dir must be set (defaults are intentionally empty). "
        "Set it in the __main__ block."
    )

    assert (
        config.use_library_search_for_ground_truth
        or config.use_mass_list_for_ground_truth
    ), (
        "At least one ground-truth source must be enabled: "
        "use_library_search_for_ground_truth and/or use_mass_list_for_ground_truth."
    )

    if config.use_library_search_for_ground_truth:
        assert len(config.spectral_library_paths) > 0, (
            "use_library_search_for_ground_truth=True but no spectral_library_paths were provided."
        )

    if config.use_mass_list_for_ground_truth:
        assert config.ground_truth_csv_path is not None, (
            "use_mass_list_for_ground_truth=True but ground_truth_csv_path is None."
        )
        assert config.ground_truth_csv_path.exists(), (
            f"ground_truth_csv_path does not exist: {config.ground_truth_csv_path}. "
            "Provide a CSV file with columns 'Molecular Formula' and 'Monoisotopic Mass'."
        )

    # Output file path
    output_file_path = config.output_file

    # Ensure directory exists or use current dir if not in experiments
    if not output_file_path.parent.exists():
        output_file_path = Path(output_file_path.name)

    # Clear previous output
    if output_file_path.exists():
        output_file_path.unlink()

    chromatogram_paths = list(config.chromatogram_dir.glob(config.chromatogram_glob))

    # --- Execution ---
    with open(output_file_path, "w") as f:
        f.write("Running optimization...\n")
        f.write(f"Chromatograms: {[str(p) for p in chromatogram_paths]}\n")
        f.write(f"Config: {config}\n")

    # 1. Load Chromatograms
    chromatogram_lfs: list[pl.LazyFrame] = []
    for path in chromatogram_paths:
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue

        lf = (
            get_chromatogram(str(path))
            .filter(
                pl.col("ms1_isotopes_m/z").is_not_null(),
                pl.col("msms_m/z").is_not_null(),
                pl.col("Isotope").eq(0),  # only monoisotopic peaks
            )
            .lazy()
            .with_columns(
                nominal_mass=pl.col("Precursor_mz_MSDIAL").round(0).cast(pl.Int64),
                # Track source file for potential debugging, though not strictly required by downstream
                source_file=pl.lit(path.name),
            )
        )
        chromatogram_lfs.append(lf)

    if not chromatogram_lfs:
        raise ValueError("No valid chromatogram files found.")

    chromatogram_lf = pl.concat(chromatogram_lfs)

    calibration_hits_lfs: list[pl.LazyFrame] = []

    # Normalize both ground-truth sources to a shared, *transformed* schema before concatenation.
    # Why: Polars `concat` requires identical schemas, and library-search vs mass-list have different column sets.
    transformed_calibration_schema_columns: list[str] = [
        # identity / grouping
        "calibration_peak_id",
        "calibration_source_file",
        # masses + intensities used downstream
        "calibration_precursor_mz",
        "calibration_height",
        "calibration_ms1_isotopes_mz",
        "calibration_ms1_isotopes_intensity",
        # optional ground-truth supervision (may be null for library-search rows)
        "calibration_true_carbon_count",
        "calibration_true_sulfur_count",
        "calibration_true_chlorine_count",
        "calibration_true_bromine_count",
    ]

    # -------------------------
    # Ground-truth source (A): Library search (current behavior, with formula-consistency filter)
    # -------------------------
    if config.use_library_search_for_ground_truth:
        spectral_lib = (
            pl.union([pl.scan_parquet(p) for p in config.spectral_library_paths])
            .filter(pl.col("ion_mode").eq("P"))
            .with_columns(nominal_mass=pl.col("precursor_mz").round(0).cast(pl.Int64))
        )

        suspects_lf = (
            chromatogram_lf.join(
                other=spectral_lib,
                on="nominal_mass",
                how="inner",
            )
            .filter(
                pl.col("Precursor_mz_MSDIAL").is_close(
                    pl.col("precursor_mz"),
                    rel_tol=config.precursor_tolerance_ppm * 1e-6,
                )
            )
            # --- Clean and Calculate Info Score (BEFORE DotProd) ---
            .with_columns(
                pl.struct(
                    pl.col("precursor_formula_array").alias("precursor_formula"),
                    pl.col("msms_m/z").alias("mz"),
                    pl.col("msms_intensity").alias("intensities"),
                )
                .mass_decomposition.clean_and_normalize_spectrum(
                    raw_fragment_tolerance_ppm=config.fragment_cleaning_tolerance_ppm,
                    normalized_fragment_tolerance_ppm=config.fragment_cleaning_tolerance_ppm,
                    min_dbe=config.min_dbe,
                    max_dbe=config.max_dbe,
                    dbe_mode="half_integer",
                    water_absorption=True,
                )
                .alias("cleaned_spectrum_struct")
            )
        )

        suspects_lf = (
            suspects_lf.with_columns(
                # Unpack cleaned spectrum for potential use (though we use raw for dotprod below)
                pl.col("cleaned_spectrum_struct")
                .struct.field("normalized_masses")
                .alias("cleaned_mz"),
                pl.col("cleaned_spectrum_struct")
                .struct.field("intensities")
                .alias("cleaned_intensity"),
                # Calculate score
                pl.struct(
                    pl.col("precursor_formula_array").alias("precursor_formula"),
                    pl.col("cleaned_spectrum_struct")
                    .struct.field("formulas")
                    .alias("fragment_formulas"),
                )
                .spectral_info.spectral_info_score(
                    distance_metric="l2", ignore_hydrogens=True
                )
                .alias("spectral_info_score"),
            )
            .filter(pl.col("spectral_info_score") > config.info_score_threshold)
            # --- Search (Dot Product) ---
            .with_columns(
                pl.struct(
                    pl.col("Precursor_mz_MSDIAL").alias("precursor_mz1"),
                    pl.col("msms_m/z").alias("mz1"),
                    pl.col("msms_intensity").alias("intensities1"),
                    pl.col("precursor_mz").alias("precursor_mz2"),
                    pl.col("cleaned_normalized_mz").alias("mz2"),
                    pl.col("cleaned_normalized_intensity").alias("intensities2"),
                )
                .spectral_similarity.dotprod_similarity(
                    ms2_tolerance_in_ppm=config.ms2_tolerance_ppm, ignore_precursor=True
                )
                .alias("dotprod_similarity")
            )
            .filter(pl.col("dotprod_similarity") > config.dot_product_threshold)
            # Keep only MS2 where *all* library matches point to the same formula.
            .with_columns(
                pl.col("precursor_formula_array")
                .n_unique()
                .over("Peak ID", "source_file")
                .alias("_n_unique_formulas_for_ms2")
            )
            .filter(pl.col("_n_unique_formulas_for_ms2") == 1)
            # Now that formula consistency is guaranteed, keep only the best dot-product match per MS2.
            .filter(
                pl.col("dotprod_similarity").eq(
                    pl.col("dotprod_similarity").max().over("Peak ID", "source_file")
                )
            )
        )

        suspects_calibration_lf = suspects_lf.select(
            pl.col("Peak ID").alias("calibration_peak_id"),
            pl.col("source_file").alias("calibration_source_file"),
            pl.col("Precursor_mz_MSDIAL").alias("calibration_precursor_mz"),
            pl.col("Height").alias("calibration_height"),
            pl.col("ms1_isotopes_m/z").alias("calibration_ms1_isotopes_mz"),
            pl.col("ms1_isotopes_intensity").alias(
                "calibration_ms1_isotopes_intensity"
            ),
            # This ground-truth signal is not produced by the library-search path; keep explicit nulls.
            pl.lit(None, dtype=pl.Int64).alias("calibration_true_carbon_count"),
            pl.lit(None, dtype=pl.Int64).alias("calibration_true_sulfur_count"),
            pl.lit(None, dtype=pl.Int64).alias("calibration_true_chlorine_count"),
            pl.lit(None, dtype=pl.Int64).alias("calibration_true_bromine_count"),
        )

        # Fail fast: enforce identical schema for safe concatenation.
        suspects_calibration_lf = suspects_calibration_lf.select(
            [pl.col(c) for c in transformed_calibration_schema_columns]
        )

        calibration_hits_lfs.append(suspects_calibration_lf)

    # -------------------------
    # Ground-truth source (B): Accurate mass + formula list from CSV
    #
    # Contract:
    # - Read the CSV file and use ONLY these columns:
    #     - "Molecular Formula"
    #     - "Monoisotopic Mass"
    #   Extra columns are allowed and ignored.
    # - For each list item and each file, accept it only if it matches exactly one feature (Peak ID)
    #   within tolerance in that file. If it matches multiple peaks in the same file => drop that item for that file.
    # - Extract carbon count from formula string: digits after "C" (e.g., C6H12O6 -> 6).
    #   Implemented with Polars regex (no Python regex).
    # -------------------------
    if config.use_mass_list_for_ground_truth:
        assert config.ground_truth_csv_path is not None, (
            "use_mass_list_for_ground_truth=True but ground_truth_csv_path is None."
        )
        assert config.ground_truth_csv_path.exists(), (
            f"ground_truth_csv_path does not exist: {config.ground_truth_csv_path}. "
            "Provide a CSV file with columns 'Molecular Formula' and 'Monoisotopic Mass'."
        )

        ground_truth_df = pl.read_csv(config.ground_truth_csv_path).select(
            pl.col("Monoisotopic Mass").alias("ground_truth_precursor_mz"),
            pl.col("Molecular Formula").alias("ground_truth_formula_str"),
        )

        ground_truth_df = (
            ground_truth_df.with_columns(
                pl.col("ground_truth_precursor_mz").cast(pl.Float64),
                pl.col("ground_truth_formula_str").cast(pl.Utf8),
            )
            # Why: centralize parsing rules (missing element => 0, missing number => 1) using the
            # repo's canonical element-table regexes via `formula_to_array`.
            .pipe(
                formula_to_array,
                input_col_name="ground_truth_formula_str",
                output_col_name="_ground_truth_formula_array",
            )
            .with_columns(
                pl.col("_ground_truth_formula_array")
                .arr.get(CARBON_INDEX)
                .cast(pl.Int64)
                .alias("_ground_truth_carbon_count"),
                pl.col("_ground_truth_formula_array")
                .arr.get(SULFUR_INDEX)
                .cast(pl.Int64)
                .alias("_ground_truth_sulfur_count"),
                pl.col("_ground_truth_formula_array")
                .arr.get(CHLORINE_INDEX)
                .cast(pl.Int64)
                .alias("_ground_truth_chlorine_count"),
                pl.col("_ground_truth_formula_array")
                .arr.get(BROMINE_INDEX)
                .cast(pl.Int64)
                .alias("_ground_truth_bromine_count"),
            )
        )

        assert ground_truth_df.height > 0, (
            f"Ground-truth CSV is empty after selecting required columns: {config.ground_truth_csv_path}. "
            "Ensure it has rows and the columns 'Molecular Formula' and 'Monoisotopic Mass'."
        )

        # Fail fast if any formula is missing carbon count (after applying the explicit rules above)
        assert (
            ground_truth_df.select(
                pl.col("_ground_truth_carbon_count").is_null().any()
            ).item()
            is False
        ), (
            "Ground-truth CSV contains formula(s) that could not be parsed for carbon count. "
            "Ensure formulas are valid strings like 'C6H12O6' or 'CH4'."
        )

        # Cross join by file so we can enforce uniqueness within each file
        files_df = chromatogram_lf.select("source_file").unique()
        ground_truth_per_file_lf = files_df.join(ground_truth_df.lazy(), how="cross")

        # Join to features inside each file using an absolute tolerance derived from ppm
        # Note: We use is_close with rel_tol in ppm terms because target masses differ per row.
        mass_list_matches_lf = chromatogram_lf.join(
            ground_truth_per_file_lf,
            on="source_file",
            how="inner",
        ).filter(
            pl.col("Precursor_mz_MSDIAL").is_close(
                pl.col("ground_truth_precursor_mz"),
                rel_tol=config.ground_truth_ms1_tolerance_ppm * 1e-6,
            )
        )

        # Enforce: single matching Peak ID per (source_file, ground_truth item)
        mass_list_unique_lf = (
            mass_list_matches_lf.with_columns(
                pl.col("Peak ID")
                .n_unique()
                .over(
                    "source_file",
                    "ground_truth_precursor_mz",
                    "ground_truth_formula_str",
                )
                .alias("_n_unique_peak_ids_for_ground_truth")
            )
            .filter(pl.col("_n_unique_peak_ids_for_ground_truth") == 1)
            # Deduplicate to exactly one row per ground-truth item per file (keep highest Height if duplicates exist)
            .sort(
                [
                    "source_file",
                    "ground_truth_precursor_mz",
                    "ground_truth_formula_str",
                    "Height",
                ],
                descending=[False, False, False, True],
            )
            .unique(
                subset=[
                    "source_file",
                    "ground_truth_precursor_mz",
                    "ground_truth_formula_str",
                ],
                keep="first",
            )
        )

        mass_list_calibration_lf = mass_list_unique_lf.select(
            pl.col("Peak ID").alias("calibration_peak_id"),
            pl.col("source_file").alias("calibration_source_file"),
            pl.col("Precursor_mz_MSDIAL").alias("calibration_precursor_mz"),
            pl.col("Height").alias("calibration_height"),
            pl.col("ms1_isotopes_m/z").alias("calibration_ms1_isotopes_mz"),
            pl.col("ms1_isotopes_intensity").alias(
                "calibration_ms1_isotopes_intensity"
            ),
            pl.col("_ground_truth_carbon_count").alias("calibration_true_carbon_count"),
            pl.col("_ground_truth_sulfur_count").alias("calibration_true_sulfur_count"),
            pl.col("_ground_truth_chlorine_count").alias(
                "calibration_true_chlorine_count"
            ),
            pl.col("_ground_truth_bromine_count").alias(
                "calibration_true_bromine_count"
            ),
        )

        # Fail fast: enforce identical schema for safe concatenation.
        mass_list_calibration_lf = mass_list_calibration_lf.select(
            [pl.col(c) for c in transformed_calibration_schema_columns]
        )

        calibration_hits_lfs.append(mass_list_calibration_lf)

    assert len(calibration_hits_lfs) > 0, (
        "No calibration sources produced any LazyFrames."
    )

    calibration_hits = pl.concat(calibration_hits_lfs, how="vertical").collect(
        engine="streaming"
    )

    with open(output_file_path, "a") as f:
        f.write(f"Calibration hits total: {calibration_hits.height}\n")
        f.write(
            f"  - from library search: {int(config.use_library_search_for_ground_truth)}\n"
        )
        f.write(f"  - from mass list: {int(config.use_mass_list_for_ground_truth)}\n")

    # Fail fast if a requested ground-truth source produced zero usable matches.
    # Why: If mass-list matching yields zero features, downstream calibration will either be empty or
    # missing the required carbon-count signal, and it's better to stop with a clear action item.
    if config.use_mass_list_for_ground_truth:
        assert config.ground_truth_csv_path is not None, (
            "use_mass_list_for_ground_truth=True but ground_truth_csv_path is None."
        )
        if calibration_hits.height == 0:
            with open(output_file_path, "a") as f:
                f.write(
                    "No calibration hits were produced.\n"
                    "Mass-list mode was enabled, but it appears to have produced zero matches.\n"
                    "Action items:\n"
                    "  - Verify the CSV has columns 'Molecular Formula' and 'Monoisotopic Mass'.\n"
                    "  - Verify the masses are comparable to 'Precursor_mz_MSDIAL' (ion/adduct assumptions).\n"
                    "  - Increase ground_truth_ms1_tolerance_ppm if appropriate.\n"
                )
            return

    # 5. Optimization Data Prep (for validation/logging)
    #
    # Note: As of the schema-normalization step above, `calibration_hits` is already in the
    # transformed shared schema (prefixed with `calibration_...`). This avoids concat schema
    # conflicts when both ground-truth sources are enabled.
    optimization_data = calibration_hits.select(
        [
            "calibration_peak_id",
            "calibration_source_file",
            "calibration_precursor_mz",
            "calibration_height",
            "calibration_ms1_isotopes_mz",
            "calibration_ms1_isotopes_intensity",
            "calibration_true_carbon_count",
            "calibration_true_sulfur_count",
            "calibration_true_chlorine_count",
            "calibration_true_bromine_count",
        ]
    ).rename(
        {
            # Rename back to the canonical names expected by `fit_isotopic_tolerance_parameters`
            # so downstream logic stays unchanged.
            "calibration_peak_id": "Peak ID",
            "calibration_source_file": "source_file",
            "calibration_precursor_mz": "Precursor_mz_MSDIAL",
            "calibration_height": "Height",
            "calibration_ms1_isotopes_mz": "ms1_isotopes_m/z",
            "calibration_ms1_isotopes_intensity": "ms1_isotopes_intensity",
            "calibration_true_carbon_count": "true_carbon_count",
            "calibration_true_sulfur_count": "true_sulfur_count",
            "calibration_true_chlorine_count": "true_chlorine_count",
            "calibration_true_bromine_count": "true_bromine_count",
        }
    )

    optimization_data = optimization_data.filter(
        pl.col("true_carbon_count").is_not_null(),
        pl.col("true_carbon_count") > 0,
        pl.col("ms1_isotopes_m/z").is_not_null(),
        pl.col("ms1_isotopes_intensity").is_not_null(),
    )

    with open(output_file_path, "a") as f:
        f.write(f"Optimization dataset: {optimization_data.height} compounds\n")

    # 6. Run Optimization
    _ = fit_isotopic_tolerance_parameters(
        library_hits=optimization_data,
        output_path=output_file_path,
        ms1_mass_tolerance_ppm=config.ms1_mass_tolerance_ppm,
        isotopic_mass_tolerance_ppm=config.isotopic_mass_tolerance_ppm,
        minimum_isotopic_peak_intensity=config.minimum_isotopic_peak_intensity,
        target_success_rate=config.target_success_rate,
        mass_accuracy_threshold_da=config.mass_accuracy_threshold_da,
    )


if __name__ == "__main__":
    # Defaults intentionally live here (not in Config) so running this file does not silently
    # depend on a specific workstation path unless you opt into it.
    default_config = Config(
        chromatogram_dir=Path(
            "/home/analytit_admin/Data/raw_data/iibr_data/251224_spiked_plasma/"
        ),
        spectral_library_paths=(
            Path(
                "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"
            ),
            Path(
                "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet"
            ),
        ),
        # Enable sources as desired
        use_library_search_for_ground_truth=True,
        use_mass_list_for_ground_truth=True,
        # Ground-truth CSV (optional). Must contain columns:
        #   - "Molecular Formula"
        #   - "Monoisotopic Mass"
        ground_truth_csv_path=Path(
            "/home/analytit_admin/Data/raw_data/iibr_data/251224_spiked_plasma/compounds.csv"
        ),
        ground_truth_ms1_tolerance_ppm=2.0,
        target_success_rate=0.95,
    )
    main(default_config)
