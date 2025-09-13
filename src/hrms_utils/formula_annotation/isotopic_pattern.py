import numpy as np
import re
from dataclasses import dataclass
import polars as pl
from typing import Dict, TypeVar, overload
from .element_table import ELEMENTS, ELEMENT_SYMBOLS, DEFAULT_MIN_BOUND, DEFAULT_MAX_BOUND
from numba import njit, jit
NITROGEN_SEPARATION_RESOLUTION=1e5

@dataclass
class isotopic_pattern_config:
    mass_tolerance : float
    ms1_resolution:float
    minimum_intensity : float=5e5
    max_intensity_ratio : float=1.7
    def to_dict(self) -> dict:
        return {
            'mass_tolerance': self.mass_tolerance,
            'ms1_resolution': self.ms1_resolution,
            'minimum_intensity': self.minimum_intensity,
            'max_intensity_ratio': self.max_intensity_ratio
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'isotopic_pattern_config':
        mass_tolerance = config_dict.get('mass_tolerance')
        ms1_resolution = config_dict.get('ms1_resolution')
        if mass_tolerance is None:
            raise ValueError("mass_tolerance is required and cannot be None.")
        if ms1_resolution is None:
            raise ValueError("ms1_resolution is required and cannot be None.")
        kwargs = {}
        for field_ in cls.__dataclass_fields__:
            if field_ in config_dict and config_dict[field_] is not None:
                kwargs[field_] = config_dict[field_]
        kwargs['mass_tolerance'] = mass_tolerance
        kwargs['ms1_resolution'] = ms1_resolution
        return cls(**kwargs)
MASS_ACCURACY_PPM_TO_DA_THRESHOLD = 200.0

# Get isotopic pattern info from ELEMENTS
def get_isotopic_pattern_dict():
    pattern_dict = {}
    for i, elem in enumerate(ELEMENTS):
        if elem.isotopic_distribution is not None:
            iso = elem.isotopic_distribution
            pattern_dict[elem.symbol] = {
                "mass_difference": iso.mass_differences[0],
                "zero_isotope_probability": iso.abundances[0],
                "first_isotope_probability": iso.abundances[1],
                "index": i
            }
    return pattern_dict

isotopic_pattern_dict = get_isotopic_pattern_dict()


def fits_isotopic_pattern_batch(mzs_batch, intensities_batch, formulas, precursor_mzs, config):
    """
    mzs_batch: list of arrays, each shape (n_peaks_i,)
    intensities_batch: list of arrays, each shape (n_peaks_i,)
    formulas: list/array of formula strings, length=batch
    precursor_mzs: array, length=batch
    config: isotopic_pattern_config (same for all)
    Returns: array of bool, shape (batch,)
    """
    if len(formulas) == 0:
        return np.array([], dtype=bool)

    batch_size = len(formulas)
    # Get element numbers for all formulas (shape: batch, n_elements)
    element_numbers_batch = np.stack([get_element_numbers(f) for f in formulas])

    # Find precursor indices and intensities for each spectrum
    precursor_indices = []
    precursor_intensities = []
    for i in range(batch_size):
        mzs = mzs_batch[i]
        intensities = intensities_batch[i]
        diffs = np.abs(mzs - precursor_mzs[i])
        idx = diffs.argmin()
        precursor_indices.append(idx)
        precursor_intensities.append(intensities[idx])

    # Prepare element_fits (batch, n_elements)
    n_iso = len(isotopic_pattern_dict)
    element_fits = np.zeros((batch_size, n_iso), dtype=bool)
    iso_keys = list(isotopic_pattern_dict.keys())

    # C and N
    if config.ms1_resolution > NITROGEN_SEPARATION_RESOLUTION:
        # C
        for idx in range(batch_size):
            element_fits[idx, iso_keys.index('C')] = check_element_fit(
                config, iso_keys.index('C'), mzs_batch[idx], intensities_batch[idx],
                element_numbers_batch[idx, ELEMENT_SYMBOLS.index('C')], precursor_mzs[idx], precursor_intensities[idx]
            )
        # N
        for idx in range(batch_size):
            element_fits[idx, iso_keys.index('N')] = check_element_fit(
                config, iso_keys.index('N'), mzs_batch[idx], intensities_batch[idx],
                element_numbers_batch[idx, ELEMENT_SYMBOLS.index('N')], precursor_mzs[idx], precursor_intensities[idx]
            )
    else:
        # CN combined
        for idx in range(batch_size):
            CN_fit = check_CN_fit(
                config, mzs_batch[idx], intensities_batch[idx],
                element_numbers_batch[idx, ELEMENT_SYMBOLS.index('C')],
                element_numbers_batch[idx, ELEMENT_SYMBOLS.index('N')],
                precursor_mzs[idx], precursor_intensities[idx]
            )
            element_fits[idx, iso_keys.index('C')] = CN_fit
            element_fits[idx, iso_keys.index('N')] = CN_fit

    # S, Cl, Br
    for symbol in ['S', 'Cl', 'Br']:
        i = iso_keys.index(symbol)
        for idx in range(batch_size):
            element_fits[idx, i] = check_element_fit(
                config, i, mzs_batch[idx], intensities_batch[idx],
                element_numbers_batch[idx, ELEMENT_SYMBOLS.index(symbol)], precursor_mzs[idx], precursor_intensities[idx]
            )

    return np.all(element_fits, axis=1)

def get_element_numbers(formula:str) -> np.ndarray:
    # Returns array of element counts in the order of ELEMENT_SYMBOLS
    arr = np.zeros(len(ELEMENT_SYMBOLS), dtype=int)
    for i, symbol in enumerate(ELEMENT_SYMBOLS):
        regex = ELEMENTS[i].regex
        found = re.search(regex, formula)
        if found is not None:
            num = re.search(r'\d+', found.group())
            arr[i] = int(num.group()) if num is not None else 1
    return arr

def check_element_fit(
        config:isotopic_pattern_config,
        i:int,
        mzs:np.ndarray,
        intensities:np.ndarray,
        element_number:int,
        precursor_mz:float,
        precursor_intensity:float):
    if element_number == 0:
        return True
    iso_keys = list(isotopic_pattern_dict.keys())
    iso = isotopic_pattern_dict[iso_keys[i]]
    zero_isotope_intensity = np.power(iso["zero_isotope_probability"], element_number)
    computed_isotope_relative_intensity = iso["first_isotope_probability"] * element_number / zero_isotope_intensity
    computed_isotope_intensity = precursor_intensity * computed_isotope_relative_intensity
    if computed_isotope_intensity < config.minimum_intensity:
        return True
    computed_isotope_mass = precursor_mz + iso["mass_difference"]
    best_fit_index = (np.abs(mzs-computed_isotope_mass)).argmin()
    isotope_mass = mzs[best_fit_index]

    if not np.isclose(isotope_mass,computed_isotope_mass,rtol=config.mass_tolerance):
        return False
    else:
        isotope_intensity = intensities[best_fit_index]
        intensity_ratio = isotope_intensity/computed_isotope_intensity
        if intensity_ratio > config.max_intensity_ratio or intensity_ratio < 1/config.max_intensity_ratio:
            return False
        else:
            return True

def check_CN_fit(
        config:isotopic_pattern_config,
        mzs:np.ndarray,
        intensities:np.ndarray,
        C_number:int,
        N_number:int,
        precursor_mz:float,
        precursor_intensity:float):
    '''checks for fit of the combined N and C isotopic ratio'''
    iso_C = isotopic_pattern_dict['C']
    iso_N = isotopic_pattern_dict['N']
    if N_number == 0:
        return check_element_fit(config, list(isotopic_pattern_dict.keys()).index('C'), mzs, intensities, C_number, precursor_mz, precursor_intensity)
    if C_number == 0:
        return check_element_fit(config, list(isotopic_pattern_dict.keys()).index('N'), mzs, intensities, N_number, precursor_mz, precursor_intensity)

    zero_isotope_intensity = np.power(iso_C["zero_isotope_probability"], C_number) * np.power(iso_N["zero_isotope_probability"], N_number)
    computed_isotope_relative_intensity = (iso_C["first_isotope_probability"]*C_number + iso_N["first_isotope_probability"]*N_number) / zero_isotope_intensity
    computed_isotope_intensity = precursor_intensity * computed_isotope_relative_intensity
    if computed_isotope_intensity < config.minimum_intensity:
        return True
    computed_isotope_mass = precursor_mz + iso_C["mass_difference"]
    best_fit_index = (np.abs(mzs-computed_isotope_mass)).argmin()
    isotope_mass = mzs[best_fit_index]

    if not np.isclose(isotope_mass,computed_isotope_mass,rtol=config.mass_tolerance):
        return False
    else:
        isotope_intensity = intensities[best_fit_index]
        intensity_ratio = isotope_intensity/computed_isotope_intensity
        if intensity_ratio > config.max_intensity_ratio or intensity_ratio < 1/config.max_intensity_ratio:
            return False
        else:
            return True
iso_mass_diffs = np.array([
    isotopic_pattern_dict['C']["mass_difference"],
    isotopic_pattern_dict['S']["mass_difference"],
    isotopic_pattern_dict['Cl']["mass_difference"],
    isotopic_pattern_dict['Br']["mass_difference"]
])
iso_zero_probs = np.array([
    isotopic_pattern_dict['C']["zero_isotope_probability"],
    isotopic_pattern_dict['S']["zero_isotope_probability"],
    isotopic_pattern_dict['Cl']["zero_isotope_probability"],
    isotopic_pattern_dict['Br']["zero_isotope_probability"]
])
iso_first_probs = np.array([
    isotopic_pattern_dict['C']["first_isotope_probability"],
    isotopic_pattern_dict['S']["first_isotope_probability"],
    isotopic_pattern_dict['Cl']["first_isotope_probability"],
    isotopic_pattern_dict['Br']["first_isotope_probability"]
])

# this function will work on polars series, and will return an array
def deduce_isotopic_pattern(
    precursor_mzs: pl.Series,
    ms1_mzs: pl.Series,
    ms1_intensities: pl.Series,
    ms1_mass_tolerance_ppm: float = 5.0,
    isotopic_mass_tolerance_ppm: float = 3.0,
    minimum_intensity: float = 5e4,
    intensity_absolute_tolerance: float = 5e4,
    intensity_relative_tolerance: float = 0.05,
    min_bounds: Dict[str, int] | None = None,
    max_bounds: Dict[str, int] | None = None,
)-> pl.Series:
    """
    Deduce the isotopic pattern from the given precursor and MS1 data for each precursor ion.
    Works on a complete polars DataFrame.

    Args:
        precursor_mzs (pl.Series): Precursor m/z values (length N).
        ms1_mzs (pl.Series): Each entry is a list of m/z values for the corresponding precursor (length N).
        ms1_intensities (pl.Series): Each entry is a list of intensities for the corresponding mzs (length N).
        ms1_mass_tolerance_ppm (float): Tolerance (in ppm) for matching the precursor m/z in the MS1 spectrum.
        isotopic_mass_tolerance_ppm (float): Tolerance (in ppm) for matching the expected isotopic peaks (e.g., C, S, Cl, Br) in the MS1 spectrum. It might be lower than ms1_mass_tolerance_ppm, since there is a cancellation of errors.
        minimum_intensity (float): the entire range between zero and this value is equivalent, so any peaks in this range (including any non-existent peak) will be considered to be both at zero (for lower bound) and at this value (for upper bound). hence, if a precursor is detected with intensity 5*minimum_intensity, we do expect to see its Cl and Br isotopes (so if the isotopic peaks are absent, we decide they are 0), but we don't expect to see its carbon isotopic peak if it's below ~20, so we can only say that the upper bound is 20, and the lower is 0. note that if we do see the carbon isotopic peak, we will consider it the same as 0.
        max_bounds (Dict[str, int] | None): Maximum bounds for each element's isotopic pattern, used if no other value can be obtained (which is true for most elements expect C,S,Cl,Br currently).
        min_bounds (Dict[str, int] | None): Minimum bounds for each element's isotopic pattern, used if no other value can be obtained (which is true for most elements expect C,S,Cl,Br currently).

    Returns:
        pl.Series: A series of arrays, each containing the deduced isotopic pattern for the corresponding precursor, with th
    Explanation:
        For each precursor, this function examines its MS1 spectrum (mzs and intensities).
        It searches for peaks corresponding to the expected isotopic mass differences (C, N, S, Cl, Br)
        within the given ppm tolerance    
        """
    
    # if some bound is not given, use the default.
    if min_bounds is None:
        min_bounds = DEFAULT_MIN_BOUND.copy()
    else:
        # Merge user-provided min_bounds with defaults
        min_bounds = {**DEFAULT_MIN_BOUND, **min_bounds}
        for key in min_bounds.keys():
            if key not in ELEMENT_SYMBOLS:
                print(f"Warning: The hrms_utils isotopic pattern and mass decomposition does not handle the element {key}.")
    # same for maximum
    if max_bounds is None:
        max_bounds = DEFAULT_MAX_BOUND.copy()
    else:
        # Merge user-provided max_bounds with defaults
        max_bounds = {**DEFAULT_MAX_BOUND, **max_bounds}
        for key in max_bounds.keys():
            if key not in ELEMENT_SYMBOLS:
                print(f"Warning: The hrms_utils isotopic pattern and mass decomposition does not handle the element {key}.")
    # now do some sanity checks: max >= min for all keys, min >=0 for all
    for key in ELEMENT_SYMBOLS:
        if max_bounds[key] < min_bounds[key]:
            print(f"Warning: Inconsistent bounds for {key}: max < min.")
        if min_bounds[key] < 0:
            print(f"Warning: Negative lower bound for {key}.")
    #debug:
    # print(f"Using min bounds: {min_bounds}")
    # print(f"Using max bounds: {max_bounds}")

    ms1_mzs = ms1_mzs.to_numpy()
    ms1_intensities = ms1_intensities.to_numpy()
    deduced_bounds = [None] * len(precursor_mzs)
    for i in range(len(precursor_mzs)):
        result = deduce_isotopic_pattern_inner(
            precursor_mz=precursor_mzs[i],
            ms1_mzs=ms1_mzs[i],
            ms1_intensities=ms1_intensities[i],
            ms1_mass_tolerance_ppm=ms1_mass_tolerance_ppm,
            isotopic_mass_tolerance_ppm=isotopic_mass_tolerance_ppm,
            minimum_intensity=minimum_intensity,
            intensity_absolute_tolerance=intensity_absolute_tolerance,
            intensity_relative_tolerance=intensity_relative_tolerance,
            iso_mass_diffs=iso_mass_diffs,
            iso_zero_probs=iso_zero_probs,
            iso_first_probs=iso_first_probs
        )
        # print(result)
        if result is None:
            # Fill with NaNs or zeros
            result = [-1] * 8  # or [0.0] * 8
        deduced_bounds[i] = result
    # convert to a 2d array of shape (N, 8)
    deduced_bounds = np.array(deduced_bounds, dtype=np.float64).reshape(-1, 8)
    # construct one row of the base bound, from min_bounds and max bounds
    base_bounds = np.zeros((1, 30), dtype=np.float64)
    # fill in the default values now:
    # Fill in the base_bounds array with min_bounds (first 15) and max_bounds (last 15)
    for idx, symbol in enumerate(ELEMENT_SYMBOLS):
        base_bounds[0, idx] = min_bounds[symbol]
        base_bounds[0, idx + 15] = max_bounds[symbol]
    # Now create the bounds_array by repeating base_bounds for each precursor
    bounds_array = np.repeat(base_bounds, len(deduced_bounds), axis=0)
    
    # now we copy the values, using vectorized operations
    # Copy deduced bounds for C, S, Cl, Br into the bounds_array.
    # The deduced_bounds array has shape (N, 8): [C_lower, S_lower, Cl_lower, Br_lower, C_upper, S_upper, Cl_upper, Br_upper]
    # The bounds_array has shape (N, 30): [min_bounds..., max_bounds...]
    # Fill in the appropriate indices for C, S, Cl, Br (both lower and upper bounds).
    for idx, symbol in enumerate(['C', 'S', 'Cl', 'Br']):
        bounds_array[:, ELEMENT_SYMBOLS.index(symbol)] = np.ceil(deduced_bounds[:, idx])      # lower bound
        bounds_array[:, ELEMENT_SYMBOLS.index(symbol) + 15] = np.floor(deduced_bounds[:, idx + 4])  # upper bound

    return pl.Series(values=bounds_array.astype(np.int32), dtype=pl.Array(inner=pl.Int32, shape=(30,)))

@jit(nopython=False)
def deduce_isotopic_pattern_inner(
        precursor_mz: float,
        ms1_mzs: np.ndarray,
        ms1_intensities: np.ndarray,
        ms1_mass_tolerance_ppm: float,
        isotopic_mass_tolerance_ppm: float,
        minimum_intensity: float,
        intensity_absolute_tolerance: float,
        intensity_relative_tolerance: float,
        iso_mass_diffs: np.ndarray,
        iso_zero_probs: np.ndarray,
        iso_first_probs: np.ndarray
):
    MASS_ACCURACY_PPM_TO_DA_THRESHOLD = 200
    ms1_mzs = np.atleast_1d(ms1_mzs)
    ms1_intensities = np.atleast_1d(ms1_intensities)
    # use this to detect the precursor
    MS1_absolute_tolerance = np.max(np.array([precursor_mz, MASS_ACCURACY_PPM_TO_DA_THRESHOLD])) * ms1_mass_tolerance_ppm * 1e-6
    # use this to detect the isotopic peaks
    isotopic_absolute_tolerance = np.max(np.array([precursor_mz, MASS_ACCURACY_PPM_TO_DA_THRESHOLD])) * isotopic_mass_tolerance_ppm * 1e-6
    precursor_idx = np.where(np.atleast_1d(np.isclose(ms1_mzs, precursor_mz, atol=MS1_absolute_tolerance, rtol=0.0)))[0]
    if len(precursor_idx) == 0:
        return None
    
    #the real mass measured for the precursor in the ms1
    precursor_ms1_mz = ms1_mzs[precursor_idx[ms1_intensities[precursor_idx].argmax()]] 
    precursor_ms1_intensity = ms1_intensities[precursor_idx].max()

    # C
    c_peak_mz = precursor_ms1_mz + iso_mass_diffs[0]
    c_peaks_idx = np.where(np.atleast_1d(np.isclose(ms1_mzs, c_peak_mz, atol=isotopic_absolute_tolerance, rtol=0.0)))[0]
    C_peak_total_intensities = ms1_intensities[c_peaks_idx].max() if len(c_peaks_idx) > 0 else 0
    if C_peak_total_intensities < minimum_intensity:
        C_lower = 0
        C_upper = (minimum_intensity * iso_zero_probs[0]) / (iso_first_probs[0] * precursor_ms1_intensity)
    else:
        C_lower = ((C_peak_total_intensities * (1 - intensity_relative_tolerance) - intensity_absolute_tolerance) * iso_zero_probs[0]) / (iso_first_probs[0] * precursor_ms1_intensity)
        C_upper = ((C_peak_total_intensities * (1 + intensity_relative_tolerance) + intensity_absolute_tolerance) * iso_zero_probs[0]) / (iso_first_probs[0] * precursor_ms1_intensity)

    # S
    s_peak_mz = precursor_ms1_mz + iso_mass_diffs[1]
    s_peaks_idx = np.where(np.atleast_1d(np.isclose(ms1_mzs, s_peak_mz, atol=isotopic_absolute_tolerance, rtol=0.0)))[0]
    S_peak_total_intensities = ms1_intensities[s_peaks_idx].max() if len(s_peaks_idx) > 0 else 0
    if S_peak_total_intensities < minimum_intensity:
        S_lower = 0
        S_upper = (minimum_intensity * iso_zero_probs[1]) / (iso_first_probs[1] * precursor_ms1_intensity)
    else:
        S_lower = ((S_peak_total_intensities * (1 - intensity_relative_tolerance) - intensity_absolute_tolerance) * iso_zero_probs[1]) / (iso_first_probs[1] * precursor_ms1_intensity)
        S_upper = ((S_peak_total_intensities * (1 + intensity_relative_tolerance) + intensity_absolute_tolerance) * iso_zero_probs[1]) / (iso_first_probs[1] * precursor_ms1_intensity)

    # Cl
    cl_peak_mz = precursor_ms1_mz + iso_mass_diffs[2]
    cl_peaks_idx = np.where(np.atleast_1d(np.isclose(ms1_mzs, cl_peak_mz, atol=isotopic_absolute_tolerance, rtol=0.0)))[0]
    Cl_peak_total_intensities = ms1_intensities[cl_peaks_idx].max() if len(cl_peaks_idx) > 0 else 0
    if Cl_peak_total_intensities < minimum_intensity:
        Cl_lower = 0
        Cl_upper = (minimum_intensity * iso_zero_probs[2]) / (iso_first_probs[2] * precursor_ms1_intensity)
    else:
        Cl_lower = ((Cl_peak_total_intensities * (1 - intensity_relative_tolerance) - intensity_absolute_tolerance) * iso_zero_probs[2]) / (iso_first_probs[2] * precursor_ms1_intensity)
        Cl_upper = ((Cl_peak_total_intensities * (1 + intensity_relative_tolerance) + intensity_absolute_tolerance) * iso_zero_probs[2]) / (iso_first_probs[2] * precursor_ms1_intensity)
    # second Cl peak- the M+4
    # we need this because 2 Cl and 1 Br look very similar
    # we do this only if we got a positive estimation on Cl_lower, and at least 2 on Cl_upper
    if Cl_lower > 0 and Cl_upper >= 2:
        second_cl_peak_mz = precursor_ms1_mz + iso_mass_diffs[2] * 2
        cl_peaks_idx = np.where(np.atleast_1d(np.isclose(ms1_mzs, second_cl_peak_mz, atol=isotopic_absolute_tolerance, rtol=0.0)))[0]
        second_Cl_peak_total_intensities = ms1_intensities[cl_peaks_idx].max() if len(cl_peaks_idx) > 0 else 0
        # now, if truly there are at least 2 Cls, then we can get a ratio between the first Cl peak and the second
        expected_Cl_ratio = iso_first_probs[2] / (2*iso_zero_probs[2])
        actual_Cl_ratio = second_Cl_peak_total_intensities / Cl_peak_total_intensities
        if expected_Cl_ratio * Cl_peak_total_intensities > minimum_intensity * (1+intensity_relative_tolerance): # we expect to see the second peak, even with the noise of the intesity
            Cl_deviation = np.abs((actual_Cl_ratio - expected_Cl_ratio) / expected_Cl_ratio)
            if Cl_deviation > intensity_relative_tolerance:
                Cl_lower = 0
                Cl_upper = 1

    # Br
    br_peak_mz = precursor_ms1_mz + iso_mass_diffs[3]
    br_peaks_idx = np.where(np.atleast_1d(np.isclose(ms1_mzs, br_peak_mz, atol=isotopic_absolute_tolerance, rtol=0.0)))[0]
    Br_peak_total_intensities = ms1_intensities[br_peaks_idx].max() if len(br_peaks_idx) > 0 else 0
    if Br_peak_total_intensities < minimum_intensity:
        Br_lower = 0
        Br_upper = (minimum_intensity * iso_zero_probs[3]) / (iso_first_probs[3] * precursor_ms1_intensity)
    else:
        Br_lower = ((Br_peak_total_intensities * (1 - intensity_relative_tolerance) - intensity_absolute_tolerance) * iso_zero_probs[3]) / (iso_first_probs[3] * precursor_ms1_intensity)
        Br_upper = ((Br_peak_total_intensities * (1 + intensity_relative_tolerance) + intensity_absolute_tolerance) * iso_zero_probs[3]) / (iso_first_probs[3] * precursor_ms1_intensity)
    # TODO: Add support for second isotope for Br (M+4)

    return [C_lower, S_lower, Cl_lower, Br_lower, C_upper, S_upper, Cl_upper, Br_upper]


