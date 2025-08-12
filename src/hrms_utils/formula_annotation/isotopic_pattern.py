import numpy as np
import re
from dataclasses import dataclass
import polars as pl
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
MASS_ACCURACY_PPM_TO_DA_THRESHOLD = 200.0001
isotopic_pattern_arr = np.array( 
    #mass difference, zero isotope probability, first isoptope probability
    [
        [1.0034,0.989,0.011], #C
        [0.99703,0.996,0.004], #N
        [1.9958,0.95,0.042], #S
        [1.9971,0.758,0.242], #Cl
        [1.998,0.507,0.493], #Br
    ]
)
isotopic_pattern_dict = {
    'C': {"mass_difference": isotopic_pattern_arr[0][0], "zero_isotope_probability": isotopic_pattern_arr[0][1], "first_isotope_probability": isotopic_pattern_arr[0][2]},
    'N': {"mass_difference": isotopic_pattern_arr[1][0], "zero_isotope_probability": isotopic_pattern_arr[1][1], "first_isotope_probability": isotopic_pattern_arr[1][2]},
    'S': {"mass_difference": isotopic_pattern_arr[2][0], "zero_isotope_probability": isotopic_pattern_arr[2][1], "first_isotope_probability": isotopic_pattern_arr[2][2]},
    'Cl': {"mass_difference": isotopic_pattern_arr[3][0], "zero_isotope_probability": isotopic_pattern_arr[3][1], "first_isotope_probability": isotopic_pattern_arr[3][2]},
    'Br': {"mass_difference": isotopic_pattern_arr[4][0], "zero_isotope_probability": isotopic_pattern_arr[4][1], "first_isotope_probability": isotopic_pattern_arr[4][2]},
}

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
    # Get element numbers for all formulas (shape: batch, 5)
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

    # Prepare element_fits (batch, 5)
    element_fits = np.zeros_like(element_numbers_batch, dtype=bool)

    # C and N
    if config.ms1_resolution > NITROGEN_SEPARATION_RESOLUTION:
        # C
        for idx in range(batch_size):
            element_fits[idx, 0] = check_element_fit(
                config, 0, mzs_batch[idx], intensities_batch[idx],
                element_numbers_batch[idx, 0], precursor_mzs[idx], precursor_intensities[idx]
            )
        # N
        for idx in range(batch_size):
            element_fits[idx, 1] = check_element_fit(
                config, 1, mzs_batch[idx], intensities_batch[idx],
                element_numbers_batch[idx, 1], precursor_mzs[idx], precursor_intensities[idx]
            )
    else:
        # CN combined
        for idx in range(batch_size):
            CN_fit = check_CN_fit(
                config, mzs_batch[idx], intensities_batch[idx],
                element_numbers_batch[idx, 0], element_numbers_batch[idx, 1],
                precursor_mzs[idx], precursor_intensities[idx]
            )
            element_fits[idx, 0] = CN_fit
            element_fits[idx, 1] = CN_fit

    # S, Cl, Br
    for i in [2, 3, 4]:
        for idx in range(batch_size):
            element_fits[idx, i] = check_element_fit(
                config, i, mzs_batch[idx], intensities_batch[idx],
                element_numbers_batch[idx, i], precursor_mzs[idx], precursor_intensities[idx]
            )

    return np.all(element_fits, axis=1)

def get_element_numbers(formula:str) -> np.ndarray:
    has_C = re.search(r"C((\d+)|[A-Z]|$)",formula)
    if has_C is None:
        num_C = 0
    else:
        num_C = re.search(r"C(\d+)",has_C.group(0))
        if num_C is None:
            num_C = 1
        else:
            num_C = int(num_C.group(1))
    has_N = re.search(r"N((\d+)|[A-Z]|$)",formula)
    if has_N is None:
        num_N = 0
    else:
        num_N = re.search(r"N(\d+)",has_N.group(0))
        if num_N is None:
            num_N = 1
        else:
            num_N = int(num_N.group(1))
    has_S = re.search(r"S((\d+)|[A-Z]|$)",formula)
    if has_S is None:
        num_S = 0
    else:
        num_S = re.search(r"S(\d+)",has_S.group(0))
        if num_S is None:
            num_S = 1
        else:
            num_S = int(num_S.group(1))
    has_Cl = re.search(r"Cl((\d+)|[A-Z]|$)",formula)
    if has_Cl is None:
        num_Cl = 0
    else:
        num_Cl = re.search(r"Cl(\d+)",has_Cl.group(0))
        if num_Cl is None:
            num_Cl = 1
        else:
            num_Cl = int(num_Cl.group(1))
    has_Br = re.search(r"Br((\d+)|[A-Z]|$)",formula)
    if has_Br is None:
        num_Br = 0
    else:
        num_Br = re.search(r"Br(\d+)",has_Br.group(0))
        if num_Br is None:
            num_Br = 1
        else:
            num_Br = int(num_Br.group(1))

    return np.array([num_C,num_N,num_S,num_Cl,num_Br])

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
    else:
        zero_isotope_intesnsity = np.power(isotopic_pattern_arr[i][1],element_number)
        computed_isotope_relative_intensity = isotopic_pattern_arr[i][2]*element_number/zero_isotope_intesnsity
        computed_isotope_intensity = precursor_intensity*computed_isotope_relative_intensity
        if computed_isotope_intensity < config.minimum_intensity: #meaning the intensity would be too small, and we won't be able to detect it reliably anyway.
            return True
        computed_isotope_mass = precursor_mz + isotopic_pattern_arr[i][0]
        best_fit_index = (np.abs(mzs-computed_isotope_mass)).argmin()
        isotope_mass = mzs[best_fit_index]
        
        if not np.isclose(isotope_mass,computed_isotope_mass,rtol=config.mass_tolerance): # then the element probably isn't even present, or the intensity is too low.
            return False
        else:  
            isotope_intensity = intensities[best_fit_index]
            intensity_ratio = isotope_intensity/computed_isotope_intensity
            if intensity_ratio > config.max_intensity_ratio or intensity_ratio < 1/config.max_intensity_ratio: # so we have the right isotope mass, but the ratio is off, meaning we don't have the correct number
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
    if N_number == 0:
        return check_element_fit(config,0,mzs,intensities,C_number,precursor_mz,precursor_intensity)
    if C_number == 0:
        return check_element_fit(config,1,mzs,intensities,N_number,precursor_mz,precursor_intensity)
    
    zero_isotope_intesnsity = np.power(isotopic_pattern_arr[0][1],C_number)*np.power(isotopic_pattern_arr[1][1],N_number)
    computed_isotope_relative_intensity = (isotopic_pattern_arr[0][2]*C_number+isotopic_pattern_arr[1][2]*N_number)/zero_isotope_intesnsity
    computed_isotope_intensity = precursor_intensity*computed_isotope_relative_intensity
    if computed_isotope_intensity < config.minimum_intensity: #meaning the intensity would be too small, and we won't be able to detect it reliably anyway.
        return True
    computed_isotope_mass = precursor_mz + isotopic_pattern_arr[0][0] # the mass of C will be the main mass, since the N peak is smaller and will be merged to it.
    best_fit_index = (np.abs(mzs-computed_isotope_mass)).argmin()
    isotope_mass = mzs[best_fit_index]
    
    if not np.isclose(isotope_mass,computed_isotope_mass,rtol=config.mass_tolerance): # then the element probably isn't even present, or the intensity is too low.
        return False
    else:  
        isotope_intensity = intensities[best_fit_index]
        intensity_ratio = isotope_intensity/computed_isotope_intensity
        if intensity_ratio > config.max_intensity_ratio or intensity_ratio < 1/config.max_intensity_ratio: # so we have the right isotope mass, but the ratio is off, meaning we don't have the correct number
            return False
        else:
            return True

# this function will work on polars series, and will return an array
def deduce_isotopic_pattern(
    precursor_mzs: pl.Series,
    ms1_mzs: pl.Series,
    ms1_intensities: pl.Series,
    mass_tolerance_ppm: float = 5.0,
    minimum_intensity: float = 5e4,
    intensity_relative_tolerance: float = 0.5
)-> pl.Series:
    """
    Deduce the isotopic pattern from the given precursor and MS1 data for each precursor ion.
    Works on a complete polars DataFrame.

    Args:
        precursor_mzs (pl.Series): Precursor m/z values (length N).
        ms1_mzs (pl.Series): Each entry is a list of m/z values for the corresponding precursor (length N).
        ms1_intensities (pl.Series): Each entry is a list of intensities for the corresponding mzs (length N).
        tolerance_ppm (float): m/z tolerance in ppm for matching isotopic peaks.
        minimum_intensity (float): the entire range between zero and this value is equivalent, so any peaks in this range (including any non-existent peak) will be considered to be both at zero (for lower bound) and at this value (for upper bound). hence, if a precursor is detected with intensity 5*minimum_intensity, we do expect to see its Cl and Br isotopes (so if the isotopic peaks are absent, we decide they are 0), but we don't expect to see its carbon isotopic peak if it's below ~20, so we can only say that the upper bound is 20, and the lower is 0. note that if we do see the carbon isotopic peak, we will consider it the same as 0.

    Explanation:
        For each precursor, this function examines its MS1 spectrum (mzs and intensities).
        It searches for peaks corresponding to the expected isotopic mass differences (C, N, S, Cl, Br)
        within the given ppm tolerance    
        """
    bounds = [None] * len(precursor_mzs)
    ms1_mzs = ms1_mzs.to_numpy()
    ms1_intensities = ms1_intensities.to_numpy()
    for i in range(len(precursor_mzs)):
        bounds[i] = deduce_isotopic_pattern_inner(
            precursor_mz=precursor_mzs[i],
            ms1_mzs=ms1_mzs[i],
            ms1_intensities=ms1_intensities[i],
            mass_tolerance_ppm=mass_tolerance_ppm,
            minimum_intensity=minimum_intensity,
            intensity_relative_tolerance=intensity_relative_tolerance
        )
    bounds = pl.Series(bounds, dtype=pl.Array(inner=pl.Float64,shape=(8,)))
    return bounds

def deduce_isotopic_pattern_inner(
        precursor_mz: float,
        ms1_mzs: np.ndarray,
        ms1_intensities: np.ndarray,
        mass_tolerance_ppm: float,
        minimum_intensity: float,
        intensity_relative_tolerance: float
):
    """
    Inner function to deduce the isotopic pattern for a single precursor ion.
    returns the isotopic pattern an array, with structure:
    [
    C_lower
    S_lower
    Cl_lower
    Br_lower
    C_upper
    S_upper
    Cl_upper
    Br_upper
    ]
    or None if the precursor itself can't be found. shouldn't happen, might become an exception in the future.
    """
    absolute_tolerance = np.max([precursor_mz * mass_tolerance_ppm * 1e-6, MASS_ACCURACY_PPM_TO_DA_THRESHOLD* mass_tolerance_ppm * 1e-6])
    precursor_idx = np.where(np.isclose(ms1_mzs, precursor_mz, atol=absolute_tolerance, rtol=0))[0]
    if len(precursor_idx) == 0:
        print(f"Precursor m/z {precursor_mz} not found in MS1 data.")
        return None
    precursor_ms1_mz = ms1_mzs[precursor_idx[ms1_intensities[precursor_idx].argmax()]]
    # print(precursor_ms1_mz)
    precursor_ms1_intensity = ms1_intensities[precursor_idx].max()

    # C
    c_peak_mz = precursor_ms1_mz + isotopic_pattern_dict['C']["mass_difference"]
    c_peaks_idx = np.where(np.isclose(ms1_mzs, c_peak_mz, atol=absolute_tolerance, rtol=0))[0]
    C_peak_total_intensities = ms1_intensities[c_peaks_idx].max() if len(c_peaks_idx) > 0 else 0
    if C_peak_total_intensities < minimum_intensity:
        C_lower = 0
        C_upper = (minimum_intensity * isotopic_pattern_dict['C']["zero_isotope_probability"]) / (isotopic_pattern_dict['C']["first_isotope_probability"]* precursor_ms1_intensity)
    else:
        C_lower = (C_peak_total_intensities* (1-intensity_relative_tolerance) * isotopic_pattern_dict['C']["zero_isotope_probability"]) / (isotopic_pattern_dict['C']["first_isotope_probability"]* precursor_ms1_intensity)
        C_upper = (C_peak_total_intensities *(1+intensity_relative_tolerance) * isotopic_pattern_dict['C']["zero_isotope_probability"]) / (isotopic_pattern_dict['C']["first_isotope_probability"]* precursor_ms1_intensity)

    # S
    s_peak_mz = precursor_ms1_mz + isotopic_pattern_dict['S']["mass_difference"]
    s_peaks_idx = np.where(np.isclose(ms1_mzs, s_peak_mz, atol=absolute_tolerance, rtol=0))[0]
    S_peak_total_intensities = ms1_intensities[s_peaks_idx].max() if len(s_peaks_idx) > 0 else 0
    if S_peak_total_intensities < minimum_intensity:
        S_lower = 0
        S_upper = (minimum_intensity * isotopic_pattern_dict['S']["zero_isotope_probability"]) / (isotopic_pattern_dict['S']["first_isotope_probability"]* precursor_ms1_intensity)
    else:
        S_lower = (S_peak_total_intensities* (1-intensity_relative_tolerance) * isotopic_pattern_dict['S']["zero_isotope_probability"]) / (isotopic_pattern_dict['S']["first_isotope_probability"]* precursor_ms1_intensity)
        S_upper = (S_peak_total_intensities *(1+intensity_relative_tolerance) * isotopic_pattern_dict['S']["zero_isotope_probability"]) / (isotopic_pattern_dict['S']["first_isotope_probability"]* precursor_ms1_intensity)

    # Cl
    cl_peak_mz = precursor_ms1_mz + isotopic_pattern_dict['Cl']["mass_difference"]
    cl_peaks_idx = np.where(np.isclose(ms1_mzs, cl_peak_mz, atol=absolute_tolerance, rtol=0))[0]
    Cl_peak_total_intensities = ms1_intensities[cl_peaks_idx].max() if len(cl_peaks_idx) > 0 else 0
    if Cl_peak_total_intensities < minimum_intensity:
        Cl_lower = 0
        Cl_upper = (minimum_intensity * isotopic_pattern_dict['Cl']["zero_isotope_probability"]) / (isotopic_pattern_dict['Cl']["first_isotope_probability"]* precursor_ms1_intensity)
    else:
        Cl_lower = (Cl_peak_total_intensities* (1-intensity_relative_tolerance) * isotopic_pattern_dict['Cl']["zero_isotope_probability"]) / (isotopic_pattern_dict['Cl']["first_isotope_probability"]* precursor_ms1_intensity)
        Cl_upper = (Cl_peak_total_intensities *(1+intensity_relative_tolerance) * isotopic_pattern_dict['Cl']["zero_isotope_probability"]) / (isotopic_pattern_dict['Cl']["first_isotope_probability"]* precursor_ms1_intensity)
    # TODO: Add support for second isotope for Cl (M+4)

    # Br
    br_peak_mz = precursor_ms1_mz + isotopic_pattern_dict['Br']["mass_difference"]
    br_peaks_idx = np.where(np.isclose(ms1_mzs, br_peak_mz, atol=absolute_tolerance, rtol=0))[0]
    Br_peak_total_intensities = ms1_intensities[br_peaks_idx].max() if len(br_peaks_idx) > 0 else 0
    if Br_peak_total_intensities < minimum_intensity:
        Br_lower = 0
        Br_upper = (minimum_intensity * isotopic_pattern_dict['Br']["zero_isotope_probability"]) / (isotopic_pattern_dict['Br']["first_isotope_probability"]* precursor_ms1_intensity)
    else:
        Br_lower = (Br_peak_total_intensities* (1-intensity_relative_tolerance) * isotopic_pattern_dict['Br']["zero_isotope_probability"]) / (isotopic_pattern_dict['Br']["first_isotope_probability"]* precursor_ms1_intensity)
        Br_upper = (Br_peak_total_intensities *(1+intensity_relative_tolerance) * isotopic_pattern_dict['Br']["zero_isotope_probability"]) / (isotopic_pattern_dict['Br']["first_isotope_probability"]* precursor_ms1_intensity)
    # TODO: Add support for second isotope for Br (M+4)

    return [C_lower, S_lower, Cl_lower, Br_lower, C_upper, S_upper, Cl_upper, Br_upper]


if __name__ == "__main__":
    # Test the function with some example data
    # precursor_mzs = pl.Series([100.0, 200.1], dtype=pl.Float64)
    # ms1_mzs = pl.Series([[100.0, 100.1, 100.2], [200.0, 200.1, 200.1+isotopic_pattern_dict["C"]["mass_difference"]]], dtype=pl.List(pl.Float64))
    # ms1_intensities = pl.Series([[1000, 1100, 1200], [2000, 2100, 2200]], dtype=pl.List(pl.Float64))
    from hrms_utils.interfaces.msdial import get_chromatogram
    chromatogram = get_chromatogram("/home/analytit_admin/Data/iibr_data/250515_017.txt")
    chromatogram = chromatogram.with_columns(
        bounds = pl.struct(
            pl.col("Precursor_mz_MSDIAL"),
            pl.col("ms1_isotopes_m/z"),
            pl.col("ms1_isotopes_intensity")
        ).map_batches(
            function= lambda x: deduce_isotopic_pattern(
                x.struct.field("Precursor_mz_MSDIAL"),
                x.struct.field("ms1_isotopes_m/z"),
                x.struct.field("ms1_isotopes_intensity"),
                mass_tolerance_ppm=5,
                minimum_intensity=2e5,
                intensity_relative_tolerance=0.1
            ), return_dtype=pl.Array(inner=pl.Float64, shape=(8,))
        )
    )
    # print(chromatogram.select(pl.col("bounds")).head(10).to_init_repr())
    print(chromatogram.filter(
        pl.col("bounds").arr.get(1) > 0
    ).select([pl.col("bounds"), pl.col("Precursor_mz_MSDIAL")]).head(10).to_init_repr())

 