import numpy as np
import re
from dataclasses import dataclass

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

def fits_isotopic_pattern_batch(mzs_batch, intensities_batch, formulas, precursor_mzs, config):
    """
    mzs_batch: shape (batch, n_peaks)
    intensities_batch: shape (batch, n_peaks)
    formulas: list/array of formula strings, length=batch
    precursor_mzs: array, length=batch
    config: isotopic_pattern_config (same for all)
    Returns: array of bool, shape (batch,)
    """
    # sometimes we might get an empty batch, so we need to handle that
    if len(formulas) == 0:
        return np.array([], dtype=bool)

    mzs_batch = np.asarray(mzs_batch)
    intensities_batch = np.asarray(intensities_batch)
    precursor_mzs = np.asarray(precursor_mzs)
    batch_size = len(formulas)

    # Get element numbers for all formulas (shape: batch, 5)
    element_numbers_batch = np.stack([get_element_numbers(f) for f in formulas])

    # Find precursor indices and intensities for each spectrum
    precursor_indices = np.abs(mzs_batch - precursor_mzs[:, None]).argmin(axis=1)
    precursor_intensities = intensities_batch[np.arange(batch_size), precursor_indices]

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



if __name__ == "__main__":
    mzs = []#[246.12357, 247.12074, 247.12688, 248.12413, 248.13025]	
    intensities =[]#[230812832.0, 2362436.0, 30514068.0, 473412.0, 2097496.0]
    formula = "C13H15N3O2"
    precursor_mz=246.124
    config = isotopic_pattern_config(mass_tolerance=3e-6,ms1_resolution=1.2e5,minimum_intensity=5e5,max_intensity_ratio=1.7)
    fit=fits_isotopic_pattern(mzs=mzs,intensities=intensities,precursor_mz=precursor_mz,formula=formula,config=config)
    print(fit)
    print(fit.shape)
    print(type(fit))