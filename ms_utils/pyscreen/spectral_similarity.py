import numpy as np
import polars as pl
from numba import njit
from ms_entropy import calculate_entropy_similarity
from dataclasses import dataclass, field

@dataclass
class search_config:
    polarity:str
    ms1_mass_tolerance:float=5e-6
    ms2_mass_tolerance:float=10e-6 # high, but it's because NIST can have pretty high mass error.
    DotProd_threshold:dict[int:int|float]=field(default_factory=lambda:{
                0:650,
                1:700,
                2:800,
                3:900})
    search_engine:str='entropy'
    noise_threshold:float=0.005

# yeah this is a bit of a mess, but splitting to multiple functions didn't behave nicely with numba
@njit
def identity_score_NIST_like(
        spec1_mz:np.ndarray, spec1_intensity:np.ndarray,
        spec2_mz:np.ndarray, spec2_intensity:np.ndarray,
        config:search_config
        )-> np.float64:
    """ """
    ms2_mass_tolerance_search = config.ms2_mass_tolerance

    shift=np.float64(0.0)
    mz_power=np.float64(2)
    intensity_power=np.float64(0.5)
    #### find_peak_matches
    lowest_idx = 0
    matches = []
    for i in range(spec1_mz.shape[0]):
        mz1 = spec1_mz[i]
        mz_min = mz1 - ms2_mass_tolerance_search*mz1
        mz_max = mz1 + ms2_mass_tolerance_search*mz1
        for j in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[j] + shift
            if mz2 > mz_max:
                break
            if mz2 < mz_min:
                lowest_idx = j + 1
            else:

                matches.append((i,j))
    #### end of "find_peak_matches"
    idx1 = np.array([x[0] for x in matches])
    idx2 = np.array([x[1] for x in matches])

    if len(idx1) == 0:
        return 0.0
    score = 0.0
    for i in range(idx1.shape[0]):
        score += (
            np.power(spec1_intensity[idx1[i]]*spec2_intensity[idx2[i]],intensity_power) * 
            np.power((spec1_mz[idx1[i]]*spec2_mz[idx2[i]]),mz_power)
            )

    

    # spec1_norm = cosine_score_self(mz1, spec_intensity1, mz_tol_ppm, mz_power, intensity_power)
    
    # this is "find_peak_matches_self" for spec1
    lowest_idx = 0
    matches = []
    for i in range(spec1_mz.shape[0]):
        mz1 = spec1_mz[i]
        mz_min = mz1 - ms2_mass_tolerance_search*mz1
        mz_max = mz1 + ms2_mass_tolerance_search*mz1
        for j in range(lowest_idx, spec1_mz.shape[0]):
            mz2 = spec1_mz[j]
            if mz2 > mz_max:
                break
            if mz2 < mz_min:
                lowest_idx = j + 1
            else:
                matches.append((i,j))
    idx1 = np.array([x[0] for x in matches])
    idx2 = np.array([x[1] for x in matches])
    spec1_norm = np.float64(0.0)
    for i in range(idx1.shape[0]):
        spec1_norm += (
            np.power(spec1_intensity[idx1[i]]*spec1_intensity[idx2[i]],intensity_power) * 
            np.power((spec1_mz[idx1[i]]*spec1_mz[idx2[i]]),mz_power))

    # this is "find_peak_matches_self" for spec2
    lowest_idx = 0
    matches = []
    for i in range(spec2_mz.shape[0]):
        mz1 = spec2_mz[i]
        mz_min = mz1 - ms2_mass_tolerance_search*mz1
        mz_max = mz1 + ms2_mass_tolerance_search*mz1
        for j in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[j]
            if mz2 > mz_max:
                break
            if mz2 < mz_min:
                lowest_idx = j + 1
            else:
                matches.append((i,j))
    idx1 = np.array([x[0] for x in matches])
    idx2 = np.array([x[1] for x in matches])
    spec2_norm = np.float64(0.0)
    for i in range(idx1.shape[0]):
        spec2_norm += (
            np.power(spec2_intensity[idx1[i]]*spec2_intensity[idx2[i]],intensity_power) * 
            np.power((spec2_mz[idx1[i]]*spec2_mz[idx2[i]]),mz_power))


    score = score / np.sqrt(spec1_norm*spec2_norm)

    return score
identity_score_NIST_like_batch = np.vectorize(identity_score_NIST_like)



def entropy_score(
        spec1_mz:np.ndarray, spec1_intensity:np.ndarray,
        spec2_mz:np.ndarray, spec2_intensity:np.ndarray,
        config:search_config) -> np.float64:
    if any(x is None for x in [spec1_mz,spec2_mz,spec1_intensity,spec2_intensity]):
        return -1
    spec1 = np.column_stack((spec1_mz,spec1_intensity))
    spec1 = np.array(spec1,dtype=np.float32)
    spec2 = np.column_stack((spec2_mz,spec2_intensity))
    spec2 = np.array(spec2,dtype=np.float32)
    score = calculate_entropy_similarity(
        spec1,spec2,
        ms2_tolerance_in_ppm=config.ms2_mass_tolerance*10e6,
        clean_spectra=True,
        noise_threshold=config.noise_threshold)
    score = np.float64(score)
    return score
entropy_score_batch=np.vectorize(entropy_score)

def NIST_filtering_mock(NIST:pl.DataFrame) -> pl.DataFrame:
    NIST_filtered = NIST.filter(
        pl.col('Instrument_type').eq('HCD'),
        pl.col('MultiCharge').not_()
    ).select(
        ['Name','NIST_ID','PrecursorMZ','Precursor_type',
         'raw_spectrum_intensity','normalized_spectrum_mz']
    )
    return NIST_filtered

def NIST_search_mock(
        query_df: pl.DataFrame, 
        NIST:pl.DataFrame,
        mz_tol_ppm=5) -> pl.DataFrame:
    results = NIST.join_where(
        query_df,
        pl.col('PrecursorMZ').ge(pl.col('PrecursorMZ_query').mul(1-mz_tol_ppm/1e6)),
        pl.col('PrecursorMZ').le(pl.col('PrecursorMZ_query').mul(1+mz_tol_ppm/1e6)),
        suffix='_query'
    )

    results = results.with_columns(
        pl.struct(
            pl.col('raw_spectrum_intensity'),
            pl.col('raw_spectrum_mz'),
            pl.col('raw_spectrum_intensity_query'),
            pl.col('raw_spectrum_mz_query')
        ).map_batches(
            lambda spectra: identity_score_NIST_like_batch(
                spectra.struct.field('raw_spectrum_mz').to_numpy(),
                spectra.struct.field('raw_spectrum_intensity').to_numpy(),
                spectra.struct.field('raw_spectrum_mz_query').to_numpy(),
                spectra.struct.field('raw_spectrum_intensity_query').to_numpy())
        ).alias('cosine_score')
    )

    results = results.select(['NIST_ID','NIST_ID_query','cosine_score']).sort('cosine_score',descending=True)
    return results




