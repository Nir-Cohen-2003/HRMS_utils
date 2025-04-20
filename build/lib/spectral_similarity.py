import numpy as np
import polars as pl
from numba import njit, vectorize, guvectorize, jit
from time import time
import ms_entropy as me

# yeah this is a bit of a mess, but splitting to multiple functions didn't behave nicely with numba
@njit
def cosine_score(
        spec1_mz:np.ndarray, spec_intensity1:np.ndarray,
        spec2_mz:np.ndarray, spec_intensity2:np.ndarray,
        )-> np.float64:
    """ """
    mz_tol_ppm=np.float64(5.0)
    shift=np.float64(0.0)
    mz_power=np.float64(0.0)
    intensity_power=np.float64(1.0)
    #### find_peak_matches
    mz_tol = mz_tol_ppm/1e6
    lowest_idx = 0
    matches = []
    for i in range(spec1_mz.shape[0]):
        mz1 = spec1_mz[i]
        mz_min = mz1 - mz_tol*mz1
        mz_max = mz1 + mz_tol*mz1
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
            np.power(spec_intensity1[idx1[i]]*spec_intensity2[idx2[i]],intensity_power) * 
            np.power((spec1_mz[idx1[i]]*spec2_mz[idx2[i]]),mz_power)
            )

    

    # spec1_norm = cosine_score_self(mz1, spec_intensity1, mz_tol_ppm, mz_power, intensity_power)
    
    # this is "find_peak_matches_self" for spec1
    lowest_idx = 0
    matches = []
    for i in range(spec1_mz.shape[0]):
        mz1 = spec1_mz[i]
        mz_min = mz1 - mz_tol*mz1
        mz_max = mz1 + mz_tol*mz1
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
            np.power(spec_intensity1[idx1[i]]*spec_intensity1[idx2[i]],intensity_power) * 
            np.power((spec1_mz[idx1[i]]*spec1_mz[idx2[i]]),mz_power))

    # this is "find_peak_matches_self" for spec2
    lowest_idx = 0
    matches = []
    for i in range(spec2_mz.shape[0]):
        mz1 = spec2_mz[i]
        mz_min = mz1 - mz_tol*mz1
        mz_max = mz1 + mz_tol*mz1
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
            np.power(spec_intensity2[idx1[i]]*spec_intensity2[idx2[i]],intensity_power) * 
            np.power((spec2_mz[idx1[i]]*spec2_mz[idx2[i]]),mz_power))

    # spec2_norm = cosine_score_self(mz2, spec_intensity2, mz_tol_ppm, mz_power, intensity_power)

    score = score / np.sqrt(spec1_norm*spec2_norm)
    score = np.float32(score)
    return score
cosine_score_batch = np.vectorize(cosine_score)


def entropy_score(spec1_mz, spec_intensity1, spec2_mz, spec_intensity2) -> np.float32:
    spec1 = np.column_stack((spec1_mz, spec_intensity1))
    spec1 = np.array(spec1, dtype=np.float32)
    spec2 = np.column_stack((spec2_mz, spec_intensity2))
    spec2 = np.array(spec2, dtype=np.float32)
    score = me.calculate_entropy_similarity(spec1, spec2,ms2_tolerance_in_ppm=5,clean_spectra=True,noise_threshold=0.005)
    return score
entropy_score_batch = np.vectorize(entropy_score)

def NIST_filtering(NIST:pl.DataFrame) -> pl.DataFrame:
    NIST_filtered = NIST.filter(
        pl.col('Instrument_type').eq('HCD'),
        pl.col('MultiCharge').not_()
    ).select(
        ['Name','NIST_ID','PrecursorMZ',
         'raw_spectrum_intensity','raw_spectrum_mz']
    )
    return NIST_filtered

def NIST_search(query_df: pl.DataFrame, NIST:pl.DataFrame,engine:str,mz_tol_ppm=5) -> pl.DataFrame:
    if engine == 'entropy':
        score_batch = entropy_score_batch
    elif engine == 'cosine':
        score_batch = cosine_score_batch
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
            lambda spectra: score_batch(
                spectra.struct.field('raw_spectrum_mz').to_numpy(),
                spectra.struct.field('raw_spectrum_intensity').to_numpy(),
                spectra.struct.field('raw_spectrum_mz_query').to_numpy(),
                spectra.struct.field('raw_spectrum_intensity_query').to_numpy()),
            return_dtype=pl.Float32,
            is_elementwise=True
        ).alias('score')
    )

    results = results.select(['NIST_ID','NIST_ID_query','score']).sort('score',descending=True)
    return results


if __name__ == "__main__":
    pl.set_random_seed(42)
    start = time()
    mz_tol_ppm = 5
    engine = 'entropy'
    NIST = pl.scan_parquet(r"/home/analytit_admin/Data/NIST_hr_msms/NIST_DB.parquet").select(
        ['Name','NIST_ID','PrecursorMZ',
         'raw_spectrum_intensity','raw_spectrum_mz',
         'Instrument_type','MultiCharge']
    ).collect()
    NIST = NIST_filtering(NIST)
    print(time()-start)
    after_read = time()



    query_df = NIST.sample(100000,seed=42)
    time1 = time()
    results = NIST_search(query_df, NIST,engine,mz_tol_ppm)
    print(results)

    best_matches = results.filter(
        pl.col('NIST_ID_query').ne(pl.col('NIST_ID'))
    ).group_by('NIST_ID_query').agg(
        pl.col('NIST_ID').first().alias('NIST_ID'),
        pl.col('score').first().alias('score')
        ).sort('score',descending=True)
    print(best_matches)
    print(time()-after_read)


