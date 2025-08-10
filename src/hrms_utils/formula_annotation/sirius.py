import requests
from time import time
import polars as pl
from aiohttp import ClientSession
import asyncio
from ..formula_annotation.utils import format_formula_string_to_array
from pathlib import Path
import numpy as np
from typing import Any, List, Tuple, Dict, Optional
num_elements = 15

def get_all_compounds(project_name):
    sirius_base_url = _get_sirius_base_url()
    url = sirius_base_url + '/' + project_name + '/aligned-features'
    response = requests.get(url)
    features = response.json()
    features = pl.DataFrame(features)
    return features

async def _get_all_formulas(project_name):
    sirius_base_url: str = _get_sirius_base_url()
    url: str = sirius_base_url + '/' + project_name + '/aligned-features'
    aligned_features = requests.get(url).json()
    feature_ids = [feature['alignedFeatureId'] for feature in aligned_features]
    formulas = []
    tasks = []
    base_url= sirius_base_url + '/' + project_name + '/aligned-features/'
    async with ClientSession() as session:
        for feature_id in feature_ids:
            tasks.append(asyncio.create_task(_get_formulas_for_feature(session, base_url, feature_id)))
        results = await asyncio.gather(*tasks)
        for feature_formulas in results:
            if feature_formulas:
                formulas.extend(feature_formulas)

    formulas_df: pl.DataFrame = pl.DataFrame(formulas)

    return formulas_df


def get_all_formulas(project_name: str) -> pl.DataFrame:
    """
    Get all formulas for a given project name from the Sirius API.
    
    Args:
        project_name (str): The name of the Sirius project.
    
    Returns:
        pl.DataFrame: A Polars DataFrame containing all formulas.
    """
    formulas: pl.DataFrame = asyncio.run(_get_all_formulas(project_name))
    formulas = formulas.with_columns(
        pl.col('molecularFormula').map_elements(
            function=format_formula_string_to_array,
            return_dtype=pl.List(pl.Int64)
        ).list.to_array(width=num_elements).alias('molecularFormula_array')
    )
    return formulas

async def _get_all_frag_trees(project_name: str) -> pl.DataFrame:
    features: pl.DataFrame = get_all_compounds(project_name)
    feature_ids: np.ndarray[tuple[int], Any] = features.select('alignedFeatureId').to_numpy().flatten()
    all_frag_trees: list[Any] = []
    tasks = []
    sirius_base_url: str = _get_sirius_base_url()
    base_url: str= sirius_base_url + '/' + project_name + '/aligned-features/'

    async with ClientSession() as session:
        for feature_id in feature_ids:
            tasks.append(asyncio.create_task(_get_frag_trees_for_feature(session, base_url, feature_id)))
        results = await asyncio.gather(*tasks)
        for frag_trees in results:
            if frag_trees is not None and len(frag_trees) > 0:
                all_frag_trees.extend(frag_trees)
    frag_trees_df: pl.DataFrame = pl.DataFrame(all_frag_trees)
    return frag_trees_df

async def _get_formulas_for_feature(session: ClientSession, base_url : str, feature_id : str) -> list:
    url =  base_url+ feature_id + '/formulas'
    async with session.get(url) as response:
        # Check if the response is successful
        if response.status != 200:
            return []
        feature_formulas = await response.json()
        if not feature_formulas: # Check if there are any formulas
            return []
        for formula in feature_formulas:
            formula['featureId'] = feature_id
        return feature_formulas


async def _get_frag_trees_for_feature(session: ClientSession,base_url:str, feature_id:str) -> list:
    url: str =  base_url+ feature_id + '/formulas'
    async with session.get(url) as response:
        if response.status != 200: # Check if the response is successful
            return []
        formulas : List[Dict[str,Any]] = await response.json()
        if not formulas: # Check if there are any formulas
            return []
        frag_trees: List[dict[str, Any]] = []
        for formula in formulas:
            frag_tree_url: str = url + '/' + formula['formulaId'] + '/fragtree'
            async with session.get(frag_tree_url) as frag_tree_response: 
                if frag_tree_response.status != 200: # if it failed to get the frag tree, skip
                    continue
                frag_tree: Dict[str,Any]= await frag_tree_response.json()
                frag_tree['formulaId'] = formula['formulaId']
                frag_tree['featureId'] = feature_id
                frag_trees.append(frag_tree)
    
    return frag_trees


def get_clean_spectra(sirius_project_name: str) -> pl.DataFrame:

    frag_trees: pl.DataFrame = asyncio.run(_get_all_frag_trees(sirius_project_name))
    frag_trees = frag_trees.with_columns(
        pl.col('fragments').list.eval(
            pl.element().struct.field('adduct').str.replace('M',pl.element().struct.field('molecularFormula')).str.extract(r'\[(.+)\]',1)
        ).alias('fragment_formulas'),
        pl.col('fragments').list.eval(
            pl.element().struct.field('intensity')
        ).alias('fragments_intensities')
    )
    frag_trees = frag_trees.with_columns(
        pl.col('fragment_formulas').list.eval(
            pl.element().map_elements(
                function=format_formula_string_to_array,
                return_dtype=pl.List(pl.Int64)
            ).list.to_array(width=num_elements)
    ).alias('fragment_formulas_array')
    ).drop(['fragments', 'losses','treeScore','fragment_formulas'])

    return frag_trees

def _get_sirius_port():
    port_file_path = Path.home().joinpath('.sirius').joinpath('sirius-6.port')
    try:
        with open(port_file_path, 'r') as file:
            port = file.read().strip()
            return port
    except FileNotFoundError:
        print(f"Port file not found at {port_file_path}")
        return None

def _get_sirius_base_url() -> str:
    sirius_port: str | None = _get_sirius_port()
    if sirius_port:
        sirius_base_url: str = f'http://127.0.0.1:{sirius_port}/api/projects'
    else:
        raise RuntimeError("Sirius port could not be determined.")
    return sirius_base_url


def get_all_info(
    sirius_project_name: str,
    only_with_msms: bool=True,
    max_formulas_per_feature:int=1,
    discard_failed_annotations: bool=True
    ) -> pl.DataFrame:
    """
    Get all features with their formulas and frag trees for a given project name from the Sirius API.
    Args:
        project_name (str): The name of the Sirius project.
        only_with_msms (bool): If True, only features with MS/MS data are returned.
        max_formulas_per_feature (int): Maximum number of formulas per feature to return.
        discard
    Returns:
        pl.DataFrame: A Polars DataFrame containing all features with their formulas and frag trees.
    """
    ## some notes:
    # this can be done more efficiently by getting all formulas and frag trees in one go, or by getting only those that pass the filter. however, for ~1000 features, this takes ~4 seconds, which is acceptable.
    compounds: pl.DataFrame = get_all_compounds(sirius_project_name).select(
        [
            'alignedFeatureId',
            'externalFeatureId',
            'ionMass',
            'charge',
            'detectedAdducts',
            'hasMsMs',
            'rtApexSeconds'
        ]
    )
    if only_with_msms:
        compounds = compounds.filter(pl.col('hasMsMs'))

    formulas: pl.DataFrame = get_all_formulas(sirius_project_name).select(
        pl.col('formulaId'),
        pl.col('molecularFormula'),
        pl.col('molecularFormula_array'),
        pl.col('rank'),
        pl.col('siriusScoreNormalized'),
        pl.col('siriusScore'),
        pl.col('isotopeScore'),
        pl.col('treeScore')
    ).filter(
        pl.col('rank').le(max_formulas_per_feature)
    )

    cleaned_spectra: pl.DataFrame = get_clean_spectra(sirius_project_name).drop(
        'molecularFormula'
    )


    combined_info_per_formula: pl.DataFrame = cleaned_spectra.join(
        formulas,
        on='formulaId',
        how='left',
    )

    combined_info_per_compound: pl.DataFrame = compounds.join(
        combined_info_per_formula,
        left_on='alignedFeatureId',
        right_on='featureId',
        how='left'
    )
    if discard_failed_annotations:
        combined_info_per_compound = combined_info_per_compound.filter(
            pl.col('siriusScore').is_not_null()
        )

    return combined_info_per_compound


if __name__ == '__main__':
    start: float = time()
    sirius_project_name = 'pfas_low_conc'
    sirius_base_url: str = _get_sirius_base_url()
    print(sirius_base_url)
    # compounds = get_all_compounds(sirius_project_name)
    # print(compounds)
    # print(compounds.schema)
    all_info = get_all_info(sirius_project_name)
    print(all_info)
    print(all_info.schema)
    
    print('time:', time()-start)
