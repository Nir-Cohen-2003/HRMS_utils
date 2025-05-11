import requests
from time import time
import polars as pl
from aiohttp import ClientSession
import asyncio
from MS_utils.formula import format_formula_string_to_array
from pathlib import Path

num_elements = 15

def get_all_compounds(project_name):
    sirius_base_url = _get_sirius_base_url(project_name)
    url = sirius_base_url + '/' + project_name + '/aligned-features'
    response = requests.get(url)
    features = response.json()
    features = pl.DataFrame(features)
    return features

async def _get_all_formulas(project_name):
    sirius_base_url = _get_sirius_base_url(project_name)
    url = sirius_base_url + '/' + project_name + '/aligned-features'
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

    formulas = pl.DataFrame(formulas)

    return formulas

async def _get_all_frag_trees(project_name: str) -> pl.DataFrame:
    features = get_all_compounds(project_name)
    feature_ids = features.select('alignedFeatureId').to_numpy().flatten()
    all_frag_trees = []
    tasks = []
    sirius_base_url = _get_sirius_base_url(project_name)
    base_url= sirius_base_url + '/' + project_name + '/aligned-features/'

    async with ClientSession() as session:
        for feature_id in feature_ids:
            tasks.append(asyncio.create_task(_get_frag_trees_for_feature(session, base_url, feature_id)))
        results = await asyncio.gather(*tasks)
        for frag_trees in results:
            if frag_trees:
                all_frag_trees.extend(frag_trees)
    frag_trees = pl.DataFrame(all_frag_trees)
    return frag_trees

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
    url =  base_url+ feature_id + '/formulas'
    async with session.get(url) as response:
        if response.status != 200: # Check if the response is successful
            return []
        formulas = await response.json()
        if not formulas: # Check if there are any formulas
            return []
        frag_trees = []
        for formula in formulas:
            frag_tree_url = url + '/' + formula['formulaId'] + '/fragtree'
            async with session.get(frag_tree_url) as frag_tree_response: 
                if frag_tree_response.status != 200: # if it failed to get the frag tree, skip
                    continue
                frag_tree = await frag_tree_response.json()
                frag_tree['formulaId'] = formula['formulaId']
                frag_tree['featureId'] = feature_id
                frag_trees.append(frag_tree)
    
    return frag_trees


def get_clean_spectra(sirius_project_name: str) -> pl.DataFrame:

    frag_trees = asyncio.run(_get_all_frag_trees(sirius_project_name))
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

def _get_sirius_base_url(sirius_project_name:str) -> str:
    sirius_port = _get_sirius_port()
    if sirius_port:
        sirius_base_url = f'http://127.0.0.1:{sirius_port}/api/projects'
    else:
        raise RuntimeError("Sirius port could not be determined.")
    return sirius_base_url


if __name__ == '__main__':
    start = time()
    sirius_project_name = '10ppb'
    sirius_base_url = _get_sirius_base_url(sirius_project_name)
    print(sirius_base_url)
    # compounds = get_all_compounds(sirius_project_name)
    # print(compounds)
    # print(compounds.schema)

    compounds = get_all_compounds(sirius_project_name)
    print(compounds)
    print(compounds.schema)

    formulas = asyncio.run(_get_all_formulas(sirius_project_name))
    print(formulas)
    print(formulas.schema)

    cleaned_spectra = get_clean_spectra(sirius_project_name)
    print(cleaned_spectra)
    print(cleaned_spectra.schema)

    print('time:', time()-start)
