from rdkit import Chem
from rdkit.Chem import rdmolops, rdChemReactions, MolStandardize




def eliminate_water(smiles:str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    rdmolops.RemoveStereochemistry(mol) # in place operation

    eliminateable_hydroxide = Chem.MolFromSmarts("[C;h]-[C]-[OH]")
    can_eliminate = mol.GetSubstructMatch(eliminateable_hydroxide)
    if not can_eliminate:
        # print(f"Cannot eliminate water from {smiles}")
        return None
    water_elimiantion_smarts = "[C;h:1]-[C:2]-[OH]>>[C:1]=[C:2]"
    # water_elimiantion_smarts = "[C!H0:1][C:2][O:3]>>[C:1]=[C:2].[O:3]"  # both work
    water_elimination = rdChemReactions.ReactionFromSmarts(water_elimiantion_smarts)


    mol = Chem.MolFromSmiles(smiles)
    res = water_elimination.RunReactants((mol,))

    if len(res) == 0:
        # print(f"No product found for {smiles}")
        return None
    elif len(res) > 1:
        # print(f"More than one product found for {smiles}")
        return "multiple products"
    elif len(res)==1:
        product = res[0][0]
        rdmolops.RemoveStereochemistry(product)
        MolStandardize.rdMolStandardize.CanonicalTautomer(product)
        return Chem.MolToSmiles(product)


def elimiante_ammonia(smiles:str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    rdmolops.RemoveStereochemistry(mol) # in place operation

    eliminateable_NH2 = Chem.MolFromSmarts("[C;h]-[C]-[NH2]")
    can_eliminate = mol.GetSubstructMatch(eliminateable_NH2)
    if not can_eliminate:
        # print(f"Cannot eliminate water from {smiles}")
        return None
    ammonia_elimination_smarts = "[C;h:1]-[C:2]-[NH2]>>[C:1]=[C:2]"
    # water_elimiantion_smarts = "[C!H0:1][C:2][O:3]>>[C:1]=[C:2].[O:3]"  # both work
    ammonia_elimiantion = rdChemReactions.ReactionFromSmarts(ammonia_elimination_smarts)


    mol = Chem.MolFromSmiles(smiles)
    res = ammonia_elimiantion.RunReactants((mol,))

    if len(res) == 0:
        # print(f"No product found for {smiles}")
        return None
    elif len(res) > 1:
        # print(f"More than one product found for {smiles}")
        return "multiple products"
    elif len(res)==1:
        product = res[0][0]
        rdmolops.RemoveStereochemistry(product)
        MolStandardize.rdMolStandardize.CanonicalTautomer(product)
        return Chem.MolToSmiles(product)

if __name__ == "__main__":
    smiles = ["OCC", "Oc1ccccc1", "CC(N)C(O)c1ccccc1","C(O)CC(O)c1ccccc1"]
    reaction_smarts = "[C;h:1]-[C:2]-[OH]>>[C:1]=[C:2]"
    water_elimination = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    for smile in smiles:  
        print(f"Original: {smile}")
        modified_smile = eliminate_water(smile)
        print(f"Modified: {modified_smile}")
        