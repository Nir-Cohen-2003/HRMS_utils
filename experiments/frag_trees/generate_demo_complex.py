"""
Generate a more complex synthetic MGF file with proper isobaric pairs.
Uses validated isobaric pairs from the search.
"""

from pathlib import Path
from hrms_utils.formula_annotation.element_table import ELEMENT_SYMBOLS, ELEMENT_MASSES

# Element mass lookup
def emass(symbol):
    return ELEMENT_MASSES[ELEMENT_SYMBOLS.index(symbol)]

def calc_formula_mass(formula_dict):
    return sum(formula_dict[e] * emass(e) for e in formula_dict)

def formula_to_string(formula_dict):
    order = ['C', 'H', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    parts = []
    for e in order:
        if e in formula_dict and formula_dict[e] > 0:
            if formula_dict[e] == 1:
                parts.append(e)
            else:
                parts.append(f"{e}{formula_dict[e]}")
    return "".join(parts)

def is_superset(parent, child):
    return all(parent.get(e, 0) >= child.get(e, 0) for e in set(parent) | set(child))

# Use validated isobaric pairs:
# C7H2N (100.0187) vs C3H5N2P (100.0190) - 3.1 ppm
# C8H5 (101.0391) vs C4H8NP (101.0394) - 3.1 ppm  (but P is not in our element set for child, let's use C4H8NP)
# Actually let's use C7H2N / C3H5N2P as the main isobaric pair

# Compound: C12H15N3O2P
P = {"C": 12, "H": 15, "N": 3, "O": 2, "P": 1}

# MS2 fragments
F1 = {"C": 7, "H": 2, "N": 1}        # m/z 100.0187 - isobaric with F2!
F2 = {"C": 3, "H": 5, "N": 2, "P": 1}  # m/z 100.0190 - isobaric with F1!
F3 = {"C": 6, "H": 5, "N": 1, "O": 1}  # m/z 119.0371
F4 = {"C": 5, "H": 3, "N": 1, "O": 1}  # m/z 93.0215
F5 = {"C": 4, "H": 3, "N": 1}        # m/z 65.0265
F6 = {"C": 3, "H": 1, "N": 1}        # m/z 51.0109

# MS3 from F1 (C7H2N)
F1a = {"C": 5, "H": 3, "N": 1, "O": 1}  # m/z 93.0215 - same as F4
F1b = {"C": 4, "H": 3, "N": 1}        # m/z 65.0265 - same as F5
F1c = {"C": 3, "H": 1, "N": 1}        # m/z 51.0109 - same as F6

# MS3 from F2 (C3H5N2P)
F2a = {"C": 2, "H": 3, "N": 1, "P": 1}  # m/z 72.9842
F2b = {"C": 2, "H": 1, "N": 1}        # m/z 39.0109

# MS3 from F3 (C6H5NO)
F3a = {"C": 4, "H": 3, "N": 1}        # m/z 65.0265 - same as F5
F3b = {"C": 3, "H": 1, "N": 1}        # m/z 51.0109 - same as F6

# MS4 from F1a (C5H3NO) -> but F1a is same as F4, so MS4 from F4
F4a = {"C": 3, "H": 1, "N": 1}        # m/z 51.0109 - same as F6

# Calculate masses
masses = {}
for name, formula in [
    ("P", P), ("F1", F1), ("F2", F2), ("F3", F3), ("F4", F4), ("F5", F5), ("F6", F6),
    ("F1a", F1a), ("F1b", F1b), ("F1c", F1c),
    ("F2a", F2a), ("F2b", F2b),
    ("F3a", F3a), ("F3b", F3b),
    ("F4a", F4a),
]:
    masses[name] = calc_formula_mass(formula)

print("Masses:")
for name in ["P", "F1", "F2", "F3", "F4", "F5", "F6", "F1a", "F1b", "F1c", "F2a", "F2b", "F3a", "F3b", "F4a"]:
    formula = eval(name)
    print(f"  {name}: {formula_to_string(formula)} = {masses[name]:.4f} Da")

# Check isobaric pairs
print(f"\nF1 vs F2 mass diff: {abs(masses['F2'] - masses['F1']) / masses['F1'] * 1e6:.2f} ppm")
print(f"F4 vs F1a mass diff: {abs(masses['F1a'] - masses['F4']) / masses['F4'] * 1e6:.2f} ppm")
print(f"F5 vs F1b mass diff: {abs(masses['F1b'] - masses['F5']) / masses['F5'] * 1e6:.2f} ppm")
print(f"F6 vs F1c mass diff: {abs(masses['F1c'] - masses['F6']) / masses['F6'] * 1e6:.2f} ppm")

# Verify ALL superset relationships
print("\nSuperset checks:")
all_formulas = {
    "P": P, "F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6,
    "F1a": F1a, "F1b": F1b, "F1c": F1c,
    "F2a": F2a, "F2b": F2b,
    "F3a": F3a, "F3b": F3b,
    "F4a": F4a,
}
for p_name, p_form in all_formulas.items():
    for c_name, c_form in all_formulas.items():
        if p_name != c_name and is_superset(p_form, c_form):
            print(f"  {p_name} contains {c_name}")

# Build MGF content
mgf_content = f"""BEGIN IONS
NAME=DemoComplex_MS2
DESCRIPTION=Synthetic complex demo for mass-based fragmentation trees
EXACTMASS={masses['P'] + emass('H'):.5f}
FORMULA={formula_to_string({"C": 12, "H": 16, "N": 3, "O": 2, "P": 1})}
INCHI=InChI=1S/C12H16N3O2P/c1-9-5-10(2)14(11(3)6-9)12(16)13-7-8-15(4)17/h5-7H,1-4,8H2,(H,13,16)
INCHIAUX=DEMOCOMPLEX123-UHFFFAOYSA-N
SMILES=CC1=CC(C)=N(C(C)C1)C(=O)NCCN(C)P
FEATURE_ID=demo_complex_ms2
MSLEVEL=2
RTINSECONDS=150.0
ADDUCT=[M-H]-
PEPMASS={masses['P']:.5f}
CHARGE=-1
FEATURE_MS1_HEIGHT=1.0E6
SPECTYPE=MS2
COLLISION_ENERGY=[25.0]
FRAGMENTATION_METHOD=HCD
ISOLATION_WINDOW=1.0
ACQUISITION=Commercial
INSTRUMENT_TYPE=Orbitrap
SOURCE_INSTRUMENT=Orbitrap ID-X
IMS_TYPE=none
ION_SOURCE=ESI
IONMODE=Negative
PI=Demo
DATACOLLECTOR=Demo
DATASET_ID=MSV000000000
USI=[mzspec:MSV000000000:demo_complex:scan:1]
SCANS=1
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=6
{masses['F1']:.5f} 100.0
{masses['F2']:.5f} 90.0
{masses['F3']:.5f} 85.0
{masses['F4']:.5f} 70.0
{masses['F5']:.5f} 50.0
{masses['F6']:.5f} 30.0
END IONS

BEGIN IONS
NAME=DemoComplex_MS3_F1
DESCRIPTION=Synthetic demo MS3 from F1
EXACTMASS={masses['P'] + emass('H'):.5f}
FORMULA={formula_to_string({"C": 12, "H": 16, "N": 3, "O": 2, "P": 1})}
INCHI=InChI=1S/C12H16N3O2P/c1-9-5-10(2)14(11(3)6-9)12(16)13-7-8-15(4)17/h5-7H,1-4,8H2,(H,13,16)
INCHIAUX=DEMOCOMPLEX123-UHFFFAOYSA-N
SMILES=CC1=CC(C)=N(C(C)C1)C(=O)NCCN(C)P
FEATURE_ID=demo_complex_ms3_f1
MSLEVEL=3
RTINSECONDS=150.0
ADDUCT=[M-H]-
PEPMASS={masses['F1']:.5f}
CHARGE=-1
FEATURE_MS1_HEIGHT=1.0E6
SPECTYPE=MS3
COLLISION_ENERGY=[35.0]
FRAGMENTATION_METHOD=HCD
ISOLATION_WINDOW=1.0
ACQUISITION=Commercial
INSTRUMENT_TYPE=Orbitrap
SOURCE_INSTRUMENT=Orbitrap ID-X
IMS_TYPE=none
ION_SOURCE=ESI
IONMODE=Negative
PI=Demo
DATACOLLECTOR=Demo
DATASET_ID=MSV000000000
USI=[mzspec:MSV000000000:demo_complex:scan:2]
SCANS=2
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=3
{masses['F1a']:.5f} 100.0
{masses['F1b']:.5f} 60.0
{masses['F1c']:.5f} 40.0
END IONS

BEGIN IONS
NAME=DemoComplex_MS3_F2
DESCRIPTION=Synthetic demo MS3 from F2
EXACTMASS={masses['P'] + emass('H'):.5f}
FORMULA={formula_to_string({"C": 12, "H": 16, "N": 3, "O": 2, "P": 1})}
INCHI=InChI=1S/C12H16N3O2P/c1-9-5-10(2)14(11(3)6-9)12(16)13-7-8-15(4)17/h5-7H,1-4,8H2,(H,13,16)
INCHIAUX=DEMOCOMPLEX123-UHFFFAOYSA-N
SMILES=CC1=CC(C)=N(C(C)C1)C(=O)NCCN(C)P
FEATURE_ID=demo_complex_ms3_f2
MSLEVEL=3
RTINSECONDS=150.0
ADDUCT=[M-H]-
PEPMASS={masses['F2']:.5f}
CHARGE=-1
FEATURE_MS1_HEIGHT=1.0E6
SPECTYPE=MS3
COLLISION_ENERGY=[40.0]
FRAGMENTATION_METHOD=HCD
ISOLATION_WINDOW=1.0
ACQUISITION=Commercial
INSTRUMENT_TYPE=Orbitrap
SOURCE_INSTRUMENT=Orbitrap ID-X
IMS_TYPE=none
ION_SOURCE=ESI
IONMODE=Negative
PI=Demo
DATACOLLECTOR=Demo
DATASET_ID=MSV000000000
USI=[mzspec:MSV000000000:demo_complex:scan:3]
SCANS=3
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=2
{masses['F2a']:.5f} 100.0
{masses['F2b']:.5f} 50.0
END IONS

BEGIN IONS
NAME=DemoComplex_MS3_F3
DESCRIPTION=Synthetic demo MS3 from F3
EXACTMASS={masses['P'] + emass('H'):.5f}
FORMULA={formula_to_string({"C": 12, "H": 16, "N": 3, "O": 2, "P": 1})}
INCHI=InChI=1S/C12H16N3O2P/c1-9-5-10(2)14(11(3)6-9)12(16)13-7-8-15(4)17/h5-7H,1-4,8H2,(H,13,16)
INCHIAUX=DEMOCOMPLEX123-UHFFFAOYSA-N
SMILES=CC1=CC(C)=N(C(C)C1)C(=O)NCCN(C)P
FEATURE_ID=demo_complex_ms3_f3
MSLEVEL=3
RTINSECONDS=150.0
ADDUCT=[M-H]-
PEPMASS={masses['F3']:.5f}
CHARGE=-1
FEATURE_MS1_HEIGHT=1.0E6
SPECTYPE=MS3
COLLISION_ENERGY=[45.0]
FRAGMENTATION_METHOD=HCD
ISOLATION_WINDOW=1.0
ACQUISITION=Commercial
INSTRUMENT_TYPE=Orbitrap
SOURCE_INSTRUMENT=Orbitrap ID-X
IMS_TYPE=none
ION_SOURCE=ESI
IONMODE=Negative
PI=Demo
DATACOLLECTOR=Demo
DATASET_ID=MSV000000000
USI=[mzspec:MSV000000000:demo_complex:scan:4]
SCANS=4
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=2
{masses['F3a']:.5f} 100.0
{masses['F3b']:.5f} 50.0
END IONS

BEGIN IONS
NAME=DemoComplex_MS4_F4
DESCRIPTION=Synthetic demo MS4 from F4
EXACTMASS={masses['P'] + emass('H'):.5f}
FORMULA={formula_to_string({"C": 12, "H": 16, "N": 3, "O": 2, "P": 1})}
INCHI=InChI=1S/C12H16N3O2P/c1-9-5-10(2)14(11(3)6-9)12(16)13-7-8-15(4)17/h5-7H,1-4,8H2,(H,13,16)
INCHIAUX=DEMOCOMPLEX123-UHFFFAOYSA-N
SMILES=CC1=CC(C)=N(C(C)C1)C(=O)NCCN(C)P
FEATURE_ID=demo_complex_ms4_f4
MSLEVEL=4
RTINSECONDS=150.0
ADDUCT=[M-H]-
PEPMASS={masses['F4']:.5f}
CHARGE=-1
FEATURE_MS1_HEIGHT=1.0E6
SPECTYPE=MS4
COLLISION_ENERGY=[55.0]
FRAGMENTATION_METHOD=HCD
ISOLATION_WINDOW=1.0
ACQUISITION=Commercial
INSTRUMENT_TYPE=Orbitrap
SOURCE_INSTRUMENT=Orbitrap ID-X
IMS_TYPE=none
ION_SOURCE=ESI
IONMODE=Negative
PI=Demo
DATACOLLECTOR=Demo
DATASET_ID=MSV000000000
USI=[mzspec:MSV000000000:demo_complex:scan:5]
SCANS=5
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=1
{masses['F4a']:.5f} 100.0
END IONS
"""

output_path = Path("/home/ser/dev/HRMS_utils/demo_complex.mgf")
output_path.write_text(mgf_content)
print(f"\nSaved complex synthetic MGF to {output_path}")
print(f"File size: {output_path.stat().st_size} bytes")
