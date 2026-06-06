"""
Generate a clean synthetic MGF demonstrating mass-based merging
of the SAME fragment at slightly different masses, with proper MSn pruning.

Design:
- Compound: C8H10N2O2
- Precursor [M-H]-: m/z 165.0771

MS2 fragments (4 fragments):
  F1: C6H6N2 at m/z 106.0531
  F2: C5H5NO at m/z 107.0371
  F3: C4H4N2 at m/z 80.0374
  F4: C3H3N at m/z 53.0265

MS3 from F1 (precursor m/z 106.0535 - slightly different from MS2!):
  F1a: C4H4N2 at m/z 80.0374 - same as F3
  F1b: C3H3N at m/z 53.0265 - same as F4

MS4 from F3 (precursor m/z 80.0374):
  F3a: C3H3N at m/z 53.0265 - same as F4

The key: F1 appears at 106.0531 in MS2 and 106.0535 in MS3 (4 ppm difference).
At 0 ppm tolerance, these are separate nodes.
At 5 ppm tolerance, they merge into one node.
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

# Compound: C8H10N2O2
P = {"C": 8, "H": 10, "N": 2, "O": 2}

# MS2 fragments
F1 = {"C": 6, "H": 6, "N": 2}      # m/z 106.0531
F2 = {"C": 5, "H": 5, "N": 1, "O": 1}  # m/z 107.0371
F3 = {"C": 4, "H": 4, "N": 2}      # m/z 80.0374
F4 = {"C": 3, "H": 3, "N": 1}      # m/z 53.0265

# MS3 from F1 (slightly different mass to simulate measurement error)
F1_ms3_mass = 106.0535  # 4 ppm higher than true mass

# MS3 fragments from F1
F1a = {"C": 4, "H": 4, "N": 2}      # same as F3
F1b = {"C": 3, "H": 3, "N": 1}      # same as F4

# MS4 from F3
F3a = {"C": 3, "H": 3, "N": 1}      # same as F4

# Calculate masses
masses = {}
for name, formula in [
    ("P", P), ("F1", F1), ("F2", F2), ("F3", F3), ("F4", F4),
    ("F1a", F1a), ("F1b", F1b), ("F3a", F3a),
]:
    masses[name] = calc_formula_mass(formula)

print("True masses:")
for name in ["P", "F1", "F2", "F3", "F4", "F1a", "F1b", "F3a"]:
    formula = eval(name)
    print(f"  {name}: {formula_to_string(formula)} = {masses[name]:.4f} Da")

print(f"\nF1 MS2 vs MS3 mass diff: {abs(F1_ms3_mass - masses['F1']) / masses['F1'] * 1e6:.2f} ppm")

# Verify superset relationships
print("\nSuperset checks:")
checks = [
    ("P contains F1", P, F1), ("P contains F2", P, F2),
    ("P contains F3", P, F3), ("P contains F4", P, F4),
    ("F1 contains F1a", F1, F1a), ("F1 contains F1b", F1, F1b),
    ("F3 contains F3a", F3, F3a),
]
for name, parent, child in checks:
    ok = is_superset(parent, child)
    print(f"  {name}: {ok}")

# Build MGF content
mgf_content = f"""BEGIN IONS
NAME=DemoClean_MS2
DESCRIPTION=Synthetic clean demo for mass-based fragmentation trees
EXACTMASS={masses['P'] + emass('H'):.5f}
FORMULA={formula_to_string({"C": 8, "H": 11, "N": 2, "O": 2})}
INCHI=InChI=1S/C8H11N2O2/c1-6(11)9-5-7(10)8(2)12/h5H,1-2H2,(H,9,11)
INCHIAUX=DEMOCLEAN12345-UHFFFAOYSA-N
SMILES=CC(=O)NCC(=O)C(C)O
FEATURE_ID=demo_clean_ms2
MSLEVEL=2
RTINSECONDS=120.0
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
USI=[mzspec:MSV000000000:demo_clean:scan:1]
SCANS=1
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=4
{masses['F1']:.5f} 100.0
{masses['F2']:.5f} 80.0
{masses['F3']:.5f} 60.0
{masses['F4']:.5f} 40.0
END IONS

BEGIN IONS
NAME=DemoClean_MS3_F1
DESCRIPTION=Synthetic demo MS3 from F1 with slightly different mass
EXACTMASS={masses['P'] + emass('H'):.5f}
FORMULA={formula_to_string({"C": 8, "H": 11, "N": 2, "O": 2})}
INCHI=InChI=1S/C8H11N2O2/c1-6(11)9-5-7(10)8(2)12/h5H,1-2H2,(H,9,11)
INCHIAUX=DEMOCLEAN12345-UHFFFAOYSA-N
SMILES=CC(=O)NCC(=O)C(C)O
FEATURE_ID=demo_clean_ms3_f1
MSLEVEL=3
RTINSECONDS=120.0
ADDUCT=[M-H]-
PEPMASS={F1_ms3_mass:.5f}
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
USI=[mzspec:MSV000000000:demo_clean:scan:2]
SCANS=2
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=2
{masses['F1a']:.5f} 100.0
{masses['F1b']:.5f} 50.0
END IONS

BEGIN IONS
NAME=DemoClean_MS4_F3
DESCRIPTION=Synthetic demo MS4 from F3
EXACTMASS={masses['P'] + emass('H'):.5f}
FORMULA={formula_to_string({"C": 8, "H": 11, "N": 2, "O": 2})}
INCHI=InChI=1S/C8H11N2O2/c1-6(11)9-5-7(10)8(2)12/h5H,1-2H2,(H,9,11)
INCHIAUX=DEMOCLEAN12345-UHFFFAOYSA-N
SMILES=CC(=O)NCC(=O)C(C)O
FEATURE_ID=demo_clean_ms4_f3
MSLEVEL=4
RTINSECONDS=120.0
ADDUCT=[M-H]-
PEPMASS={masses['F3']:.5f}
CHARGE=-1
FEATURE_MS1_HEIGHT=1.0E6
SPECTYPE=MS4
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
USI=[mzspec:MSV000000000:demo_clean:scan:3]
SCANS=3
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=1
{masses['F3a']:.5f} 100.0
END IONS
"""

output_path = Path("/home/ser/dev/HRMS_utils/demo_clean.mgf")
output_path.write_text(mgf_content)
print(f"\nSaved clean synthetic MGF to {output_path}")
print(f"File size: {output_path.stat().st_size} bytes")
