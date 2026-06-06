"""
Generate a synthetic MGF file to demonstrate mass-based vs formula-based
fragmentation tree building with MSn pruning.

Design:
- Compound with molecular formula C10H13N5OP (contains P for isobaric demo)
- MS2: fragments C7H2N (m/z 100.0187), C3H5N2P (m/z 100.0190, isobaric!), C3H2N (m/z 42.0344), C2HN (m/z 39.0215)
- MS3 (precursor C7H2N): fragments C3H2N, C2HN
- MS3 (precursor C3H5N2P): fragments C3H2N, C2HN
- MS4 (precursor C3H2N): fragment C2HN

The isobaric pair C7H2N / C3H5N2P (3.1 ppm difference) will merge in mass-based
but stay separate in formula-based. The MSn truncation then prunes edges
differently.
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

# Define formulas
P = {"C": 10, "H": 13, "N": 5, "O": 1, "P": 1}  # [M-H]- precursor
F1 = {"C": 7, "H": 2, "N": 1}  # Fragment from MS2
F2 = {"C": 3, "H": 5, "N": 2, "P": 1}  # Isobaric fragment from MS3
F3 = {"C": 3, "H": 2, "N": 1}  # Common child
F4 = {"C": 2, "H": 1, "N": 1}  # Common child

mass_P = calc_formula_mass(P)
mass_F1 = calc_formula_mass(F1)
mass_F2 = calc_formula_mass(F2)
mass_F3 = calc_formula_mass(F3)
mass_F4 = calc_formula_mass(F4)

print(f"Precursor [M-H]-: {formula_to_string(P)} = {mass_P:.4f} Da")
print(f"F1: {formula_to_string(F1)} = {mass_F1:.4f} Da")
print(f"F2: {formula_to_string(F2)} = {mass_F2:.4f} Da")
print(f"Mass difference F1 vs F2: {abs(mass_F2 - mass_F1) / mass_F1 * 1e6:.2f} ppm")
print(f"F3: {formula_to_string(F3)} = {mass_F3:.4f} Da")
print(f"F4: {formula_to_string(F4)} = {mass_F4:.4f} Da")

# Verify superset relationships
print("\nSuperset checks:")
for name, parent, child in [
    ("P contains F1", P, F1),
    ("P contains F2", P, F2),
    ("P contains F3", P, F3),
    ("P contains F4", P, F4),
    ("F1 contains F3", F1, F3),
    ("F1 contains F4", F1, F4),
    ("F2 contains F3", F2, F3),
    ("F2 contains F4", F2, F4),
    ("F3 contains F4", F3, F4),
]:
    ok = all(parent.get(e, 0) >= child.get(e, 0) for e in set(parent) | set(child))
    print(f"  {name}: {ok}")

# Build MGF content
mgf_content = f"""BEGIN IONS
NAME=DemoCompound_MS2
DESCRIPTION=Synthetic demo for mass-based fragmentation trees
EXACTMASS={mass_P + emass('H'):.5f}
FORMULA={formula_to_string({"C": 10, "H": 14, "N": 5, "O": 1, "P": 1})}
INCHI=InChI=1S/C10H14N5OP/c11-6-1-2-7(12)10(16)13-8(14)9(15)5-3-4/h1-2,5H,3-4H2,(H2,11,12,13,14,15,16)
INCHIAUX=DEMOKEY123456-UHFFFAOYSA-N
SMILES=O=P1(N=C(N)N=C(N)N1)C=CC=C
FEATURE_ID=demo_ms2
MSLEVEL=2
RTINSECONDS=100.0
ADDUCT=[M-H]-
PEPMASS={mass_P:.5f}
CHARGE=-1
FEATURE_MS1_HEIGHT=1.0E6
SPECTYPE=MS2
COLLISION_ENERGY=[20.0]
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
USI=[mzspec:MSV000000000:demo:scan:1]
SCANS=1
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=4
{mass_F1:.5f} 100.0
{mass_F2:.5f} 80.0
{mass_F3:.5f} 60.0
{mass_F4:.5f} 40.0
END IONS

BEGIN IONS
NAME=DemoCompound_MS3_F1
DESCRIPTION=Synthetic demo MS3 from F1
EXACTMASS={mass_P + emass('H'):.5f}
FORMULA={formula_to_string({"C": 10, "H": 14, "N": 5, "O": 1, "P": 1})}
INCHI=InChI=1S/C10H14N5OP/c11-6-1-2-7(12)10(16)13-8(14)9(15)5-3-4/h1-2,5H,3-4H2,(H2,11,12,13,14,15,16)
INCHIAUX=DEMOKEY123456-UHFFFAOYSA-N
SMILES=O=P1(N=C(N)N=C(N)N1)C=CC=C
FEATURE_ID=demo_ms3_f1
MSLEVEL=3
RTINSECONDS=100.0
ADDUCT=[M-H]-
PEPMASS={mass_F1:.5f}
CHARGE=-1
FEATURE_MS1_HEIGHT=1.0E6
SPECTYPE=MS3
COLLISION_ENERGY=[30.0]
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
USI=[mzspec:MSV000000000:demo:scan:2]
SCANS=2
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=2
{mass_F3:.5f} 100.0
{mass_F4:.5f} 50.0
END IONS

BEGIN IONS
NAME=DemoCompound_MS3_F2
DESCRIPTION=Synthetic demo MS3 from F2
EXACTMASS={mass_P + emass('H'):.5f}
FORMULA={formula_to_string({"C": 10, "H": 14, "N": 5, "O": 1, "P": 1})}
INCHI=InChI=1S/C10H14N5OP/c11-6-1-2-7(12)10(16)13-8(14)9(15)5-3-4/h1-2,5H,3-4H2,(H2,11,12,13,14,15,16)
INCHIAUX=DEMOKEY123456-UHFFFAOYSA-N
SMILES=O=P1(N=C(N)N=C(N)N1)C=CC=C
FEATURE_ID=demo_ms3_f2
MSLEVEL=3
RTINSECONDS=100.0
ADDUCT=[M-H]-
PEPMASS={mass_F2:.5f}
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
USI=[mzspec:MSV000000000:demo:scan:3]
SCANS=3
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=2
{mass_F3:.5f} 100.0
{mass_F4:.5f} 50.0
END IONS

BEGIN IONS
NAME=DemoCompound_MS4_F3
DESCRIPTION=Synthetic demo MS4 from F3
EXACTMASS={mass_P + emass('H'):.5f}
FORMULA={formula_to_string({"C": 10, "H": 14, "N": 5, "O": 1, "P": 1})}
INCHI=InChI=1S/C10H14N5OP/c11-6-1-2-7(12)10(16)13-8(14)9(15)5-3-4/h1-2,5H,3-4H2,(H2,11,12,13,14,15,16)
INCHIAUX=DEMOKEY123456-UHFFFAOYSA-N
SMILES=O=P1(N=C(N)N=C(N)N1)C=CC=C
FEATURE_ID=demo_ms4_f3
MSLEVEL=4
RTINSECONDS=100.0
ADDUCT=[M-H]-
PEPMASS={mass_F3:.5f}
CHARGE=-1
FEATURE_MS1_HEIGHT=1.0E6
SPECTYPE=MS4
COLLISION_ENERGY=[50.0]
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
USI=[mzspec:MSV000000000:demo:scan:4]
SCANS=4
PRECURSOR_PURITY=1.0
QUALITY_CHIMERIC=PASSED
QUALITY_EXPLAINED_INTENSITY=0.95
QUALITY_EXPLAINED_SIGNALS=0.90
Num peaks=1
{mass_F4:.5f} 100.0
END IONS
"""

output_path = Path("/home/ser/dev/HRMS_utils/demo_isobaric.mgf")
output_path.write_text(mgf_content)
print(f"\nSaved synthetic MGF to {output_path}")
print(f"File size: {output_path.stat().st_size} bytes")
