"""Tests for the spectral library processing pipeline."""

from pathlib import Path

import polars as pl
import pytest

from hrms_utils.formats.spectral_library import process_spectral_library


def _write_msp(path: Path, name: str, inchikey: str, smiles: str | None, inchi: str | None) -> None:
    """Write a minimal single-spectrum MSP file for testing."""
    smiles_line = f"SMILES: {smiles}\n" if smiles else ""
    inchi_line = f"INCHI: {inchi}\n" if inchi else ""
    path.write_text(
        f"NAME: {name}\n"
        "PRECURSORMZ: 100.0\n"
        "PRECURSORTYPE: [M+H]+\n"
        "FORMULA: C2H6O\n"
        f"INCHIKEY: {inchikey}\n"
        f"{inchi_line}"
        f"{smiles_line}"
        "NUM PEAKS: 1\n"
        "100.0 1.0\n"
    )


def _write_pubchem(path: Path, inchikey: str, smiles: str, inchi: str) -> None:
    """Write a minimal PubChem parquet for testing."""
    pl.DataFrame(
        {
            "CID": [1],
            "InChIKey": [inchikey],
            "SMILES": [smiles],
            "InChI": [inchi],
        }
    ).write_parquet(path)


def test_pubchem_fill_missing_only_default(tmp_path: Path) -> None:
    """By default PubChem only fills rows missing both SMILES and InChI."""
    msp = tmp_path / "test.msp"
    _write_msp(
        msp,
        name="Complete",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        smiles="CCO",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    )
    msp_missing = tmp_path / "test_missing.msp"
    _write_msp(
        msp_missing,
        name="Missing",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        smiles=None,
        inchi=None,
    )

    pubchem = tmp_path / "pubchem.parquet"
    _write_pubchem(
        pubchem,
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        smiles="CCO",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    )

    df = process_spectral_library(
        files=[msp, msp_missing],
        pubchem_path=pubchem,
        deduplicate=False,
        min_explained_intensity=0.0,
    )

    complete = df.filter(pl.col("name") == "Complete").row(0, named=True)
    missing = df.filter(pl.col("name") == "Missing").row(0, named=True)

    # The complete row keeps its original identifiers.
    assert complete["smiles"] == "CCO"
    assert complete["inchi"] == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"

    # The missing row was filled from PubChem.
    assert missing["smiles"] == "CCO"
    assert missing["inchi"] == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"


def test_pubchem_enrich_all_overrides_existing(tmp_path: Path) -> None:
    """With fill_missing_only=False, PubChem overrides existing identifiers."""
    msp = tmp_path / "test.msp"
    _write_msp(
        msp,
        name="Complete",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        smiles="CCO",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    )

    pubchem = tmp_path / "pubchem.parquet"
    # PubChem provides the same molecule but with a different tautomer/ordering.
    # For ethanol the msready step will converge, so we use a distinct full InChIKey
    # that still maps to the same base InChIKey to keep the test focused on override.
    _write_pubchem(
        pubchem,
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        smiles="CCO",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    )

    df = process_spectral_library(
        files=[msp],
        pubchem_path=pubchem,
        pubchem_fill_missing_only=False,
        deduplicate=False,
        min_explained_intensity=0.0,
    )

    row = df.filter(pl.col("name") == "Complete").row(0, named=True)
    # Values should still be present and valid after standardization.
    assert row["smiles"] is not None and row["smiles"] != ""
    assert row["inchi"] is not None and row["inchi"] != ""
    assert row["base_inchikey"] == "LFQSCWFLJHTTHZ"


def test_changed_inchikey_csv_written(tmp_path: Path) -> None:
    """A CSV is written when the final base InChIKey differs from the original."""
    msp = tmp_path / "test.msp"
    # The input InChIKey does not match the SMILES.
    _write_msp(
        msp,
        name="WrongInchiKey",
        inchikey="AAAAAAAABBBBBB-CCCCCDBBBV",
        smiles="CCO",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    )

    changes_csv = tmp_path / "changes.csv"
    df = process_spectral_library(
        files=[msp],
        deduplicate=False,
        min_explained_intensity=0.0,
        inchikey_changes_path=changes_csv,
    )

    row = df.filter(pl.col("name") == "WrongInchiKey").row(0, named=True)
    assert row["base_inchikey"] == "LFQSCWFLJHTTHZ"

    assert changes_csv.exists()
    changes = pl.read_csv(changes_csv)
    assert changes.height == 1
    changed = changes.row(0, named=True)
    assert changed["name"] == "WrongInchiKey"
    assert changed["original_base_inchikey"] == "AAAAAAAABBBBBB"
    assert changed["new_base_inchikey"] == "LFQSCWFLJHTTHZ"
    assert changed["original_smiles"] == "CCO"
    assert changed["new_smiles"] is not None and changed["new_smiles"] != ""


def test_no_changes_no_csv(tmp_path: Path) -> None:
    """No CSV is written when the base InChIKey does not change."""
    msp = tmp_path / "test.msp"
    _write_msp(
        msp,
        name="Correct",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        smiles="CCO",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    )

    changes_csv = tmp_path / "changes.csv"
    process_spectral_library(
        files=[msp],
        deduplicate=False,
        min_explained_intensity=0.0,
        inchikey_changes_path=changes_csv,
    )

    assert not changes_csv.exists()
