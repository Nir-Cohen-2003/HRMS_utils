"""Strict precursor parsing and conservative RDKit single-bond recovery."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import numpy.typing as npt
import polars as pl
from rdkit import Chem

from hrms_utils.formula_annotation.element_table import ELEMENT_SYMBOLS
from hrms_utils.formula_annotation.utils import formula_to_array
from parallel_rdkit import smiles_to_formula

FormulaVector = npt.NDArray[np.int64]  # shape: (12,)
FORMULA_WIDTH = 12

PERMITTED_LOSS_PREFIXES = {
    "M", "M+H", "M-H", "M+Na", "M+K", "M+NH4", "M+Cl", "M+FA-H", "M+CHO2"
}
ORDINARY_PRECURSOR_BODIES = PERMITTED_LOSS_PREFIXES | {"M-H2O-H"}
_ELEMENT_INDEX = {symbol: index for index, symbol in enumerate(ELEMENT_SYMBOLS)}
# Match two-letter symbols before their one-letter prefixes (Na/N, Cl/C, Br/B).
_TOKEN_PATTERN = re.compile(r"(?:Na|Cl|Br|H|C|N|O|F|P|S|K|I)(?:[1-9][0-9]*)?")
_FORMULA_PATTERN = _TOKEN_PATTERN


@dataclass(frozen=True)
class ValidatedParent:
    original_smiles: str | None
    molecule: Chem.Mol | None
    eligibility_reason: str | None
    canonical_smiles: str | None


@dataclass(frozen=True)
class FragmentComponent:
    bond_index: int
    atom_indices: tuple[int, int]
    component_index: int
    component_smiles: str
    companion_smiles: str
    formal_charge: int


@dataclass(frozen=True)
class ParentEnumeration:
    eligible_bond_count: int
    fragmentation_failure_bond_count: int
    radical_rejected_bond_count: int
    components: list[FragmentComponent]


def _normalise_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", "", text)


def _array(value: object) -> FormulaVector | None:
    if value is None:
        return None
    try:
        values = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if values.shape != (FORMULA_WIDTH,) or np.any(values < 0) or int(values.sum()) == 0:
        return None
    return values


def _parse_formula_literal(text: str) -> FormulaVector | None:
    """Validate one literal, then delegate element extraction to the shared utility."""
    if not text or text.startswith(("+", "-")) or text[0].isdigit():
        return None
    matches = list(_TOKEN_PATTERN.finditer(text))
    if not matches or "".join(match.group(0) for match in matches) != text:
        return None
    symbols = [re.match(r"[A-Z][a-z]?", match.group(0)).group(0) for match in matches]
    if len(symbols) != len(set(symbols)):
        return None
    checked = formula_to_array(pl.DataFrame({"_formula_literal": [text]}), "_formula_literal", "_formula_array")
    result = np.asarray(checked.get_column("_formula_array")[0], dtype=np.int64)
    if result.shape != (FORMULA_WIDTH,) or int(result.sum()) == 0 or np.any(result < 0):
        return None
    return result


def _validated_formula_lookup(values: pl.Series) -> pl.DataFrame:
    """Return arrays for distinct, grammar-validated formula literals."""
    normalized = list(dict.fromkeys(value for value in (_normalise_text(item) for item in values.to_list()) if value))
    valid_literals = []
    for value in normalized:
        matches = list(_TOKEN_PATTERN.finditer(value))
        symbols = [re.match(r"[A-Z][a-z]?", match.group(0)).group(0) for match in matches]
        if matches and "".join(match.group(0) for match in matches) == value and len(symbols) == len(set(symbols)):
            valid_literals.append(value)
    if not valid_literals:
        return pl.DataFrame(schema={"_formula_literal": pl.String, "_formula_array": pl.Array(pl.Int64, FORMULA_WIDTH)})
    checked = formula_to_array(pl.DataFrame({"_formula_literal": valid_literals}), "_formula_literal", "_formula_array")
    return checked.filter(pl.col("_formula_array").arr.sum() > 0)


def _formula_string(values: object) -> str | None:
    if values is None:
        return None
    try:
        vector = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (FORMULA_WIDTH,) or np.any(vector < 0):
        return None
    symbols = list(ELEMENT_SYMBOLS)
    if vector[1] > 0:
        ordered = ["C", "H"] + sorted(symbol for symbol in symbols if symbol not in {"C", "H"})
    else:
        ordered = sorted(symbols)
    return "".join(symbol + (str(int(vector[_ELEMENT_INDEX[symbol]])) if vector[_ELEMENT_INDEX[symbol]] != 1 else "") for symbol in ordered if vector[_ELEMENT_INDEX[symbol]] > 0) or "0"


def formula_array_to_string_expression(array_column: str) -> pl.Expr:
    # The fixed-width array is converted row-wise only for presentation; chemistry arithmetic remains vectorized.
    return pl.col(array_column).map_elements(_formula_string, return_dtype=pl.String)


def add_validated_formula_arrays(formulas: pl.DataFrame, formula_column: str, array_column: str) -> pl.DataFrame:
    normalized = formulas.with_columns(pl.col(formula_column).map_elements(_normalise_text, return_dtype=pl.String).alias("_formula_literal"))
    lookup = _validated_formula_lookup(normalized.get_column("_formula_literal"))
    base = normalized.drop(array_column) if array_column in normalized.columns else normalized
    return base.join(lookup.rename({"_formula_array": array_column}), on="_formula_literal", how="left").drop("_formula_literal")


def normalize_precursor_types(library: pl.LazyFrame) -> pl.LazyFrame:
    return library.with_columns(
        pl.col("precursor_type").cast(pl.String, strict=False).str.replace_all("−|–|—", "-").str.replace_all(r"\s+", "").alias("normalized_precursor_type")
    )


def _parse_precursor(value: object) -> dict[str, object]:
    normalized = _normalise_text(value)
    result: dict[str, object] = {
        "normalized_precursor_type": normalized,
        "precursor_selection_status": "unsupported_precursor",
        "precursor_parse_reason": "missing_precursor" if not normalized else "unsupported_precursor",
        "prefix": None, "neutral_loss_formula": None, "neutral_loss_formula_array": None,
    }
    if not normalized:
        return result
    bracketed = normalized.startswith("[")
    if bracketed != normalized.endswith("]+") and bracketed != normalized.endswith("]-"):
        # Continue below for bracketed charge forms; this catches unmatched brackets and bare bracketed bodies.
        pass
    body = normalized
    charge: str | None = None
    if bracketed:
        match = re.fullmatch(r"\[(.*)\](1?[+-])", normalized)
        if match is None:
            result["precursor_parse_reason"] = "malformed_brackets_or_charge"
            return result
        body, charge = match.group(1), match.group(2)
        if charge in {"1+", "1-"}:
            pass
    else:
        if normalized.count("[") or normalized.count("]"):
            result["precursor_parse_reason"] = "malformed_brackets"
            return result
        match = re.fullmatch(r"(.*?)([+-])?", normalized)
        if match is None:
            result["precursor_parse_reason"] = "malformed_charge"
            return result
        body, charge = match.group(1), match.group(2)
        if charge and body and body[-1].isdigit():
            result["precursor_parse_reason"] = "ambiguous_unbracketed_charge"
            return result
    if not body or body.endswith(("+", "-")) or "+" in body[1:] and body.count("+") > 2:
        result["precursor_parse_reason"] = "malformed_charge"
        return result
    if body in ORDINARY_PRECURSOR_BODIES:
        result["precursor_selection_status"] = "not_source_loss"
        result["precursor_parse_reason"] = "ordinary_precursor_body"
        return result
    # Strictly split one final minus-literal, with no other loss operation accepted in the prefix.
    match = re.fullmatch(r"(.+)-([A-Za-z0-9]+)", body)
    if match is None:
        result["precursor_parse_reason"] = "unsupported_precursor_body"
        return result
    prefix, loss_text = match.groups()
    if "H2O" in prefix:
        result["precursor_parse_reason"] = "existing_neutral_loss_prefix"
        return result
    if prefix not in PERMITTED_LOSS_PREFIXES:
        result["precursor_parse_reason"] = "unsupported_loss_prefix"
        return result
    loss = _parse_formula_literal(loss_text)
    if loss is None:
        result["precursor_parse_reason"] = "invalid_neutral_loss_formula"
        return result
    result.update(
        precursor_selection_status="source_loss", precursor_parse_reason=None,
        prefix=prefix, neutral_loss_formula=loss_text,
        neutral_loss_formula_array=loss,
    )
    return result


def parse_source_loss_precursors(precursors: pl.DataFrame) -> pl.DataFrame:
    values = precursors.get_column("precursor_type").to_list()
    records = []
    for value in values:
        record = _parse_precursor(value)
        record["precursor_type"] = value
        records.append(record)
    return pl.DataFrame(records).with_columns(
        pl.col("neutral_loss_formula_array").cast(pl.Array(pl.Int64, FORMULA_WIDTH))
    )


def _metadata_formula(row: dict[str, object]) -> tuple[FormulaVector | None, str | None, str | None]:
    array_present = row.get("molecular_formula_array") is not None
    raw_array = row.get("molecular_formula_array")
    array = _array(raw_array)
    text_value = _normalise_text(row.get("molecular_formula"))
    text = _parse_formula_literal(text_value or "")
    if array_present and array is None:
        try:
            if np.asarray(raw_array).shape == (FORMULA_WIDTH,) and int(np.asarray(raw_array, dtype=np.int64).sum()) == 0:
                return None, None, "empty_parent_formula"
        except (TypeError, ValueError):
            pass
        return None, None, "invalid_parent_formula_array"
    if array is not None and text is not None and not np.array_equal(array, text):
        return array, "molecular_formula_array", "parent_formula_disagreement"
    if array is not None:
        return array, "molecular_formula_array", None
    if row.get("molecular_formula") is not None and text is None:
        return None, None, "empty_parent_formula" if not text_value else "invalid_parent_formula_text"
    if text is not None:
        return text, "molecular_formula", None
    return None, None, "missing_formula"


def _source_column(frame: pl.DataFrame, name: str, dtype: pl.DataType) -> pl.Expr:
    expression = pl.col(name).cast(dtype, strict=False) if name in frame.columns else pl.lit(None, dtype=dtype)
    return expression.alias(name)


def prepare_source_candidates(library: pl.DataFrame | pl.LazyFrame, precursor_lookup: pl.DataFrame) -> pl.DataFrame:
    """Filter and annotate candidates with Polars; spectra never cross the Python boundary."""
    frame = library.collect() if isinstance(library, pl.LazyFrame) else library
    if "library_row_id" not in frame.columns:
        raise ValueError("library_row_id is required before source-candidate filtering")
    lookup = precursor_lookup.select(
        "normalized_precursor_type", "precursor_selection_status", "precursor_parse_reason",
        "neutral_loss_formula", "neutral_loss_formula_array",
    )
    working = frame.join(lookup, on="normalized_precursor_type", how="inner", suffix="_parsed").filter(
        pl.col("precursor_selection_status") == "source_loss"
    )
    if working.is_empty():
        return _candidate_frame([])
    formula_text = _source_column(working, "molecular_formula", pl.String).str.replace_all(r"\s+", "")
    text_lookup = _validated_formula_lookup(working.select(formula_text.alias("_formula_literal")).get_column("_formula_literal"))
    working = working.with_columns(formula_text.alias("_formula_literal")).join(text_lookup, on="_formula_literal", how="left")
    if "_formula_array" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Array(pl.Int64, FORMULA_WIDTH)).alias("_formula_array"))
    array_value = _source_column(working, "molecular_formula_array", pl.Array(pl.Int64, FORMULA_WIDTH))
    array_present = array_value.is_not_null()
    array_valid = (array_present & (array_value.arr.sum() > 0) & (array_value.arr.min() >= 0)).fill_null(False)
    text_present = pl.col("molecular_formula").is_not_null() if "molecular_formula" in working.columns else pl.lit(False)
    text_valid = pl.col("_formula_array").is_not_null()
    disagreement = array_valid & text_valid & (array_value != pl.col("_formula_array"))
    working = working.with_columns(
        array_value.alias("_array_value"),
        pl.when(array_present & ~array_valid).then(pl.lit("invalid_parent_formula_array"))
        .when(disagreement).then(pl.lit("parent_formula_disagreement"))
        .when(text_present & ~text_valid).then(pl.when(pl.col("_formula_literal") == "").then(pl.lit("empty_parent_formula")).otherwise(pl.lit("invalid_parent_formula_text")))
        .when(~array_present & ~text_present).then(pl.lit("missing_formula"))
        .otherwise(pl.lit(None, dtype=pl.String)).alias("_metadata_reason"),
        pl.when(array_valid).then(array_value).otherwise(pl.col("_formula_array")).alias("_metadata_formula"),
    ).with_columns(
        (pl.col("_metadata_formula") - pl.col("neutral_loss_formula_array")).alias("_target_raw"),
    ).with_columns(
        pl.when(pl.col("_metadata_reason").is_null() & (pl.col("_target_raw").arr.min() < 0)).then(pl.lit("negative_retained_target"))
        .otherwise(pl.col("_metadata_reason")).alias("_metadata_reason"),
    )
    metadata_formula = pl.col("_metadata_formula")
    target = pl.when(pl.col("_metadata_reason").is_null()).then(pl.col("_target_raw")).otherwise(pl.lit(None, dtype=pl.Array(pl.Int64, FORMULA_WIDTH)))
    eligibility_reason = pl.col("_metadata_reason").fill_null(
        pl.when(_source_column(working, "smiles", pl.String).is_null() | (_source_column(working, "smiles", pl.String).str.strip_chars() == ""))
        .then(pl.lit("missing_smiles")).otherwise(pl.lit(None, dtype=pl.String))
    )
    working = working.with_columns(
        pl.when(pl.col("_metadata_reason") == "missing_formula").then(pl.lit("missing_formula"))
        .when(pl.col("_metadata_reason").is_not_null()).then(pl.lit("invalid_formula"))
        .when(eligibility_reason == "missing_smiles").then(pl.lit("missing_smiles"))
        .otherwise(pl.lit("eligible")).alias("eligibility_status"),
        eligibility_reason.alias("eligibility_reason"),
        metadata_formula.alias("parent_formula_used_array"),
        pl.when(array_valid).then(pl.lit("molecular_formula_array")).when(text_valid).then(pl.lit("molecular_formula")).otherwise(pl.lit(None, dtype=pl.String)).alias("parent_formula_source"),
        target.alias("retained_target_formula_array"),
    )
    null_string = pl.lit(None, dtype=pl.String)
    selected = working.select(
        pl.col("library_row_id").alias("source_row_id"), _source_column(working, "name", pl.String), _source_column(working, "source_file", pl.String),
        _source_column(working, "smiles", pl.String), _source_column(working, "precursor_type", pl.String), _source_column(working, "ion_mode", pl.String),
        _source_column(working, "molecular_formula", pl.String), _source_column(working, "molecular_formula_array", pl.Array(pl.Int64, FORMULA_WIDTH)), _source_column(working, "precursor_formula_array", pl.Array(pl.Int64, FORMULA_WIDTH)),
        _source_column(working, "cleaned_normalized_mz", pl.List(pl.Float64)), _source_column(working, "cleaned_normalized_intensity", pl.List(pl.Float64)), _source_column(working, "precursor_mz", pl.Float64),
        _source_column(working, "mslevel", pl.Int64), _source_column(working, "clean_precursor", pl.Boolean), _source_column(working, "inchi", pl.String), _source_column(working, "inchikey", pl.String), _source_column(working, "base_inchikey", pl.String),
        pl.col("normalized_precursor_type"), pl.col("eligibility_status"), pl.col("eligibility_reason"), pl.col("precursor_parse_reason"), pl.col("neutral_loss_formula"), pl.col("neutral_loss_formula_array"),
        pl.col("parent_formula_used_array"), pl.col("parent_formula_source"), pl.lit(None, dtype=pl.Array(pl.Int64, FORMULA_WIDTH)).alias("parent_graph_formula_array"), pl.col("retained_target_formula_array"),
        pl.col("retained_target_formula_array").map_elements(_formula_string, return_dtype=pl.String).alias("retained_target_formula"),
        pl.lit(None, dtype=pl.UInt32).alias("eligible_bond_count"), pl.lit(None, dtype=pl.UInt32).alias("matching_bond_count"), pl.lit(None, dtype=pl.UInt32).alias("matching_component_count"), pl.lit(None, dtype=pl.List(pl.UInt32)).alias("matching_bond_indices"),
        pl.lit(None, dtype=pl.UInt32).alias("fragmentation_failure_bond_count"), pl.lit(None, dtype=pl.UInt32).alias("radical_rejected_bond_count"), pl.lit(None, dtype=pl.String).alias("recovery_status"), pl.lit(None, dtype=pl.String).alias("recovered_smiles_raw"), pl.lit(None, dtype=pl.UInt32).alias("recovered_bond_index"),
        pl.lit(None, dtype=pl.Array(pl.UInt32, 2)).alias("recovered_bond_atom_indices"), pl.lit(None, dtype=pl.Array(pl.Int64, FORMULA_WIDTH)).alias("dropped_component_formula_array"), null_string.alias("dropped_component_formula"), pl.lit(None, dtype=pl.Int32).alias("recovered_formal_charge"),
        null_string.alias("recovered_smiles_msready"), null_string.alias("recovered_inchikey"), null_string.alias("recovered_base_inchikey"), pl.lit("not_attempted").alias("standardization_status"),
    )
    return selected


def _candidate_frame(records: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "source_row_id": pl.UInt64, "name": pl.String, "source_file": pl.String, "smiles": pl.String,
        "precursor_type": pl.String, "ion_mode": pl.String, "molecular_formula": pl.String,
        "molecular_formula_array": pl.Array(pl.Int64, 12), "precursor_formula_array": pl.Array(pl.Int64, 12),
        "cleaned_normalized_mz": pl.List(pl.Float64), "cleaned_normalized_intensity": pl.List(pl.Float64), "precursor_mz": pl.Float64,
        "mslevel": pl.Int64, "clean_precursor": pl.Boolean, "inchi": pl.String, "inchikey": pl.String, "base_inchikey": pl.String,
        "normalized_precursor_type": pl.String, "eligibility_status": pl.String, "eligibility_reason": pl.String, "precursor_parse_reason": pl.String,
        "neutral_loss_formula": pl.String, "neutral_loss_formula_array": pl.Array(pl.Int64, 12), "parent_formula_used_array": pl.Array(pl.Int64, 12),
        "parent_formula_source": pl.String, "parent_graph_formula_array": pl.Array(pl.Int64, 12), "retained_target_formula_array": pl.Array(pl.Int64, 12), "retained_target_formula": pl.String,
        "eligible_bond_count": pl.UInt32, "matching_bond_count": pl.UInt32, "matching_component_count": pl.UInt32, "matching_bond_indices": pl.List(pl.UInt32),
        "fragmentation_failure_bond_count": pl.UInt32, "radical_rejected_bond_count": pl.UInt32, "recovery_status": pl.String, "recovered_smiles_raw": pl.String,
        "recovered_bond_index": pl.UInt32, "recovered_bond_atom_indices": pl.Array(pl.UInt32, 2), "dropped_component_formula_array": pl.Array(pl.Int64, 12),
        "dropped_component_formula": pl.String, "recovered_formal_charge": pl.Int32, "recovered_smiles_msready": pl.String, "recovered_inchikey": pl.String,
        "recovered_base_inchikey": pl.String, "standardization_status": pl.String,
    }
    if not records:
        return pl.DataFrame(schema=schema)
    normalized_records: list[dict[str, object]] = []
    for record in records:
        normalized_record = dict(record)
        for key, value in normalized_record.items():
            if isinstance(value, np.ndarray):
                normalized_record[key] = value.tolist()
            elif isinstance(value, tuple):
                normalized_record[key] = list(value)
        normalized_records.append(normalized_record)
    return pl.DataFrame(normalized_records, schema=schema, strict=False)


def validate_parent_structure(parent_smiles: str | None) -> ValidatedParent:
    if parent_smiles is None or not str(parent_smiles).strip():
        return ValidatedParent(parent_smiles, None, "missing_smiles", None)
    molecule = Chem.MolFromSmiles(str(parent_smiles))
    if molecule is None:
        return ValidatedParent(parent_smiles, None, "invalid_smiles", None)
    if len(Chem.GetMolFrags(molecule)) != 1:
        return ValidatedParent(parent_smiles, None, "disconnected_parent", None)
    for atom in molecule.GetAtoms():
        # Boron is intentionally excluded: the twelve-element formula backend has no B slot.
        if atom.GetAtomicNum() not in {1, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53}:
            return ValidatedParent(parent_smiles, None, "unsupported_structure", None)
        if atom.GetIsotope() or atom.GetNumRadicalElectrons():
            return ValidatedParent(parent_smiles, None, "unsupported_structure", None)
    return ValidatedParent(parent_smiles, molecule, None, Chem.MolToSmiles(molecule, isomericSmiles=True))


def build_parent_graph_formula_lookup(validated_parent_smiles: pl.Series) -> pl.DataFrame:
    values = list(dict.fromkeys(value for value in validated_parent_smiles.to_list() if value))
    if not values:
        return pl.DataFrame(schema={"smiles": pl.String, "parent_graph_formula_array": pl.Array(pl.Int64, 12)})
    formula = smiles_to_formula(pl.Series("smiles", values))
    if len(formula) != len(values) or formula.dtype != pl.Array(pl.Int64, FORMULA_WIDTH):
        raise RuntimeError("parallel_rdkit returned an invalid parent formula contract")
    if any(int(np.asarray(value).sum()) <= 0 or np.any(np.asarray(value) < 0) for value in formula.to_list()):
        raise RuntimeError("parallel_rdkit returned an invalid formula for a validated parent graph")
    return pl.DataFrame({"smiles": values, "parent_graph_formula_array": formula})


def eligible_bond_indices(parent: Chem.Mol) -> list[int]:
    return [bond.GetIdx() for bond in parent.GetBonds() if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing() and bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1]


def enumerate_parent_components(parent: Chem.Mol) -> ParentEnumeration:
    eligible = eligible_bond_indices(parent)
    failures = radicals = 0
    components: list[FragmentComponent] = []
    for index in eligible:
        bond = parent.GetBondWithIdx(index)
        endpoints = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        mapping: list[tuple[int, ...]] = []
        try:
            cut = Chem.FragmentOnBonds(parent, [index], addDummies=False)
            fragments = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=mapping)
            if len(fragments) != 2:
                failures += 1
                continue
            if any(sum(atom.GetNumRadicalElectrons() for atom in fragment.GetAtoms()) for fragment in fragments):
                radicals += 1
                continue
            smiles = [Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True) for fragment in fragments]
            for component_index, fragment in enumerate(fragments):
                components.append(FragmentComponent(index, endpoints, component_index, smiles[component_index], smiles[1 - component_index], Chem.GetFormalCharge(fragment)))
        except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException, RuntimeError, ValueError):
            failures += 1
    return ParentEnumeration(len(eligible), failures, radicals, components)


def build_component_formula_lookup(component_smiles: pl.Series) -> pl.DataFrame:
    values = list(dict.fromkeys(value for value in component_smiles.to_list() if value))
    if not values:
        return pl.DataFrame(schema={"component_smiles": pl.String, "component_formula_array": pl.Array(pl.Int64, 12)})
    formula = smiles_to_formula(pl.Series("smiles", values))
    if len(formula) != len(values) or formula.dtype != pl.Array(pl.Int64, FORMULA_WIDTH):
        raise RuntimeError("parallel_rdkit returned an invalid component formula contract")
    if any(int(np.asarray(value).sum()) <= 0 or np.any(np.asarray(value) < 0) for value in formula.to_list()):
        raise RuntimeError("parallel_rdkit returned an invalid formula for a validated fragment")
    return pl.DataFrame({"component_smiles": values, "component_formula_array": formula})


def recover_candidate_structures(candidates: pl.DataFrame) -> pl.DataFrame:
    """Validate parents with joins, then enumerate only distinct exact-agreement graphs."""
    if candidates.is_empty():
        return candidates
    distinct_smiles = candidates.filter(pl.col("smiles").is_not_null()).select("smiles").unique()
    validated = {value: validate_parent_structure(value) for value in distinct_smiles.get_column("smiles").to_list()}
    validation_lookup = pl.DataFrame({
        "smiles": list(validated),
        "_parent_validation_reason": [record.eligibility_reason for record in validated.values()],
    }, schema={"smiles": pl.String, "_parent_validation_reason": pl.String})
    valid_smiles = [value for value, record in validated.items() if record.molecule is not None]
    graph_lookup = build_parent_graph_formula_lookup(pl.Series("smiles", valid_smiles))
    if len(graph_lookup) != len(valid_smiles):
        raise RuntimeError("parallel_rdkit returned incomplete parent graph formulas")
    working = candidates.join(validation_lookup, on="smiles", how="left").join(graph_lookup.rename({"parent_graph_formula_array": "_graph_formula"}), on="smiles", how="left")
    parent_formula_matches = pl.col("parent_formula_used_array") == pl.col("_graph_formula")
    working = working.with_columns(
        pl.col("_graph_formula").alias("parent_graph_formula_array"),
        pl.when((pl.col("eligibility_status") == "eligible") & pl.col("_parent_validation_reason").is_not_null())
        .then(pl.col("_parent_validation_reason"))
        .when((pl.col("eligibility_status") == "eligible") & ~parent_formula_matches)
        .then(pl.lit("invalid_formula"))
        .otherwise(pl.col("eligibility_status")).alias("eligibility_status"),
        pl.when((pl.col("eligibility_status") == "eligible") & pl.col("_parent_validation_reason").is_not_null())
        .then(pl.col("_parent_validation_reason"))
        .when((pl.col("eligibility_status") == "eligible") & ~parent_formula_matches)
        .then(pl.lit("parent_graph_formula_disagreement"))
        .otherwise(pl.col("eligibility_reason")).alias("eligibility_reason"),
    ).with_columns(
        pl.when(pl.col("eligibility_status") == "eligible").then(pl.col("retained_target_formula_array")).otherwise(pl.lit(None, dtype=pl.Array(pl.Int64, FORMULA_WIDTH))).alias("retained_target_formula_array"),
    ).with_columns(
        pl.col("retained_target_formula_array").map_elements(_formula_string, return_dtype=pl.String).alias("retained_target_formula"),
    ).drop("_parent_validation_reason", "_graph_formula")
    exact_smiles = working.filter(pl.col("eligibility_status") == "eligible").select("smiles").unique().get_column("smiles").to_list()
    enumerations = {
        value: enumerate_parent_components(validated[value].molecule)
        for value in exact_smiles
        if validated[value].molecule is not None
    }
    enumeration_records = [{
        "smiles": value, "eligible_bond_count": enumeration.eligible_bond_count,
        "fragmentation_failure_bond_count": enumeration.fragmentation_failure_bond_count,
        "radical_rejected_bond_count": enumeration.radical_rejected_bond_count,
    } for value, enumeration in enumerations.items()]
    enumeration_lookup = pl.DataFrame(enumeration_records, schema={"smiles": pl.String, "eligible_bond_count": pl.UInt32, "fragmentation_failure_bond_count": pl.UInt32, "radical_rejected_bond_count": pl.UInt32}) if enumeration_records else pl.DataFrame(schema={"smiles": pl.String, "eligible_bond_count": pl.UInt32, "fragmentation_failure_bond_count": pl.UInt32, "radical_rejected_bond_count": pl.UInt32})
    component_records = [{
        "smiles": value, "bond_index": component.bond_index, "bond_atom_indices": list(component.atom_indices),
        "component_index": component.component_index, "component_smiles": component.component_smiles,
        "companion_smiles": component.companion_smiles, "recovered_formal_charge": component.formal_charge,
    } for value, enumeration in enumerations.items() for component in enumeration.components]
    components = pl.DataFrame(component_records, schema={"smiles": pl.String, "bond_index": pl.UInt32, "bond_atom_indices": pl.Array(pl.UInt32, 2), "component_index": pl.UInt32, "component_smiles": pl.String, "companion_smiles": pl.String, "recovered_formal_charge": pl.Int32}) if component_records else pl.DataFrame(schema={"smiles": pl.String, "bond_index": pl.UInt32, "bond_atom_indices": pl.Array(pl.UInt32, 2), "component_index": pl.UInt32, "component_smiles": pl.String, "companion_smiles": pl.String, "recovered_formal_charge": pl.Int32})
    component_smiles = pl.concat([components.select(pl.col("component_smiles").alias("component_smiles")), components.select(pl.col("companion_smiles").alias("component_smiles"))]).unique() if not components.is_empty() else pl.DataFrame(schema={"component_smiles": pl.String})
    component_formulas = build_component_formula_lookup(component_smiles.get_column("component_smiles"))
    components = components.join(component_formulas, on="component_smiles", how="left").rename({"component_formula_array": "recovered_formula_array"})
    companion_formulas = component_formulas.rename({"component_smiles": "companion_smiles", "component_formula_array": "dropped_component_formula_array"})
    components = components.join(companion_formulas, on="companion_smiles", how="left")
    if components.is_empty():
        diagnostic_columns = [
            "eligible_bond_count", "fragmentation_failure_bond_count", "radical_rejected_bond_count",
            "matching_bond_count", "matching_component_count", "matching_bond_indices",
        ]
        working = working.drop([column for column in diagnostic_columns if column in working.columns])
        return working.join(enumeration_lookup, on="smiles", how="left").with_columns(
            pl.when(pl.col("eligibility_status") == "eligible").then(pl.lit(0, dtype=pl.UInt32)).otherwise(pl.lit(None, dtype=pl.UInt32)).alias("matching_bond_count"),
            pl.when(pl.col("eligibility_status") == "eligible").then(pl.lit(0, dtype=pl.UInt32)).otherwise(pl.lit(None, dtype=pl.UInt32)).alias("matching_component_count"),
            pl.when(pl.col("eligibility_status") == "eligible").then(pl.lit([], dtype=pl.List(pl.UInt32))).otherwise(pl.lit(None, dtype=pl.List(pl.UInt32))).alias("matching_bond_indices"),
            pl.when(pl.col("eligibility_status") == "eligible").then(pl.lit("no_answer")).otherwise(pl.lit(None, dtype=pl.String)).alias("recovery_status"),
        ).select(_candidate_frame([]).columns)
    matched = working.select("source_row_id", "smiles", "retained_target_formula_array").join(components, on="smiles", how="inner").filter(
        pl.col("recovered_formula_array") == pl.col("retained_target_formula_array")
    ).unique(subset=["source_row_id", "bond_index", "component_smiles"], maintain_order=True)
    match_summary = matched.group_by("source_row_id").agg(
        pl.col("bond_index").n_unique().cast(pl.UInt32).alias("matching_bond_count"),
        pl.len().cast(pl.UInt32).alias("matching_component_count"),
        pl.col("bond_index").unique().sort().alias("matching_bond_indices"),
    )
    # Candidate rows carry typed null placeholders for these diagnostics.  Remove
    # them before joining the computed values; otherwise Polars suffixes the
    # joined columns and the placeholders would be filled with zero instead.
    diagnostic_columns = [
        "eligible_bond_count", "fragmentation_failure_bond_count", "radical_rejected_bond_count",
        "matching_bond_count", "matching_component_count", "matching_bond_indices",
    ]
    working = working.drop([column for column in diagnostic_columns if column in working.columns])
    working = working.join(enumeration_lookup, on="smiles", how="left").join(match_summary, on="source_row_id", how="left").with_columns(
        pl.when(pl.col("eligibility_status") == "eligible").then(pl.col("matching_bond_count").fill_null(0).cast(pl.UInt32)).otherwise(pl.lit(None, dtype=pl.UInt32)).alias("matching_bond_count"),
        pl.when(pl.col("eligibility_status") == "eligible").then(pl.col("matching_component_count").fill_null(0).cast(pl.UInt32)).otherwise(pl.lit(None, dtype=pl.UInt32)).alias("matching_component_count"),
        pl.when(pl.col("eligibility_status") == "eligible").then(pl.col("matching_bond_indices").fill_null([]).cast(pl.List(pl.UInt32))).otherwise(pl.lit(None, dtype=pl.List(pl.UInt32))).alias("matching_bond_indices"),
    ).with_columns(
        pl.when(pl.col("eligibility_status") != "eligible").then(pl.lit(None, dtype=pl.String))
        .when(pl.col("matching_component_count") == 0).then(pl.lit("no_answer"))
        .when((pl.col("matching_component_count") == 1) & (pl.col("matching_bond_count") == 1)).then(pl.lit("unique"))
        .otherwise(pl.lit("several_answers")).alias("recovery_status"),
    )
    unique = matched.group_by("source_row_id").agg(pl.len().alias("_match_count")).filter(pl.col("_match_count") == 1).drop("_match_count")
    unique_match = unique.join(matched, on="source_row_id", how="inner")
    unique_columns = unique_match.select(
        "source_row_id", pl.col("component_smiles").alias("_unique_component_smiles"),
        pl.col("bond_index").alias("_unique_bond_index"), pl.col("bond_atom_indices").alias("_unique_bond_atom_indices"),
        pl.col("dropped_component_formula_array").alias("_unique_dropped_formula"), pl.col("recovered_formal_charge").alias("_unique_charge"),
    )
    working = working.join(unique_columns, on="source_row_id", how="left").with_columns(
        pl.when(pl.col("recovery_status") == "unique").then(pl.col("_unique_component_smiles")).otherwise(pl.lit(None, dtype=pl.String)).alias("recovered_smiles_raw"),
        pl.when(pl.col("recovery_status") == "unique").then(pl.col("_unique_bond_index")).otherwise(pl.lit(None, dtype=pl.UInt32)).alias("recovered_bond_index"),
        pl.when(pl.col("recovery_status") == "unique").then(pl.col("_unique_bond_atom_indices")).otherwise(pl.lit(None, dtype=pl.Array(pl.UInt32, 2))).alias("recovered_bond_atom_indices"),
        pl.when(pl.col("recovery_status") == "unique").then(pl.col("_unique_dropped_formula")).otherwise(pl.lit(None, dtype=pl.Array(pl.Int64, FORMULA_WIDTH))).alias("dropped_component_formula_array"),
        pl.when(pl.col("recovery_status") == "unique").then(pl.col("_unique_dropped_formula").map_elements(_formula_string, return_dtype=pl.String)).otherwise(pl.lit(None, dtype=pl.String)).alias("dropped_component_formula"),
        pl.when(pl.col("recovery_status") == "unique").then(pl.col("_unique_charge")).otherwise(pl.lit(None, dtype=pl.Int32)).alias("recovered_formal_charge"),
    ).drop("_unique_component_smiles", "_unique_bond_index", "_unique_bond_atom_indices", "_unique_dropped_formula", "_unique_charge")
    return working.select(_candidate_frame([]).columns)
