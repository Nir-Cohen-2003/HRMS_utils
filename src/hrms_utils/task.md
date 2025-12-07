# Task: Fragmentation Graph Construction from MSP Data

## Objective
Create a script that reads MSP file parsing results and constructs a directed fragmentation graph (parent → child) based on parent-child relationships between fragment formulas, accounting for MS levels and water absorption.

---

## Input
- **Data source**: Output from `nist_mspec.py::read_MSPEC_file()` or equivalent MSP parser
- **Expected columns** (required subset):
  - `base_inchikey`: Compound identifier (string)
  - `ion_mode`: Polarity (P/N, string)
  - `precursor_formula_array`: Array of element counts (Array[Int32, num_elements])
  - `mslevel`: MS fragmentation level (Int, e.g., 2 for MS², 3 for MS³)
  - `cleaned_fragment_formulas`: List of fragment formula arrays (List[Array[Int32, num_elements]])
  - `cleaned_fragment_formulas_str`: List of fragment formula strings (List[String])

---

## Processing Steps

### 1. Group Spectra
- **Group by**: `base_inchikey` and `ion_mode`
- **Rationale**: Each compound-polarity combination represents an independent fragmentation tree

### 2. Combine Spectra by Precursor Formula
- **Within each compound-polarity group**:
  - Combine all spectra sharing the same `precursor_formula_array`
  - Collect all fragment formulas across all MS levels
  - **Note**: Discard intensity information; only fragment formulas matter for graph construction

### 3. Merge Fragments Across MS Levels
- **Fragment identity**: Defined solely by formula array (element counts)
- **Merge rule**: If the same fragment formula appears in multiple MS levels (e.g., MS² and MS³), treat as a single node
- **Rationale**: Information from multiple MS levels helps deduce the true parent-child relationships

### 4. Add Precursor as Root Node
- **Rule**: The precursor formula is the root node of each fragmentation tree
- **Deduplication**: If the precursor formula appears in the fragment list (which can occur), merge them into a single node
- **Extension**: For MS³ and higher levels, the precursor of that MS level should be added as a node and similarly deduplicated if it appears in the fragment list
- **Guaranteed relationship**: All fragments will have at least the precursor as a potential parent (ensured by the pipeline)

### 5. Construct Directed Fragmentation Graph

#### 5.1 Parent-Child Relationship Rules
A directed edge `parent → child` exists if:

**Direct loss**:
```
parent_formula - child_formula ≥ 0 (element-wise)
AND
parent_formula - child_formula ≠ [0, 0, 0, ...] (non-zero difference)
```

**Water absorption loss** (if `water_absorption=True`):
```
parent_formula - child_formula - H₂O ≥ 0 (element-wise)
where H₂O = [0, 2, 1, 0, 0, ...] (H=2, O=1 in element array)
AND
parent_formula - child_formula - H₂O ≠ [0, 0, 0, ...] (non-zero difference after H₂O)
```

#### 5.2 MS Level Filtering Rule
**Key constraint**: A fragment in MS^N can only have parents that appear in the same MS^N spectrum or are precursors at level MS^(N-1).

**Implementation logic**:
- For each fragment in MS^N where N > 2:
  - Identify all potential parents based on formula subtraction (5.1)
  - **Cut edges** to parent candidates that are:
    - Observed in MS^(N-1) or lower levels, **AND**
    - Not present in the same MS^N spectrum, **AND**
    - Not the direct precursor of the MS^N experiment

**Example** (from original prompt):
```
MS² spectrum: [N₂H₂, O₂H₂, H₂]
MS³ spectrum (precursor: N₂H₂): [H₂]

Potential edges before filtering:
- N₂H₂ → H₂ (valid: N₂H₂ is MS³ precursor)
- O₂H₂ → H₂ (invalid: O₂H₂ is in MS² but not in MS³, and not the MS³ precursor)

Action: Remove edge O₂H₂ → H₂
Rationale: O₂H₂ cannot be the parent of H₂ in MS³ because O₂H₂ was not selected 
           as the precursor for MS³ and does not appear in the MS³ fragment list
```

**Why this matters**: This filtering ensures the graph reflects the actual fragmentation cascade. A fragment in MS³ can only arise from:
1. The MS³ precursor (selected from MS²)
2. Other fragments present in the same MS³ spectrum

---

## Output Requirements

### Graph Representation
Return a Polars DataFrame with columns:
- `base_inchikey`: Compound identifier (string)
- `ion_mode`: Polarity (string, P or N)
- `precursor_formula_array`: Root precursor formula array (Array[Int32, num_elements])
- `precursor_formula_str`: Root precursor formula string (string)
- `parent_formula_array`: Parent fragment formula array (Array[Int32, num_elements])
- `parent_formula_str`: Parent fragment formula string (string)
- `parent_observed_mslevels`: List of MS levels where parent was observed (List[Int])
- `child_formula_array`: Child fragment formula array (Array[Int32, num_elements])
- `child_formula_str`: Child fragment formula string (string)
- `child_observed_mslevels`: List of MS levels where child was observed (List[Int])
- `loss_type`: Either `"direct"` or `"water_absorption"` (string)
- `loss_formula_str`: Formula difference (parent - child, as string, e.g., "CH₂", "H₂O")
- `loss_formula_array`: Formula difference as array (Array[Int32, num_elements])

**Note on mslevels columns**: Since fragments are merged across MS levels, each fragment node exists across multiple levels. The `observed_mslevels` columns track where each fragment was actually observed.

---

## Implementation Requirements

### Input Validation
```python
required_columns = [
    "base_inchikey", 
    "ion_mode", 
    "precursor_formula_array",
    "mslevel",
    "cleaned_fragment_formulas",
    "cleaned_fragment_formulas_str"
]
missing = set(required_columns) - set(dataframe.columns)
assert not missing, (
    f"Missing required columns for fragmentation graph construction: {missing}. "
    f"Expected columns: {required_columns}"
)
```

### Technology Stack
- **Dataframe operations**: Use Polars exclusively
- **Type hints**: All functions must have explicit return types and parameter types
- **Array operations**: after merging the needed spectra with polars list.set_union, export to numpy and use numpy operations on a list of arrays, this is acceptable for performance and makes the code clearer.

### Configuration
needed flags for the function:
water_absorption: If True, allow fragment relationships with H₂O loss (parent - child - H₂O ≥ 0 element-wise)
element_array_width: Number of elements in formula arrays (default: num_elements)

### Naming Conventions
- Function names: Descriptive and long
  - Good: `construct_fragmentation_graph_from_msp_data`
  - Bad: `make_graph`
- Variable names: Clear and unambiguous
  - Good: `parent_child_edge_candidates`
  - Bad: `edges` or `e`

### Error Handling
- **Fail fast**: Assert all required columns exist with descriptive messages
- **Validate formula arrays**: Check that all arrays have shape `(element_array_width,)`
- **Document assumptions**: Add comments explaining why MS level filtering works the way it does

---

## Testing Criteria

### Unit Tests Required
1. **Single MS² spectrum**: Verify basic parent-child relationships
2. **MS² + MS³ cascade**: Test MS level filtering (use N₂H₂/O₂H₂/H₂ example)
3. **Precursor deduplication**: Ensure precursor appearing in fragments is merged correctly
4. **Water absorption**: Verify separate edges created for H₂O loss when enabled
5. **Multiple compounds**: Test grouping by `base_inchikey` and `ion_mode`
6. **Empty spectra**: Handle case with no fragments gracefully
7. **Isolated fragments**: Fragments with only precursor as parent

### Edge Cases
- Precursor formula identical to a fragment formula
- MS³ precursor also appearing in MS³ fragment list
- Fragment appearing in MS², MS³, and MS⁴
- No valid parent-child pairs (only precursor and unrelated fragments)

---

## Implementation Steps (Suggested Order)

1. **Input validation**: Check required columns, validate array dimensions
2. **Group and combine**: Group by compound/polarity, merge by precursor formula
3. **Fragment deduplication**: Create unique fragment set across all MS levels
4. **Add precursor nodes**: Insert precursor as root, deduplicate if needed
5. **Generate candidate edges**: Apply formula subtraction rules (direct + water absorption)
6. **Apply MS level filtering**: Remove invalid parent-child pairs based on MS level rules
7. **Format output**: Create final DataFrame with all required columns
8. **Document**: Add module-level docstring explaining the algorithm and its constraints

---

## Questions Resolved
1. ✓ Graph is directed (parent → child)
2. ✓ Precursor included as root node, deduplicated if it appears in fragments
3. ✓ Fragments from different MS levels are merged (fragment = formula identity)
4. ✓ All fragments guaranteed to have precursor as a potential parent (pipeline ensures this)

---

## Domain-Specific Notes

**Why MS level filtering matters**: In tandem MS experiments, MS³ fragments can only arise from the precursor ion selected from MS². A fragment observed in MS² but not selected as the MS³ precursor cannot directly produce MS³ fragments, even if the formulas suggest a valid subtraction relationship.

**Why merge fragments across MS levels**: A fragment with formula CH₄ appearing in both MS² and MS³ is the same chemical entity. Its appearance in both levels provides information about the fragmentation cascade that helps deduce true parent-child relationships.

**Water absorption caveat**: Some MS instruments show artifacts where fragments appear to have gained water. This is typically a measurement artifact, but the flag allows modeling these apparent relationships if needed for certain instrument types or experimental conditions.