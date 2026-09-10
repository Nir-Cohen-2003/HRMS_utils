# Mass decomposition

Mass decomposition returns candidate elemental formulas for an observed m/z. The
Polars plugin supports unbounded or per-row bounded decomposition:

```python
import polars as pl
from hrms_utils import hrms_core  # registers the Polars namespaces

masses = pl.DataFrame({"mass": [180.0634]})
result = masses.with_columns(
    pl.col("mass").mass_decomposition.decompose_mass(
        tolerance_ppm=3.0,
        dbe_mode="half_integer",
    ).alias("decomposition")
)
```

In the MS-DIAL annotation workflow, isotope evidence is passed as per-feature
`min_bounds` and `max_bounds` to `decompose_mass_with_bounds()`. Formula arrays are
`Array(Int32, 12)` in the fixed order `H, C, N, O, F, Na, P, S, Cl, K, Br, I`;
formula strings and ppm errors are returned alongside them. One input feature can
produce multiple candidate rows after exploding the formula list.

The annotation implementation performs this bounded decomposition after
`deduce_isotopic_pattern()`, then cleans MS/MS fragments for each candidate. See the
[MS-DIAL workflow](01-msdial-chromatogram-annotation.md) and the [data-structure
reference](../reference/data-structures.md) for the complete schema.
