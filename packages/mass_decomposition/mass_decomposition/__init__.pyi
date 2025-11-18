import polars as pl

__version__: str
NUM_ELEMENTS: int

@pl.api.register_expr_namespace("mass_decomposition")
class MassDecomposerUtils:
    def __init__(self, expr: pl.Expr): ...

    def decompose_mass(
        self,
        tolerance_ppm: float = 5.0,
        min_dbe: float = 0.0,
        max_dbe: float = 40.0,
        dbe_mode: str = "integer",
        min_bounds: list[int] | None = None,
        max_bounds: list[int] | None = None,
    ) -> pl.Expr: ...

    def decompose_mass_with_bounds(
        self,
        tolerance_ppm: float = 5.0,
        min_dbe: float = -0.5,
        max_dbe: float = 40.0,
        dbe_mode: str = "integer",
    ) -> pl.Expr: ...

    def clean_and_normalize_spectrum(
        self,
        raw_fragment_tolerance_ppm: float = 5.0,
        normalized_fragment_tolerance_ppm: float = 2.0,
        min_dbe: float = -10.0,
        max_dbe: float = 100.0,
        dbe_mode: str = "any",
        water_absorption: bool = False,
    ) -> pl.Expr: ...
