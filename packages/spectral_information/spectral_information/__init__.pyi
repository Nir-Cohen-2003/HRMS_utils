import polars as pl

__version__: str
NUM_ELEMENTS: int

@pl.api.register_expr_namespace("spectral_info")
class SpectralInfoNamespace:
    def __init__(self, expr: pl.Expr): ...

    def spectral_info_score(
        self,
        *,
        distance_metric: str = "l2",
        ignore_hydrogens: bool = True,
    ) -> pl.Expr: ...
