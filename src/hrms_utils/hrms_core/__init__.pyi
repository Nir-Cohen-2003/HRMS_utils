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


@pl.api.register_expr_namespace("spectral_info")
class SpectralInfoNamespace:
    def __init__(self, expr: pl.Expr): ...

    def spectral_info_score(
        self,
        *,
        distance_metric: str = "l2",
        ignore_hydrogens: bool = True,
    ) -> pl.Expr: ...


@pl.api.register_expr_namespace("spectral_similarity")
class SpectralUtils:
    def __init__(self, expr : pl.Expr): ...

    def entropy_similarity(
        self,
        ms2_tolerance_in_ppm: float,
        clean_spectra_first: bool = True,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
    ) -> pl.Expr: ...

    def general_cosine_similarity(
        self,
        intensity_power: float,
        mass_power: float,
        ms2_tolerance_in_ppm: float,
        clean_spectra_first: bool = True,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
    ) -> pl.Expr: ...

    def mass_weighted_cosine_similarity(
        self,
        ms2_tolerance_in_ppm: float,
        clean_spectra_first: bool = True,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
    ) -> pl.Expr: ...

    def dotprod_similarity(
        self,
        ms2_tolerance_in_ppm: float,
        clean_spectra_first: bool = True,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
    ) -> pl.Expr: ...

    def explained_intensity(
        self,
        ms2_tolerance_in_ppm: float,
        intensity_power: float = 1.0,
        mass_power: float = 0.0,
        clean_spectra_first: bool = False,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
        permissive: bool = False,
    ) -> pl.Expr: ...
