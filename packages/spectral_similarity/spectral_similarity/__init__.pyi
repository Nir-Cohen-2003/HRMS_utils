import polars as pl

__version__: str

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
