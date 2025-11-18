from .__init__ import SpectralInfoNamespace

class Expr:
    @property
    def spectral_info(self) -> SpectralInfoNamespace: ...
