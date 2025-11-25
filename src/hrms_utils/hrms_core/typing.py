from typing import TYPE_CHECKING, Union

import polars as pl

IntoExprColumn = Union[str, pl.Expr, pl.Series]

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    from polars.datatypes import DataType, DataTypeClass

    IntoExprColumn: TypeAlias = Union[pl.Expr, str, pl.Series]
    PolarsDataType: TypeAlias = Union[DataType, DataTypeClass]
