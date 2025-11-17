
from typing import Union

import polars as pl

IntoExprColumn = Union[str, pl.Expr, pl.Series]
