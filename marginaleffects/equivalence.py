import polars as pl
import numpy as np
from scipy.stats import norm, t
from typing import Union

def get_equivalence(x: pl.DataFrame, equivalence: Union[list, None], df: float = np.inf) -> pl.DataFrame:
    if equivalence is None:
        return x

    assert len(equivalence)== 2 and isinstance(equivalence[0], float) and isinstance(equivalence[1], float), "The `equivalence` argument must be None or a list of two 'float' values."

    if not all(col in x for col in ["estimate", "std_error"]):
        msg = "The `equivalence` argument is not supported for `marginaleffects` commands which do not produce standard errors."
        raise ValueError(msg)

    delta = np.abs(np.diff(equivalence)) / 2
    null = np.min(equivalence) + delta

    x = x.with_columns(
        ((x["estimate"] - equivalence[0]) / x["std_error"]).alias("statistic_noninf"),
        ((x["estimate"] - equivalence[1]) / x["std_error"]).alias("statistic_nonsup"),
    )

    if np.isinf(df):
        x = x.with_columns(
            pl.col("statistic_noninf").apply(lambda x: norm.sf(x)).alias("p_value_noninf"),
            pl.col("statistic_nonsup").apply(lambda x: norm.cdf(x)).alias("p_value_nonsup"),
        )
    else:
        x = x.with_columns(
            pl.col("statistic.noninf").apply(lambda x: t.sf(x)).alias("p_value_noninf"),
            pl.col("statistic.nonsup").apply(lambda x: t.cdf(x)).alias("p_value_nonsup"),
        )

    x = x.with_columns(pl.max(["p_value_nonsup", "p_value_noninf"]).alias("p_value_equiv"))

    return x