from .by import *
from .utils import *
from .sanity import *
from .hypothesis import *
from .uncertainty import *
from .sanitize_variables import *
from .estimands import *
import polars as pl
import pandas as pd
import numpy as np
import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


def convert_int_columns_to_float32(dfs: list) -> list:
    numeric_types = [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]

    converted_dfs = []
    for df in dfs:
        new_columns = []
        if df is not None:
            for col in df:
                if col.dtype in numeric_types:
                    new_columns.append(col.cast(pl.Float32).alias(col.name))
                else:
                    new_columns.append(col)
            converted_df = df.with_columns(new_columns)
            converted_dfs.append(converted_df)
    return converted_dfs

    
def comparisons(
        model,
        variables = None,
        newdata = None,
        comparison = "difference",
        vcov = True,
        conf_int = 0.95,
        by = False,
        hypothesis = None,
        eps = 1e-4):

    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata)

    # after sanitize_newdata() 
    variables = sanitize_variables(variables=variables, model=model, newdata=newdata, comparison=comparison, eps=eps, by=by)

    # pad for character/categorical variables in patsy
    pad = []
    hi = []
    lo = []
    for v in variables:
        hi.append(newdata.with_columns(
            v.hi.alias(v.variable),
            pl.lit(v.variable).alias("term"),
            pl.lit(v.lab).alias("contrast"),
            pl.lit(v.comparison).alias("marginaleffects_comparison")))
        lo.append(newdata.with_columns(
            v.lo.alias(v.variable),
            pl.lit(v.variable).alias("term"),
            pl.lit(v.lab).alias("contrast"),
            pl.lit(v.comparison).alias("marginaleffects_comparison")))
        pad.append(get_pad(newdata, v.variable, v.pad)) 

    # ugly hack, but polars is very strict and `value / 2`` is float
    hi = convert_int_columns_to_float32(hi)
    lo = convert_int_columns_to_float32(lo)
    pad = convert_int_columns_to_float32(pad)
    hi = pl.concat(hi, how = "vertical_relaxed")
    lo = pl.concat(lo, how = "vertical_relaxed")
    pad = [x for x in pad if x is not None]
    if len(pad) == 0:
        pad = pl.DataFrame()
    else:
        pad = pl.concat(pad).unique()
        hi = pl.concat([pad, hi], how = "diagonal")
        lo = pl.concat([pad, lo], how = "diagonal")

    # model matrices
    y, hi_X = patsy.dmatrices(model.model.formula, hi.to_pandas())
    y, lo_X = patsy.dmatrices(model.model.formula, lo.to_pandas())

    # unpad
    if pad.shape[0] > 0:
        hi_X = hi_X[pad.shape[0]:]
        lo_X = lo_X[pad.shape[0]:]
        hi = hi[pad.shape[0]:]
        lo = lo[pad.shape[0]:]

    # TODO: fix derivatives
    yvar = pl.Series(np.repeat(None, newdata.shape[0]))

    baseline = hi.clone()

    def inner(coefs, by):

        # we don't want a pandas series
        try:
            coefs = coefs.to_numpy()
        except:
            pass

        # estimates
        tmp = baseline.with_columns(
            pl.Series(model.model.predict(coefs, lo_X)).alias("predicted_lo"),
            pl.Series(model.model.predict(coefs, hi_X)).alias("predicted_hi"),
            # pl.lit(xvar).alias("marginaleffects_xvar"),
            # pl.lit(yvar).alias("marginaleffects_yvar"),
        )

        if isinstance(by, str):
            by = ["term", "contrast"] + [by]
        elif isinstance(by, list):
            by = ["term", "contrast"] + by
        else:
            by = ["term", "contrast"]

        def applyfun(x, by = by):
            comp = x["marginaleffects_comparison"][0]
            xvar = x[x["term"][0]]
            est = estimands[comp](
                hi = x["predicted_hi"],
                lo = x["predicted_lo"],
                eps = eps,
                x = xvar,
                y = None, 
            )
            if est.shape[0] == 1:
                est = est.item()
                tmp = x.select(by) \
                       .unique() \
                       .with_columns(pl.lit(est).alias("estimate"))
            else:
                tmp = x.with_columns(pl.lit(est).alias("estimate"))
            return tmp 

        # maintain_order is extremely important
        tmp = tmp.groupby(by, maintain_order = True).apply(applyfun)
        return tmp 

    outer = lambda x: inner(x, by = by)

    out = outer(model.params.to_numpy())

    if vcov is not None and vcov is not False:
        J = get_jacobian(func = outer, coefs = model.params.to_numpy())
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(out, model, conf_int=conf_int)

    out = sort_columns(out, by = by)
    return out