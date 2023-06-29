import polars as pl
from .estimands import * 


def sanitize_newdata(model, newdata, wts = None):
    if newdata is None:
        out = model.model.data.frame
    try:
        out = pl.from_pandas(out)
    except:
        pass
    reserved = ["group", "rowid"]
    if any(x in reserved for x in out.columns):
        reserved = ", ".join(reserved)
        raise ValueError("These column names are reserved and must not appear in `newdata`: {reserved}")
    if wts is not None:
        if isinstance(wts, str) is False or wts not in out.columns:
            raise ValueError("`wts` must be a column name in `newdata`.")
        out = out.with_columns(wts)
    return out


def sanitize_vcov(vcov, model):
    if isinstance(vcov, bool):
        if vcov is True:
            V = model.cov_params()
        else:
            V = None
    elif isinstance(vcov, str):
        lab = f"cov_{vcov}"
        if (hasattr(model, lab)):
            V = getattr(model, lab)
        else:
            raise ValueError(f"The model object has no {lab} attribute.")
    else:
        raise ValueError('`vcov` must be a boolean or a string like "HC3", which corresponds to an attribute of the model object such as "vcov_HC3".')
    # mnlogit returns pandas
    try:
        V = V.to_numpy()
    except:
        pass
    return V
