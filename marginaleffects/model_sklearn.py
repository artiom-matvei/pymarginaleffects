import numpy as np
import warnings
import polars as pl
from .utils import ingest
from .model_abstract import ModelAbstract


class ModelSklearn(ModelAbstract):
    def __init__(self, model):
        if not hasattr(model, "data"):
            raise ValueError("Model must have a 'data' attribute")
        else:
            self.data = ingest(model.data)
        if not hasattr(model, "formula"):
            raise ValueError("Model must have a 'formula' attribute")
        else:
            self.formula = model.formula
        super().__init__(model)

    def get_predict(self, params, newdata: pl.DataFrame):
        if isinstance(newdata, np.ndarray):
            exog = newdata
        else:
            try:
                import formulaic

                if isinstance(newdata, formulaic.ModelMatrix):
                    exog = newdata.to_numpy()
                else:
                    if isinstance(newdata, pl.DataFrame):
                        nd = newdata.to_pandas()
                    else:
                        nd = newdata
                    y, exog = formulaic.model_matrix(self.model.formula, nd)
                    exog = exog.to_numpy()
            except ImportError:
                raise ImportError(
                    "The formulaic package is required to use this feature."
                )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                p = self.model.predict_proba(exog)
                # only keep the second column for binary classification since it is redundant info
                if p.shape[1] == 2:
                    p = p[:, 1]
        except (AttributeError, NotImplementedError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                p = self.model.predict(exog)

        if p.ndim == 1:
            p = pl.DataFrame({"rowid": range(newdata.shape[0]), "estimate": p})
        elif p.ndim == 2:
            colnames = {f"column_{i}": v for i, v in enumerate(self.model.classes_)}
            p = (
                pl.DataFrame(p)
                .rename(colnames)
                .with_columns(
                    pl.Series(range(p.shape[0]), dtype=pl.Int32).alias("rowid")
                )
                .melt(id_vars="rowid", variable_name="group", value_name="estimate")
            )
        else:
            raise ValueError(
                "The `predict()` method must return an array with 1 or 2 dimensions."
            )

        p = p.with_columns(pl.col("rowid").cast(pl.Int32))

        return p