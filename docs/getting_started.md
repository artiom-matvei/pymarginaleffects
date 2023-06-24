# Install

``` sh
git clone https://github.com/vincentarelbundock/pymarginaleffects
cd pymarginaleffects
pip install .
```

# Dev environment + Test suite

Install poetry and dependencies:

``` sh
pip install poetry
cd pymarginaleffects
poetry install
```

Make file help:

``` sh
make help
```

Run test suite:

``` sh
make test
```

# Linear model

``` python
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import statsmodels.api as sm
from marginaleffects import *

np.random.seed(1024)

df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = pl.from_pandas(df)

# load data
df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = pl.from_pandas(df)

# recode
df = df.with_columns((pl.col("Area") > pl.col("Area").median()).alias("Bool"))
df = df.with_columns((pl.col("Distance") > pl.col("Distance").median()).alias("Bin"))
df = df.with_columns(df['Bin'].apply(lambda x: int(x), return_dtype=pl.Int32).alias('Bin'))
df = df.with_columns(pl.Series(np.random.choice(["a", "b", "c"], df.shape[0])).alias("Char"))

# fit model
mod = smf.ols("Literacy ~ Pop1831 * Desertion + Bool + Bin + Char", df)
fit = mod.fit()
```

# `comparisons()`

``` python
# `comparison`
comparisons(fit)
comparisons(fit, variables = "Pop1831", comparison = "differenceavg")
comparisons(fit, variables = "Pop1831", comparison = "difference").head()
comparisons(fit, variables = "Pop1831", comparison = "ratio").head()

# `by`
comparisons(fit, variables = "Pop1831", comparison = "difference", by = "Region")

# `vcov`
comparisons(fit, vcov = False, comparison = "differenceavg")
comparisons(fit, vcov = "HC3", comparison = "differenceavg")

# `variables` argument
comparisons(fit)
comparisons(fit, variables = "Pop1831")
comparisons(fit, variables = ["Pop1831", "Desertion"])
comparisons(fit, variables = {"Pop1831": 1000, "Desertion": 2})
comparisons(fit, variables = {"Pop1831": [100, 2000]})
```

# `predictions()`

``` python
predictions(fit).head()
predictions(fit, by = "Region")

# `hypothesis` lincome matrices
hyp = np.array([1, 0, -1, 0, 0, 0])
predictions(fit, by = "Region", hypothesis = hyp)

hyp = np.vstack([
    [1, 0, -1, 0, 0, 0],
    [1, 0, 0, -1, 0, 0]
]).T
predictions(fit, by = "Region", hypothesis = hyp)

# equivalent to:
p = predictions(fit, by = "Region")
print(p["estimate"][0] - p["estimate"][2])
print(p["estimate"][0] - p["estimate"][3])
```

# `hypotheses()`

``` python
hyp = np.array([0, 0, 1, -1, 0, 0, 0, 0])
hypotheses(fit, hypothesis = hyp)
```

# GLM

``` python
df = sm.datasets.get_rdataset("Guerry", "HistData").data
mod = smf.ols("Literacy ~ Pop1831 * Desertion", df)
fit = mod.fit()
df["bin"] = df["Literacy"] > df["Literacy"].median()
df["bin"] = df["bin"].replace({True: 1, False: 0})
mod = smf.glm("bin ~ Pop1831 * Desertion", df, family = sm.families.Binomial())
fit = mod.fit()
comparisons(fit)
```