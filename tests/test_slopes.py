import statsmodels.formula.api as smf
from marginaleffects import *
from .utilities import *
from marginaleffects.comparisons import estimands
from polars.testing import assert_series_equal
import polars as pl

mtcars = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
mod_py = smf.ols("mpg ~ wt * hp", mtcars).fit()

def test_comparison_derivatives():
    est = ["dydx", "dydxavg", "dyex", "dyexavg", "eydx", "eydxavg", "eyex", "eyexavg"]
    for e in est:
        cmp_py = comparisons(mod_py, comparison = e)
        cmp_r = pl.read_csv(f"tests/r/test_slopes_01_{e}.csv")
        compare_r_to_py(cmp_r, cmp_py, msg = e, tolr = 1e-5, tola = 1e-5)

test_comparison_derivatives()


def test_slopes():
    est = ["dydx", "dydx", "dyex", "eydx", "eyex"]
    for e in est:
        cmp_py = slopes(mod_py, slope = e)
        cmp_r = pl.read_csv(f"tests/r/test_slopes_01_{e}.csv")
        compare_r_to_py(cmp_r, cmp_py, msg = e, tolr = 1e-5, tola = 1e-5)

    
def test_slopes_padding():
    dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv") \
        .with_columns(pl.col("cyl").cast(pl.Utf8))
    mod = smf.ols("mpg ~ cyl + hp", dat.to_pandas()).fit()
    s = slopes(mod, newdata = "mean")
    assert s.shape[0] == 3


def test_bug_newdata_variables():
    dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv") \
        .with_columns(pl.col("cyl").cast(pl.Utf8))
    mod = smf.ols("mpg ~ cyl + hp", dat.to_pandas()).fit()
    s = slopes(mod, newdata = "mean", variables = "hp")
    assert s.shape[0] == 1