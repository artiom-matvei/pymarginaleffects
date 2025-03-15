import polars as pl
import marginaleffects as me
from polars.testing import assert_series_equal
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from tests.helpers import wage_panel_pd

data = wage_panel_pd

# Fit two different types of models to test
formula = "lwage ~ exper * hours * educ * married - 1"
mod_linear = me.fit_sklearn(formula, data, engine=LinearRegression)
mod_rf = me.fit_sklearn(formula, data, engine=RandomForestRegressor)


def test_predictions_linear():  # std_error is missing for some reason
    pred = me.predictions(mod_linear)
    assert isinstance(pred, pl.DataFrame)
    assert "estimate" in pred.columns
    # assert "std_error" in pred.columns # std_error is missing for some reason
    assert len(pred) == len(data)


def test_predictions_random_forest():
    pred = me.predictions(mod_rf)
    assert isinstance(pred, pl.DataFrame)
    assert "estimate" in pred.columns
    # assert "std_error" in pred.columns # std_error is missing for some reason
    assert len(pred) == len(data)


def test_comparisons_linear():
    comp = me.comparisons(mod_linear, variables="exper").sort(
        ["term", "contrast", "rowid"]
    )
    assert isinstance(comp, pl.DataFrame)
    assert "estimate" in comp.columns
    # assert "std_error" in comp.columns # std_error is missing for some reason
    assert len(comp) == len(data)


def test_comparisons_random_forest():
    comp = me.comparisons(mod_rf, variables="exper").sort(["term", "contrast", "rowid"])
    assert isinstance(comp, pl.DataFrame)
    assert "estimate" in comp.columns
    # assert "std_error" in comp.columns # std_error is missing for some reason
    assert len(comp) == len(data)


def test_slopes_linear():
    slopes = me.slopes(mod_linear, variables="exper").sort(
        ["term", "contrast", "rowid"]
    )
    assert isinstance(slopes, pl.DataFrame)
    assert "estimate" in slopes.columns
    # assert "std_error" in slopes.columns # std_error is missing for some reason
    assert len(slopes) == len(data)


def test_slopes_random_forest():
    slopes = me.slopes(mod_rf, variables="exper").sort(["term", "contrast", "rowid"])
    assert isinstance(slopes, pl.DataFrame)
    assert "estimate" in slopes.columns
    # assert "std_error" in slopes.columns # std_error is missing for some reason
    assert len(slopes) == len(data)
