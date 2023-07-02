# Get Started

## Installation

Install the latest PyPi release:

``` python
pip install marginaleffects
```

## Estimands: Predictions, Comparisons, and Slopes

The `marginaleffects` package allows `python` users to compute and plot
three principal quantities of interest: (1) predictions, (2)
comparisons, and (3) slopes. In addition, the package includes a
convenience function to compute a fourth estimand, “marginal means”,
which is a special case of averaged predictions. `marginaleffects` can
also average (or “marginalize”) unit-level (or “conditional”) estimates
of all those quantities, and conduct hypothesis tests on them.

[*Predictions*:](predictions.html)

> The outcome predicted by a fitted model on a specified scale for a
> given combination of values of the predictor variables, such as their
> observed values, their means, or factor levels. a.k.a. Fitted values,
> adjusted predictions. `predictions()`, `avg_predictions()`,
> `plot_predictions()`.

[*Comparisons*:](comparisons.html)

> Compare the predictions made by a model for different regressor values
> (e.g., college graduates vs. others): contrasts, differences, risk
> ratios, odds, etc. `comparisons()`, `avg_comparisons()`,
> `plot_comparisons()`.

[*Slopes*:](slopes.html)

> Partial derivative of the regression equation with respect to a
> regressor of interest. a.k.a. Marginal effects, trends. `slopes()`,
> `avg_slopes()`, `plot_slopes()`.

[Hypothesis and Equivalence Tests:](hypothesis.html)

> Hypothesis and equivalence tests can be conducted on linear or
> non-linear functions of model coefficients, or on any of the
> quantities computed by the `marginaleffects` packages (predictions,
> slopes, comparisons, marginal means, etc.). Uncertainy estimates can
> be obtained via the delta method (with or without robust standard
> errors), bootstrap, or simulation.

Predictions, comparisons, and slopes are fundamentally unit-level (or
“conditional”) quantities. Except in the simplest linear case, estimates
will typically vary based on the values of all the regressors in a
model. Each of the observations in a dataset is thus associated with its
own prediction, comparison, and slope estimates. Below, we will see that
it can be useful to marginalize (or “average over”) unit-level estimates
to report an “average prediction”, “average comparison”, or “average
slope”.

We now apply `marginaleffects` functions to compute each of the
estimands described above. First, we fit a linear regression model with
multiplicative interactions:

#### Predictions

``` python
import numpy as np
import polars as pl
from marginaleffects import *
import statsmodels.formula.api as smf
mtcars = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
mod = smf.ols("mpg ~ hp * wt * am", data = mtcars).fit()

print(mod.summary().as_text())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    mpg   R-squared:                       0.896
    Model:                            OLS   Adj. R-squared:                  0.866
    Method:                 Least Squares   F-statistic:                     29.55
    Date:                Sun, 02 Jul 2023   Prob (F-statistic):           2.60e-10
    Time:                        08:45:36   Log-Likelihood:                -66.158
    No. Observations:                  32   AIC:                             148.3
    Df Residuals:                      24   BIC:                             160.0
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     40.3272     13.008      3.100      0.005      13.480      67.175
    hp            -0.0888      0.065     -1.372      0.183      -0.222       0.045
    wt            -4.7968      4.002     -1.199      0.242     -13.056       3.462
    hp:wt          0.0145      0.019      0.755      0.458      -0.025       0.054
    am            12.8371     14.222      0.903      0.376     -16.517      42.191
    hp:am         -0.0326      0.089     -0.366      0.717      -0.216       0.151
    wt:am         -5.3620      4.597     -1.166      0.255     -14.851       4.127
    hp:wt:am       0.0178      0.026      0.680      0.503      -0.036       0.072
    ==============================================================================
    Omnibus:                        1.875   Durbin-Watson:                   2.205
    Prob(Omnibus):                  0.392   Jarque-Bera (JB):                1.588
    Skew:                           0.528   Prob(JB):                        0.452
    Kurtosis:                       2.721   Cond. No.                     3.32e+04
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 3.32e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.

Then, we call the `predictions()` function. As noted above, predictions
are unit-level estimates, so there is one specific prediction per
observation. By default, the `predictions()` function makes one
prediction per observation in the dataset that was used to fit the
original model. Since `mtcars` has 32 rows, the `predictions()` outcome
also has 32 rows:

``` python
pre = predictions(mod)

pre.shape

print(pre.head())
```

    | rowid | estimate  | std_error | statistic | … | vs  | am  | gear | carb |
    |-------|-----------|-----------|-----------|---|-----|-----|------|------|
    | 0     | 22.488569 | 0.884149  | 25.43528  | … | 0   | 1   | 4    | 4    |
    | 1     | 20.801859 | 1.194205  | 17.419002 | … | 0   | 1   | 4    | 4    |
    | 2     | 25.264652 | 0.708531  | 35.657806 | … | 1   | 1   | 4    | 1    |
    | 3     | 20.255492 | 0.704464  | 28.753051 | … | 1   | 0   | 3    | 1    |
    | 4     | 16.997817 | 0.711866  | 23.877839 | … | 0   | 0   | 3    | 2    |

#### Comparisons: Differences, Ratios, Log-Odds, Lift, etc.

Now, we use the `comparisons()` function to compute the difference in
predicted outcome when each of the predictors is incremented by 1 unit
(one predictor at a time, holding all others constant). Once again,
comparisons are unit-level quantities. And since there are 3 predictors
in the model and our data has 32 rows, we obtain 96 comparisons:

``` python
cmp = comparisons(mod)

cmp.shape

print(cmp.head())
```

    | rowid | term | contrast | estimate  | … | carb | predicted | predicted_lo | predicted_hi |
    |-------|------|----------|-----------|---|------|-----------|--------------|--------------|
    | 0.0   | wt   | +1       | -6.614549 | … | 4.0  | 22.488569 | 25.795843    | 19.181294    |
    | 1.0   | wt   | +1       | -6.614549 | … | 4.0  | 20.801859 | 24.109133    | 17.494584    |
    | 2.0   | wt   | +1       | -7.162295 | … | 1.0  | 25.264652 | 28.845799    | 21.683504    |
    | 3.0   | wt   | +1       | -3.206559 | … | 1.0  | 20.255492 | 21.858772    | 18.652213    |
    | 4.0   | wt   | +1       | -2.266852 | … | 2.0  | 16.997817 | 18.131243    | 15.864392    |

The `comparisons()` function allows customized queries. For example,
what happens to the predicted outcome when the `hp` variable increases
from 100 to 120?

``` python
cmp = comparisons(mod, variables = {"hp": [120, 100]})
print(cmp)
```

    | rowid | term | contrast  | estimate | … | carb | predicted | predicted_lo | predicted_hi |
    |-------|------|-----------|----------|---|------|-----------|--------------|--------------|
    | 0.0   | hp   | 100 - 120 | 0.738111 | … | 4.0  | 22.488569 | 22.119513    | 22.857624    |
    | 1.0   | hp   | 100 - 120 | 0.573787 | … | 4.0  | 20.801859 | 20.514965    | 21.088752    |
    | 2.0   | hp   | 100 - 120 | 0.931433 | … | 1.0  | 25.264652 | 24.007217    | 24.93865     |
    | 3.0   | hp   | 100 - 120 | 0.845426 | … | 1.0  | 20.255492 | 19.83278     | 20.678205    |
    | …     | …    | …         | …        | … | …    | …         | …            | …            |
    | 28.0  | hp   | 100 - 120 | 0.383687 | … | 4.0  | 15.896176 | 18.658723    | 19.04241     |
    | 29.0  | hp   | 100 - 120 | 0.64145  | … | 6.0  | 19.411674 | 21.175661    | 21.817111    |
    | 30.0  | hp   | 100 - 120 | 0.125924 | … | 8.0  | 14.7881   | 16.141785    | 16.267709    |
    | 31.0  | hp   | 100 - 120 | 0.635006 | … | 2.0  | 21.461991 | 21.112738    | 21.747744    |

What happens to the predicted outcome when the `wt` variable increases
by 1 standard deviation about its mean?

``` python
cmp = comparisons(mod, variables = {"hp": "sd"})
print(cmp)
```

    | rowid | term | contrast         | estimate  | … | carb | predicted | predicted_lo | predicted_hi |
    |-------|------|------------------|-----------|---|------|-----------|--------------|--------------|
    | 0.0   | hp   | +68.562868489320 | -2.530351 | … | 4.0  | 22.488569 | 23.753744    | 21.223393    |
    |       |      | 59               |           |   |      |           |              |              |
    | 1.0   | hp   | +68.562868489320 | -1.967025 | … | 4.0  | 20.801859 | 21.785371    | 19.818346    |
    |       |      | 59               |           |   |      |           |              |              |
    | 2.0   | hp   | +68.562868489320 | -3.193087 | … | 1.0  | 25.264652 | 26.861195    | 23.668108    |
    |       |      | 59               |           |   |      |           |              |              |
    | 3.0   | hp   | +68.562868489320 | -2.89824  | … | 1.0  | 20.255492 | 21.704613    | 18.806372    |
    |       |      | 59               |           |   |      |           |              |              |
    | …     | …    | …                | …         | … | …    | …         | …            | …            |
    | 28.0  | hp   | +68.562868489320 | -1.315334 | … | 4.0  | 15.896176 | 16.553843    | 15.238509    |
    |       |      | 59               |           |   |      |           |              |              |
    | 29.0  | hp   | +68.562868489320 | -2.198983 | … | 6.0  | 19.411674 | 20.511165    | 18.312182    |
    |       |      | 59               |           |   |      |           |              |              |
    | 30.0  | hp   | +68.562868489320 | -0.431686 | … | 8.0  | 14.7881   | 15.003943    | 14.572257    |
    |       |      | 59               |           |   |      |           |              |              |
    | 31.0  | hp   | +68.562868489320 | -2.176891 | … | 2.0  | 21.461991 | 22.550437    | 20.373545    |
    |       |      | 59               |           |   |      |           |              |              |

The `comparisons()` function also allows users to specify arbitrary
functions of predictions, with the `comparison` argument. For example,
what is the average ratio between predicted Miles per Gallon after an
increase of 50 units in Horsepower?

``` python
cmp = comparisons(
  mod,
  variables = {"hp": 50},
  comparison = "ratioavg")
print(cmp)
```

    | term | contrast | estimate | std_error | … | p_value | s_value | conf_low | conf_high |
    |------|----------|----------|-----------|---|---------|---------|----------|-----------|
    | hp   | +50      | 0.909534 | 0.029058  | … | 0.0     | inf     | 0.84956  | 0.969507  |

#### Slopes: Derivatives and elasticities

Consider a logistic regression model with a single predictor:

``` python
url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"
mtcars = pl.read_csv(url)
mod = smf.logit("am ~ mpg", data = mtcars).fit()
```

    Optimization terminated successfully.
             Current function value: 0.463674
             Iterations 6

We can estimate the slope of the prediction function with respect to the
`mpg` variable at any point in the data space. For example, what is the
slope of the prediction function at `mpg = 24`?

``` python
mfx = slopes(mod, newdata = datagrid(mpg = 24, newdata = mtcars))
print(mfx)
```

    | term | contrast | estimate | std_error | … | p_value  | s_value   | conf_low | conf_high |
    |------|----------|----------|-----------|---|----------|-----------|----------|-----------|
    | mpg  | +0.0001  | 0.066534 | 0.01779   | … | 0.000776 | 10.331677 | 0.030203 | 0.102866  |

This is equivalent to the result we obtain by taking the analytical
derivative using the chain rule:

``` python
from scipy.stats import logistic
beta_0 = mod.params.iloc[0]
beta_1 = mod.params.iloc[1]
print(beta_1 * logistic.pdf(beta_0 + beta_1 * 24))
```

    0.06653436463892946

This computes a “marginal effect at the mean” (or “slope at the mean”):

``` python
mfx = slopes(mod, newdata = datagrid(newdata = mtcars))
print(mfx)
```

    | term | contrast | estimate | std_error | … | p_value  | s_value  | conf_low | conf_high |
    |------|----------|----------|-----------|---|----------|----------|----------|-----------|
    | mpg  | +0.0001  | 0.073235 | 0.028289  | … | 0.014712 | 6.086849 | 0.015461 | 0.13101   |

We can also compute an “average slope” or “average marginaleffects”

``` python
mfx = avg_slopes(mod)
print(mfx)
```

    | term | contrast | estimate | std_error | … | p_value  | s_value   | conf_low | conf_high |
    |------|----------|----------|-----------|---|----------|-----------|----------|-----------|
    | mpg  | +0.0001  | 0.046486 | 0.008864  | … | 0.000012 | 16.384139 | 0.028382 | 0.06459   |

Which again is equivalent to the analytical result:

``` python
np.mean(beta_1 * logistic.pdf(beta_0 + beta_1 * mtcars["mpg"]))
```

    0.04648596405936302

## Grid

Predictions, comparisons, and slopes are typically “conditional”
quantities which depend on the values of all the predictors in the
model. By default, `marginaleffects` functions estimate quantities of
interest for the empirical distribution of the data (i.e., for each row
of the original dataset). However, users can specify the exact values of
the predictors they want to investigate by using the `newdata` argument.

`newdata` accepts data frames like this:

``` python
pre = predictions(mod, newdata = mtcars.tail(2))
print(pre)
```

    | rowid | estimate | std_error | statistic | … | vs  | am  | gear | carb |
    |-------|----------|-----------|-----------|---|-----|-----|------|------|
    | 0     | 0.119402 | 0.077817  | 1.534391  | … | 0   | 1   | 5    | 8    |
    | 1     | 0.49172  | 0.119614  | 4.110899  | … | 1   | 1   | 4    | 2    |

The [`datagrid` function gives us a powerful way to define a grid of
predictors.](https://vincentarelbundock.github.io/marginaleffects/reference/datagrid.html)
All the variables not mentioned explicitly in `datagrid()` are fixed to
their mean or mode:

``` python
pre = predictions(
  mod,
  newdata = datagrid(
    newdata = mtcars,
    am = [0, 1],
    wt = [mtcars["wt"].max(), mtcars["wt"].min()]))

print(pre)
```

    | rowid | estimate | std_error | statistic | … | qsec     | vs  | gear | carb |
    |-------|----------|-----------|-----------|---|----------|-----|------|------|
    | 0     | 0.3929   | 0.108367  | 3.625643  | … | 17.84875 | 0   | 3    | 2    |
    | 1     | 0.3929   | 0.108367  | 3.625643  | … | 17.84875 | 0   | 3    | 2    |
    | 2     | 0.3929   | 0.108367  | 3.625643  | … | 17.84875 | 0   | 3    | 2    |
    | 3     | 0.3929   | 0.108367  | 3.625643  | … | 17.84875 | 0   | 3    | 2    |

## Averaging

Since predictions, comparisons, and slopes are conditional quantities,
they can be a bit unwieldy. Often, it can be useful to report a
one-number summary instead of one estimate per observation. Instead of
presenting “conditional” estimates, some methodologists recommend
reporting “marginal” estimates, that is, an average of unit-level
estimates.

(This use of the word “marginal” as “averaging” should not be confused
with the term “marginal effect” which, in the econometrics tradition,
corresponds to a partial derivative, or the effect of a “small/marginal”
change.)

To marginalize (average over) our unit-level estimates, we can use the
`by` argument or the one of the convenience functions:
`avg_predictions()`, `avg_comparisons()`, or `avg_slopes()`. For
example, both of these commands give us the same result: the average
predicted outcome in the `mtcars` dataset:

``` python
pre = avg_predictions(mod)
print(pre)
```

    | estimate | std_error | statistic | p_value  | s_value   | conf_low | conf_high |
    |----------|-----------|-----------|----------|-----------|----------|-----------|
    | 0.40625  | 0.068786  | 5.906026  | 0.000002 | 19.072686 | 0.265771 | 0.546729  |

This is equivalent to manual computation by:

``` python
np.mean(mod.predict())
```

    0.40624999999999994

The main `marginaleffects` functions all include a `by` argument, which
allows us to marginalize within sub-groups of the data. For example,

``` python
cmp = avg_comparisons(mod, by = "am")
print(cmp)
```

    | am  | term | contrast | estimate | … | p_value   | s_value   | conf_low | conf_high |
    |-----|------|----------|----------|---|-----------|-----------|----------|-----------|
    | 1.0 | mpg  | +1       | 0.044926 | … | 1.4198e-7 | 22.747797 | 0.031486 | 0.058365  |
    | 0.0 | mpg  | +1       | 0.04751  | … | 0.000284  | 11.779712 | 0.023884 | 0.071135  |

Marginal Means are a special case of predictions, which are marginalized
(or averaged) across a balanced grid of categorical predictors. To
illustrate, we estimate a new model with categorical predictors:

``` python
dat = mtcars \
  .with_columns(
    pl.col("am").cast(pl.Boolean),
    pl.col("cyl").cast(pl.Utf8)
  )
mod_cat = smf.ols("mpg ~ am + cyl + hp", data = dat).fit()
```

We can compute marginal means manually using the functions already
described:

``` python
pre = avg_predictions(
  mod_cat,
  newdata = datagrid(
    newdata = dat,
    cyl = dat["cyl"].unique(),
    am = dat["am"].unique()),
  by = "am")

print(pre)
```

``` python
cmp = avg_comparisons(mod_cat)
print(cmp)
```

    | term | contrast     | estimate  | std_error | … | p_value  | s_value  | conf_low  | conf_high |
    |------|--------------|-----------|-----------|---|----------|----------|-----------|-----------|
    | cyl  | 6 - 4        | -3.924578 | 1.537515  | … | 0.016663 | 5.907182 | -7.079298 | -0.769859 |
    | cyl  | 8 - 4        | -3.533414 | 2.502788  | … | 0.169433 | 2.561213 | -8.668711 | 1.601883  |
    | hp   | +1           | -0.044244 | 0.014576  | … | 0.005266 | 7.569022 | -0.074151 | -0.014337 |
    | am   | mean(True) - | 4.157856  | 1.25655   | … | 0.00266  | 8.554463 | 1.579629  | 6.736084  |
    |      | mean(False)  |           |           |   |          |          |           |           |

## Hypothesis and equivalence tests

The `hypotheses()` function and the `hypothesis` argument can be used to
conduct linear and non-linear hypothesis tests on model coefficients, or
on any of the quantities computed by the functions introduced above.

Consider this model:

``` python
mod = smf.ols("mpg ~ qsec * drat", data = mtcars).fit()
mod.params
```

    Intercept    12.337199
    qsec         -1.024118
    drat         -3.437146
    qsec:drat     0.597315
    dtype: float64

Can we reject the null hypothesis that the `drat` coefficient is 2 times
the size of the `qsec` coefficient?

``` python
hyp = hypotheses(mod, "b3 = 2 * b2")
print(hyp)
```

    | term    | estimate  | std_error | statistic | p_value  | s_value  | conf_low   | conf_high |
    |---------|-----------|-----------|-----------|----------|----------|------------|-----------|
    | b3=2*b2 | -1.388909 | 10.77593  | -0.12889  | 0.898366 | 0.154625 | -23.462402 | 20.684583 |

The main functions in `marginaleffects` all have a `hypothesis`
argument, which means that we can do complex model testing. For example,
consider two slope estimates:

``` python
range = lambda x: [x.max(), x.min()]
cmp = comparisons(
  mod,
  variables = "drat",
  newdata = datagrid(newdata = mtcars, qsec = range(mtcars["qsec"])))
print(cmp)
```

    | rowid | term | contrast | estimate  | … | carb | predicted | predicted_lo | predicted_hi |
    |-------|------|----------|-----------|---|------|-----------|--------------|--------------|
    | 0.0   | drat | +1       | 10.241374 | … | 4.0  | 25.718632 | 20.597945    | 30.839319    |
    | 1.0   | drat | +1       | 5.223926  | … | 4.0  | 16.275659 | 13.663696    | 18.887621    |

Are these two contrasts significantly different from one another? To
test this, we can use the `hypothesis` argument:

``` python
cmp = comparisons(
  mod,
  hypothesis = "b1 = b2",
  variables = "drat",
  newdata = datagrid(newdata = mtcars, qsec = range(mtcars["qsec"])))
print(cmp)
```

    | term  | estimate | std_error | statistic | p_value  | s_value  | conf_low   | conf_high |
    |-------|----------|-----------|-----------|----------|----------|------------|-----------|
    | b1=b2 | 5.017448 | 8.519298  | 0.588951  | 0.560616 | 0.834915 | -12.433542 | 22.468439 |
