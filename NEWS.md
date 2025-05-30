# 0.1.2

* `datagrid(grid_type="counterfactual")` now accepts lambda functions.
* `get_dataset()` no longer requires the `package` argument.
* Group formula hypotheses are now supported `hypothesis="difference~reference|group"`
* `np.*()` is allowed in string/equation `hypothesis`.
* Better printing when `datagrid()` is used to specify column values explicitly.

# 0.1.1

Bugs:

* Allow predictors with missing values with `newdata=None`

New:

* `datagrid()` accepts functions: `datagrid(x = np.mean, y = lambda x: np.quantile(x, [0, .4]))`
* Major refactor and improvement of the `fit_*()` functions.

# 0.1.0

Breaking change:

* `hypothesis="reference"` and friends are deprecated. Use the formula syntax instead: `hypothesis=difference~reference`

New functions: 

* `get_dataset()`
* `fit_sklearn()`, `fit_statsmodels()`, `fit_linearmodels()`

New:

* `datagrid()` gets new arguments: `FUN_other`, `FUN_binary`, `FUN_numeric`, `FUN_character`
* `variables` is available in `avg_predictions()`
* `variables` accepts strings and list of strings in `predictions()`
* Regex supported in `joint_index` argument of `hypotheses()`. Issue #191.
* `comparisons()` allows reverse binary contrast by manually specifying `variables`. Issue #197.

Bugs:

* `datagrid(grid_type='balanced')` takes unique values of binary and categorical variables. Issue #156.
* `datagrid(grid_type='balanced')` does not return duplicates based on response. Issue 169.
* `comparisons(mod, variables="iqr")` and `minmax` now work. Issue #198.

# 0.0.14

* Thanks to Narwhals, marginaleffects can now ingest data frames in multiple formats and convert them to the Polars representation that we need internally. This no longer requires external dependencies like Pandas or DuckDB. Thanks to @artiom-matvei.

# 0.0.13

* Formulas should not include scale() or center(). Thanks to @alexjonesphd for reporting Issue #113.

Breaking change:

* `hypothesis` and `hypothesis` now index in a Python-like style by counting from 0 instead of counting from 1 as in R. Example code before the change  `predictions(mod, hypothesis = "b1 - b2 = 0")`; example correct code after change `predictions(mod, hypothesis = "b0 - b1 = 0")`
 
# 0.0.12

* Bug in datagrid() prevented "balanced" grid type. Thanks to @danielkberry for the fix (PR #104).
* Bug: Missing values leading to ValueError: "Something went wrong" in predictions() (Issue #83)
 
# 0.0.11

* Workaround for upstream regression in Polars.
* Bugfix for p value calculation in equivalence tests. Results could be incorrect.

# 0.0.10

* Polars 0.20.7 introduced a breaking change by error. Pinning version until thi is fixed. https://github.com/pola-rs/polars/issues/14401

# 0.0.9

* Issue #90: Informative error on reserved keyword like 'group'.
* Issue #91: find_variables() in class ModelStatsmodels does not return all variables which causes errors

# 0.0.8

* PyFixest: Fixed effects variables are recognized as categorical by `datagrid()`
* `MarginalEffectsDataFrame` class now has a `jacobian` attribute.

# 0.0.7

Breaking change:

* `datagridcf()` is deprecated. Use `datagrid(grid_type='counterfactual')` instead.

New:

* Support the `PyFixest` package. https://s3alfisc.github.io/pyfixest/
- `datagrid()` no longer requires specifying the `model` argument when called inside another marginaleffects function like `predictions()`.
* `eps_vcov` argument to control the step size in the computation of the Jacobian used for standard errors.
* plot_*() use `plotnine` instead of raw `matplotlib`
* plot_*() `condition` argument gets string shortcuts for numeric variables: "threenum", "fivenum", "minmax".
* `datagrid()` gets a `grid_type` argument: 'mean_or_mode', 'balance', 'counterfactual'
* Plot labels are sorted for clarity and consistency.
* `hypotheses()` function now supports raw models for conducting (non)-linear hypothesis tests on coefficients.

Misc:

* Refactor and several bug fixes in the `plot_*()` functions.
* Many bug fixes.
* Upgraded dependency on the `polars` package, with a shift from `.apply()` to `.map_*()` due to deprecation.
* Removed `pandas` dependency.


# 0.0.6

* `hypothesis` accepts a float or integer to specify a different null hypothesis.
* Better column order in printout when using `datagrid()` or `by`
* Version bump for dependencies.
* Equivalence test bug with duplicated column names.
* Minor bugs in plot_*() with unknown consequences.
* Linting.

# 0.0.5

* `predictions()` supports categorical predictors when `newdata` does not include all levels (internal padding).
* Better sorting of output, using the `by` argument.

# 0.0.4

* New function: `datagridcf()`
* `predictions()` supports categorical predictors when `newdata` does not include all levels (internal padding).

# 0.0.3

Breaking change:

* Rename argument to match `R` `marginaleffects`: `conf_int` -> `conf_level`

Misc:

* `MarginaleffectsDataFrame` class inherits from `pl.DataFrame` for better printing and to host useful attributes.

# 0.0.2

Misc:

* Better step size selection for the numerical derivatives used to compute delta method standard errors.

Bugs:

* When newdata was an unseen dataframe, out.columns would be referenced in sanity.py prior to assignment. Thanks to @Vinnie-Palazeti for PR #25.


# 0.0.1

Initial release
