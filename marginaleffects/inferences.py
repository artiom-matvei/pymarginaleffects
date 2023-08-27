import numpy as np
import polars as pl
from marginaleffects.utils import *
from marginaleffects import predictions

def inferences(x, method="conformal_cv+", R=10, conformal_test=None, conformal_calibration=None, conformal_score="residual_abs"):
    assert conformal_score in ["residual_abs", "residual_sq", "softmax"]
    assert method in ["conformal_cv+", "conformal_split"]
    assert R <= 25
    if method == "conformal_cv+":
        out = conformal_cv_plus(x, test=conformal_test, R=R, score=conformal_score, conf_level=0.95) 
    else:
        conformal_split(x, test=conformal_test, calibration=conformal_calibration, score=conformal_score, conf_level=0.95)
    return 2.




def get_conformal_score(x, score):
    response_name = x.model.model.endog_names
    response = x[response_name]
    if not isinstance(response, (int, float)) and score != "softmax":
        raise ValueError("The response must be numeric. Did you want to use `conformal_score='softmax'`?")
    if score == "residual_abs":
        out = abs(response - x.estimate)
    elif score == "residual_sq":
        out = (response - x.estimate) ** 2
    elif score == "softmax":
        model = x.model
        response = x[insight.find_response(model)]
        if isinstance(response, (int, float)) and all(val in [0, 1] for val in response):
            # See p.4 of Angelopoulos, Anastasios N., and Stephen Bates. 2022. “A
            # Gentle Introduction to Conformal Prediction and Distribution-Free
            # Uncertainty Quantification.” arXiv.
            # https://doi.org/10.48550/arXiv.2107.07511.
            # 1 minus the softmax output of the true class
            out = 1 - np.where(response == 1, x.estimate, 1 - x.estimate)
        elif "group" in x.columns:
            # HACK: is this fragile? I think `group` should always be character.
            idx = response.astype(str) == x["group"].astype(str)
            out = 1 - x.estimate[idx]
        else:
            raise ValueError("Failed to compute the conformity score.")
    return out


def get_conformal_bounds(x, score, conf_level):
    d = min(score[score > np.quantile(score, conf_level)])
    # if "group" in x.columns:
    #     response_name = x.model.model.endog_names
    #     response = x[response_name]
    #     q = np.quantile(score, (len(score) + 1) * conf_level / len(score))
    #     out = x[x.estimate > (1 - q)]
    #     out = out.groupby(["rowid", response_name])["group"].unique().reset_index()
    #     out = out.sort_values("rowid")
    #     return out
    # else:
    # continuous outcome: conformity half-width
    x["pred.low"] = x["estimate"] - d
    x["pred.high"] = x["estimate"] + d
    return x


def conformal_split(x, test, calibration, score, conf_level):
    # calibration
    # use original model---fitted on the training set---to make predictions in the calibration set
    # p_calib is the `predictions()` call, which we re-evaluate on newdata=calibration
    p_calib = x["call_args"]
    p_calib["newdata"] = calibration
    p_calib["vcov"] = False  # faster
    p_calib = predictions(**p_calib)
    score = get_conformal_score(p_calib, score=score)

    # test
    # use original model to make predictions in the test set
    p_test = x["call_args"]
    p_test["newdata"] = test
    p_test = predictions(**p_test)

    # bounds
    out = get_conformal_bounds(p_test, score=score, conf_level=conf_level)

    return out


def conformal_cv_plus(x, test, R, score, conf_level):
    # cross-validation
    train = get_modeldata(x.call_args["model"])
    idx = np.array_split(np.random.permutation(range(train.shape[0])), R)
    scores = []
    for i in idx:
        not_i = [x for x in range(train.shape[0]) if x not in i]
        data_cv = train.rows(not_i)
        # re-fit the original model on training sets withholding the CV fold
        # model_cv = call_args["model"]update(x["model"], data=data_cv)
        # use the updated model to make out-of-fold predictions
        # call_cv is the `predictions()` call, which we re-evaluate in-fold: newdata=train[i,]
        call_cv = x["call"].copy()
        call_cv["model"] = model_cv
        call_cv["newdata"] = train.loc[i]
        call_cv["vcov"] = False  # faster
        pred_cv = predictions(call_cv)
        # save the scores form each fold
        scores.append(get_conformal_score(pred_cv, score=score))

    # test
    out = x["call"].copy()
    out["newdata"] = test
    return get_conformal_bounds(out, score=np.concatenate(scores), conf_level=conf_level)