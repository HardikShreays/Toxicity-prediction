"""A2 - LR / RF / XGBoost, one model per endpoint, plus the train-only descriptor preprocessing."""
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from tox.constants import CONFIGS

CFG = yaml.safe_load((CONFIGS / "aml.yaml").read_text())


def _inf_to_nan(x):
    return np.where(np.isinf(x), np.nan, x)


def desc_pipeline():
    """inf -> NaN -> median impute -> drop columns constant on train -> standardize.
    Always fit on train rows only; wrapping it in a Pipeline means that can't go wrong by accident."""
    return make_pipeline(
        FunctionTransformer(_inf_to_nan, feature_names_out="one-to-one"),
        SimpleImputer(strategy="median"),
        VarianceThreshold(0.0),
        StandardScaler(),
    )


def grid(algo):
    """Hyper-parameter grid for one algorithm (small on purpose, tuned on val PR-AUC)."""
    c = CFG[algo]
    if algo == "lr":
        return [{"C": C} for C in c["C"]]
    if algo == "rf":
        return [{"min_samples_leaf": m} for m in c["min_samples_leaf"]]
    return [{"max_depth": d} for d in c["max_depth"]]


def make_model(algo, params, seed, y_train, early_stop=True):
    if algo == "lr":
        return LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000,
                                  random_state=seed, **params)
    if algo == "rf":
        return RandomForestClassifier(n_estimators=CFG["rf"]["n_estimators"], max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=seed, **params)
    c = CFG["xgb"]
    n_pos = int(y_train.sum())
    return XGBClassifier(
        n_estimators=c["n_estimators"], learning_rate=c["learning_rate"], subsample=c["subsample"],
        colsample_bytree=c["colsample_bytree"], tree_method="hist", eval_metric="aucpr",
        scale_pos_weight=(len(y_train) - n_pos) / n_pos,        # n_neg / n_pos, from train only
        early_stopping_rounds=c["early_stopping_rounds"] if early_stop else None,
        n_jobs=-1, random_state=seed, **params)


def xgb_gain(model):
    """Gain importance from the trees predict_proba actually uses. With early stopping the booster keeps
    the `early_stopping_rounds` trees grown after best_iteration, which must not count."""
    booster = model.get_booster()
    try:
        booster = booster[: model.best_iteration + 1]
    except AttributeError:                      # no early stopping -> every tree is used
        pass
    return booster.get_score(importance_type="gain")
