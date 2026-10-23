import numpy as np
import pytest
import xgboost as xgb

from tox.models_aml import make_model, xgb_gain

rng = np.random.default_rng(0)
X = rng.random((400, 10))
y = (X[:, 0] + 0.3 * rng.random(400) > 0.6).astype(float)
Xv, yv = rng.random((200, 10)), rng.integers(0, 2, 200).astype(float)   # noise val -> stops early


@pytest.fixture(scope="module")
def stopped():
    m = make_model("xgb", {"max_depth": 3}, 0, y)
    return m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)


def test_early_stopping_leaves_extra_trees(stopped):
    # the case the fix is about: the booster holds more trees than predict_proba uses
    assert stopped.get_booster().num_boosted_rounds() > stopped.best_iteration + 1


def test_gain_only_counts_trees_up_to_best_iteration(stopped):
    kept = stopped.get_booster()[: stopped.best_iteration + 1]
    assert xgb_gain(stopped) == kept.get_score(importance_type="gain")
    assert xgb_gain(stopped) != stopped.get_booster().get_score(importance_type="gain")


def test_kept_trees_are_the_ones_predict_proba_uses(stopped):
    kept = stopped.get_booster()[: stopped.best_iteration + 1]
    assert np.allclose(stopped.predict_proba(Xv)[:, 1], kept.predict(xgb.DMatrix(Xv)))


def test_no_early_stopping_uses_every_tree():
    m = make_model("xgb", {"max_depth": 3}, 0, y, early_stop=False).set_params(n_estimators=20).fit(X, y)
    assert xgb_gain(m) == m.get_booster().get_score(importance_type="gain")
