import numpy as np

from tox.constants import TASKS
from tox.metrics import score

rng = np.random.default_rng(0)


def _labels(n=50):
    y = rng.integers(0, 2, size=(n, len(TASKS))).astype(float)
    y[0, :], y[1, :] = 0, 1          # make sure every task has both classes
    return y


def test_perfect_predictions_give_auc_1():
    y = _labels()
    df = score(y, y.copy())
    assert np.allclose(df.roc_auc, 1.0)
    assert np.allclose(df.pr_auc, 1.0)


def test_nan_labels_are_excluded():
    y = _labels()
    p = y.copy()
    y[10:20, 0] = np.nan
    p[10:20, 0] = 1 - np.nan_to_num(p[10:20, 0])   # garbage preds on the masked rows
    df = score(y, p)
    assert df.loc[0, "n"] == 40
    assert df.loc[0, "roc_auc"] == 1.0


def test_single_class_task_returns_nan_not_crash():
    y = _labels()
    y[:, 3] = 0
    df = score(y, rng.random(y.shape))
    assert np.isnan(df.loc[3, "roc_auc"]) and np.isnan(df.loc[3, "pr_auc"])
    assert df.loc[3, "n_pos"] == 0


def test_pos_rate_is_reported():
    y = _labels()
    df = score(y, rng.random(y.shape))
    assert np.allclose(df.pos_rate, np.nanmean(y, axis=0))
