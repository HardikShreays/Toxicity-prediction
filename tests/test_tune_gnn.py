import json

import numpy as np
import pytest
import torch

from tox.constants import SEEDS
from tox.featurize_graph import build_graphs
from tox.train_gnn import (GNN_PARAMS, RUNS, TUNED_KEYS, frozen_config, load_config,
                           load_split_graphs, run_grid)
from tox.tune_gnn import GRID, SELECTION_METRIC, tune

SMILES = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)O",
          "CC(=O)N", "CCOC", "CCSC", "C1CCCCC1", "C1CCCC1", "CC(C)O", "CC(C)N", "OCCO", "NCCN"]


@pytest.fixture(scope="module")
def graphs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (len(SMILES), 12)).astype(np.float32)
    y[:, 0] = np.arange(len(SMILES)) % 2
    y[rng.random(y.shape) < 0.3] = np.nan
    g = build_graphs(SMILES, y)
    return {"train": g[:12], "val": g[12:]}                      # NO test key on purpose


def test_grid_is_the_plan_grid():
    assert GRID == {"hidden": [64, 128, 256], "n_layers": [3, 4], "dropout": [0.1, 0.3]}
    assert TUNED_KEYS == ("hidden", "n_layers", "dropout")
    assert SELECTION_METRIC == "val_mean_pr_auc"


def test_tuning_picks_highest_val_pr_auc_not_roc(graphs, tmp_path, monkeypatch):
    scripted = iter([  # (pr, roc) per trial in grid order; best PR is trial 2, best ROC is trial 3
        (0.30, 0.60), (0.50, 0.70), (0.40, 0.90), (0.10, 0.50),
    ])
    def fake_fit(model, tr, va, cfg, seed, pw=None):
        pr, roc = next(scripted)
        return {"best_val_pr_auc": pr, "best_val_roc_auc": roc, "best_epoch": 1, "epochs_run": 5}
    monkeypatch.setattr("tox.tune_gnn.fit", fake_fit)
    grid = {"hidden": [8, 16], "n_layers": [1, 2], "dropout": [0.1]}
    path = tmp_path / "p.json"
    best = tune("gcn", graphs, grid=grid, params_path=path, base_cfg=load_config(), log=lambda *_: None)
    assert best["val_pr_auc"] == 0.50 and (best["hidden"], best["n_layers"]) == (8, 2)   # trial 2
    state = json.loads(path.read_text())["gcn"]
    assert len(state["trials"]) == 4 and state["selection_metric"] == "val_mean_pr_auc"
    assert state["early_stopping"] == "val_mean_roc_auc" and state["split"] == "scaffold_det"


def test_tuning_is_resumable_and_does_not_repeat_trials(graphs, tmp_path, monkeypatch):
    calls = []
    def fake_fit(model, tr, va, cfg, seed, pw=None):
        calls.append(cfg["hidden"])
        return {"best_val_pr_auc": 0.1 * len(calls), "best_val_roc_auc": 0.5, "best_epoch": 0, "epochs_run": 1}
    monkeypatch.setattr("tox.tune_gnn.fit", fake_fit)
    grid = {"hidden": [8, 16], "n_layers": [1], "dropout": [0.1]}
    path = tmp_path / "p.json"
    tune("gcn", graphs, grid=grid, params_path=path, log=lambda *_: None)
    assert len(calls) == 2
    tune("gcn", graphs, grid=grid, params_path=path, log=lambda *_: None)
    assert len(calls) == 2                                       # nothing re-run
    tune("gcn", graphs, grid={**grid, "hidden": [8, 16, 32]}, params_path=path, log=lambda *_: None)
    assert len(calls) == 3                                       # only the new config
    assert len(json.loads(path.read_text())["gcn"]["trials"]) == 3


def test_tuning_real_fit_without_any_test_data(graphs, tmp_path):
    cfg = dict(load_config(), batch_size=4, max_epochs=3, patience=2)
    grid = {"hidden": [8], "n_layers": [1, 2], "dropout": [0.1]}
    best = tune("mpnn", graphs, grid=grid, params_path=tmp_path / "p.json", base_cfg=cfg, log=lambda *_: None)
    assert set(TUNED_KEYS) <= set(best) and 0 <= best["val_pr_auc"] <= 1


def test_frozen_config_overlays_tuned_values_and_requires_tuning(tmp_path):
    path = tmp_path / "p.json"
    with pytest.raises(FileNotFoundError, match="tox.tune_gnn"):
        frozen_config("gcn", path)
    path.write_text(json.dumps({"gcn": {"best": {"hidden": 256, "n_layers": 4, "dropout": 0.3,
                                                 "val_pr_auc": 0.4}}}))
    cfg = frozen_config("gcn", path)
    assert (cfg["hidden"], cfg["n_layers"], cfg["dropout"]) == (256, 4, 0.3)
    assert cfg["max_epochs"] == 200 and cfg["patience"] == 25      # everything else stays default
    with pytest.raises(FileNotFoundError):
        frozen_config("mpnn", path)                                # tuned per model


def test_runs_match_the_aml_grid_and_aggregator_convention():
    from tox.train_aml import RUNS as AML_RUNS
    assert RUNS == AML_RUNS and len(RUNS) == 9
    assert [s for sp, s in RUNS if sp == "scaffold_det"] == SEEDS


def _params(tmp_path, models=("gcn", "mpnn", "afp")):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({m: {"best": {"hidden": 32, "n_layers": 3, "dropout": 0.1}} for m in models}))
    return p


def test_run_grid_runs_every_split_seed_with_frozen_params(tmp_path, monkeypatch):
    monkeypatch.setattr("tox.train_gnn.load_split_graphs", lambda split, roles=None: {"split": split})
    calls = []
    runner = lambda m, split, seed, cfg, out, graphs: calls.append((m, split, seed, cfg["hidden"], graphs["split"]))
    done = run_grid(["gcn", "attentivefp"], runner=runner, preds_dir=tmp_path / "preds",
                    params_path=_params(tmp_path))
    assert len(done) == len(calls) == 18
    assert [(sp, s) for m, sp, s, *_ in calls if m == "gcn"] == RUNS
    assert {m for m, *_ in calls} == {"gcn", "afp"}              # alias resolved to the plan token
    assert all(h == 32 and g == sp for _, sp, _, h, g in calls)


def test_run_grid_skips_existing_prediction_files_unless_forced(tmp_path, monkeypatch):
    monkeypatch.setattr("tox.train_gnn.load_split_graphs", lambda split, roles=None: {})
    preds = tmp_path / "preds"; preds.mkdir()
    (preds / "gcn__scaffold_det__seed0.parquet").write_text("x")
    calls = []
    runner = lambda m, split, seed, cfg, out, graphs: calls.append((split, seed))
    run_grid(["gcn"], runner=runner, preds_dir=preds, params_path=_params(tmp_path))
    assert ("scaffold_det", 0) not in calls and len(calls) == 8
    calls.clear()
    run_grid(["gcn"], runner=runner, preds_dir=preds, params_path=_params(tmp_path), force=True)
    assert len(calls) == 9


def test_run_grid_refuses_to_start_untuned(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_grid(["gcn"], runner=lambda *a, **k: None, preds_dir=tmp_path, params_path=tmp_path / "none.json")


def test_loading_only_train_and_val_never_touches_test():
    from tox.constants import CLEAN
    if not CLEAN.exists():
        pytest.skip("C1 not built")
    g = load_split_graphs("scaffold_det", roles=("train", "val"))
    assert set(g) == {"train", "val"}
