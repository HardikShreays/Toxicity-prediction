import numpy as np
import pandas as pd
import pytest
import torch

from tox.constants import TASKS
from tox.featurize_graph import build_graphs
from tox.metrics import score
from tox.models_gnn import build_model
from tox.train_gnn import (compute_pos_weight, export_predictions, fit, labels_of,
                           load_config, predict_probs, run, val_score)

SMILES = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)O",
          "CC(=O)N", "CCOC", "CCSC", "C1CCCCC1", "C1CCCC1", "CC(C)O", "CC(C)N", "OCCO", "NCCN",
          "CCCCO", "CCCCN", "c1ccncc1", "c1ccsc1", "CC=O", "CC#N"]


def _labels(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, (n, len(TASKS))).astype(np.float32)
    y[rng.random(y.shape) < 0.3] = np.nan          # missing labels
    y[:, 0] = np.where(np.arange(n) % 2 == 0, 1.0, 0.0)   # learnable-ish, both classes present
    return y


@pytest.fixture(scope="module")
def graphs():
    g = build_graphs(SMILES, _labels(len(SMILES), 0))
    return {"train": g[:12], "val": g[12:18], "test": g[18:]}


CFG = dict(load_config(), hidden=16, n_layers=2, batch_size=4, max_epochs=6, patience=3)


def test_config_records_pos_weight_off_by_default():
    assert load_config()["pos_weight"]["enabled"] is False


def test_compute_pos_weight_uses_given_labels_and_clips():
    y = np.array([[1, 0, np.nan], [0, 0, np.nan], [0, 0, np.nan], [0, 1, np.nan]], dtype=np.float32)
    w = compute_pos_weight(y, clip_min=1.0, clip_max=2.5)
    # tasks 0,1: 3 neg / 1 pos = 3 -> clipped to 2.5; task 2 has no labels -> 1
    assert w.tolist() == pytest.approx([2.5, 2.5, 1.0])
    assert compute_pos_weight(y, 1.0, 10.0).tolist() == pytest.approx([3.0, 3.0, 1.0])


def test_pos_weight_enabled_without_clip_raises(graphs):
    cfg = dict(CFG, pos_weight={"enabled": True, "clip_min": None, "clip_max": None})
    with pytest.raises(ValueError, match="clip"):
        run("gcn", "x", 0, cfg, graphs=graphs)


@pytest.mark.parametrize("name", ["gcn", "mpnn", "attentivefp"])
def test_fit_runs_and_restores_best_val_epoch(name, graphs):
    torch.manual_seed(0)
    m = build_model(name, hidden=16, n_layers=2)
    info = fit(m, graphs["train"], graphs["val"], CFG, seed=0)
    hist = [h["val_pr_auc"] for h in info["history"]]
    assert info["best_epoch"] == int(np.argmax(hist))                 # picked by val only
    assert info["best_val_pr_auc"] == pytest.approx(max(hist))
    assert val_score(m, graphs["val"]) == pytest.approx(max(hist), abs=1e-6)   # weights restored
    assert info["epochs_run"] <= CFG["max_epochs"]
    assert all(np.isfinite(h["train_loss"]) for h in info["history"])


def test_early_stopping_triggers_on_patience(graphs, monkeypatch):
    scores = iter([0.5, 0.4, 0.3, 0.2, 0.1, 0.0] * 10)       # best at epoch 0, never improves
    monkeypatch.setattr("tox.train_gnn.val_score", lambda model, g: next(scores))
    m = build_model("gcn", hidden=8, n_layers=1)
    info = fit(m, graphs["train"], graphs["val"], dict(CFG, max_epochs=50, patience=2), seed=0)
    assert info["epochs_run"] == 3 and info["best_epoch"] == 0


def test_fit_signature_has_no_test_input():
    import inspect
    assert "test" not in " ".join(inspect.signature(fit).parameters)


def test_test_labels_do_not_affect_training_or_val_predictions(graphs, tmp_path):
    """run() with scrambled TEST labels must give identical training history and val predictions."""
    cfg = dict(CFG, max_epochs=3)
    scrambled = dict(graphs, test=build_graphs(SMILES[18:], 1 - np.nan_to_num(_labels(len(SMILES), 0)[18:])))
    ia = run("mpnn", "demo", 3, cfg, out=tmp_path / "a", log_dir=tmp_path / "a", graphs=graphs)
    ib = run("mpnn", "demo", 3, cfg, out=tmp_path / "b", log_dir=tmp_path / "b", graphs=scrambled)
    assert ia["history"] == ib["history"]
    pa = pd.read_parquet(tmp_path / "a" / "mpnn__demo__seed3.parquet")
    pb = pd.read_parquet(tmp_path / "b" / "mpnn__demo__seed3.parquet")
    pd.testing.assert_frame_equal(pa, pb)       # labels never enter predictions at all


def test_same_seed_same_result(graphs):
    def go():
        torch.manual_seed(0)
        m = build_model("mpnn", hidden=16, n_layers=2)
        info = fit(m, graphs["train"], graphs["val"], CFG, seed=5)
        return [h["train_loss"] for h in info["history"]]
    assert go() == go()


def test_export_matches_c3_contract_and_scorer(graphs, tmp_path):
    m = build_model("gcn", hidden=16, n_layers=2)
    fit(m, graphs["train"], graphs["val"], CFG, seed=0)
    df = export_predictions(m, graphs, tmp_path / "gcn__demo__seed0.parquet")
    back = pd.read_parquet(tmp_path / "gcn__demo__seed0.parquet")
    assert list(back.columns) == ["smiles", "role"] + TASKS
    assert set(back.role) == {"val", "test"} and len(back) == len(graphs["val"]) + len(graphs["test"])
    assert back[TASKS].to_numpy().min() >= 0 and back[TASKS].to_numpy().max() <= 1
    assert back.smiles.tolist() == SMILES[12:]                      # val then test, in order
    # shared scorer accepts it, handles NaN labels and single-class endpoints without error
    s = score(labels_of(graphs["test"]), back[back.role == "test"][TASKS].to_numpy())
    assert list(s.columns) == ["task", "n", "n_pos", "pos_rate", "roc_auc", "pr_auc"] and len(s) == 12


def test_run_writes_preds_log_and_records_seed(graphs, tmp_path):
    cfg = dict(CFG, max_epochs=2)
    run("gcn", "demo", 7, cfg, out=tmp_path / "preds", log_dir=tmp_path / "log", graphs=graphs)
    assert (tmp_path / "preds" / "gcn__demo__seed7.parquet").exists()
    import json
    log = json.loads((tmp_path / "log" / "gcn__demo__seed7.json").read_text())
    assert log["seed"] == 7 and log["config"]["hidden"] == 16 and log["history"]


def test_real_splits_are_disjoint_and_cover_c1_labels():
    from tox.constants import CLEAN, SPLITS
    import json
    if not CLEAN.exists():
        pytest.skip("C1 not built")
    sp = json.loads((SPLITS / "scaffold_det.json").read_text())
    r = {k: set(sp[k]) for k in ("train", "val", "test")}
    assert not (r["train"] & r["val"] or r["train"] & r["test"] or r["val"] & r["test"])
    assert (r["train"] | r["val"] | r["test"]) <= set(pd.read_parquet(CLEAN).smiles)
