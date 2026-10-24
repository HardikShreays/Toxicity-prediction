import numpy as np
import pandas as pd
import pytest
import torch

from tox.constants import TASKS
from tox.featurize_graph import build_graphs
from tox.metrics import score
from tox.models_gnn import build_model
from tox.train_gnn import (compute_pos_weight, export_predictions, fit, labels_of,
                           load_config, plot_curves, predict_probs, run, val_scores)

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


@pytest.fixture(autouse=True)
def _no_figures_in_repo(tmp_path_factory, monkeypatch):
    """run() must never write curve PNGs into the real results/figures/ during tests."""
    monkeypatch.setattr("tox.train_gnn.FIGURES", tmp_path_factory.mktemp("figs"))


CFG = dict(load_config(), hidden=16, n_layers=2, batch_size=4, max_epochs=6, patience=3)


def test_config_enables_pos_weight_capped_at_30_by_default():
    pw = load_config()["pos_weight"]
    assert pw["enabled"] is True and pw["clip_max"] == 30


def test_compute_pos_weight_ratio_cap_and_nan_handling():
    # task0: 3 neg / 1 pos = 3; task1: 90 neg / 1 pos = 90 -> capped at 30; task2: no labels -> 1
    y = np.full((91, 3), np.nan, dtype=np.float32)
    y[:4, 0] = [1, 0, 0, 0]
    y[:, 1] = 0.0
    y[0, 1] = 1.0
    w = compute_pos_weight(y)                              # default cap = 30
    assert w.tolist() == pytest.approx([3.0, 30.0, 1.0])
    assert compute_pos_weight(y, clip_max=2.5).tolist() == pytest.approx([2.5, 2.5, 1.0])
    assert compute_pos_weight(y, clip_max=100.0)[1].item() == pytest.approx(90.0)


def test_pos_weight_uses_training_labels_only(graphs, tmp_path):
    """Scrambling val/test labels must not change the logged pos_weight; scrambling train must."""
    import json
    cfg = dict(CFG, max_epochs=1)
    flipped = lambda gs: build_graphs([g.smiles for g in gs], 1 - np.nan_to_num(labels_of(gs)))
    other = dict(graphs, val=flipped(graphs["val"]), test=flipped(graphs["test"]))
    run("gcn", "d", 0, cfg, out=tmp_path / "a", log_dir=tmp_path / "a", graphs=graphs)
    run("gcn", "d", 0, cfg, out=tmp_path / "b", log_dir=tmp_path / "b", graphs=other)
    pa = json.loads((tmp_path / "a" / "gcn__d__seed0.json").read_text())["pos_weight"]
    pb = json.loads((tmp_path / "b" / "gcn__d__seed0.json").read_text())["pos_weight"]
    assert pa == pb == compute_pos_weight(labels_of(graphs["train"])).tolist()
    changed = dict(graphs, train=flipped(graphs["train"]))
    run("gcn", "d", 0, cfg, out=tmp_path / "c", log_dir=tmp_path / "c", graphs=changed)
    assert json.loads((tmp_path / "c" / "gcn__d__seed0.json").read_text())["pos_weight"] != pa


def test_pos_weight_enabled_without_clip_raises(graphs):
    cfg = dict(CFG, pos_weight={"enabled": True, "clip_max": None})
    with pytest.raises(ValueError, match="clip_max"):
        run("gcn", "x", 0, cfg, graphs=graphs)


def test_pos_weight_off_logs_none(graphs, tmp_path):
    import json
    cfg = dict(CFG, max_epochs=1, pos_weight={"enabled": False, "clip_max": 30})
    run("gcn", "d", 0, cfg, out=tmp_path, log_dir=tmp_path, graphs=graphs)
    assert json.loads((tmp_path / "gcn__d__seed0.json").read_text())["pos_weight"] is None


def _scripted(monkeypatch, rocs, prs=None):
    """Make val_scores return a scripted sequence so stopping logic is tested exactly."""
    prs = prs or [0.1] * len(rocs)
    it = iter(zip(rocs, prs))
    monkeypatch.setattr("tox.train_gnn.val_scores",
                        lambda model, g: dict(zip(("roc_auc", "pr_auc"), next(it))))


def test_config_matches_plan_b4():
    cfg = load_config()
    assert (cfg["max_epochs"], cfg["patience"], cfg["lr"], cfg["weight_decay"], cfg["batch_size"]) \
        == (200, 25, 1e-3, 1e-5, 64)


@pytest.mark.parametrize("name", ["gcn", "mpnn", "attentivefp"])
def test_fit_runs_and_restores_best_val_epoch(name, graphs):
    torch.manual_seed(0)
    m = build_model(name, hidden=16, n_layers=2)
    info = fit(m, graphs["train"], graphs["val"], CFG, seed=0)
    roc = [h["val_roc_auc"] for h in info["history"]]
    assert info["best_epoch"] == int(np.argmax(roc))                 # picked by val ROC-AUC
    assert info["best_val_roc_auc"] == pytest.approx(max(roc))
    assert val_scores(m, graphs["val"])["roc_auc"] == pytest.approx(max(roc), abs=1e-6)  # restored
    assert info["epochs_run"] <= CFG["max_epochs"]
    assert all(np.isfinite(h["train_loss"]) for h in info["history"])


def test_early_stopping_uses_roc_auc_not_pr_auc(graphs, monkeypatch):
    # ROC peaks at epoch 1; PR keeps rising and would peak at the last epoch if it were the criterion
    _scripted(monkeypatch, rocs=[0.5, 0.9, 0.8, 0.7, 0.6, 0.5], prs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    m = build_model("gcn", hidden=8, n_layers=1)
    info = fit(m, graphs["train"], graphs["val"], dict(CFG, max_epochs=50, patience=3), seed=0)
    assert info["best_epoch"] == 1 and info["best_val_roc_auc"] == 0.9
    assert info["epochs_run"] == 5                      # stops 3 epochs after the best
    assert info["best_val_pr_auc"] == 0.2              # PR-AUC reported at the ROC-best epoch


def test_patience_counts_epochs_without_improvement(graphs, monkeypatch):
    _scripted(monkeypatch, rocs=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0] * 3)
    m = build_model("gcn", hidden=8, n_layers=1)
    info = fit(m, graphs["train"], graphs["val"], dict(CFG, max_epochs=50, patience=2), seed=0)
    assert info["epochs_run"] == 3 and info["best_epoch"] == 0


def test_improvement_resets_patience(graphs, monkeypatch):
    _scripted(monkeypatch, rocs=[0.5, 0.4, 0.6, 0.5, 0.5, 0.5, 0.5])
    m = build_model("gcn", hidden=8, n_layers=1)
    info = fit(m, graphs["train"], graphs["val"], dict(CFG, max_epochs=50, patience=3), seed=0)
    assert info["best_epoch"] == 2 and info["epochs_run"] == 6


def test_max_epochs_is_a_hard_limit(graphs, monkeypatch):
    _scripted(monkeypatch, rocs=list(np.linspace(0.5, 0.9, 50)))   # always improving
    m = build_model("gcn", hidden=8, n_layers=1)
    info = fit(m, graphs["train"], graphs["val"], dict(CFG, max_epochs=7, patience=25), seed=0)
    assert info["epochs_run"] == 7 and info["best_epoch"] == 6


def test_unscorable_validation_raises_instead_of_silently_passing(graphs, monkeypatch):
    _scripted(monkeypatch, rocs=[float("nan")] * 10)
    m = build_model("gcn", hidden=8, n_layers=1)
    with pytest.raises(RuntimeError, match="never computable"):
        fit(m, graphs["train"], graphs["val"], dict(CFG, max_epochs=3), seed=0)


def test_checkpoints_every_epoch_and_best(graphs, tmp_path):
    m = build_model("gcn", hidden=16, n_layers=2)
    info = fit(m, graphs["train"], graphs["val"], CFG, seed=0, ckpt_dir=tmp_path)
    last = torch.load(tmp_path / "last.pt", weights_only=False)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert last["epoch"] == info["epochs_run"] - 1 and len(last["history"]) == info["epochs_run"]
    assert last["done"] is True and {"model", "opt", "rng"} <= set(last)
    assert best["epoch"] == info["best_epoch"]
    assert best["val_roc_auc"] == pytest.approx(info["best_val_roc_auc"])
    assert not list(tmp_path.glob("*.tmp"))                          # atomic writes leave no temp files


def test_last_checkpoint_is_written_each_epoch(graphs, tmp_path, monkeypatch):
    seen = []
    real = val_scores
    def spy(model, g):
        seen.append(torch.load(tmp_path / "last.pt", weights_only=False)["epoch"]
                    if (tmp_path / "last.pt").exists() else -1)
        return real(model, g)
    monkeypatch.setattr("tox.train_gnn.val_scores", spy)
    fit(build_model("gcn", hidden=8, n_layers=1), graphs["train"], graphs["val"],
        dict(CFG, max_epochs=4, patience=10), seed=0, ckpt_dir=tmp_path)
    assert seen == [-1, 0, 1, 2]       # before epoch k's validation, last.pt holds epoch k-1


def test_resume_after_interruption_equals_uninterrupted_run(graphs, tmp_path, monkeypatch):
    cfg = dict(CFG, max_epochs=6, patience=10)
    def fresh():
        torch.manual_seed(0)
        return build_model("mpnn", hidden=16, n_layers=2)       # dropout -> uses the RNG
    full = fit(fresh(), graphs["train"], graphs["val"], cfg, seed=4, ckpt_dir=tmp_path / "a")

    calls, real = [], val_scores
    def flaky(model, g):
        calls.append(1)
        if len(calls) == 4:
            raise ConnectionError("colab disconnected")
        return real(model, g)
    monkeypatch.setattr("tox.train_gnn.val_scores", flaky)
    with pytest.raises(ConnectionError):
        fit(fresh(), graphs["train"], graphs["val"], cfg, seed=4, ckpt_dir=tmp_path / "b")
    monkeypatch.setattr("tox.train_gnn.val_scores", real)
    m2 = fresh()
    resumed = fit(m2, graphs["train"], graphs["val"], cfg, seed=4, ckpt_dir=tmp_path / "b")
    assert resumed["history"] == full["history"]
    assert resumed["best_epoch"] == full["best_epoch"]


def test_resume_with_different_config_or_seed_is_refused(graphs, tmp_path):
    cfg = dict(CFG, max_epochs=2)
    fit(build_model("gcn", hidden=8, n_layers=1), graphs["train"], graphs["val"], cfg, 0, ckpt_dir=tmp_path)
    with pytest.raises(ValueError, match="different seed/config"):
        fit(build_model("gcn", hidden=8, n_layers=1), graphs["train"], graphs["val"], dict(cfg, lr=0.5), 0,
            ckpt_dir=tmp_path)
    with pytest.raises(ValueError, match="different seed/config"):
        fit(build_model("gcn", hidden=8, n_layers=1), graphs["train"], graphs["val"], cfg, 1, ckpt_dir=tmp_path)


def test_finished_run_is_not_retrained_on_resume(graphs, tmp_path):
    cfg = dict(CFG, max_epochs=3, patience=10)
    a = fit(build_model("gcn", hidden=8, n_layers=1), graphs["train"], graphs["val"], cfg, 0, ckpt_dir=tmp_path)
    b = fit(build_model("gcn", hidden=8, n_layers=1), graphs["train"], graphs["val"], cfg, 0, ckpt_dir=tmp_path)
    assert a["history"] == b["history"]


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


def test_same_seed_same_result_and_other_seed_differs(graphs):
    def go(seed):
        torch.manual_seed(0)
        m = build_model("mpnn", hidden=16, n_layers=2)
        return fit(m, graphs["train"], graphs["val"], CFG, seed=seed)["history"]
    assert go(5) == go(5)
    assert go(5) != go(6)


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


def test_run_writes_preds_log_checkpoints_and_curve(graphs, tmp_path):
    import json
    cfg = dict(CFG, max_epochs=2)
    run("gcn", "demo", 7, cfg, out=tmp_path / "preds", log_dir=tmp_path / "log", graphs=graphs,
        fig_dir=tmp_path / "fig")
    assert (tmp_path / "preds" / "gcn__demo__seed7.parquet").exists()
    log = json.loads((tmp_path / "log" / "gcn__demo__seed7.json").read_text())
    assert log["seed"] == 7 and log["config"]["hidden"] == 16 and log["history"]
    assert log["early_stopping"] == {"metric": "val_mean_roc_auc", "patience": CFG["patience"], "max_epochs": 2}
    assert {"val_roc_auc", "val_pr_auc"} <= set(log["history"][0])
    assert log["best_val_pr_auc"] == log["history"][log["best_epoch"]]["val_pr_auc"]
    assert (tmp_path / "log" / "gcn__demo__seed7.pt").exists()
    assert (tmp_path / "log" / "ckpt" / "gcn__demo__seed7" / "last.pt").exists()
    png = tmp_path / "fig" / "gcn__demo__seed7_curve.png"
    assert png.exists() and png.stat().st_size > 1000


def test_plot_curves_writes_png(tmp_path):
    info = {"best_epoch": 1, "history": [{"epoch": i, "train_loss": 1 / (i + 1), "val_roc_auc": 0.5 + i / 10,
                                          "val_pr_auc": 0.1} for i in range(3)]}
    plot_curves(info, tmp_path / "c.png")
    assert (tmp_path / "c.png").read_bytes()[:4] == b"\x89PNG"


def test_real_splits_are_disjoint_and_cover_c1_labels():
    from tox.constants import CLEAN, SPLITS
    import json
    if not CLEAN.exists():
        pytest.skip("C1 not built")
    sp = json.loads((SPLITS / "scaffold_det.json").read_text())
    r = {k: set(sp[k]) for k in ("train", "val", "test")}
    assert not (r["train"] & r["val"] or r["train"] & r["test"] or r["val"] & r["test"])
    assert (r["train"] | r["val"] | r["test"]) <= set(pd.read_parquet(CLEAN).smiles)
