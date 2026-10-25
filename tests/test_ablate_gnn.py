import json

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("torch", reason="GNN tests need torch (not in requirements.txt)")
pytest.importorskip("torch_geometric", reason="GNN tests need torch_geometric (not in requirements.txt)")
import torch

from tox.ablate_gnn import GROUPS, SPLIT, run_ablations, run_single_task, variants
from tox.constants import SEEDS, TASKS
from tox.featurize_graph import build_graphs
from tox.metrics import score
from tox.train_gnn import fit, load_config, labels_of, val_scores
from tox.models_gnn import build_model

SMILES = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)O",
          "CC(=O)N", "CCOC", "CCSC", "C1CCCCC1", "C1CCCC1", "CC(C)O", "CC(C)N", "OCCO", "NCCN",
          "CCCCO", "CCCCN", "c1ccncc1", "c1ccsc1", "CC=O", "CC#N"]


def _graphs(y=None):
    if y is None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, (len(SMILES), 12)).astype(np.float32)
        y[:, 0] = np.arange(len(SMILES)) % 2
        y[:, 5] = np.arange(len(SMILES)) % 3 == 0
        y[rng.random(y.shape) < 0.2] = np.nan
        y[[12, 13, 18, 19], :] = np.array([1, 0, 1, 0])[:, None]       # keep both classes in val/test
    g = build_graphs(SMILES, y)
    return {"train": g[:12], "val": g[12:18], "test": g[18:]}, y


CFG = dict(load_config(), hidden=8, n_layers=1, batch_size=4, max_epochs=3, patience=2)


# ---------------- variant definitions (plan B5) ----------------

def test_mpnn_variants_cover_the_five_ablations_with_plan_tokens():
    v = variants("mpnn")
    assert set(v) == {"mpnn-l2", "mpnn-l3", "mpnn-l4", "mpnn-l5", "mpnn-noedge", "mpnn-pmean",
                      "mpnn-pmax", "mpnn-nopw", "mpnn-single"}
    assert v["mpnn-noedge"] == {"use_bond_features": False}             # the plan's example token
    assert [v[f"mpnn-l{n}"]["n_layers"] for n in (2, 3, 4, 5)] == [2, 3, 4, 5]
    assert {v["mpnn-pmean"]["pooling"], v["mpnn-pmax"]["pooling"]} == {"mean", "max"}
    assert v["mpnn-nopw"]["pos_weight"]["enabled"] is False
    assert v["mpnn-single"] == {"single_task": True}
    assert "mpnn" not in v                                              # baseline = the main run


def test_inapplicable_ablations_are_refused_not_silently_skipped():
    assert "gcn-noedge" not in variants("gcn") and "gcn-pmean" in variants("gcn")
    assert "afp-pmean" not in variants("afp") and "afp-noedge" in variants("afp")
    with pytest.raises(ValueError, match="does not apply"):
        variants("gcn", ["bond"])
    with pytest.raises(ValueError, match="does not apply"):
        variants("afp", ["pooling"])
    with pytest.raises(ValueError, match="unknown ablation"):
        variants("mpnn", ["depth"])
    assert variants("attentivefp", ["layers"]).keys() == {"afp-l2", "afp-l3", "afp-l4", "afp-l5"}


# ---------------- runner ----------------

def _params(tmp_path, model="mpnn"):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({model: {"best": {"hidden": 32, "n_layers": 3, "dropout": 0.1}}}))
    return p


def test_run_ablations_runs_every_variant_seed_on_scaffold_det_with_overrides(tmp_path):
    calls, singles = [], []
    runner = lambda m, split, seed, cfg, **kw: calls.append((kw["tag"], split, seed, cfg, kw))
    single = lambda m, split, seed, cfg, **kw: singles.append((kw["tag"], seed, cfg))
    done = run_ablations("mpnn", runner=runner, single_runner=single, out_dir=tmp_path / "o",
                         params_path=_params(tmp_path), graphs={"g": 1})
    assert len(done) == 9 * 3 == len(calls) + len(singles) and len(singles) == 3
    assert {c[1] for c in calls} == {SPLIT} == {"scaffold_det"}
    assert {c[2] for c in calls} == set(SEEDS)
    by_tag = {c[0]: c[3] for c in calls}
    assert by_tag["mpnn-noedge"]["use_bond_features"] is False and by_tag["mpnn-l5"]["n_layers"] == 5
    assert by_tag["mpnn-nopw"]["pos_weight"]["enabled"] is False
    assert by_tag["mpnn-pmax"]["hidden"] == 32 and by_tag["mpnn-pmax"]["dropout"] == 0.1   # frozen config kept
    assert all("single_task" not in c[3] for c in calls) and all("single_task" not in s[2] for s in singles)
    kw = calls[0][4]
    assert kw["out"] == tmp_path / "o" / "preds" and kw["log_dir"] == tmp_path / "o" / "logs"


def test_run_ablations_resumes_and_force_reruns(tmp_path):
    out = tmp_path / "o"; (out / "preds").mkdir(parents=True)
    (out / "preds" / "mpnn-noedge__scaffold_det__seed1.parquet").write_text("x")
    calls = []
    f = lambda m, split, seed, cfg, **kw: calls.append((kw["tag"], seed))
    run_ablations("mpnn", ["bond"], runner=f, out_dir=out, params_path=_params(tmp_path), graphs={})
    assert calls == [("mpnn-noedge", 0), ("mpnn-noedge", 2)]
    calls.clear()
    run_ablations("mpnn", ["bond"], runner=f, out_dir=out, params_path=_params(tmp_path), graphs={}, force=True)
    assert len(calls) == 3


def test_run_ablations_needs_tuned_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="tune_gnn"):
        run_ablations("mpnn", runner=lambda *a, **k: None, out_dir=tmp_path, params_path=tmp_path / "x.json", graphs={})


def test_default_output_dir_is_separate_from_the_fingerprint_ablation():
    from tox.ablate_gnn import OUT
    from tox.constants import ABLATIONS
    assert OUT == ABLATIONS / "gnn" and OUT != ABLATIONS / "metrics"


# ---------------- single-task training ----------------

def test_fit_single_task_uses_only_its_own_endpoint(tmp_path):
    g, y = _graphs()
    y2 = y.copy(); y2[:12, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]] = 1 - np.nan_to_num(y2[:12, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]])
    g2, _ = _graphs(y2)                                       # only task 5's labels are unchanged
    def go(gr):
        torch.manual_seed(0)
        m = build_model("gcn", hidden=8, n_layers=1, n_tasks=1)
        return fit(m, gr["train"], gr["val"], CFG, seed=3, task=5)["history"]
    assert go(g) == go(g2)


def test_val_scores_single_task_scores_exactly_that_column():
    g, y = _graphs()
    torch.manual_seed(0)
    m = build_model("gcn", hidden=8, n_layers=1, n_tasks=1).eval()
    got = val_scores(m, g["val"], task=0)
    from tox.train_gnn import predict_probs
    p = predict_probs(m, g["val"])[:, 0]
    full_y = np.full((len(g["val"]), 12), np.nan); full_y[:, 0] = labels_of(g["val"])[:, 0]
    full_p = np.zeros((len(g["val"]), 12)); full_p[:, 0] = p
    s = score(full_y, full_p)
    assert got["roc_auc"] == pytest.approx(float(s.roc_auc[0]))
    assert got["pr_auc"] == pytest.approx(float(s.pr_auc[0]))


def test_run_single_task_writes_valid_c3_with_independent_endpoint_models(tmp_path):
    g, y = _graphs()
    run_single_task("gcn", "demo", 0, CFG, out=tmp_path / "p", log_dir=tmp_path / "l", graphs=g, tag="gcn-single")
    df = pd.read_parquet(tmp_path / "p" / "gcn-single__demo__seed0.parquet")
    assert list(df.columns) == ["smiles", "role"] + TASKS and set(df.role) == {"val", "test"}
    assert df[TASKS].to_numpy().min() >= 0 and df[TASKS].to_numpy().max() <= 1
    assert df.smiles.tolist() == SMILES[12:]
    log = json.loads((tmp_path / "l" / "gcn-single__demo__seed0.json").read_text())
    assert [t["task"] for t in log["tasks"]] == TASKS and log["single_task"] is True
    # changing ONLY endpoint 5's training labels changes ONLY column 5 (12 truly separate models)
    y2 = y.copy(); y2[:12, 5] = 1 - np.nan_to_num(y2[:12, 5])
    g2, _ = _graphs(y2)
    run_single_task("gcn", "demo", 0, CFG, out=tmp_path / "p2", log_dir=tmp_path / "l2", graphs=g2, tag="gcn-single")
    df2 = pd.read_parquet(tmp_path / "p2" / "gcn-single__demo__seed0.parquet")
    other = [t for t in TASKS if t != TASKS[5]]
    pd.testing.assert_frame_equal(df[["smiles", "role"] + other], df2[["smiles", "role"] + other])
    assert not np.allclose(df[TASKS[5]], df2[TASKS[5]])


def test_run_single_task_respects_pos_weight_switch(tmp_path):
    g, _ = _graphs()
    for on in (True, False):
        cfg = dict(CFG, pos_weight={"enabled": on, "clip_max": 30})
        run_single_task("gcn", "d", 0, cfg, out=tmp_path / str(on), log_dir=tmp_path / str(on), graphs=g, tag="t")
        assert (json.loads((tmp_path / str(on) / "t__d__seed0.json").read_text())["pos_weight"] is None) is (not on)


def test_ablation_run_with_real_trainer_writes_tagged_c3_file(tmp_path):
    from tox.train_gnn import run
    g, _ = _graphs()
    run("mpnn", "demo", 0, dict(CFG, use_bond_features=False), out=tmp_path / "p", log_dir=tmp_path / "l",
        graphs=g, fig_dir=tmp_path / "f", tag="mpnn-noedge")
    assert (tmp_path / "p" / "mpnn-noedge__demo__seed0.parquet").exists()
    assert json.loads((tmp_path / "l" / "mpnn-noedge__demo__seed0.json").read_text())["tag"] == "mpnn-noedge"


# ---------------- summary (rolls up C4 files written by tox.metrics) ----------------

def _c4(path, roc, pr=None, n_pos=30, pos_rate=0.1):
    pd.DataFrame({"task": TASKS, "n": 700, "n_pos": n_pos, "pos_rate": pos_rate,
                  "roc_auc": roc, "pr_auc": roc / 2 if pr is None else pr}).to_csv(path, index=False)


@pytest.fixture()
def c4_dirs(tmp_path):
    abl, base = tmp_path / "abl", tmp_path / "base"
    abl.mkdir(); base.mkdir()
    for s in range(3):
        _c4(base / f"mpnn__scaffold_det__seed{s}.csv", 0.70 + 0.01 * s)
        _c4(abl / f"mpnn-noedge__scaffold_det__seed{s}.csv", 0.60 + 0.01 * s)
        _c4(abl / f"mpnn-l2__scaffold_det__seed{s}.csv", 0.65)
    _c4(abl / "mpnn-single__scaffold_det__seed0.csv", np.linspace(0.5, 0.9, 12))
    _c4(base / "gcn__scaffold_det__seed0.csv", 0.1)                      # other model: must be ignored
    return abl, base


def test_summary_has_baseline_and_variants_with_headline_recipe(c4_dirs):
    from tox.ablate_gnn import summarize
    abl, base = c4_dirs
    s = summarize("mpnn", abl, base).set_index("variant")
    assert s.loc["mpnn", "group"] == "baseline" and s.loc["mpnn-noedge", "group"] == "bond"
    assert s.loc["mpnn-l2", "group"] == "layers" and s.loc["mpnn-single", "group"] == "single"
    assert s.loc["mpnn", "n_runs"] == 3 and s.loc["mpnn-single", "n_runs"] == 1
    assert s.loc["mpnn", "roc_auc_mean"] == pytest.approx(0.71)         # mean of per-seed 12-task means
    assert s.loc["mpnn", "roc_auc_std"] == pytest.approx(np.std([0.70, 0.71, 0.72], ddof=1))
    assert s.loc["mpnn-single", "roc_auc_mean"] == pytest.approx(np.linspace(0.5, 0.9, 12).mean())
    assert "gcn" not in s.index and s.index[0] == "mpnn"                # baseline first


def test_per_task_summary_one_row_per_variant_and_endpoint(c4_dirs):
    from tox.ablate_gnn import per_task_summary
    abl, base = c4_dirs
    t = per_task_summary("mpnn", abl, base)
    assert len(t) == 4 * 12 and set(t.task) == set(TASKS)
    single = t[t.variant == "mpnn-single"].set_index("task")
    assert single.loc[TASKS[0], "roc_auc_mean"] == pytest.approx(0.5)
    assert single.loc[TASKS[-1], "roc_auc_mean"] == pytest.approx(0.9)
    assert {"n_pos", "pos_rate"} <= set(t.columns)


def test_summary_alias_and_missing_files(c4_dirs, tmp_path):
    from tox.ablate_gnn import group_of, summarize
    abl, base = c4_dirs
    with pytest.raises(FileNotFoundError, match="tox.metrics"):
        summarize("afp", abl, base)
    assert group_of("afp-pmax", "afp") == "pooling" and group_of("afp-nopw", "afp") == "posweight"
    with pytest.raises(ValueError):
        group_of("mpnn-weird", "mpnn")
    for tag in variants("mpnn"):
        assert group_of(tag, "mpnn") in GROUPS                          # every runner token is classified
