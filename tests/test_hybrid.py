import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from tox.constants import TASKS
from tox.featurize_graph import build_graphs
from tox.hybrid import (Fusion, encoder_embeddings, stack_features, stack_one, warm_start,
                        with_fps)
from tox.models_gnn import build_model, pooled_dim

SMILES = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)O",
          "CC(=O)N", "CCOC", "CCSC", "C1CCCCC1", "C1CCCC1", "CC(C)O", "CC(C)N", "OCCO", "NCCN",
          "CCCCO", "CCCCN", "c1ccncc1", "c1ccsc1", "CC=O", "CC#N"]
FP_DIM = 2048


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


def _fps(graphs):
    rng = np.random.default_rng(1)
    return {r: with_fps(g, rng.integers(0, 2, (len(g), FP_DIM)).astype(np.uint8),
                        [x.smiles for x in g])
            for r, g in graphs.items()}


def test_fusion_forward_shape(graphs):
    fps = _fps(graphs)
    for use_fp, use_graph, in_dim in ((True, True, pooled_dim(128, "meanmax") + 128),
                                      (True, False, 128), (False, True, pooled_dim(128, "meanmax"))):
        model = Fusion(hidden=128, fp_hidden=128, use_fp=use_fp, use_graph=use_graph)
        out = model(next(iter(DataLoader(fps["val"], batch_size=6))))
        assert out.shape == (6, len(TASKS))
        assert torch.isfinite(out).all()
        assert model.head.net[0].in_features == in_dim


def test_fusion_needs_a_branch():
    with pytest.raises(ValueError):
        Fusion(use_fp=False, use_graph=False)


def test_fusion_needs_attached_fingerprints(graphs):
    model = Fusion(use_fp=True, use_graph=False)
    with pytest.raises(RuntimeError):
        model(next(iter(DataLoader(graphs["val"], batch_size=6))))


def test_with_fps_attaches_rows(graphs):
    fp = np.eye(FP_DIM, dtype=np.uint8)[: len(graphs["val"])]
    out = with_fps(graphs["val"], fp, [g.smiles for g in graphs["val"]])
    for g, orig in zip(out, graphs["val"]):
        assert g.fp.shape == (1, FP_DIM)
        assert g.fp[0].sum() == 1                      # the eye rows are attached in order
        assert torch.equal(torch.nan_to_num(g.y), torch.nan_to_num(orig.y))
    assert not any(hasattr(g, "fp") for g in graphs["val"])     # originals untouched


def test_batch_fp_is_collated_per_molecule():
    g = build_graphs(SMILES, _labels(len(SMILES), 0))
    fp = np.arange(len(g), dtype=np.uint8).reshape(-1, 1).repeat(FP_DIM, axis=1)
    loader = DataLoader(with_fps(g, fp, [x.smiles for x in g]), batch_size=8)
    b = next(iter(loader))
    assert b.fp.shape == (8, FP_DIM)
    assert (b.fp[:, 0].numpy() == np.arange(8)).all()  # row k of the batch is molecule k


def test_warm_start_fills_the_graph_branch_and_skips_the_head(tmp_path):
    src = build_model("mpnn", hidden=128, n_layers=3, pooling="meanmax", dropout=0.2,
                      use_bond_features=True).state_dict()
    ckpt = tmp_path / "mpnn.pt"
    torch.save(src, ckpt)
    model = Fusion(hidden=128, fp_hidden=128, use_fp=True, use_graph=True)
    own = model.state_dict()
    expected = sum(1 for k, v in src.items() if k in own and own[k].shape == v.shape)
    n = warm_start(model, ckpt, log=lambda *_: None)
    assert n == expected
    for k, v in src.items():
        if k.startswith(("convs", "norms")):
            assert torch.equal(model.state_dict()[k], v)
    # shape-mismatched keys (the parent's 256->12 head vs the 384-input fusion head) are
    # skipped and the fresh head dimensions are untouched
    assert model.head.net[0].in_features == pooled_dim(128, "meanmax") + 128


def test_encoder_embeddings_match_readout_dim(graphs):
    model = build_model("mpnn", hidden=64, n_layers=3, pooling="meanmax", dropout=0.2)
    emb = encoder_embeddings(model, graphs["val"])
    assert emb.shape == (len(graphs["val"]), pooled_dim(64, "meanmax"))
    assert np.isfinite(emb).all()


def test_stack_fit_on_val_only_and_fallback():
    rng = np.random.default_rng(0)
    n_val, n_test = 200, 60
    X = {r: rng.random((n, 4 * len(TASKS))) for r, n in (("val", n_val), ("test", n_test))}
    for r in X:                                        # columns 12-23 mirror 0-11 (parent B = parent A)
        X[r][:, len(TASKS):2 * len(TASKS)] = X[r][:, :len(TASKS)]
    y_val = np.full((n_val, len(TASKS)), np.nan)
    y_val[:, 0] = (X["val"][:, 0] > 0.5).astype(float)                       # learnable endpoint
    y_val[:, 2:] = (X["val"][:, 2:len(TASKS)] > 0.5).astype(float)           # other endpoints: both classes
    y_val[::2, 1] = 1.0                                                     # single class only
    y_test = np.full((n_test, len(TASKS)), np.nan)
    y_test[:, 0] = (X["test"][:, 0] > 0.5).astype(float)
    probs, fallback = stack_one(X, {"val": y_val, "test": y_test}, 0)
    assert fallback == [TASKS[1]]
    acc = np.mean((probs["test"][:, 0] > 0.5) == y_test[:, 0])
    assert acc > 0.8                                 # the meta-learner learned the shared signal
    assert np.all((probs["test"] >= 0) & (probs["test"] <= 1))
    # fallback endpoint: mean of the parents, nothing fitted
    assert np.allclose(probs["test"][:, 1], X["test"][:, [1, 1 + len(TASKS)]].mean(axis=1))


def test_stack_features_aligns_by_smiles(tmp_path):
    labels = pd.DataFrame({"smiles": SMILES, **{t: np.nan for t in TASKS}}).set_index("smiles")
    labels.loc[SMILES[0], TASKS[0]] = 1.0
    rows = [{"smiles": s, "role": "val", **{t: 0.5 for t in TASKS}} for s in SMILES[:6]]
    a = tmp_path / "a.parquet"
    pd.DataFrame(rows).to_parquet(a)
    b = tmp_path / "b.parquet"
    pd.DataFrame([{**r, TASKS[0]: 0.9} for r in rows]).to_parquet(b)
    X, y = stack_features(a, b, "val", labels)
    assert X.shape == (6, 2 * len(TASKS))
    assert X[0, 0] == 0.5 and X[0, len(TASKS)] == 0.9
    assert y[0, 0] == 1.0
    # parent files with different row order are rejected loudly
    pd.DataFrame(rows[::-1]).to_parquet(b)
    with pytest.raises(AssertionError):
        stack_features(a, b, "val", labels)
