import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from tox.constants import TASKS
from tox.featurize_graph import (ATOM_DIM, BOND_DIM, build_graphs, smiles_to_graph)


def test_ethanol_shapes_and_dtypes():
    g = smiles_to_graph("CCO")                     # 3 heavy atoms, 2 bonds
    assert g.x.shape == (3, ATOM_DIM) and g.x.dtype == torch.float32
    assert g.edge_index.shape == (2, 4) and g.edge_index.dtype == torch.long
    assert g.edge_attr.shape == (4, BOND_DIM) and g.edge_attr.dtype == torch.float32
    assert g.y.shape == (1, len(TASKS)) and g.y.dtype == torch.float32


def test_every_bond_in_both_directions():
    g = smiles_to_graph("c1ccccc1C(=O)O")          # benzoic acid: 9 atoms, 9 bonds
    edges = set(map(tuple, g.edge_index.t().tolist()))
    assert g.edge_index.shape[1] == 18
    assert all((j, i) in edges for i, j in edges)
    assert all(i != j for i, j in edges)
    assert g.edge_index.max() < g.num_nodes
    # the two directions of one bond carry identical features
    assert torch.equal(g.edge_attr[0::2], g.edge_attr[1::2])


def test_onehot_blocks_are_valid():
    g = smiles_to_graph("CC(=O)[O-]")
    assert torch.isfinite(g.x).all() and torch.isfinite(g.edge_attr).all()
    assert set(g.x.unique().tolist()) <= {0.0, 1.0}
    # first block is the atomic-number one-hot: exactly one hot entry per atom
    from tox.featurize_graph import ATOMS
    assert (g.x[:, : len(ATOMS) + 1].sum(1) == 1).all()


def test_features_distinguish_chemistry():
    benz = smiles_to_graph("c1ccccc1")
    cyc = smiles_to_graph("C1CCCCC1")
    assert not torch.equal(benz.x, cyc.x)              # aromatic / hybridization differ
    assert not torch.equal(benz.edge_attr, cyc.edge_attr)
    anion, neutral = smiles_to_graph("CC(=O)[O-]"), smiles_to_graph("CC(=O)O")
    assert not torch.equal(anion.x, neutral.x)         # charge is visible


def test_unknown_element_goes_to_other_bucket():
    from tox.featurize_graph import ATOMS
    g = smiles_to_graph("[Pt]")
    assert g.x[0, len(ATOMS)] == 1.0


def test_single_atom_has_no_edges():
    g = smiles_to_graph("[Na+]")
    assert g.x.shape == (1, ATOM_DIM)
    assert g.edge_index.shape == (2, 0) and g.edge_attr.shape == (0, BOND_DIM)


@pytest.mark.parametrize("bad", ["not_a_smiles", "C(C", "", "c1ccc"])
def test_invalid_smiles_returns_none(bad):
    assert smiles_to_graph(bad) is None


def test_build_graphs_invalid_is_loud_unless_skipped():
    with pytest.raises(ValueError, match="cannot parse"):
        build_graphs(["CCO", "garbage"])
    gs = build_graphs(["CCO", "garbage", "CC"], skip_invalid=True)
    assert [g.smiles for g in gs] == ["CCO", "CC"]


def test_missing_labels_preserved_as_nan_and_batch_shape():
    y = np.full((3, len(TASKS)), np.nan, dtype=np.float32)
    y[0, 0], y[0, 5] = 1.0, 0.0
    y[2, :] = 1.0
    graphs = build_graphs(["CCO", "c1ccccc1", "CC(=O)O"], labels=y)
    assert torch.isnan(graphs[1].y).all()
    assert graphs[0].y[0, 0] == 1.0 and graphs[0].y[0, 5] == 0.0
    assert torch.isnan(graphs[0].y[0, 1])
    batch = Batch.from_data_list(graphs)
    assert batch.y.shape == (3, len(TASKS))
    np.testing.assert_array_equal(np.isnan(batch.y.numpy()), np.isnan(y))
    assert batch.edge_index.max() < batch.num_nodes   # PyG offsets indices per molecule


def test_labels_wrong_shape_rejected():
    with pytest.raises(ValueError):
        build_graphs(["CCO"], labels=np.zeros((1, 11)))


def test_no_labels_gives_all_nan():
    assert torch.isnan(smiles_to_graph("CCO").y).all()


def test_real_split_smiles_all_parse():
    """Every SMILES in the committed split files (contract C2) must convert."""
    import json
    from tox.constants import SPLITS
    d = json.loads((SPLITS / "scaffold_det.json").read_text())
    smiles = d["train"][:300] + d["val"][:100] + d["test"][:100]
    assert len(build_graphs(smiles)) == len(smiles)
