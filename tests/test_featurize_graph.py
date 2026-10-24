import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from tox.constants import TASKS
from tox.featurize_graph import (ATOM_DIM, ATOMS, BOND_DIM, build_graphs, smiles_to_graph)


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
    assert (g.x[:, : len(ATOMS) + 1].sum(1) == 1).all()


def test_features_distinguish_chemistry():
    benz = smiles_to_graph("c1ccccc1")
    cyc = smiles_to_graph("C1CCCCC1")
    assert not torch.equal(benz.x, cyc.x)              # aromatic / hybridization differ
    assert not torch.equal(benz.edge_attr, cyc.edge_attr)
    anion, neutral = smiles_to_graph("CC(=O)[O-]"), smiles_to_graph("CC(=O)O")
    assert not torch.equal(anion.x, neutral.x)         # charge is visible


def test_unknown_element_goes_to_other_bucket():
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


# ---- exact plan schema (B1, p. 14) --------------------------------------------------------

def test_dimensions_are_exactly_the_plan_schema():
    # atomic number 12+other, degree 0-5, charge {-1,0,+1,other}, hybridization 5+other,
    # aromatic, total Hs 0-4, in ring
    assert ATOM_DIM == 13 + 6 + 4 + 6 + 1 + 5 + 1 == 36
    assert BOND_DIM == 4 + 1 + 1 == 6                       # type, conjugated, in ring
    assert ATOMS == [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 34]


def _hot(row, lo, hi):
    return row[lo:hi].nonzero().flatten().tolist()


def test_aromatic_carbon_feature_layout():
    x = smiles_to_graph("c1ccccc1").x[0]                   # aromatic CH: degree 2, 1 H, SP2, ring
    assert _hot(x, 0, 13) == [0]                            # carbon = first atomic-number slot
    assert _hot(x, 13, 19) == [2]                           # degree 2
    assert _hot(x, 19, 23) == [1]                           # formal charge 0 -> middle slot
    assert _hot(x, 23, 29) == [1]                           # SP2
    assert x[29] == 1.0                                     # aromatic
    assert _hot(x, 30, 35) == [1]                           # total Hs = 1
    assert x[35] == 1.0                                     # in ring


def test_bond_feature_layout():
    g = smiles_to_graph("C=CC#N")                           # double (conjugated, no ring), single, triple
    types = {tuple(r[:4].tolist()) for r in g.edge_attr}
    assert types == {(0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0)}
    ring = smiles_to_graph("c1ccccc1").edge_attr[0]
    assert ring.tolist() == [0, 0, 0, 1, 1, 1]              # aromatic, conjugated, in ring
    chain = smiles_to_graph("CC").edge_attr[0]
    assert chain.tolist() == [1, 0, 0, 0, 0, 0]


def test_charge_other_and_hybridization_other_slots():
    x = smiles_to_graph("[Zn+2]").x[0]                      # charge +2; RDKit hybridization 'S'
    assert _hot(x, 19, 23) == [3]                           # charge 'other'
    assert _hot(x, 23, 29) == [5]                           # hybridization 'other'
    assert _hot(smiles_to_graph("CC(=O)[O-]").x[3], 19, 23) == [0]     # -1
    assert _hot(smiles_to_graph("C[NH3+]").x[1], 19, 23) == [2]        # +1


def test_degree_above_five_has_no_slot_so_block_is_zero():
    x = smiles_to_graph("FS(F)(F)(F)(F)F").x                # sulfur hexafluoride: S has degree 6
    s = x[1]
    assert s[13:19].sum() == 0
    assert _hot(s, 23, 29) == [4]                           # SP3D2 is a planned hybridization


def test_chirality_and_double_bond_stereo_are_not_encoded():
    a, b = smiles_to_graph("C[C@H](N)O"), smiles_to_graph("C[C@@H](N)O")
    assert torch.equal(a.x, b.x) and torch.equal(a.edge_attr, b.edge_attr)
    e, z = smiles_to_graph("C/C=C/C"), smiles_to_graph("C/C=C\\C")
    assert torch.equal(e.edge_attr, z.edge_attr)


def test_whole_clean_dataset_featurizes_to_planned_dims():
    from tox.constants import CLEAN
    if not CLEAN.exists():
        pytest.skip("C1 not built")
    import pandas as pd
    smiles = pd.read_parquet(CLEAN).smiles.tolist()[:1500]
    for g in build_graphs(smiles):
        assert g.x.shape[1] == 36 and g.edge_attr.shape[1] == 6
        assert set(g.x.unique().tolist()) <= {0.0, 1.0}
