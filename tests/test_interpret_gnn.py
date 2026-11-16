"""P3 (GNN side) - tests for the attribution helpers.

Everything here runs on an untrained model: we are testing the plumbing (task selection, the
explainer adapter, atom alignment, rendering), not the quality of an explanation. The checks
that read the committed C1/C3 files are skipped when the parquet has not been generated.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="GNN tests need torch")
pytest.importorskip("torch_geometric", reason="GNN tests need torch_geometric")
import torch
from rdkit import Chem
from torch_geometric.data import Data

from tox.analysis.interpret_gnn import _TaskView, atom_attributions, draw, pick_molecules
from tox.constants import CLEAN, PREDS, TASKS
from tox.featurize_graph import smiles_to_graph
from tox.models_gnn import build_model

TASK = "SR-MMP"
needs_data = pytest.mark.skipif(not CLEAN.exists(), reason="run python -m tox.data first")
needs_preds = pytest.mark.skipif(not (PREDS / "mpnn__scaffold_det__seed0.parquet").exists(),
                                 reason="needs the committed mpnn C3 file")


@pytest.fixture(scope="module")
def model():
    return build_model("mpnn", hidden=16, n_layers=2, pooling="meanmax", dropout=0.0,
                       use_bond_features=True).eval()


# --- the explainer adapter ------------------------------------------------------------------

def test_task_view_returns_one_column_matching_the_full_model(model):
    """The adapter must expose exactly the selected endpoint, untouched."""
    g = smiles_to_graph("CC(=O)Nc1ccc(O)cc1")
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    with torch.no_grad():
        full = model(g)
        for k in (0, 5, len(TASKS) - 1):
            got = _TaskView(model, k)(g.x, g.edge_index, edge_attr=g.edge_attr, batch=g.batch)
            assert got.shape == (1, 1)
            assert torch.allclose(got, full[:, [k]])


def test_task_view_defaults_to_a_single_graph_when_no_batch_is_given(model):
    g = smiles_to_graph("CCO")
    out = _TaskView(model, 0)(g.x, g.edge_index, edge_attr=g.edge_attr)
    assert out.shape == (1, 1) and torch.isfinite(out).all()


# --- attributions ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1"])
def test_attributions_give_one_finite_weight_per_atom(model, smiles):
    """Weights must align 1:1 with RDKit atom order, which is what SimilarityMaps assumes."""
    w = atom_attributions(model, smiles, TASKS.index(TASK), epochs=5)
    assert w.shape == (Chem.MolFromSmiles(smiles).GetNumAtoms(),)
    assert np.isfinite(w).all()


def test_attributions_reject_unparseable_smiles(model):
    with pytest.raises(ValueError, match="cannot parse"):
        atom_attributions(model, "not_a_molecule", 0, epochs=5)


# --- rendering ------------------------------------------------------------------------------

def test_draw_writes_a_png(tmp_path):
    smiles = "CC(=O)Nc1ccc(O)cc1"
    w = np.linspace(-1, 1, Chem.MolFromSmiles(smiles).GetNumAtoms())
    out = tmp_path / "map.png"
    draw(smiles, w, out, "title")
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_draw_survives_constant_weights(tmp_path):
    """A flat attribution must not divide by zero."""
    smiles = "CCO"
    out = tmp_path / "flat.png"
    draw(smiles, np.zeros(Chem.MolFromSmiles(smiles).GetNumAtoms()), out)
    assert out.exists()


# --- molecule selection ---------------------------------------------------------------------

@needs_data
@needs_preds
def test_pick_molecules_returns_confident_tps_then_fps():
    sel = pick_molecules("mpnn", "scaffold_det", 0, TASK, n=5)
    assert list(sel.kind) == ["tp"] * 5 + ["fp"] * 5
    assert (sel[sel.kind == "tp"].y == 1.0).all()
    assert (sel[sel.kind == "fp"].y == 0.0).all()
    for kind in ("tp", "fp"):
        probs = sel[sel.kind == kind].prob.tolist()
        assert probs == sorted(probs, reverse=True)      # most confident first


@needs_data
@needs_preds
def test_pick_molecules_only_uses_test_rows_with_a_label():
    sel = pick_molecules("mpnn", "scaffold_det", 0, TASK, n=5)
    preds = pd.read_parquet(PREDS / "mpnn__scaffold_det__seed0.parquet")
    test_smiles = set(preds[preds.role == "test"].smiles)
    assert set(sel.smiles) <= test_smiles
    assert sel.y.notna().all()
    assert sel.smiles.is_unique
