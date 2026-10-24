import pytest
import torch
from torch_geometric.data import Batch

from tox.constants import TASKS
from tox.featurize_graph import build_graphs
from tox.models_gnn import MODELS, build_model

SMILES = ["CCO", "c1ccccc1C(=O)O", "CC(=O)[O-]", "[Na+]", "C1CCCCC1N", "CN(C)C=O"]


@pytest.fixture(scope="module")
def batch():
    return Batch.from_data_list(build_graphs(SMILES))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_forward_shape_and_finite(name, batch):
    out = build_model(name, hidden=32)(batch)
    assert out.shape == (len(SMILES), len(TASKS))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(MODELS))
def test_gradients_flow_to_all_parameters(name, batch):
    m = build_model(name, hidden=32)
    m(batch).sum().backward()
    no_grad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not no_grad, no_grad


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_layer_counts(name, n_layers, batch):
    assert build_model(name, hidden=16, n_layers=n_layers)(batch).shape == (len(SMILES), 12)


@pytest.mark.parametrize("name", ["gcn", "mpnn"])
@pytest.mark.parametrize("pooling", ["mean", "sum", "max"])
def test_pooling_options(name, pooling, batch):
    assert build_model(name, hidden=16, pooling=pooling)(batch).shape == (len(SMILES), 12)


@pytest.mark.parametrize("name", ["mpnn", "attentivefp"])
def test_bond_features_toggle_changes_output_only_when_on(name, batch):
    torch.manual_seed(0)
    on = build_model(name, hidden=16, use_bond_features=True).eval()
    off = build_model(name, hidden=16, use_bond_features=False).eval()
    off.load_state_dict(on.state_dict())               # same weights, only the input differs
    zeroed = batch.clone(); zeroed.edge_attr = torch.zeros_like(batch.edge_attr)
    assert torch.allclose(off(batch), on(zeroed), atol=1e-6)    # off == feeding zero bonds
    assert not torch.allclose(on(batch), off(batch), atol=1e-6)  # and bonds matter when on


@pytest.mark.parametrize("name", sorted(MODELS))
def test_eval_is_independent_of_batch_mates(name):
    """A molecule's logits must not depend on what else is in the batch (eval mode)."""
    gs = build_graphs(SMILES)
    m = build_model(name, hidden=16).eval()
    with torch.no_grad():
        full = m(Batch.from_data_list(gs))
        alone = m(Batch.from_data_list([gs[1]]))
    assert torch.allclose(full[1], alone[0], atol=1e-5)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model("transformer")
