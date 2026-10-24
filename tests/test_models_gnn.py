import pytest
import torch
from torch_geometric.data import Batch

from tox.constants import TASKS
from tox.featurize_graph import build_graphs
from tox.models_gnn import (ALIASES, MODELS, POOLING_MODES, build_model, canonical_name, pooled_dim,
                            readout)

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
@pytest.mark.parametrize("pooling", ["mean", "max", "meanmax"])
def test_pooling_options(name, pooling, batch):
    assert build_model(name, hidden=16, pooling=pooling)(batch).shape == (len(SMILES), 12)


@pytest.mark.parametrize("name", ["mpnn", "afp"])
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


def test_model_tokens_match_the_plan_and_alias_still_works():
    assert set(MODELS) == {"gcn", "mpnn", "afp"}                    # C3 filename tokens, plan p. 5
    assert canonical_name("attentivefp") == "afp" and canonical_name("gcn") == "gcn"
    assert type(build_model("attentivefp", hidden=16)) is type(build_model("afp", hidden=16))
    assert set(ALIASES.values()) <= set(MODELS)


def test_default_pooling_is_meanmax_and_head_is_256_128_12():
    for name in ("gcn", "mpnn"):
        m = build_model(name)                                       # all defaults
        lin = [l for l in m.head.net if isinstance(l, torch.nn.Linear)]
        assert m.pooling == "meanmax"
        assert [(l.in_features, l.out_features) for l in lin] == [(256, 128), (128, 12)]


@pytest.mark.parametrize("name", ["gcn", "mpnn"])
@pytest.mark.parametrize("pooling,dim", [("mean", 128), ("max", 128), ("meanmax", 256)])
def test_head_input_width_follows_pooling(name, pooling, dim):
    m = build_model(name, pooling=pooling)
    assert m.head.net[0].in_features == dim == pooled_dim(128, pooling)


def test_meanmax_readout_is_mean_concat_max():
    h = torch.tensor([[1., 4.], [3., 2.], [10., -1.]])
    idx = torch.tensor([0, 0, 1])                                   # molecule 0: atoms 0,1; molecule 1: atom 2
    out = readout(h, idx, "meanmax")
    assert out.shape == (2, 4)
    assert torch.equal(out[0], torch.tensor([2., 3., 3., 4.]))      # mean(1,3), mean(4,2), max(1,3), max(4,2)
    assert torch.equal(out[1], torch.tensor([10., -1., 10., -1.]))
    assert torch.equal(readout(h, idx, "mean"), out[:, :2])
    assert torch.equal(readout(h, idx, "max"), out[:, 2:])


@pytest.mark.parametrize("name", ["gcn", "mpnn"])
def test_unsupported_pooling_rejected(name):
    assert set(POOLING_MODES) == {"mean", "max", "meanmax"}         # 'sum' is not in the plan
    for bad in ("sum", "attention", ""):
        with pytest.raises(ValueError, match="pooling"):
            build_model(name, pooling=bad)


@pytest.mark.parametrize("name", ["gcn", "mpnn"])
def test_head_width_scales_with_hidden_size(name):
    m = build_model(name, hidden=64)                                # tuning grid hidden in {64,128,256}
    lin = [l for l in m.head.net if isinstance(l, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in lin] == [(128, 64), (64, 12)]
