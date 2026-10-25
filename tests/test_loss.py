import math

import pytest
pytest.importorskip("torch", reason="GNN tests need torch (not in requirements.txt)")
pytest.importorskip("torch_geometric", reason="GNN tests need torch_geometric (not in requirements.txt)")
import torch
import torch.nn.functional as F

from tox.train_gnn import masked_bce, masked_bce_with_logits


def test_no_missing_equals_plain_bce():
    torch.manual_seed(0)
    logits, y = torch.randn(5, 12), torch.randint(0, 2, (5, 12)).float()
    assert torch.allclose(masked_bce_with_logits(logits, y),
                          F.binary_cross_entropy_with_logits(logits, y))


def test_missing_labels_are_ignored_in_value():
    logits = torch.tensor([[0.3, -1.2, 2.0], [1.0, 0.5, -0.7]])
    y = torch.tensor([[1.0, float("nan"), 0.0], [float("nan"), 1.0, float("nan")]])
    ref = F.binary_cross_entropy_with_logits(
        logits[[0, 0, 1], [0, 2, 1]], y[[0, 0, 1], [0, 2, 1]])   # only the 3 labelled cells
    assert torch.allclose(masked_bce_with_logits(logits, y), ref)


def test_masked_entries_have_zero_gradient():
    logits = torch.randn(4, 12, requires_grad=True)
    y = torch.randint(0, 2, (4, 12)).float()
    mask = torch.rand(4, 12) < 0.5
    mask[0, 0] = False                                # make sure something is both kinds
    mask[1, 1] = True
    y[~mask] = float("nan")
    masked_bce_with_logits(logits, y).backward()
    assert torch.all(logits.grad[~mask] == 0)
    assert torch.any(logits.grad[mask] != 0)
    assert torch.isfinite(logits.grad).all()


def test_changing_a_masked_logit_changes_nothing():
    y = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    a = torch.tensor([[0.2, 5.0], [-0.4, 0.9]])
    b = a.clone(); b[0, 1] = -50.0
    assert masked_bce_with_logits(a, y) == masked_bce_with_logits(b, y)


def test_fully_masked_batch_is_zero_and_finite():
    logits = torch.randn(3, 12, requires_grad=True)
    y = torch.full((3, 12), float("nan"))
    loss = masked_bce_with_logits(logits, y)
    loss.backward()
    assert loss.item() == 0.0 and math.isfinite(loss.item())
    assert torch.all(logits.grad == 0)


def test_pos_weight_scales_positive_terms_only():
    logits = torch.zeros(2, 2)
    y = torch.tensor([[1.0, 0.0], [1.0, float("nan")]])
    pw = torch.tensor([3.0, 5.0])
    ln2 = math.log(2)
    # labelled cells: (0,0) pos w3, (0,1) neg, (1,0) pos w3  -> (3ln2 + ln2 + 3ln2) / 3
    assert masked_bce_with_logits(logits, y, pw).item() == pytest.approx(7 * ln2 / 3, rel=1e-5)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        masked_bce_with_logits(torch.zeros(2, 12), torch.zeros(2, 11))


def test_plan_name_and_old_name_are_the_same_function():
    assert masked_bce is masked_bce_with_logits


def test_masked_entries_zero_gradient_with_pos_weight_and_cap():
    from tox.train_gnn import compute_pos_weight
    y = torch.randint(0, 2, (6, 12)).float()
    y[torch.rand(6, 12) < 0.4] = float("nan")
    y[0, 0], y[1, 1] = float("nan"), 1.0
    pw = compute_pos_weight(y.numpy())   # capped at 30
    assert float(pw.max()) <= 30.0
    logits = torch.randn(6, 12, requires_grad=True)
    masked_bce(logits, y, pw).backward()
    assert torch.all(logits.grad[torch.isnan(y)] == 0)
    assert torch.isfinite(logits.grad).all() and torch.any(logits.grad != 0)


def test_fully_masked_batch_with_pos_weight_is_finite():
    logits = torch.randn(3, 12, requires_grad=True)
    loss = masked_bce(logits, torch.full((3, 12), float("nan")), torch.full((12,), 30.0))
    loss.backward()
    assert loss.item() == 0.0 and torch.all(logits.grad == 0)
