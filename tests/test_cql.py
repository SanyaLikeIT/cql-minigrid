import math
import pytest
import torch

from cql_minigrid.offline import conservative_penalty, dqn_loss, q_learning_loss


def test_known_q_values_and_large_common_shift():
    q = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    action = torch.tensor([1])
    expected = torch.tensor(math.log(math.exp(1) + math.exp(2) + math.exp(3)) - 2, dtype=torch.float64)
    torch.testing.assert_close(conservative_penalty(q, action), expected)
    torch.testing.assert_close(conservative_penalty(q + 10000, action), expected, atol=1e-10, rtol=1e-10)


def test_alpha_zero_matches_dqn_loss_and_gradients_exactly():
    q = torch.tensor([[1., 2., 3.], [-2., 0., 1.]], requires_grad=True)
    actions = torch.tensor([1, 0])
    target = torch.tensor([0.5, -1.])
    baseline = dqn_loss(q, actions, target)
    cql, td, penalty = q_learning_loss(q, actions, target, alpha=0.)
    assert torch.equal(cql, baseline) and torch.equal(td, baseline)
    a = torch.autograd.grad(baseline, q, retain_graph=True)[0]
    b = torch.autograd.grad(cql, q)[0]
    assert torch.equal(a, b)
    total, td, penalty = q_learning_loss(q, actions, target, alpha=0.7)
    torch.testing.assert_close(total, td + .7 * penalty)


def test_negative_conservative_weight_is_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        q_learning_loss(torch.zeros(1,3), torch.tensor([0]), torch.tensor([0.]), alpha=-1)
