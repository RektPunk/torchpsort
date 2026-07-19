import math

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from torchpsort.isotonic import (
    isotonic_kl_backward,
    isotonic_kl_forward,
    isotonic_l2_backward,
    isotonic_l2_forward,
)


def test_l2_forward_already_sorted():
    y = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    expected = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    out = isotonic_l2_forward(y)
    assert_close(out, expected)


def test_l2_forward_strictly_increasing():
    y = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[2.5, 2.5, 2.5, 2.5]])
    out = isotonic_l2_forward(y)
    assert_close(out, expected)


def test_l2_forward_mixed_blocks():
    y = torch.tensor([[3.0, 4.0, 1.0, 2.0]])
    expected = torch.tensor([[3.5, 3.5, 1.5, 1.5]])
    out = isotonic_l2_forward(y)
    assert_close(out, expected)


def test_l2_forward_negative_values():
    y = torch.tensor([[-1.0, -3.0, -2.0]])
    expected = torch.tensor([[-1.0, -2.5, -2.5]])
    out = isotonic_l2_forward(y)
    assert_close(out, expected)


def test_l2_forward_single_element():
    y = torch.tensor([[42.0]])
    expected = torch.tensor([[42.0]])
    out = isotonic_l2_forward(y)
    assert_close(out, expected)


def test_l2_forward_batch_independence():
    y = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    expected = torch.tensor([[2.0, 2.0, 2.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    out = isotonic_l2_forward(y)
    assert_close(out, expected)


def test_l2_forward_monotonic():
    random_samples = torch.randn(100, 20, dtype=torch.double)
    sol = isotonic_l2_forward(random_samples)
    assert torch.all(sol[:, 1:] <= sol[:, :-1])


def test_l2_forward_idempotent():
    random_samples = torch.randn(100, 20, dtype=torch.double)
    sol1 = isotonic_l2_forward(random_samples)
    sol2 = isotonic_l2_forward(sol1)
    assert_close(sol1, sol2)


def test_l2_forward_constant():
    y = torch.full((1, 5), 3.14)
    out = isotonic_l2_forward(y)
    assert_close(out, y)


def test_l2_backward_no_blocks():
    sol = torch.tensor([[3.0, 2.0, 1.0]])
    grad_output = torch.tensor([[0.5, 1.5, -1.0]])
    expected_grad_input = torch.tensor([[0.5, 1.5, -1.0]])
    grad_input = isotonic_l2_backward(sol, grad_output)
    assert_close(grad_input, expected_grad_input)


def test_l2_backward_single_block():
    sol = torch.tensor([[2.5, 2.5, 2.5, 2.5]])
    grad_output = torch.tensor([[1.0, 3.0, -2.0, 6.0]])
    expected_grad_input = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    grad_input = isotonic_l2_backward(sol, grad_output)
    assert_close(grad_input, expected_grad_input)


def test_l2_backward_mixed_blocks():
    sol = torch.tensor([[3.5, 3.5, 1.5, 1.5, 0.5]])
    grad_output = torch.tensor([[2.0, 4.0, 5.0, -1.0, 3.0]])
    expected_grad_input = torch.tensor([[3.0, 3.0, 2.0, 2.0, 3.0]])
    grad_input = isotonic_l2_backward(sol, grad_output)
    assert_close(grad_input, expected_grad_input)


def test_l2_backward_preserves_block_average():
    grad_output = torch.randn(1, 10)
    sol = torch.ones(1, 10)
    grad_input = isotonic_l2_backward(sol, grad_output)
    expected = grad_output.mean().expand_as(grad_input)
    assert_close(grad_input, expected)


def test_kl_forward_already_sorted():
    y = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    w = torch.zeros_like(y)
    expected = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    out = isotonic_kl_forward(y, w)
    assert_close(out, expected)


def test_kl_forward_strictly_increasing():
    y = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    w = torch.zeros_like(y)
    expected_val = torch.logsumexp(y, dim=1) - math.log(4)
    expected = expected_val.unsqueeze(1).expand_as(y)
    out = isotonic_kl_forward(y, w)
    assert_close(out, expected)


def test_kl_forward_mixed_blocks():
    y = torch.tensor([[3.0, 4.0, 1.0, 2.0]])
    w = torch.zeros_like(y)
    b1 = torch.logsumexp(torch.tensor([3.0, 4.0]), dim=0) - math.log(2)
    b2 = torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0) - math.log(2)
    expected = torch.tensor([[b1, b1, b2, b2]])
    out = isotonic_kl_forward(y, w)
    assert_close(out, expected)


def test_kl_forward_negative_values():
    y = torch.tensor([[-1.0, -3.0, -2.0]])
    w = torch.zeros_like(y)
    b1 = -1.0
    b2 = torch.logsumexp(torch.tensor([-3.0, -2.0]), dim=0) - math.log(2)
    expected = torch.tensor([[b1, b2, b2]])
    out = isotonic_kl_forward(y, w)
    assert_close(out, expected)


def test_kl_forward_single_element():
    y = torch.tensor([[42.0]])
    w = torch.zeros_like(y)
    expected = torch.tensor([[42.0]])
    out = isotonic_kl_forward(y, w)
    assert_close(out, expected)


def test_kl_forward_batch_independence():
    y = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    w = torch.zeros_like(y)
    row0_val = torch.logsumexp(torch.tensor([1.0, 2.0, 3.0]), dim=0) - math.log(3)
    expected = torch.tensor(
        [[row0_val, row0_val, row0_val], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]],
    )
    out = isotonic_kl_forward(y, w)
    assert_close(out, expected)


def test_kl_forward_monotonic():
    random_samples = torch.randn(100, 20, dtype=torch.double)
    w = torch.zeros_like(random_samples)
    sol = isotonic_kl_forward(random_samples, w)
    assert torch.all(sol[:, 1:] <= sol[:, :-1] + 1e-6)


def test_kl_forward_idempotent():
    random_samples = torch.randn(100, 20, dtype=torch.double)
    w = torch.zeros_like(random_samples)
    sol1 = isotonic_kl_forward(random_samples, w)
    sol2 = isotonic_kl_forward(sol1, torch.zeros_like(sol1))
    assert_close(sol1, sol2)


def test_kl_forward_constant():
    y = torch.full((1, 5), 3.14)
    w = torch.zeros_like(y)
    out = isotonic_kl_forward(y, w)
    assert_close(out, y)


def test_kl_backward_no_blocks():
    s = torch.tensor([[3.0, 2.0, 1.0]])
    sol = torch.tensor([[3.0, 2.0, 1.0]])
    grad_output = torch.tensor([[0.5, 1.5, -1.0]])
    expected_grad_input = torch.tensor([[0.5, 1.5, -1.0]])
    grad_input = isotonic_kl_backward(s, sol, grad_output)
    assert_close(grad_input, expected_grad_input)


def test_kl_backward_single_block():
    s = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    sol = torch.tensor([[2.5, 2.5, 2.5, 2.5]])
    grad_output = torch.tensor([[1.0, 3.0, -2.0, 6.0]])
    weights = F.softmax(s, dim=1)
    grad_sum = grad_output.sum()
    expected_grad_input = weights * grad_sum
    grad_input = isotonic_kl_backward(s, sol, grad_output)
    assert_close(grad_input, expected_grad_input)


def test_kl_backward_mixed_blocks():
    s = torch.tensor([[3.0, 4.0, 1.0, 2.0, 0.5]])
    sol = torch.tensor([[2.0, 2.0, 1.0, 1.0, 0.0]])
    grad_output = torch.tensor([[2.0, 4.0, 5.0, -1.0, 3.0]])
    exp_b1 = F.softmax(torch.tensor([3.0, 4.0]), dim=0) * (2.0 + 4.0)
    exp_b2 = F.softmax(torch.tensor([1.0, 2.0]), dim=0) * (5.0 - 1.0)
    exp_b3 = F.softmax(torch.tensor([0.5]), dim=0) * 3.0
    expected_grad_input = torch.cat([exp_b1, exp_b2, exp_b3]).unsqueeze(0)
    grad_input = isotonic_kl_backward(s, sol, grad_output)
    assert_close(grad_input, expected_grad_input)


def test_kl_backward_preserves_gradient_sum_weighted():
    s = torch.randn(1, 10)
    sol = torch.ones(1, 10)
    grad_output = torch.randn(1, 10)
    grad_input = isotonic_kl_backward(s, sol, grad_output)
    expected = F.softmax(s, dim=1) * grad_output.sum()
    assert_close(grad_input, expected)
