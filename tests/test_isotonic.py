import torch
from torch.testing import assert_close

from torchpuresort.isotonic import isotonic_backward, isotonic_forward


def test_forward_already_sorted():
    y = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    expected = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    out = isotonic_forward(y)
    assert_close(out, expected)


def test_forward_strictly_increasing():
    y = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[2.5, 2.5, 2.5, 2.5]])
    out = isotonic_forward(y)
    assert_close(out, expected)


def test_forward_mixed_blocks():
    y = torch.tensor([[3.0, 4.0, 1.0, 2.0]])
    expected = torch.tensor([[3.5, 3.5, 1.5, 1.5]])
    out = isotonic_forward(y)
    assert_close(out, expected)


def test_forward_negative_values():
    y = torch.tensor([[-1.0, -3.0, -2.0]])
    expected = torch.tensor([[-1.0, -2.5, -2.5]])
    out = isotonic_forward(y)
    assert_close(out, expected)


def test_forward_single_element():
    y = torch.tensor([[42.0]])
    expected = torch.tensor([[42.0]])
    out = isotonic_forward(y)
    assert_close(out, expected)


def test_forward_batch_independence():
    y = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    expected = torch.tensor([[2.0, 2.0, 2.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    out = isotonic_forward(y)
    assert_close(out, expected)


def test_forward_monotonic():
    random_samples = torch.randn(100, 20, dtype=torch.double)
    sol = isotonic_forward(random_samples)
    assert torch.all(sol[:, 1:] <= sol[:, :-1])


def test_forward_idempotent():
    random_samples = torch.randn(100, 20, dtype=torch.double)
    sol1 = isotonic_forward(random_samples)
    sol2 = isotonic_forward(sol1)
    assert_close(sol1, sol2)


def test_forward_constant():
    y = torch.full((1, 5), 3.14)
    out = isotonic_forward(y)
    assert_close(out, y)


def test_backward_no_blocks():
    sol = torch.tensor([[3.0, 2.0, 1.0]])
    grad_output = torch.tensor([[0.5, 1.5, -1.0]])
    expected_grad_input = torch.tensor([[0.5, 1.5, -1.0]])
    grad_input = isotonic_backward(sol, grad_output)
    assert_close(grad_input, expected_grad_input)


def test_backward_single_block():
    sol = torch.tensor([[2.5, 2.5, 2.5, 2.5]])
    grad_output = torch.tensor([[1.0, 3.0, -2.0, 6.0]])
    expected_grad_input = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    grad_input = isotonic_backward(sol, grad_output)
    assert_close(grad_input, expected_grad_input)


def test_backward_mixed_blocks():
    sol = torch.tensor([[3.5, 3.5, 1.5, 1.5, 0.5]])
    grad_output = torch.tensor([[2.0, 4.0, 5.0, -1.0, 3.0]])
    expected_grad_input = torch.tensor([[3.0, 3.0, 2.0, 2.0, 3.0]])
    grad_input = isotonic_backward(sol, grad_output)
    assert_close(grad_input, expected_grad_input)


def test_backward_preserves_block_average():
    grad_output = torch.randn(1, 10)
    sol = torch.ones(1, 10)
    grad_input = isotonic_backward(sol, grad_output)
    expected = grad_output.mean().expand_as(grad_input)
    assert_close(grad_input, expected)
