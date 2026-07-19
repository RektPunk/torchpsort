import pytest
import torch
from torch.autograd import gradcheck

from torchpsort.ops import soft_rank, soft_sort


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(42)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_rank(reg):
    x = torch.tensor([[10.0, 1.0, 5.0]], dtype=torch.float64)
    tau = 1e-3
    out = soft_rank(x, reg=reg, tau=tau)
    hard_ranks = torch.argsort(torch.argsort(x, descending=True))
    soft_ranks = torch.argsort(torch.argsort(out, descending=True))
    assert torch.equal(hard_ranks, soft_ranks)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_rank_gradcheck(gen, reg):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, generator=gen)
    tau = torch.tensor(1.0)
    assert gradcheck(soft_rank, (x, reg, tau), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort(reg):
    x = torch.randn(10, 5, requires_grad=True, dtype=torch.double)
    sorted_out = soft_sort(x, reg=reg, tau=0.1)
    assert torch.all(sorted_out[:, 1:] >= sorted_out[:, :-1] - 1e-5)
    sorted_out.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort_values(reg):
    x = torch.tensor([[10.0, 1.0, 5.0]], dtype=torch.double)
    sorted_out = soft_sort(x, reg=reg, tau=1e-3)
    expected = torch.tensor([[1.0, 5.0, 10.0]], dtype=torch.double)
    assert torch.allclose(sorted_out, expected, atol=1e-2)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort_gradcheck(gen, reg):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, generator=gen)
    tau = torch.tensor(1.0)
    assert gradcheck(soft_sort, (x, reg, tau), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort_hard_limit(reg):
    x = torch.tensor([[10.0, 1.0, 5.0, 8.0]], dtype=torch.float64)
    tau = 1e-3
    out = soft_sort(x, reg=reg, tau=tau)
    expected, _ = torch.sort(x, descending=False)
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort_permutation_invariance(gen, reg):
    x = torch.randn(1, 10, dtype=torch.float64, generator=gen)
    permutation = torch.randperm(10, generator=gen)
    x_shuffled = x[:, permutation]
    out_original = soft_sort(x, reg=reg, tau=1.0)
    out_shuffled = soft_sort(x_shuffled, reg=reg, tau=1.0)
    torch.testing.assert_close(out_original, out_shuffled)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_rank_permutation_equivariance(gen, reg):
    x = torch.randn(1, 10, dtype=torch.float64, generator=gen)
    permutation = torch.randperm(10, generator=gen)
    x_shuffled = x[:, permutation]
    out_original = soft_rank(x, reg=reg, tau=1.0)
    out_shuffled = soft_rank(x_shuffled, reg=reg, tau=1.0)
    out_original_shuffled = out_original[:, permutation]
    torch.testing.assert_close(out_shuffled, out_original_shuffled)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_ties_do_not_produce_nans(reg):
    x = torch.tensor(
        [[2.0, 2.0, 2.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True
    )
    out = soft_sort(x, reg=reg, tau=0.5)
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(out).any(), "Forward pass produced NaNs on ties"
    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "Backward pass produced NaNs on ties"
