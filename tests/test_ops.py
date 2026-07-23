import pytest
import torch
from torch.autograd import gradcheck

from torchpsort import (
    soft_kth_value,
    soft_max,
    soft_median,
    soft_min,
    soft_quantile,
    soft_rank,
    soft_sort,
    soft_topk_values,
)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(42)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_rank(reg):
    x = torch.tensor([[10.0, 1.0, 5.0]], dtype=torch.float64)
    out = soft_rank(x, reg=reg)
    hard_ranks = torch.argsort(torch.argsort(x, descending=True))
    soft_ranks = torch.argsort(torch.argsort(out, descending=True))
    assert torch.equal(hard_ranks, soft_ranks)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_rank_permutation_equivariance(gen, reg):
    x = torch.randn(1, 10, dtype=torch.float64, generator=gen)
    permutation = torch.randperm(10, generator=gen)
    x_shuffled = x[:, permutation]
    out_original = soft_rank(x, reg=reg)
    out_shuffled = soft_rank(x_shuffled, reg=reg)
    out_original_shuffled = out_original[:, permutation]
    torch.testing.assert_close(out_shuffled, out_original_shuffled)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_rank_gradcheck(gen, reg):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, generator=gen)
    assert gradcheck(soft_rank, (x, torch.tensor(1.0), reg))


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort(reg):
    x = torch.randn(10, 5, requires_grad=True, dtype=torch.double)
    sorted_out = soft_sort(x, reg=reg)
    assert torch.all(sorted_out[:, 1:] >= sorted_out[:, :-1] - 1e-5)
    sorted_out.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort_values(reg):
    x = torch.tensor([[10.0, 1.0, 5.0]], dtype=torch.double)
    sorted_out = soft_sort(x, tau=1e-3, reg=reg)
    expected = torch.tensor([[1.0, 5.0, 10.0]], dtype=torch.double)
    assert torch.allclose(sorted_out, expected)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort_gradcheck(gen, reg):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, generator=gen)
    assert gradcheck(soft_sort, (x, torch.tensor(1.0), reg))


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort_hard_limit(reg):
    x = torch.tensor([[10.0, 1.0, 5.0, 8.0]], dtype=torch.float64)
    out = soft_sort(x, tau=1e-3, reg=reg)
    expected, _ = torch.sort(x, descending=False)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort_permutation_invariance(gen, reg):
    x = torch.randn(1, 10, dtype=torch.float64, generator=gen)
    permutation = torch.randperm(10, generator=gen)
    x_shuffled = x[:, permutation]
    out_original = soft_sort(x, reg=reg)
    out_shuffled = soft_sort(x_shuffled, reg=reg)
    torch.testing.assert_close(out_original, out_shuffled)


@pytest.mark.parametrize("reg", ["l2", "kl"])
def test_soft_sort_with_tiea(reg):
    x = torch.tensor(
        [[2.0, 2.0, 2.0, 1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    out = soft_sort(x, tau=1e-3, reg=reg)
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(out).any(), "Forward pass produced NaNs on ties"
    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "Backward pass produced NaNs on ties"


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_soft_kth_value_matches_sort(gen, n):
    x = torch.randn(10, n, dtype=torch.float64, generator=gen)
    expected, _ = torch.sort(x)
    for k in range(1, n + 1):
        torch.testing.assert_close(soft_kth_value(x, k=k, tau=1e-3), expected[:, k - 1])


@pytest.mark.parametrize(
    "x",
    [
        [1.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 3.0],
        [2.0, 2.0, 2.0, 2.0],
    ],
)
def test_soft_kth_value_with_ties(x):
    x = torch.tensor([x], dtype=torch.float64)
    expected, _ = torch.sort(x)
    for k in range(1, x.shape[1] + 1):
        torch.testing.assert_close(soft_kth_value(x, k=k, tau=1e-3), expected[:, k - 1])


def test_soft_kth_value_gradcheck(gen):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, generator=gen)
    for k in range(1, 6):
        assert gradcheck(lambda t, kk=k: soft_kth_value(t, kk), (x,))


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_soft_topk_values_matches_sort(gen, n):
    x = torch.randn(4, n, dtype=torch.float64, generator=gen)
    expected, _ = torch.sort(x)
    for k in range(1, n + 1):
        torch.testing.assert_close(soft_topk_values(x, k, tau=1e-3), expected[:, -k:])


@pytest.mark.parametrize(
    "x",
    [
        [1.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 3.0],
        [2.0, 2.0, 2.0, 2.0],
    ],
)
def test_soft_topk_values_ties(x):
    x = torch.tensor([x], dtype=torch.float64)
    expected, _ = torch.sort(x)
    for k in range(1, x.shape[1] + 1):
        torch.testing.assert_close(soft_topk_values(x, k, tau=1e-3), expected[:, -k:])


def test_soft_topk_values_gradcheck(gen):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, generator=gen)

    for k in range(1, 6):
        assert gradcheck(lambda t, kk=k: soft_topk_values(t, kk), (x,))


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        ([10.0, 1.0, 5.0, 8.0], (1.0, 10.0)),
        ([10.0, 1.0, 1.0, 5.0, 8.0], (1.0, 10.0)),
        ([-3.0, 2.0, -1.0], (-3.0, 2.0)),
    ],
)
def test_soft_extrema(x, expected):
    x = torch.tensor([x], dtype=torch.float64)
    _min, _max = expected
    expected_min = torch.tensor([_min], dtype=torch.float64)
    expected_max = torch.tensor([_max], dtype=torch.float64)
    torch.testing.assert_close(soft_min(x, tau=1e-3), expected_min)
    torch.testing.assert_close(soft_max(x, tau=1e-3), expected_max)


def test_soft_extrema_gradcheck(gen):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, generator=gen)
    assert gradcheck(lambda t: soft_min(t), (x,))
    assert gradcheck(lambda t: soft_max(t), (x,))


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_soft_quantile_matches_torch(gen, q):
    x = torch.randn(8, 10, dtype=torch.float64, generator=gen)
    expected = torch.quantile(x, q, dim=1)
    torch.testing.assert_close(soft_quantile(x, q=q, tau=1e-3), expected)


@pytest.mark.parametrize(
    "x",
    [
        [1.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 3.0],
        [2.0, 2.0, 2.0, 2.0],
    ],
)
def test_soft_quantile_with_ties(x):
    x = torch.tensor([x], dtype=torch.float64)
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        expected = torch.quantile(x, q, dim=1)
        torch.testing.assert_close(soft_quantile(x, q=q, tau=1e-3), expected)


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_soft_quantile_gradcheck(gen, q):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, generator=gen)
    assert gradcheck(lambda t, qq=q: soft_quantile(t, qq), (x,))


def test_soft_median(gen):
    x = torch.randn(4, 9, dtype=torch.float64, generator=gen)
    torch.testing.assert_close(
        soft_median(x, tau=1e-3), soft_quantile(x, q=0.5, tau=1e-3)
    )


def test_soft_median_gradcheck(gen):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, generator=gen)
    assert gradcheck(lambda t: soft_median(t), (x,))
