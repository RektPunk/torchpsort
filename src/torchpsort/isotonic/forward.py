import torch
import torch.nn.functional as F
from torch import Tensor

_MASK_CACHE: dict[tuple[torch.device, int], Tensor] = {}
_COUNT_JK_CACHE: dict[tuple[torch.device, torch.dtype, int], Tensor] = {}
_COUNTS_CACHE: dict[tuple[torch.device, torch.dtype, int], Tensor] = {}


def _get_mask(device: torch.device, p: int) -> Tensor:
    key = (device, p)
    mask = _MASK_CACHE.get(key)
    if mask is None:
        mask = torch.ones(p, p, device=device, dtype=torch.bool).triu_()
        _MASK_CACHE[key] = mask

    return mask


def _get_count_jk(
    device: torch.device,
    dtype: torch.dtype,
    p: int,
) -> Tensor:
    key = (device, dtype, p)
    count = _COUNT_JK_CACHE.get(key)
    if count is None:
        # Compute the length of each interval [j, k].
        idx = torch.arange(p, device=device, dtype=dtype)
        # count_jk[j, k] = k - j + 1
        count = idx[None, :] - idx[:, None]
        count.add_(1)
        _COUNT_JK_CACHE[key] = count

    return count


def _get_counts(
    device: torch.device,
    dtype: torch.dtype,
    p: int,
) -> Tensor:
    key = (device, dtype, p)
    counts = _COUNTS_CACHE.get(key)
    if counts is None:
        counts = torch.arange(p, 0, -1, device=device, dtype=dtype)
        _COUNTS_CACHE[key] = counts

    return counts


def _isotonic_l2_forward_dense(x: Tensor) -> Tensor:
    _, p = x.shape
    device = x.device
    dtype = x.dtype
    mask = _get_mask(device, p)

    # Compute interval sums for all pairs (j <= k)
    pad_cumsum = F.pad(torch.cumsum(x, dim=1), (1, 0))  # (batch_size, p + 1)

    # interval_values[b, j, k] = sum(x[b, j : k+1])
    interval_values = pad_cumsum[:, None, 1:] - pad_cumsum[:, :-1, None]

    # Compute the length of each interval [j, k].
    count_jk = _get_count_jk(device, dtype, p)

    # interval_values[b, j, k] = mean of interval [j, k]
    interval_values.div_(count_jk)

    # Mask out mathematically invalid intervals (where start j > end k).
    interval_values.masked_fill_(~mask, -float("inf"))

    # Suffix maximums: U[b, j, i] = max_{k >= i} interval_values[b, j, k]
    U = torch.flip(
        torch.cummax(torch.flip(interval_values, dims=[2]), dim=2)[0], dims=[2]
    )
    U.masked_fill_(~mask, float("inf"))

    # Prefix minimums: sol[b, i] = min_{j <= i} U[b, j, i]
    return torch.min(U, dim=1)[0]


def _isotonic_kl_forward_dense(x: Tensor, w: Tensor) -> Tensor:
    _, p = x.shape
    device = x.device
    mask = _get_mask(device, p)

    # Log-sum-exp for x across all intervals (j <= k)
    interval_values = x[:, None, :].expand(-1, p, -1).clone()
    interval_values.masked_fill_(~mask, -float("inf"))
    torch.logcumsumexp(interval_values, dim=2, out=interval_values)

    # Log-sum-exp for w across all intervals (j <= k)
    lse_w = w[:, None, :].expand(-1, p, -1).clone()
    lse_w.masked_fill_(~mask, -float("inf"))
    torch.logcumsumexp(lse_w, dim=2, out=lse_w)

    # Compute interval objective values: (batch_size, p, p)
    interval_values.sub_(lse_w)

    # Suffix maximums: U[b, j, i] = max_{k >= i} interval_values[b, j, k]
    U = torch.flip(
        torch.cummax(torch.flip(interval_values, dims=[2]), dim=2)[0], dims=[2]
    )
    U.masked_fill_(~mask, float("inf"))

    # Prefix minimums: sol[b, i] = min_{j <= i} U[b, j, i]
    return torch.min(U, dim=1)[0]


def _isotonic_l2_forward_streaming(x: Tensor) -> Tensor:
    _, p = x.shape
    device = x.device
    dtype = x.dtype
    sol = torch.full_like(x, -float("inf"))

    # Compute prefix sums for interval sum queries.
    x_cumsum = F.pad(torch.cumsum(x, dim=1), (1, 0))

    # Interval lengths for a fixed right endpoint k.
    counts = _get_counts(device, dtype, p)
    for k in range(p):
        # Compute interval means ending at k.
        interval_values = x_cumsum[:, (k + 1) : (k + 2)] - x_cumsum[:, : (k + 1)]
        count_jk = counts[(p - k - 1) :]
        interval_values.div_(count_jk)

        # Prefix minimums: V_k[b, i] = min_{j <= i} interval_values[b, j]
        prefix_min = torch.cummin(interval_values, dim=1)[0]

        # Suffix maximums: sol[b, i] = max_{k >= i} prefix_min[b, i]
        torch.maximum(sol[:, : (k + 1)], prefix_min, out=sol[:, : (k + 1)])

    return sol


def _isotonic_kl_forward_streaming(x: Tensor, w: Tensor) -> Tensor:
    _, p = x.shape
    sol = torch.full_like(x, -float("inf"))

    # Running log-sum-exp values for intervals ending at current k.
    lse_x = x.clone()
    lse_w = w.clone()
    for k in range(p):
        if k > 0:
            # Update log-sum-exp values for intervals ending at k.
            torch.logaddexp(lse_x[:, :k], x[:, k : (k + 1)], out=lse_x[:, :k])
            torch.logaddexp(lse_w[:, :k], w[:, k : (k + 1)], out=lse_w[:, :k])

        # Compute interval objective values: (batch_size, (k + 1))
        interval_values = lse_x[:, : (k + 1)] - lse_w[:, : (k + 1)]

        # Prefix minimums: prefix_min[b, i] = min_{j <= i} interval_values[b, j]
        prefix_min = torch.cummin(interval_values, dim=1)[0]

        # Suffix maximums: sol[b, i] = max_{k >= i} prefix_min[b, i]
        torch.maximum(sol[:, : (k + 1)], prefix_min, out=sol[:, : (k + 1)])

    return sol


def _should_use_dense(x: Tensor) -> bool:
    if x.device.type != "cuda":
        return False

    # Approximate peak memory used by dense kernels.
    batch_size, p = x.shape
    estimated = 20 * batch_size * p * p * x.element_size()
    free, _ = torch.cuda.mem_get_info(x.device)

    return estimated <= free


def isotonic_l2_forward(x: Tensor) -> Tensor:
    if _should_use_dense(x):
        return _isotonic_l2_forward_dense(x)
    return _isotonic_l2_forward_streaming(x)


def isotonic_kl_forward(x: Tensor, w: Tensor) -> Tensor:
    if _should_use_dense(x):
        return _isotonic_kl_forward_dense(x, w)
    return _isotonic_kl_forward_streaming(x, w)
