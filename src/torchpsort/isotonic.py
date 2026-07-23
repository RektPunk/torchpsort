#  Copyright 2007-2020 The scikit-learn developers.
#  Copyright 2020 Google LLC.
#  Copyright 2021 Teddy Koker.
#  Copyright 2026 RektPunk
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#    a. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#    b. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#    c. Neither the name of the Scikit-learn Developers  nor the names of
#       its contributors may be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#  DAMAGE.

import torch
import torch.nn.functional as F
from torch import Tensor

_SAFETY_FACTOR = 0.25


def _isotonic_l2_forward_dense(x: Tensor) -> Tensor:
    _, p = x.shape
    device = x.device
    dtype = x.dtype

    # Compute interval sums for all pairs (j <= k)
    pad_cumsum = F.pad(torch.cumsum(x, dim=1), (1, 0))  # (batch_size, p + 1)

    # sum_jk[b, j, k] = sum(x[b, j : k+1])
    sum_jk = pad_cumsum[:, None, 1:] - pad_cumsum[:, :-1, None]

    # Compute the length of each interval [j, k].
    idx = torch.arange(p, device=device, dtype=dtype)

    # count_jk[j, k] = k - j + 1
    count_jk = idx[None, :] - idx[:, None] + 1.0

    # vals[b, j, k] = mean of interval [j, k]
    vals = sum_jk / count_jk

    # Mask out mathematically invalid intervals (where start j > end k).
    mask = torch.ones(p, p, device=device, dtype=torch.bool).triu_()
    vals.masked_fill_(~mask, -float("inf"))

    # Suffix maximums: U[b, j, i] = max_{k >= i} vals[b, j, k]
    U = torch.flip(torch.cummax(torch.flip(vals, dims=[2]), dim=2)[0], dims=[2])
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
    counts = torch.arange(p, 0, -1, device=device, dtype=dtype)
    for k in range(p):
        # Compute interval means ending at k.
        sum_jk = x_cumsum[:, k + 1 : k + 2] - x_cumsum[:, : k + 1]
        count_jk = counts[p - k - 1 :]
        interval_values = sum_jk / count_jk

        # Prefix minimums: V_k[b, i] = min_{j <= i} interval_values[b, j]
        prefix_min = torch.cummin(interval_values, dim=1)[0]

        # Suffix maximums: sol[b, i] = max_{k >= i} prefix_min[b, i]
        torch.maximum(sol[:, : k + 1], prefix_min, out=sol[:, : k + 1])

    return sol


def _isotonic_kl_forward_dense(x: Tensor, w: Tensor) -> Tensor:
    _, p = x.shape
    device = x.device
    mask = torch.triu(torch.ones(p, p, device=device, dtype=torch.bool))[None, :, :]
    eps = torch.finfo(x.dtype).tiny

    # Log-sum-exp for x across all intervals (j <= k)
    diff_x = x[:, None, :] - x[:, :, None]
    diff_x = torch.where(mask, diff_x, -float("inf"))
    exp_x = torch.exp(diff_x)
    sum_exp_x = torch.cumsum(exp_x, dim=2)
    lse_x = x[:, :, None] + torch.log(torch.clamp(sum_exp_x, min=eps))

    # Log-sum-exp for w across all intervals (j <= k)
    diff_w = w[:, None, :] - w[:, :, None]
    diff_w = torch.where(mask, diff_w, -float("inf"))
    exp_w = torch.exp(diff_w)
    sum_exp_w = torch.cumsum(exp_w, dim=2)
    lse_w = w[:, :, None] + torch.log(torch.clamp(sum_exp_w, min=eps))

    # Compute interval objective values: (batch_size, p, p)
    vals = torch.where(mask, lse_x - lse_w, -float("inf"))

    # Suffix maximums: U[b, j, i] = max_{k >= i} vals[b, j, k]
    U = torch.flip(torch.cummax(torch.flip(vals, dims=[2]), dim=2)[0], dims=[2])
    U = torch.where(mask, U, float("inf"))

    # Prefix minimums: sol[b, i] = min_{j <= i} U[b, j, i]
    return torch.min(U, dim=1)[0]


def _isotonic_kl_forward_streaming(x: Tensor, w: Tensor) -> Tensor:
    _, p = x.shape
    sol = torch.full_like(x, -float("inf"))

    # Running log-sum-exp values for intervals ending at current k.
    lse_x = x.clone()
    lse_w = w.clone()
    for k in range(p):
        if k > 0:
            # Update log-sum-exp values for intervals ending at k.
            x_k = x[:, k : k + 1]
            w_k = w[:, k : k + 1]

            lse_x[:, :k] = torch.logaddexp(lse_x[:, :k], x_k)
            lse_w[:, :k] = torch.logaddexp(lse_w[:, :k], w_k)

        # Compute interval objective values: (batch_size, k + 1)
        interval_values = lse_x[:, : k + 1] - lse_w[:, : k + 1]

        # Prefix minimums: prefix_min[b, i] = min_{j <= i} interval_values[b, j]
        prefix_min = torch.cummin(interval_values, dim=1)[0]

        # Suffix maximums: sol[b, i] = max_{k >= i} prefix_min[b, i]
        torch.maximum(sol[:, : k + 1], prefix_min, out=sol[:, : k + 1])

    return sol


def _estimated_bytes(x: Tensor) -> int:
    batch_size, p = x.shape
    return 5 * batch_size * p * p * x.element_size()


def isotonic_l2_forward(x: Tensor) -> Tensor:
    if x.device.type != "cuda":
        return _isotonic_l2_forward_streaming(x)

    # Approximate peak memory of the dense implementation.
    estimated_bytes = _estimated_bytes(x)
    free_bytes, _ = torch.cuda.mem_get_info(x.device)

    if estimated_bytes <= free_bytes * _SAFETY_FACTOR:
        return _isotonic_l2_forward_dense(x)
    return _isotonic_l2_forward_streaming(x)


def isotonic_kl_forward(x: Tensor, w: Tensor) -> Tensor:
    if x.device.type != "cuda":
        return _isotonic_kl_forward_streaming(x, w)

    # Approximate peak memory of the dense implementation.
    estimated_bytes = _estimated_bytes(x)
    free_bytes, _ = torch.cuda.mem_get_info(x.device)

    if estimated_bytes <= free_bytes * _SAFETY_FACTOR:
        return _isotonic_kl_forward_dense(x, w)
    return _isotonic_kl_forward_streaming(x, w)


def _get_global_block_ids(sol: Tensor) -> Tensor:
    batch_size, p = sol.shape
    device = sol.device

    # Small tolerance for identifying pooled blocks.
    tol = torch.finfo(sol.dtype).eps * 10

    # Detect boundaries between pooled isotonic blocks
    diff = torch.abs(sol[:, 1:] - sol[:, :-1]) >= tol
    start_mask = torch.cat(
        [torch.ones(batch_size, 1, dtype=torch.bool, device=device), diff], dim=1
    )

    # Assign block ids within each batch.
    local_block_ids = torch.cumsum(start_mask.to(torch.long), dim=1) - 1

    # Offset block ids so different batches never collide.
    global_block_ids = (
        local_block_ids + p * torch.arange(batch_size, device=device).unsqueeze(1)
    ).reshape(-1)

    return global_block_ids


def isotonic_l2_backward(sol: Tensor, grad_output: Tensor) -> Tensor:
    batch_size, p = sol.shape
    device = sol.device
    dtype = sol.dtype
    max_num_blocks = batch_size * p
    global_block_ids = _get_global_block_ids(sol)

    # Sum gradients and count elements for each pooled block
    flat_grad_output = grad_output.reshape(-1)
    block_grad_sums = torch.zeros(max_num_blocks, device=device, dtype=dtype)
    block_counts = torch.zeros(max_num_blocks, device=device, dtype=dtype)

    block_grad_sums.scatter_add_(0, global_block_ids, flat_grad_output)
    block_counts.scatter_add_(0, global_block_ids, torch.ones_like(flat_grad_output))

    # Broadcast the block average back to every element
    block_grad = block_grad_sums[global_block_ids]
    block_count = block_counts[global_block_ids]
    grad_input = block_grad / torch.clamp(block_count, min=1.0)

    return grad_input.reshape(batch_size, p)


def isotonic_kl_backward(s: Tensor, sol: Tensor, grad_output: Tensor) -> Tensor:
    batch_size, p = sol.shape
    device = sol.device
    dtype = sol.dtype
    max_num_blocks = batch_size * p
    global_block_ids = _get_global_block_ids(sol)

    # Compute a numerically stable softmax within each pooled block
    flat_grad_output = grad_output.reshape(-1)
    flat_s = s.reshape(-1)
    block_max = torch.full((max_num_blocks,), -float("inf"), device=device, dtype=dtype)
    block_max.scatter_reduce_(
        0,
        global_block_ids,
        flat_s,
        reduce="amax",
        include_self=False,
    )
    flat_s_stable = flat_s - block_max[global_block_ids]
    flat_exp = torch.exp(flat_s_stable)
    block_exp_sums = torch.zeros(max_num_blocks, device=device, dtype=dtype)
    block_exp_sums.scatter_add_(0, global_block_ids, flat_exp)
    softmax_weights = flat_exp / torch.clamp(block_exp_sums[global_block_ids], min=1e-6)

    # Sum upstream gradients for each pooled block.
    block_grad_sums = torch.zeros(max_num_blocks, device=device, dtype=dtype)
    block_grad_sums.scatter_add_(0, global_block_ids, flat_grad_output)

    # Distribute the block gradient according to the block softmax.
    grad_input = softmax_weights * block_grad_sums[global_block_ids]

    return grad_input.reshape(batch_size, p)
