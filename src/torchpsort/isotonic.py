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
from torch import Tensor


def isotonic_l2_forward(x: Tensor) -> Tensor:
    _, p = x.shape
    device = x.device
    dtype = x.dtype

    # Compute interval sums and means for all pairs (j <= k)
    x_cumsum = torch.cumsum(x, dim=1)
    sum_jk = x_cumsum[:, None, :] - x_cumsum[:, :, None] + x[:, :, None]
    count_jk = torch.arange(1, p + 1, device=device, dtype=dtype)
    count_jk = count_jk[None, :] - count_jk[:, None] + 1.0
    vals = sum_jk / count_jk[None, :, :]

    # Mask invalid lower-triangle entries (j > k)
    mask = torch.triu(torch.ones(p, p, device=device, dtype=torch.bool))[None, :, :]

    # Interval means.
    # Shape: (batch_size, p, p)
    vals = torch.where(mask, vals, -float("inf"))

    # Suffix maximums: U[b, j, i] = max_{k >= i} vals[b, j, k]
    U = torch.flip(torch.cummax(torch.flip(vals, dims=[2]), dim=2)[0], dims=[2])
    U = torch.where(mask, U, float("inf"))

    # Prefix minimums: sol[b, i] = min_{j <= i} U[b, j, i]
    return torch.min(U, dim=1)[0]


def isotonic_kl_forward(x: Tensor, w: Tensor) -> Tensor:
    _, p = x.shape
    device = x.device
    mask = torch.triu(torch.ones(p, p, device=device, dtype=torch.bool))[None, :, :]
    eps = torch.finfo(torch.float32).tiny

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

    # Compute interval objective values
    # Shape: (batch_size, p, p)
    vals = torch.where(mask, lse_x - lse_w, -float("inf"))

    # Suffix maximums: U[b, j, i] = max_{k >= i} vals[b, j, k]
    U = torch.flip(torch.cummax(torch.flip(vals, dims=[2]), dim=2)[0], dims=[2])
    U = torch.where(mask, U, float("inf"))

    # Prefix minimums: sol[b, i] = min_{j <= i} U[b, j, i]
    return torch.min(U, dim=1)[0]


def _get_global_block_ids(sol: Tensor) -> Tensor:
    batch_size, p = sol.shape

    device = sol.device
    tol = torch.finfo(sol.dtype).eps * 100

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
    num_blocks = batch_size * p
    global_block_ids = _get_global_block_ids(sol)

    # Sum gradients and count elements for each pooled block
    flat_grad_output = grad_output.reshape(-1)
    block_grad_sums = torch.zeros(num_blocks, device=device, dtype=dtype)
    block_counts = torch.zeros(num_blocks, device=device, dtype=dtype)

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
    num_blocks = batch_size * p
    global_block_ids = _get_global_block_ids(sol)

    # Compute a numerically stable softmax within each pooled block
    flat_grad_output = grad_output.reshape(-1)
    flat_s = s.reshape(-1)
    block_max = torch.full((num_blocks,), -float("inf"), device=device, dtype=dtype)
    block_max.scatter_reduce_(
        0,
        global_block_ids,
        flat_s,
        reduce="amax",
        include_self=False,
    )
    flat_s_stable = flat_s - block_max[global_block_ids]
    flat_exp = torch.exp(flat_s_stable)
    block_exp_sums = torch.zeros(num_blocks, device=device, dtype=dtype)
    block_exp_sums.scatter_add_(0, global_block_ids, flat_exp)
    softmax_weights = flat_exp / torch.clamp(block_exp_sums[global_block_ids], min=1e-6)

    # Sum upstream gradients for each pooled block.
    block_grad_sums = torch.zeros(num_blocks, device=device, dtype=dtype)
    block_grad_sums.scatter_add_(0, global_block_ids, flat_grad_output)

    # Distribute the block gradient according to the block softmax.
    grad_input = softmax_weights * block_grad_sums[global_block_ids]

    return grad_input.reshape(batch_size, p)
