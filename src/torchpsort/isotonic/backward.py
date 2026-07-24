import torch
from torch import Tensor


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
        0, global_block_ids, flat_s, reduce="amax", include_self=False
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
