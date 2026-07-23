# Copyright 2020 Google LLC
# Copyright 2021 Teddy Koker
# Copyright 2026 RektPunk

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor

from .isotonic import (
    isotonic_kl_backward,
    isotonic_kl_forward,
    isotonic_l2_backward,
    isotonic_l2_forward,
)


def _descending_ranks(x: Tensor) -> Tensor:
    return torch.arange(
        x.shape[1],
        0,
        -1,
        dtype=x.dtype,
        device=x.device,
    ).expand_as(x)


def _validate_inputs(x: Tensor, tau: float, reg: str) -> None:
    if x.ndim != 2:
        raise ValueError("x must be a 2D tensor")

    if reg not in ("l2", "kl"):
        raise ValueError(f"Unknown reg: {reg}")

    if tau <= 0:
        raise ValueError("tau must be positive")


class SoftRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, tau: float = 1.0, reg: str = "l2"):
        _validate_inputs(x, tau, reg)

        ctx.scale = 1.0 / tau
        ctx.is_l2 = reg == "l2"
        w = _descending_ranks(x)
        theta = x * ctx.scale
        s, permutation = torch.sort(theta, descending=True)
        inv_permutation = permutation.argsort(dim=1)
        if ctx.is_l2:
            dual_sol = isotonic_l2_forward(s - w)
            ret = (s - dual_sol).gather(1, inv_permutation)
            ctx.save_for_backward(dual_sol, permutation, inv_permutation)
        else:
            dual_sol = isotonic_kl_forward(s, torch.log(w))
            ret = torch.exp((s - dual_sol).gather(1, inv_permutation))
            ctx.save_for_backward(ret, s, dual_sol, permutation, inv_permutation)

        return ret

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.is_l2:
            dual_sol, permutation, inv_permutation = ctx.saved_tensors
            grad_iso = isotonic_l2_backward(
                dual_sol,
                grad_outputs[0].gather(1, permutation),
            )
            grad = grad_outputs[0] - grad_iso.gather(1, inv_permutation)
        else:
            ret, s, dual_sol, permutation, inv_permutation = ctx.saved_tensors
            grad_iso = isotonic_kl_backward(
                s,
                dual_sol,
                (grad_outputs[0] * ret).gather(1, permutation),
            )
            grad = (grad_outputs[0] * ret) - grad_iso.gather(1, inv_permutation)

        return grad * ctx.scale, None, None


class SoftSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, tau: float = 1.0, reg: str = "l2"):
        _validate_inputs(x, tau, reg)

        ctx.is_l2 = reg == "l2"
        w = _descending_ranks(x) / tau
        s, permutation = torch.sort(-x, descending=True)
        inv_permutation = permutation.argsort(dim=1)
        if ctx.is_l2:
            sol = isotonic_l2_forward(w - s)
            ctx.save_for_backward(sol, inv_permutation)
        else:
            sol = isotonic_kl_forward(w, s)
            ctx.save_for_backward(s, sol, inv_permutation)

        return sol - w

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.is_l2:
            sol, inv_permutation = ctx.saved_tensors
            grad = isotonic_l2_backward(sol, grad_outputs[0])
        else:
            s, sol, inv_permutation = ctx.saved_tensors
            grad = isotonic_kl_backward(s, sol, grad_outputs[0])

        return grad.gather(1, inv_permutation), None, None


def soft_rank(x: Tensor, tau: float = 1.0, reg: str = "l2") -> Tensor:
    """Differentiable approximation of the rank operator."""
    return SoftRank.apply(x, tau, reg)


def soft_sort(x: Tensor, tau: float = 1.0, reg: str = "l2") -> Tensor:
    """Differentiable approximation of the sort operator."""
    return SoftSort.apply(x, tau, reg)


def soft_min(x: Tensor, tau: float = 1.0, reg: str = "l2") -> Tensor:
    """Differentiable approximation of the min operator."""
    return soft_sort(x, tau=tau, reg=reg)[:, 0]


def soft_max(x: Tensor, tau: float = 1.0, reg: str = "l2") -> Tensor:
    """Differentiable approximation of the max operator."""
    return soft_sort(x, tau=tau, reg=reg)[:, -1]


def soft_kth_value(x: Tensor, k: int, tau: float = 1.0, reg: str = "l2") -> Tensor:
    """Differentiable approximation of the k-th value operator."""
    if not 1 <= k <= x.shape[1]:
        raise ValueError(f"k must be in [1, {x.shape[1]}], got {k}")
    return soft_sort(x, tau=tau, reg=reg)[:, k - 1]


def soft_topk_values(x: Tensor, k: int, tau: float = 1.0, reg: str = "l2") -> Tensor:
    """Differentiable approximation of the top-k values operator."""
    if not 1 <= k <= x.shape[1]:
        raise ValueError(f"k must be in [1, {x.shape[1]}], got {k}")
    return soft_sort(x, tau=tau, reg=reg)[:, -k:]


def soft_quantile(x: Tensor, q: float, tau: float = 1.0, reg: str = "l2") -> Tensor:
    """Differentiable approximation of the quantile operator."""
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    sorted_x = soft_sort(x, tau=tau, reg=reg)
    n = sorted_x.shape[1]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    alpha = pos - lo
    return (1 - alpha) * sorted_x[:, lo] + alpha * sorted_x[:, hi]


def soft_median(x: Tensor, tau: float = 1.0, reg: str = "l2") -> Tensor:
    """Differentiable approximation of the median operator."""
    return soft_quantile(x, q=0.5, tau=tau, reg=reg)
