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


def _index_range(x: Tensor) -> Tensor:
    return torch.arange(
        x.shape[1],
        dtype=x.dtype,
        device=x.device,
    ).expand(x.shape[0], -1)


def _descending_ranks(x: Tensor) -> Tensor:
    return torch.arange(
        x.shape[1],
        0,
        -1,
        dtype=x.dtype,
        device=x.device,
    ).expand_as(x)


def _inv_permutation(permutation: Tensor) -> Tensor:
    inv_permutation = torch.empty_like(permutation)
    inv_permutation.scatter_(1, permutation, _index_range(permutation))
    return inv_permutation


def _validate_inputs(x: Tensor, reg: str, tau: float) -> None:
    if x.ndim != 2:
        raise ValueError("x must be a 2D tensor")

    if reg not in ("l2", "kl"):
        raise ValueError(f"Unknown reg: {reg}")

    if tau <= 0:
        raise ValueError("tau must be positive")


class SoftRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, reg: str = "l2", tau: float = 1.0):
        _validate_inputs(x, reg, tau)

        ctx.scale = 1.0 / tau
        ctx.is_l2 = reg == "l2"
        w = _descending_ranks(x)
        theta = x * ctx.scale
        s, permutation = torch.sort(theta, descending=True)
        inv_permutation = _inv_permutation(permutation)
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
    def forward(ctx, x: Tensor, reg: str = "l2", tau: float = 1.0):
        _validate_inputs(x, reg, tau)

        ctx.is_l2 = reg == "l2"
        w = _descending_ranks(x) / tau
        s, permutation = torch.sort(-x, descending=True)
        inv_permutation = _inv_permutation(permutation)
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


def soft_rank(x: Tensor, reg: str = "l2", tau: float = 1.0) -> Tensor:
    """Differentiable approximation of the rank operator.

    Args:
        x: Input tensor of shape (batch, p)
        reg: Regularization type ("l2" or "kl")
        tau: Regularization strength (temperature)
    """
    return SoftRank.apply(x, reg, tau)


def soft_sort(x: Tensor, reg: str = "l2", tau: float = 1.0) -> Tensor:
    """Differentiable approximation of the sort operator.

    Args:
        x: Input tensor of shape (batch, p)
        reg: Regularization type ("l2" or "kl")
        tau: Regularization strength (temperature)
    """
    return SoftSort.apply(x, reg, tau)
