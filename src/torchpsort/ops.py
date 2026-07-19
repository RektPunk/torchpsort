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

from .isotonic import isotonic_backward, isotonic_forward


def _arange_asc(x: Tensor) -> Tensor:
    return torch.arange(
        x.shape[1],
        dtype=x.dtype,
        device=x.device,
    ).expand(x.shape[0], -1)


def _arange_desc(x: Tensor) -> Tensor:
    return torch.arange(
        x.shape[1] - 1,
        -1,
        -1,
        dtype=x.dtype,
        device=x.device,
    ).expand(x.shape[0], -1)


def _inv_permutation(permutation: Tensor) -> Tensor:
    inv_permutation = torch.empty_like(permutation)
    inv_permutation.scatter_(1, permutation, _arange_asc(permutation))
    return inv_permutation


class SoftRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, tau: float = 1.0):
        ctx.scale = 1.0 / tau
        w = _arange_desc(x) + 1
        theta = x * ctx.scale
        s, permutation = torch.sort(theta, descending=True)
        inv_permutation = _inv_permutation(permutation)
        dual_sol = isotonic_forward(s - w)
        ret = (s - dual_sol).gather(1, inv_permutation)
        ctx.save_for_backward(dual_sol, permutation, inv_permutation)
        return ret

    @staticmethod
    def backward(ctx, *grad_outputs):
        dual_sol, permutation, inv_permutation = ctx.saved_tensors
        grad = grad_outputs[0].clone()
        grad_iso = isotonic_backward(dual_sol, grad.gather(1, permutation))
        grad -= grad_iso.gather(1, inv_permutation)

        return grad * ctx.scale, None


class SoftSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, tau: float = 1.0):
        w = (_arange_desc(x) + 1) / tau
        s, permutation = torch.sort(-x, descending=True)
        sol = isotonic_forward(w - s)
        ctx.save_for_backward(sol, permutation)

        return -(w - sol)

    @staticmethod
    def backward(ctx, *grad_outputs):
        sol, permutation = ctx.saved_tensors
        inv_permutation = _inv_permutation(permutation)
        grad = isotonic_backward(sol, grad_outputs[0])

        return grad.gather(1, inv_permutation), None


def soft_rank(x: Tensor, tau: float = 1.0):
    """Differentiable rank operation.

    Args:
        x: Input tensor of shape (batch, p)
        tau: Regularization strength (temperature)
    """
    return SoftRank.apply(x, tau)


def soft_sort(x: Tensor, tau: float = 1.0):
    """Differentiable sort operation.

    Args:
        x: Input tensor of shape (batch, p)
        tau: Regularization strength (temperature)
    """
    return SoftSort.apply(x, tau)
