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


@torch.jit.script
def isotonic_l2_forward(x: Tensor) -> Tensor:
    batch_size, p = x.shape
    solution = x.clone()
    block_sums = x.clone()
    block_counts = torch.ones_like(x)
    block_end = (
        torch.arange(p, device=x.device, dtype=torch.long)
        .expand(batch_size, -1)
        .clone()
    )
    for b in range(batch_size):
        block = 0
        while block < p:
            next_block = block_end[b, block] + 1
            if next_block == p:
                break
            if solution[b, block] > solution[b, next_block]:
                block = next_block
                continue

            block_sum = block_sums[b, block]
            block_count = block_counts[b, block]

            while True:
                last_value = solution[b, next_block]
                block_sum += block_sums[b, next_block]
                block_count += block_counts[b, next_block]
                next_block = block_end[b, next_block] + 1

                if next_block == p or last_value > solution[b, next_block]:
                    solution[b, block] = block_sum / block_count
                    block_sums[b, block] = block_sum
                    block_counts[b, block] = block_count
                    block_end[b, block] = next_block - 1
                    block_end[b, next_block - 1] = block

                    if block > 0:
                        block = block_end[b, block - 1]
                    break

        block = 0
        while block < p:
            next_block = block_end[b, block] + 1
            solution[b, (block + 1) : next_block] = solution[b, block]
            block = next_block

    return solution


@torch.jit.script
def isotonic_kl_forward(x: Tensor, w: Tensor) -> Tensor:
    batch_size, p = x.shape
    solution = x - w
    block_lse_x = x.clone()
    block_lse_w = w.clone()
    block_end = (
        torch.arange(p, device=x.device, dtype=torch.long)
        .expand(batch_size, -1)
        .clone()
    )

    for b in range(batch_size):
        block = 0
        while block < p:
            next_block = block_end[b, block] + 1
            if next_block == p:
                break
            if solution[b, block] > solution[b, next_block]:
                block = next_block
                continue

            curr_lse_x = block_lse_x[b, block]
            curr_lse_w = block_lse_w[b, block]

            while True:
                last_value = solution[b, next_block]
                curr_lse_x = torch.logaddexp(curr_lse_x, block_lse_x[b, next_block])
                curr_lse_w = torch.logaddexp(curr_lse_w, block_lse_w[b, next_block])
                next_block = block_end[b, next_block] + 1

                if next_block == p or last_value > solution[b, next_block]:
                    solution[b, block] = curr_lse_x - curr_lse_w
                    block_lse_x[b, block] = curr_lse_x
                    block_lse_w[b, block] = curr_lse_w
                    block_end[b, block] = next_block - 1
                    block_end[b, next_block - 1] = block

                    if block > 0:
                        block = block_end[b, block - 1]
                    break

        block = 0
        while block < p:
            next_block = block_end[b, block] + 1
            solution[b, (block + 1) : next_block] = solution[b, block]
            block = next_block

    return solution


@torch.jit.script
def isotonic_l2_backward(sol: Tensor, grad_output: Tensor) -> Tensor:
    batch_size, p = sol.shape
    grad_input = torch.empty_like(grad_output)
    tol = 1e-6 if sol.dtype == torch.float32 else 1e-12
    for b in range(batch_size):
        start = 0
        while start < p:
            end = start + 1
            while end < p and torch.abs(sol[b, end] - sol[b, start]) < tol:
                end += 1

            val = 1.0 / float(end - start)

            grad_sum = grad_output[b, start:end].sum()
            grad_input[b, start:end] = val * grad_sum

            start = end

    return grad_input


@torch.jit.script
def isotonic_kl_backward(s: Tensor, sol: Tensor, grad_output: Tensor) -> Tensor:
    batch_size, p = sol.shape
    grad_input = torch.empty_like(grad_output)
    tol = 1e-6 if sol.dtype == torch.float32 else 1e-12
    for b in range(batch_size):
        start = 0
        while start < p:
            end = start + 1
            while end < p and torch.abs(sol[b, end] - sol[b, start]) < tol:
                end += 1

            block_s = s[b, start:end]
            softmax_weights = torch.softmax(block_s, dim=0)

            grad_sum = grad_output[b, start:end].sum()
            grad_input[b, start:end] = softmax_weights * grad_sum

            start = end

    return grad_input
