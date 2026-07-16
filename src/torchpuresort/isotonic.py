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


@torch.jit.script
def isotonic_forward(x: torch.Tensor) -> torch.Tensor:
    batch_size, p = x.shape
    solution = torch.empty_like(x)
    sums = torch.empty_like(x)
    target = torch.empty_like(x, dtype=torch.long)
    counts = torch.empty_like(x)
    for b in range(batch_size):
        for j in range(p):
            counts[b, j] = 1.0
            solution[b, j] = x[b, j]
            sums[b, j] = x[b, j]
            target[b, j] = j

        j = 0
        while j < p:
            k = target[b, j] + 1
            if k == p:
                break
            if solution[b, j] > solution[b, k]:
                j = k
                continue

            sum_x = sums[b, j]
            sum_c = counts[b, j]

            while True:
                prev_x = solution[b, k]
                sum_x += sums[b, k]
                sum_c += counts[b, k]
                k = target[b, k] + 1

                if k == p or prev_x > solution[b, k]:
                    solution[b, j] = sum_x / sum_c
                    sums[b, j] = sum_x
                    counts[b, j] = sum_c
                    target[b, j] = k - 1
                    target[b, k - 1] = j

                    if j > 0:
                        j = target[b, j - 1]
                    break

        j = 0
        while j < p:
            k = target[b, j] + 1
            start_idx = int(j + 1)
            end_idx = int(k)
            solution[b, start_idx:end_idx] = solution[b, j]
            j = k

    return solution


@torch.jit.script
def isotonic_backward(sol: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    batch_size, p = sol.shape
    grad_input = torch.empty_like(grad_output)
    tol = 1e-6 if sol.dtype == torch.float32 else 1e-12
    for b in range(batch_size):
        start = 0
        while start < p:
            end = start + 1
            while end < p and torch.abs(sol[b, end] - sol[b, start]) < tol:
                end += 1

            size = float(end - start)
            val = 1.0 / size

            grad_sum = grad_output[b, start:end].sum()
            grad_input[b, start:end] = val * grad_sum

            start = end

    return grad_input
