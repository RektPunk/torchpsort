<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=torchpuresort&height=120&fontSize=90">
</div>

Fast, differentiable sorting and ranking in **pure PyTorch without C++ or CUDA**. This is a lightweight implementation of [Fast Differentiable Sorting and Ranking (Blondel et al.)](https://arxiv.org/abs/2002.08871) and inspired by [torchsort](https://github.com/teddykoker/torchsort). Unlike the [torchsort](https://github.com/teddykoker/torchsort), this version contains **no C++ or CUDA extensions**, making it easy to install, portable, and compatible with any environment where PyTorch runs. While the original C++/CUDA implementation may have a performance edge for extremely large batch sizes, this pure PyTorch version is optimized to be efficient for standard deep learning workflows.

## Installation

```bash
pip install torchpuresort
```

## Quick Start

`torchpuresort` provides two simple functions: `soft_sort` and `soft_rank`. They behave exactly like the original `torchsort` API.

```python
import torch
from torchpuresort import soft_rank, soft_sort

# Basic Example
x = torch.tensor(
    [[8.0, 0.0, 5.0, 3.0, 2.0, 1.0, 6.0, 7.0, 9.0]],
    requires_grad=True,
)

## Differentiable Sort
## the parameter tau controls the "softness" (regularization strength)
sorted_x = soft_sort(x, tau=0.1)
print(sorted_x)
# tensor([[-0., 1., 2., 3., 5., 6., 7., 8., 9.]])

## Differentiable Rank
ranks = soft_rank(x, tau=0.1)
print(ranks)
# tensor([[8., 1., 5., 4., 3., 2., 6., 7., 9.]])

## Backprop works out of the box
loss = sorted_x.sum()
loss.backward()
print(x.grad)
# tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1.]])


# Spearman Rank Correlation Example
def spearmanr(pred, target, **kw):
    pred = soft_rank(pred, **kw)
    target = soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
target = torch.tensor([[5.0, 6.0, 7.0, 8.0, 7.0]])
spearman = spearmanr(pred, target)
print(spearman)
# tensor(0.8321)

print(torch.autograd.grad(spearman, pred))
# (tensor([[-5.5470e-02,  2.9802e-09,  5.5470e-02,  1.1094e-01, -1.1094e-01]]),)
```
