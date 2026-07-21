<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=torchpsort&height=120&fontSize=80">
</div>

Fast, differentiable sorting and ranking in **pure PyTorch without C++ or CUDA**. This is a lightweight implementation of [Fast Differentiable Sorting and Ranking (Blondel et al.)](https://arxiv.org/abs/2002.08871) and inspired by [torchsort](https://github.com/teddykoker/torchsort). Unlike the [torchsort](https://github.com/teddykoker/torchsort), this version contains **no C++ or CUDA extensions**, making it easy to install and portable across platforms. For the best runtime performance, I recommend using `torch.compile`, which substantially reduces the overhead of the pure PyTorch implementation. While the original C++/CUDA implementation may have a performance edge for extremely large batch sizes, this pure PyTorch version is optimized to be efficient for standard deep learning workflows. Try it here:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RektPunk/torchpsort/blob/main/examples/notebook.ipynb)

## Installation

```bash
pip install torchpsort
```

> [!TIP]
> For the best performance, compile the operators once before using them in your training loop:
>
> ```python
> import torch
> import torchpsort
>
> soft_sort = torch.compile(torchpsort.soft_sort)
> soft_rank = torch.compile(torchpsort.soft_rank)
> ```
>
> `torch.compile` can significantly improve performance, especially for repeated forward and backward passes.
