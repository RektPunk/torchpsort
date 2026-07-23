<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=torchpsort&height=120&fontSize=80">
</div>

Fast, differentiable sorting and ranking in **pure PyTorch without C++ or CUDA**. This is a lightweight implementation of [Fast Differentiable Sorting and Ranking (Blondel et al.)](https://arxiv.org/abs/2002.08871) and inspired by [torchsort](https://github.com/teddykoker/torchsort). Unlike the [torchsort](https://github.com/teddykoker/torchsort), this version contains **no C++ or CUDA extensions**, making it easy to install and portable across platforms. While the original C++/CUDA implementation may have a performance edge for extremely large batch sizes, this pure PyTorch version is optimized to be efficient for standard deep learning workflows. Try it here:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RektPunk/torchpsort/blob/main/examples/notebook.ipynb)

## Installation

```bash
pip install torchpsort
```

> [!CAUTION]
> **Do not use `torch.compile`** with these functions. Because the implementation uses a sequential Python loop over the sequence length ($p$) to guarantee $O(Bp)$ memory efficiency, `torch.compile` will cause a graph compilation explosion for large sequences.
