import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable

from .sinc import sinc_kernel
from .upsample import _pad_to_block_2
from .fftconv import _custom_fft_conv1d

BLOCK_RATIO = 4


class Decimate(nn.Module):
    r"""Downsampling by an integer amount.

    Args:
        q (int): downsample factor
        num_zeros (int): number of zero crossing to keep in the sinc filter. Default: ``16``
        window_func (Callable): window function. Default: :meth:`hann_window <torch.hann_window>`
        **kwargs: arguments passed through to :meth:`sinc_kernel <torch.sinc_kernel>`

    Shape:
        - Input: `[*, T]`
        - Output: `[*, T / q]`

    Examples::

        >>> import torch
        >>> decimater = Decimate(4)
        >>> x = torch.randn(1000)
        >>> print(len(decimater(x)))
        250

    """

    def __init__(self,
                 q: int = 2,
                 num_zeros: int = 16,
                 window_func: Callable[[int],
                                       torch.Tensor] = torch.hann_window,
                 **kwargs):
        super().__init__()
        kernel = sinc_kernel(num_zeros, q, **kwargs)
        N = kernel.numel()
        kernel *= window_func(N, dtype=torch.float64)
        kernel /= q

        self.stride = q
        self.padding = N // 2
        self.register_buffer('kernel', kernel.view(1, 1, -1).float())

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.view(-1, 1, shape[-1])

        block_length = self.kernel.shape[-1] * self.stride * BLOCK_RATIO
        out_size = shape[-1] // self.stride
        if shape[-1] < block_length:
            x = F.pad(x, [self.padding] * 2, mode='reflect')
            y = _custom_fft_conv1d(x, self.kernel, stride=self.stride)
        else:
            x = _pad_to_block_2(x, block_length, self.padding)
            num_blocks = x.shape[-2]
            x = x.reshape(-1, 1, x.shape[-1])
            y = _custom_fft_conv1d(x, self.kernel, stride=self.stride)
            y = y.view(-1, num_blocks * y.shape[-1])[..., :out_size]

        return y.view(*shape[:-1], -1)
