import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable
from .sinc import sinc_kernel
from .fftconv import _custom_fft_conv1d

BLOCK_RATIO = 5


def _pad_to_block(x, block_size):
    offset = x.shape[-1] % block_size
    if offset:
        x = F.pad(x, [0, block_size - offset], mode='reflect')
    return x.view(*x.shape[:-1], -1, block_size)


def _pad_to_block_2(x, block_size, padding):
    offset = x.shape[-1] % block_size
    if offset:
        offset = block_size - offset
    x = F.pad(x, [padding, offset + padding], mode='reflect')
    return x.unfold(-1, block_size + 2 * padding, block_size)


class Upsample(nn.Module):
    r"""Upsampling by an integer amount.

    Args:
        q (int): upsample factor
        num_zeros (int): number of zero crossing to keep in the sinc filter. Default: ``16``
        window_func (Callable): window function. Default: :meth:`hann_window <torch.hann_window>`
        **kwargs: arguments passed through to :meth:`sinc_kernel <torch.sinc_kernel>`

    Shape:
        - Input: `[*, T]`
        - Output: `[*, T * q]`

    Examples::

        >>> import torch
        >>> upsampler = Upsample(3)
        >>> x = torch.randn(500)
        >>> print(len(upsampler(x)))
        1500

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
        kernel = F.pad(kernel.float(), [q - 1, 0]).view(-1,
                                                        q).t().reshape(q, 1, -1).flip(0)
        self.stride = q
        self.padding = N // 2 // q
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.view(-1, 1, shape[-1])

        block_length = self.kernel.shape[-1] * BLOCK_RATIO
        if shape[-1] < block_length:
            x = F.pad(x, [self.padding] * 2, mode='reflect')
            y = _custom_fft_conv1d(x, self.kernel)
        else:
            q = self.kernel.shape[0]
            x = _pad_to_block_2(x, block_length, self.padding)
            num_blocks = x.shape[-2]
            x = x.reshape(-1, 1, x.shape[-1])
            y = _custom_fft_conv1d(x, self.kernel)
            y = y.view(-1, num_blocks, q, block_length).transpose(1,
                                                                  2).reshape(-1, q, num_blocks * block_length)[..., :shape[-1]]

        y = y.transpose(1, 2).contiguous()
        return y.view(*shape[:-1], -1)
