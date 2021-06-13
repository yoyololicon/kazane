import torch
from torch import Tensor
import torch.nn.functional as F
from torch.fft import rfft, irfft


def _custom_fft_conv1d(input: Tensor, weight: Tensor,
                       stride: int = 1) -> Tensor:
    weight = weight.view(-1, weight.shape[-1])
    output_size = (input.shape[-1] - 1 *
                   (weight.shape[-1] - 1) - 1) // stride + 1

    s = max(input.shape[-1], weight.shape[-1])
    if stride % 2:
        factor = stride * 2
    else:
        factor = stride

    offset = s % factor
    if offset:
        s += factor - offset

    X = rfft(input, n=s)
    W = rfft(weight, n=s)
    W.imag.mul_(-1)
    Y = X * W

    # handle stride
    if stride > 1:
        n_fft = s
        new_n_fft = n_fft // stride
        step_size = new_n_fft // 2
        strided_Y_size = step_size + 1

        unfolded_Y_real = Y.real.unfold(-1, strided_Y_size, step_size)
        unfolded_Y_imag = Y.imag[...,
                                 1:].unfold(-1, strided_Y_size - 2, step_size)
        Y_pos_real, Y_pos_imag = unfolded_Y_real[..., ::2,
                                                 :].sum(-2), unfolded_Y_imag[..., ::2, :].sum(-2)
        Y_neg_real, Y_neg_imag = unfolded_Y_real[..., 1::2, :].sum(
            -2).flip(-1), unfolded_Y_imag[..., 1::2, :].sum(-2).flip(-1)

        Y_real = Y_pos_real.add_(Y_neg_real)
        Y_imag = Y_pos_imag.add_(Y_neg_imag, alpha=-1)
        Y_imag = F.pad(Y_imag, [1, 1])

        Y = torch.view_as_complex(
            torch.stack((Y_real, Y_imag), -1)).div_(stride)

    output = irfft(Y)

    # Remove extra padded values
    output = output[..., :output_size].contiguous()
    return output
