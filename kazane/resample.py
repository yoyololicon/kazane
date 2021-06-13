import torch
import math
import torch.nn.functional as F

from .fftconv import _custom_fft_conv1d


def _pad_to_block(x, block_size, padding):
    offset = x.shape[-1] % block_size
    if offset:
        offset = block_size - offset
    x = F.pad(x, [padding[0], offset + padding[1]], mode='replicate')
    return x.unfold(-1, block_size + sum(padding), block_size)


class Resample(torch.nn.Module):
    """
    Resampling from the sample rate `old_sr` to `new_sr`.
    """

    def __init__(self, old_sr: int, new_sr: int, zeros: int = 24, rolloff: float = 0.945):
        """
        Args:
            old_sr (int): sample rate of the input signal x.
            new_sr (int): sample rate of the output.
            zeros (int): number of zero crossing to keep in the sinc filter.
            rolloff (float): use a lowpass filter that is `rolloff * new_sr / 2`,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce anti-aliasing, but will reduce some of the
                highest frequencies.

        Shape:

            - Input: `[*, T]`
            - Output: `[*, T']` with `T' = int(new_sr * T / old_sr)


        .. caution::
            After dividing `old_sr` and `new_sr` by their GCD, both should be small
            for this implementation to be fast.

        >>> import torch
        >>> resample = ResampleFrac(4, 5)
        >>> x = torch.randn(1000)
        >>> print(len(resample(x)))
        1250
        """
        super().__init__()
        if not isinstance(old_sr, int) or not isinstance(new_sr, int):
            raise ValueError("old_sr and new_sr should be integers")
        gcd = math.gcd(old_sr, new_sr)
        self.old_sr = old_sr // gcd
        self.new_sr = new_sr // gcd
        self.zeros = zeros
        self.rolloff = rolloff

        self._init_kernels()

    def _init_kernels(self):
        if self.old_sr == self.new_sr:
            return

        kernels = []
        sr = min(self.new_sr, self.old_sr)
        # rolloff will perform antialiasing filtering by removing the highest frequencies.
        # At first I thought I only needed this when downsampling, but when upsampling
        # you will get edge artifacts without this, the edge is equivalent to zero padding,
        # which will add high freq artifacts.
        sr *= self.rolloff

        # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
        # using the sinc interpolation formula:
        #   x(t) = sum_i x[i] sinc(pi * old_sr * (i / old_sr - t))
        # We can then sample the function x(t) with a different sample rate:
        #    y[j] = x(j / new_sr)
        # or,
        #    y[j] = sum_i x[i] sinc(pi * old_sr * (i / old_sr - j / new_sr))

        # We see here that y[j] is the convolution of x[i] with a specific filter, for which
        # we take an FIR approximation, stopping when we see at least `zeros` zeros crossing.
        # But y[j+1] is going to have a different set of weights and so on, until y[j + new_sr].
        # Indeed:
        # y[j + new_sr] = sum_i x[i] sinc(pi * old_sr * ((i / old_sr - (j + new_sr) / new_sr))
        #               = sum_i x[i] sinc(pi * old_sr * ((i - old_sr) / old_sr - j / new_sr))
        #               = sum_i x[i + old_sr] sinc(pi * old_sr * (i / old_sr - j / new_sr))
        # so y[j+new_sr] uses the same filter as y[j], but on a shifted version of x by `old_sr`.
        # This will explain the F.conv1d after, with a stride of old_sr.
        self._width = math.ceil(self.zeros * self.old_sr / sr)
        # If old_sr is still big after GCD reduction, most filters will be very unbalanced, i.e.,
        # they will have a lot of almost zero values to the left or to the right...
        # There is probably a way to evaluate those filters more efficiently, but this is kept for
        # future work.
        idx = torch.arange(-self._width, self._width + self.old_sr).float()
        for i in range(self.new_sr):
            t = (-i/self.new_sr + idx/self.old_sr) * sr
            t = t.clamp_(-self.zeros, self.zeros)
            t *= math.pi
            window = torch.cos(t/self.zeros/2)**2
            kernel = torch.sinc(t / math.pi) * window
            # Renormalize kernel to ensure a constant signal is preserved.
            kernel.div_(kernel.sum())
            kernels.append(kernel)

        self.register_buffer("kernel", torch.stack(
            kernels).view(self.new_sr, 1, -1))

    def forward(self, x: torch.Tensor):
        if self.old_sr == self.new_sr:
            return x
        shape = x.shape
        length = x.shape[-1]
        x = x.reshape(-1, length)

        block_length = self.kernel.shape[-1] * self.old_sr * 2
        if length < block_length:
            x = F.pad(x[:, None], (self._width, self._width +
                                   self.old_sr), mode='replicate')
            ys = _custom_fft_conv1d(x, self.kernel, stride=self.old_sr)
        else:
            x = _pad_to_block(x[:, None], block_length, (self._width, self._width +
                                                         self.old_sr))
            num_blocks = x.shape[-2]
            x = x.reshape(-1, 1, x.shape[-1])
            ys = _custom_fft_conv1d(x, self.kernel, stride=self.old_sr)
            ys = ys.view(-1, num_blocks, self.new_sr, ys.shape[-1]).transpose(
                1, 2).reshape(-1, self.new_sr, num_blocks * ys.shape[-1])
        y = ys.transpose(1, 2).reshape(list(shape[:-1]) + [-1])
        return y[..., :int(self.new_sr * length / self.old_sr)]
