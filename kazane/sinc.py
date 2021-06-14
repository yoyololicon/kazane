import torch


def sinc_kernel(num_zeros, precision, roll_off=0.945, dtype=torch.float64):
    """
    Sinc interpolation kernel.

    Args:
        num_zeros (int): number of zero crossing to keep in the sinc filter
        precision (int): number of filter coefficients to retain for each zero-crossing
        roll_off (float): the roll-off frequency (as a fraction of nyquist). Default: ``0.945``
        dtype (torch.dtype): Default: ``torch.float64``
    """
    t = torch.arange(-num_zeros * precision, num_zeros *
                     precision + 1, dtype=dtype) / precision * roll_off
    return torch.sinc(t) * roll_off


if __name__ == '__main__':
    print(sinc_kernel(16, 2))
