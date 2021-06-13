import torch


def sinc_kernel(num_zeros, precision, roll_off=0.945, dtype=torch.float64):
    t = torch.arange(-num_zeros * precision, num_zeros *
                     precision + 1, dtype=dtype) / precision * roll_off
    return torch.sinc(t) * roll_off


if __name__ == '__main__':
    print(sinc_kernel(16, 2))
