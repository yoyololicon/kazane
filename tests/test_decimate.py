from kazane import Decimate
import torch
import pytest
from math import pi, log2

device = 'cpu'
dtype = torch.float64


def make_tone(freq, sr, duration):
    return torch.sin(2 * pi * freq / sr * torch.arange(int(sr * duration), dtype=dtype))


def make_sweep(freq, sr, duration):
    return torch.sin(torch.cumsum(2 * pi * torch.logspace(log2(2.0 / sr),
                                                          log2(
                                                              float(freq) / sr),
                                                          steps=int(duration*sr), base=2.0, dtype=dtype), dim=-1))


@pytest.mark.parametrize('q', [2, 3, 5, 7])
@pytest.mark.parametrize('zeros,rms', [
    (64, 2e-4),
    (32, 2e-4),
    (16, 4e-4)
])
def test_quality_sine(q, zeros, rms):
    FREQ = 512.0
    DURATION = 2
    sr = 44100
    sr_new = sr // q
    dec = Decimate(q, zeros, roll_off=0.945).to(dtype)

    x = make_tone(FREQ, sr, DURATION)
    y = make_tone(FREQ, sr_new, DURATION)
    y_pred = dec(x)

    idx = slice(sr_new // 2, - sr_new // 2)

    err = torch.mean(torch.abs(y[idx] - y_pred[idx])).item()
    assert err <= rms, '{:g} > {:g}'.format(err, rms)


@pytest.mark.parametrize('q', [2, 3])
@pytest.mark.parametrize('zeros,rms', [
    (64, 8e-2),
    (32, 8e-2),
    (16, 8e-2)
])
def test_quality_sweep(q, zeros, rms):
    FREQ = 4096
    DURATION = 5
    sr = 44100
    sr_new = sr // q
    dec = Decimate(q, zeros, roll_off=0.945).to(dtype)

    x = make_sweep(FREQ, sr, DURATION)
    y = make_sweep(FREQ, sr_new, DURATION)
    y_pred = dec(x)

    idx = slice(sr_new // 2, - sr_new // 2)

    err = torch.mean(torch.abs(y[idx] - y_pred[idx])).item()
    assert err <= rms, '{:g} > {:g}'.format(err, rms)
