import torch
from kazane.resample import Resample
from julius import ResampleFrac
from torch.profiler import profiler
from torch.utils.benchmark import Timer, Compare
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    device = args.device

    batch = 4
    sr = 44100
    pairs = [(44100, 48000), (48000, 44100), (22050, 44100), (44100, 22050)]
    duration = 2
    zeros = 24

    num_threads = torch.get_num_threads()
    print(f'Benchmarking on {num_threads} threads')

    results = []
    for old_sr, new_sr in pairs:
        label = 'Resample'
        sub_label = f'ratio: {old_sr, new_sr}'
        x = torch.randn(batch, int(old_sr * duration), device=device)

        results.append(Timer(
            stmt='m(x)',
            setup='',
            globals={'x': x, 'm': ResampleFrac(
                old_sr, new_sr, zeros).to(device)},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='julius',
        ).blocked_autorange(min_run_time=1))
        results.append(Timer(
            stmt='m(x)',
            setup='',
            globals={'x': x, 'm': Resample(old_sr, new_sr, zeros).to(device)},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='kazane',
        ).blocked_autorange(min_run_time=1))

    compare = Compare(results)
    compare.print()
