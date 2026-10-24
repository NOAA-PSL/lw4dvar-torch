"""Build the sample file for the static-ensemble latent B (latent_b.EnsembleB).

Samples are latent background errors at the increment's injection point
(output of the SFNO's last block, one step from the window start):
e_k = z(ERA5) - z(cycled background), for every `--stride`-th cycle of a cycled
run after `--skip` spin-up cycles (stride 5 with 6-hourly cycles rotates
through 00/06/12/18Z and decorrelates consecutive samples). The sample mean is
removed; samples are stored fp16, shape (N, C, H, W), with metadata
identifying the checkpoint (sha256) so a run with a different checkpoint
refuses the file. Localization is NOT applied here -- it is a runtime
parameter of EnsembleB.

Run on a GPU node in the ace2 env, e.g. via
  sbatch ... --wrap "python -u diagnostics/build_latent_ens_b.py --run_dir ... --path_model ... --out ..."
"""

import argparse
import datetime
import glob
import hashlib
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_latent_b import LatentProbe, era5_state, utils, ace2_ic  # noqa: E402


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--suffix', default='120h_50it')
    ap.add_argument('--path_model', required=True)
    ap.add_argument('--model_name', default='ace2_era5_ckpt.tar')
    ap.add_argument('--dataset_dir', default='/scratch4/BMC/gsienkf/whitaker/ace2_era5_1deg_8layer')
    ap.add_argument('--skip', type=int, default=40)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    path_model = os.path.join(args.path_model, '')
    model = utils.get_model({'model_backend': 'ace2', 'path_model': path_model, 'device': 'cuda'})
    dev = model.device
    hgt = ace2_ic.load_hgtsfc(path_model + 'forcing_data')
    probe = LatentProbe(model)
    C, H, W = model.latent_shape
    files = sorted(glob.glob(os.path.join(args.run_dir, f'*_control_inputs_{args.suffix}.pkl')))[args.skip::args.stride]
    N = len(files)
    print(f"{N} samples from {args.run_dir} (skip {args.skip}, stride {args.stride})", flush=True)
    samples = torch.empty((N, C, H, W), dtype=torch.float16)
    mean = torch.zeros((C, H, W), dtype=torch.float64, device=dev)
    dates = []
    for k, f in enumerate(files):
        with open(f, 'rb') as fh:
            s_b = pickle.load(fh)
        s_b.state = s_b.state.to(dev)
        s_t = era5_state(model, args.dataset_dir, s_b.date, hgt)
        e = probe(s_t) - probe(s_b)
        mean += e
        samples[k] = e.to(torch.float16).cpu()
        dates.append(f"{s_b.date:%Y-%m-%dT%H}")
        if k % 20 == 0:
            print(f"{k:4d} {dates[-1]} rms {e.pow(2).mean().sqrt().item():.4f}", flush=True)
    mean = (mean / N).to(torch.float16).cpu()
    for k in range(N):
        samples[k] -= mean
    ckpt = os.path.join(path_model, args.model_name)
    meta = {'backend': 'ace2', 'checkpoint': os.path.abspath(ckpt), 'checkpoint_sha256': sha256(ckpt),
            'injection_point': 'output of _sfno.blocks[-1], first step', 'latent_shape': (C, H, W),
            'sht_grid': model._sfno.itrans_up.grid, 'n_samples': N, 'dates': dates,
            'run_dir': os.path.abspath(args.run_dir), 'skip': args.skip, 'stride': args.stride,
            'dataset_dir': args.dataset_dir, 'mean_removed': True, 'dtype': 'float16',
            'created': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'definition': 'e_k = z(ERA5) - z(cycled background) at window start; mean removed; '
                          'B = (1/(N-1)) sum_k e_k e_k^T o L (localization applied at run time)'}
    torch.save({'samples': samples, 'mean': mean, 'meta': meta}, args.out)
    print(f"wrote {args.out}: {tuple(samples.shape)} fp16, {samples.numel() * 2 / 1e9:.1f} GB; "
          f"sample rms {samples[:10].float().pow(2).mean().sqrt().item():.4f}")


if __name__ == '__main__':
    main()
