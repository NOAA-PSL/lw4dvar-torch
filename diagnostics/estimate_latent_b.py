"""Estimate a spectral-diagonal latent-space background-error covariance for
ACE2 from a cycled lw4dvar run.

Samples: for every cycle in `--run_dir`, the background at window start
(`<date>_control_inputs_*.pkl` -- the window-start state is never modified by
the latent increment, so this is the cycled background) and ERA5 at the same
time, built from our ACE2 training dataset (`--dataset_dir`, monthly
`YYYYMM0100.nc`). PRESsfc there is valid at the dataset's HGTsfc; it is
reduced to the checkpoint's HGTsfc exactly as `ace2_ic.py` does for DA ICs.

Each state is stepped once with the latent captured at the injection point
(the output of the SFNO's last block, where ACE2Model adds the increment).
The error sample is e = z(ERA5) - z(background), shape (C, H, W). Its real
SHT (orthonormal, the SFNO's own grid) gives coefficients c[k, l, m];
accumulated per (channel, l):

    power[k, l] = |c[k,l,0]|^2 + 2 sum_{m>0} |c[k,l,m]|^2   (sum of squares
                  over the sphere, by Parseval)
    sigma2[k, l] = (mean power - power of the mean) / (2l + 1)

i.e. the per-coefficient variance of a homogeneous, isotropic B for each
latent channel, sample mean removed. Also saved: per-sample power (for
sampling noise / seasonality checks), the mean error (bias) spectrum, the
grid-space mean error and variance by (channel, latitude), and per-sample area-weighted h500
rms (background - ERA5) as a pairing sanity check.

`--check_ic YYYY-MM-DDTHH` instead compares the dataset-built ERA5 state with
`ace2_ic.py`'s cached IC for that date (field rms and the latent difference
spectrum -- the pipeline-difference noise floor).

Run on a GPU node in the ace2 env (see run_estimate_latent_b.sh).
"""

import argparse
import datetime
import glob
import os
import pickle
import sys

import numpy as np
import torch
import xarray as xr

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import long_window_4dvar_utils as utils  # noqa: E402

utils._ensure_backend_on_path('ace2')
import ace2_ic  # noqa: E402
import torch_harmonics as th  # noqa: E402


def era5_state(model, dataset_dir, date, hgt_ace2):
    """ACE2State for ERA5 at `date` from the training dataset, PRESsfc reduced
    to the checkpoint's HGTsfc (as ace2_ic.fetch_ic)."""
    path = os.path.join(dataset_dir, f"{date:%Y%m}0100.nc")
    with xr.open_dataset(path) as ds:
        snap = ds.sel(time=np.datetime64(date.replace(tzinfo=None), 'ns'))
        fields = {n: snap[n].values.astype(np.float64) for n in model._channel_names if n in snap}
        h_from = snap['HGTsfc'].values.astype(np.float64)
    tv = fields['air_temperature_7'] * (1.0 + (461.5 / ace2_ic.RDGAS - 1.0) * fields['specific_total_water_7'])
    fields['PRESsfc'] = ace2_ic._reduce_surface_pressure(fields['PRESsfc'], tv, h_from, hgt_ace2)
    return model.prepare_initial_state({'fields': fields}, date)


class LatentProbe:
    """Captures the output of the SFNO's last block (the increment's injection
    point) on a one-step no-grad advance."""

    def __init__(self, model):
        self.model = model
        self.z = None
        model._sfno.blocks[-1].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self.z = out.detach()

    @torch.no_grad()
    def __call__(self, state):
        self.z = None
        self.model.advance(state, steps=1, use_checkpoint=False)
        return self.z[0].double()   # (C, H, W)


def l_power(c):
    """(C, L, M) complex SHT coefficients -> (C, L) power, m>0 counted twice."""
    p = c.real ** 2 + c.imag ** 2
    return p[..., 0] + 2.0 * p[..., 1:].sum(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', help='cycled run output dir with *_control_inputs_*.pkl')
    ap.add_argument('--suffix', default='120h_50it', help='file suffix (default 120h_50it)')
    ap.add_argument('--dataset_dir', default='/scratch4/BMC/gsienkf/whitaker/ace2_era5_1deg_8layer')
    ap.add_argument('--path_model', required=True, help='ACE2-ERA5 dir (checkpoint + forcing_data/)')
    ap.add_argument('--out', help='output .pt')
    ap.add_argument('--stride', type=int, default=1, help='use every stride-th cycle')
    ap.add_argument('--skip', type=int, default=0, help='drop the first SKIP cycles (cycling spin-up)')
    ap.add_argument('--cross', action='store_true', help='also accumulate the per-l channel cross-covariance (L, C, C)')
    ap.add_argument('--check_ic', help='YYYY-MM-DDTHH: compare dataset ERA5 with ace2_ic cache, then exit')
    ap.add_argument('--ic_cache', default=None, help='ace2_ic cache dir (for --check_ic)')
    args = ap.parse_args()

    path_model = os.path.join(args.path_model, '')
    model = utils.get_model({'model_backend': 'ace2', 'path_model': path_model, 'device': 'cuda'})
    hgt_ace2 = ace2_ic.load_hgtsfc(path_model + 'forcing_data')
    probe = LatentProbe(model)
    sfno = model._sfno
    C, H, W = model.latent_shape
    sht = th.RealSHT(H, W, grid=sfno.itrans_up.grid, norm='ortho').to(model.device).double()
    L = sht.lmax
    print(f"latent {C}x{H}x{W}, SHT grid {sht.grid}, lmax {L}, mmax {sht.mmax}", flush=True)
    areaw = torch.as_tensor(model.area_weights, device=model.device, dtype=torch.float64)
    areaw = areaw / areaw.sum()
    h500 = model._chan_idx['h500']

    if args.check_ic:
        date = datetime.datetime.strptime(args.check_ic, '%Y-%m-%dT%H').replace(tzinfo=datetime.timezone.utc)
        s_ds = era5_state(model, args.dataset_dir, date, hgt_ace2)
        s_ic = model.prepare_initial_state(ace2_ic.build_input_state(date, args.ic_cache), date)
        for i, n in enumerate(model._prognostic):
            d = (s_ds.state[0, i] - s_ic.state[0, i]).double().reshape(-1)
            print(f"{n:36s} rms diff {torch.sqrt((areaw * d ** 2).sum()).item():.4g}  "
                  f"(field std {s_ic.state[0, i].double().std().item():.4g})")
        e = probe(s_ds) - probe(s_ic)
        torch.save({'date': args.check_ic, 'power': l_power(sht(e)).cpu()}, args.out)
        print(f"latent diff total power {l_power(sht(e)).sum().item():.4g}; wrote {args.out}")
        return

    files = sorted(glob.glob(os.path.join(args.run_dir, f'*_control_inputs_{args.suffix}.pkl')))[args.skip::args.stride]
    print(f"{len(files)} cycles from {args.run_dir}", flush=True)
    n = 0
    sum_c = torch.zeros((C, L, sht.mmax), dtype=torch.complex128, device=model.device)
    sum_p = torch.zeros((C, L), dtype=torch.float64, device=model.device)
    sum_e = torch.zeros((C, H, W), dtype=torch.float64, device=model.device)
    sum_e2 = torch.zeros((C, H, W), dtype=torch.float64, device=model.device)
    power, dates, h500_rms, z_rms = [], [], [], []
    # Cross-covariance: Re(c_k c_j*) summed over m with weight 1 (m=0), 2 (m>0)
    # -- the same convention as l_power, whose diagonal it reproduces.
    wk = torch.cat([torch.ones(1), 2 * torch.ones(sht.mmax - 1)]).repeat(2).to(model.device, torch.float64)
    def l_cross(c):
        r = torch.cat([c.real, c.imag], -1).permute(1, 0, 2).double()  # (L, C, 2M)
        return torch.bmm(r * wk, r.transpose(1, 2))                  # (L, C, C)
    sum_x = torch.zeros((L, C, C), dtype=torch.float64, device=model.device) if args.cross else None
    for f in files:
        with open(f, 'rb') as fh:
            s_b = pickle.load(fh)
        s_b.state = s_b.state.to(model.device)
        date = s_b.date
        s_t = era5_state(model, args.dataset_dir, date, hgt_ace2)
        dh = (s_b.state[0, h500] - s_t.state[0, h500]).double().reshape(-1)
        e = probe(s_t) - probe(s_b)
        c = sht(e)
        p = l_power(c)
        sum_c += c
        sum_p += p
        if args.cross:
            sum_x += l_cross(c)
        sum_e += e
        sum_e2 += e ** 2
        n += 1
        power.append(p.float().cpu())
        dates.append(f"{date:%Y-%m-%dT%H}")
        h500_rms.append(torch.sqrt((areaw * dh ** 2).sum()).item())
        z_rms.append(torch.sqrt((e ** 2).mean()).item())
        if n % 50 == 1:
            print(f"{n:4d} {dates[-1]}  h500 bg-ERA5 rms {h500_rms[-1]:.2f} m, latent err rms {z_rms[-1]:.4g}", flush=True)

    mean_c = sum_c / n
    sigma2 = (sum_p / n - l_power(mean_c)) / (2.0 * torch.arange(L, device=model.device, dtype=torch.float64) + 1.0)
    out = {
        'sigma2': sigma2.cpu(),                       # (C, L) per-coefficient variance, mean removed
        'mean_power': (sum_p / n).cpu(),              # (C, L) incl. bias
        'bias_power': l_power(mean_c).cpu(),          # (C, L) power of the mean error
        'power_samples': torch.stack(power),          # (N, C, L)
        'grid_mean': (sum_e / n).float().cpu(),       # (C, H, W) mean error (bias)
        'grid_var_lat': (sum_e2 / n - (sum_e / n) ** 2).mean(-1).cpu(),  # (C, H) zonal mean of the variance
        'cross': ((sum_x / n - l_cross(mean_c)) / (2.0 * torch.arange(L, device=model.device, dtype=torch.float64) + 1.0)[:, None, None]).float().cpu() if args.cross else None,  # (L, C, C) per-coefficient channel covariance, mean removed
        'dates': dates, 'h500_rms': h500_rms, 'latent_rms': z_rms,
        'meta': {'backend': 'ace2', 'checkpoint': path_model + 'ace2_era5_ckpt.tar', 'latent_shape': (C, H, W),
                 'sht_grid': sht.grid, 'lmax': L, 'mmax': sht.mmax, 'norm': 'ortho', 'n_samples': n, 'skip': args.skip,
                 'run_dir': os.path.abspath(args.run_dir), 'dataset_dir': args.dataset_dir,
                 'method': 'z(ERA5) - z(cycled background) at window start, output of sfno.blocks[-1]'},
    }
    torch.save(out, args.out)
    print(f"wrote {args.out}: {n} samples; mean h500 rms {np.mean(h500_rms):.2f} m; "
          f"total latent error variance {out['sigma2'].mul(2 * torch.arange(L) + 1.0).sum().item():.4g}")


if __name__ == '__main__':
    main()
