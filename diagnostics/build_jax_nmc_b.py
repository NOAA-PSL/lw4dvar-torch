"""Build NMC-method samples for the static-ensemble B (latent_b.EnsembleB,
`jb_samples`) of the spectral-state JAX backends (ngcm, neuralgcm).

For each valid time t (every `--stride` hours from `--start` to `--end`):

    e_t = x(t | ERA5 at t - long) - x(t | ERA5 at t - short)

the difference of two forecasts from ERA5 (default 36 h and 12 h) valid at
the same time (Parrish & Derber 1992), in latent units (the packed-state
difference divided by the state_scales factor, `latent_from_state_diff`;
frozen channels are 0), mapped to the Gaussian grid (`latent_to_grid`:
modal families through the model's own SHT) because EnsembleB localizes in
grid space. The sample mean is removed; samples are stored fp16, shape
(N, C, H, W), latitudes north to south, with metadata identifying the
checkpoint (`model.checkpoint_id()`), checked when the file is loaded.
Localization (`jb_loc_km`) and the overall amplitude (`jb_weight`) are
run-time parameters.

36 h - 12 h rather than the classic 12 h - 6 h: for NeuralGCM-2.8 the 12-6
differences are dominated by fast, unbalanced error (single-ob increments
with half-geostrophic winds that lose ~60% of their ps within 6 h), while
the 36-12 ones are geostrophically balanced (wind vs geostrophic wind
correlation 0.9-0.98), ~2.5x larger in variance, and retained by the model
(check_jax_nmc_b.py). A 24 h lead difference also avoids aliasing the
diurnal cycle.

The default 18 h stride rotates the valid time through 00/18/12/06 UTC, so
the diurnal cycle is sampled evenly. Use a period outside the DA test period.

Run on a GPU node in the ngcm-torch env, e.g.
  sbatch -t 1:00:00 -J nmc_b -o logs/nmc_b_%j.out run_ngcm.sh \\
      diagnostics/build_jax_nmc_b.py --config config_test_ngcm_t0inc.yml \\
      --start 2019-12-01T00 --end 2020-03-31T18 --out /path/ngcm_nmc_b.pt
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import long_window_4dvar_utils as utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='a solver config whose exp section builds the model')
    ap.add_argument('--start', required=True, help='first valid time, YYYY-MM-DDTHH')
    ap.add_argument('--end', required=True, help='last valid time, YYYY-MM-DDTHH')
    ap.add_argument('--stride', type=int, default=18, help='hours between valid times')
    ap.add_argument('--short', type=int, default=12, help='short lead (h)')
    ap.add_argument('--long', type=int, default=36, help='long lead (h)')
    ap.add_argument('--neuralgcm_input_dir', default=None,
                    help='override exp.neuralgcm_input_dir (ERA5 files covering the sample period)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        exp = yaml.safe_load(f)['exp']
    if args.neuralgcm_input_dir:
        exp['neuralgcm_input_dir'] = args.neuralgcm_input_dir
    backend = exp['model_backend']
    if backend not in ('ngcm', 'neuralgcm'):
        raise ValueError(f"build_jax_nmc_b.py is for the ngcm / neuralgcm backends, got {backend!r}")
    model = utils.get_model(exp)
    dt_h = int(round(model.timestep.total_seconds() / 3600))
    for lead in (args.short, args.long):
        if lead % dt_h:
            raise ValueError(f"lead {lead} h is not a multiple of the {dt_h} h model step")

    def forecast(valid, lead):
        init = valid - datetime.timedelta(hours=lead)
        s = model.prepare_initial_state(model.read_inputs(init), init)
        return model.advance(s, steps=lead // dt_h, use_checkpoint=False)

    t0 = datetime.datetime.strptime(args.start, '%Y-%m-%dT%H')
    t1 = datetime.datetime.strptime(args.end, '%Y-%m-%dT%H')
    times = []
    t = t0
    while t <= t1:
        times.append(t)
        t += datetime.timedelta(hours=args.stride)
    N = len(times)
    samples, dates = None, []
    with torch.no_grad():
        for k, valid in enumerate(times):
            xl, xs = forecast(valid, args.long), forecast(valid, args.short)
            assert xl.date == xs.date == valid, (xl.date, xs.date, valid)
            lat = model.latent_from_state_diff(xl.state[0] - xs.state[0])
            g = model.latent_to_grid(lat)
            if samples is None:
                # the solver maps B's grid output back with latent_from_grid:
                # check it inverts latent_to_grid on a real sample
                back = model.latent_from_grid(g)
                err = ((back - lat).norm() / lat.norm()).item()
                print(f"latent -> grid -> latent relative error {err:.2e}", flush=True)
                if not err < 1e-3:
                    raise ValueError(f"latent_from_grid does not invert latent_to_grid (rel err {err:.2e})")
                C, H, W = g.shape
                samples = torch.empty((N, C, H, W), dtype=torch.float32)
                print(f"{N} NMC samples ({args.long} h - {args.short} h) valid {times[0]:%Y-%m-%dT%H} .. "
                      f"{times[-1]:%Y-%m-%dT%H} every {args.stride} h; grid sample shape {(C, H, W)}, "
                      f"latent {tuple(model.latent_shape)}", flush=True)
            samples[k] = g.float().cpu()
            dates.append(f"{valid:%Y-%m-%dT%H}")
            if k % 20 == 0:
                print(f"{k:4d} {dates[-1]} rms {g.pow(2).mean().sqrt().item():.4g}", flush=True)
    mean = samples.mean(0)
    samples = (samples - mean).to(torch.float16)
    if not torch.isfinite(samples).all():
        raise ValueError("non-finite values in the fp16 samples (latent units overflow fp16?)")
    meta = {'backend': backend, 'checkpoint_id': model.checkpoint_id(), 'latent_shape': tuple(model.latent_shape),
            'sht_grid': model.jb_sht_grid, 'n_samples': N, 'dates': dates,
            'method': f"NMC {args.long} h - {args.short} h forecasts from ERA5",
            'short_h': args.short, 'long_h': args.long, 'stride_h': args.stride,
            'state_scales': exp.get(f'{backend}_state_scales'), 'mean_removed': True, 'dtype': 'float16',
            'created': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'definition': 'e_t = x(t | t - long) - x(t | t - short) in latent units on the Gaussian grid '
                          '(lat north-to-south); mean removed; B = (1/(N-1)) sum e e^T o L (L at run time)'}
    torch.save({'samples': samples, 'mean': mean.to(torch.float16), 'meta': meta}, args.out)
    rms = samples.float().pow(2).mean(dim=(0, 2, 3)).sqrt().numpy()
    print(f"wrote {args.out}: {tuple(samples.shape)} fp16, {samples.numel() * 2 / 1e9:.2f} GB")
    np.set_printoptions(precision=3, linewidth=150)
    print("per-channel sample rms (latent units):", rms)


if __name__ == '__main__':
    main()
