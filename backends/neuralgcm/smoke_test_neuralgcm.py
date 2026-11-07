"""Smoke test of the official-NeuralGCM JAX backend behind the torch interface.

1. The bridged per-step rollout (2-entry forcing windows) matches the plain
   JAX `model.advance` loop with the whole window's forcings.
2. z500 vs ERA5 (the input files) at t0 and after the rollout.
3. Latent gradient vs finite differences per control family, along smooth
   random directions (l <= 21) of 5 % of each family's background rms, for
   a loss linear in the z500 change.
4. Timing.
Run on a GPU node:
  sbatch -o LOG -e LOG run_ngcm.sh backends/neuralgcm/smoke_test_neuralgcm.py CKPT INPUT_DIR [T0] [HOURS]
"""

import datetime
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neuralgcm_model  # noqa: E402

import jax  # noqa: E402
import xarray as xr  # noqa: E402

NGCM_TRAIN = os.environ.get("NGCM_TRAIN", "/scratch4/BMC/gsienkf/whitaker/python/ngcm-train")


def main():
    ckpt, input_dir = sys.argv[1], sys.argv[2]
    t0 = datetime.datetime.strptime(sys.argv[3] if len(sys.argv) > 3 else "2015-01-02T00", "%Y-%m-%dT%H")
    hours = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    model = neuralgcm_model.NeuralGCMModel(ckpt, input_dir, ngcm_train_path=NGCM_TRAIN)
    state0 = model.prepare_initial_state(model.read_inputs(t0), t0)
    print(f"{ckpt}; step {model.timestep}; packed {tuple(state0.state.shape)}; latent {model.latent_shape}")
    print("  families:", {f: (c[0], len(c)) for f, c in model._family_channels.items()})
    w = torch.as_tensor(model.area_weights, device=model.device)

    def z500_err(state, label):
        d = model.decode_state(state)
        truth = model.read_verif(state.date)["z500"]
        e = (d["z500"] - truth) / neuralgcm_model.GRAV
        print(f"  {label}: z500 vs ERA5 rms {float(torch.sqrt((w * e ** 2).mean())):.2f} m, "
              f"bias {float((w * e).mean()):+.2f} m")
        return d

    z500_err(state0, "t0 (encode/decode round trip)")
    with torch.no_grad():
        t = time.time()
        s = model.advance(state0, steps=1)
        torch.cuda.synchronize()
        print(f"first step (incl. compile) {time.time() - t:.1f} s")
        t = time.time()
        s = model.advance(s, steps=hours - 1)
        torch.cuda.synchronize()
        print(f"{hours - 1} more steps {(time.time() - t) / (hours - 1):.4f} s/step")
    z500_err(s, f"+{hours} h")

    # 1. NeuralGCM's own unroll over the whole window's forcings.
    t0n = neuralgcm_model._np_datetime(t0).astype("datetime64[h]")
    files = [model._file(t0n + np.timedelta64(h, "h")) for h in range(0, hours + 7, 6)]
    all_forcings = model.model.forcings_from_xarray(xr.concat(files, "time"))
    inputs, forcings = model.read_inputs(t0)
    ref = model.model.encode(inputs, forcings)
    ref, _ = model.model.unroll(ref, all_forcings, steps=hours, timedelta=np.timedelta64(1, "h"))
    ref_packed = torch.from_dlpack(model._pack_jit(ref))
    diff = (s.state[0] - ref_packed).abs()
    n = model._n_control
    print(f"bridged vs plain JAX (all forcings): max |diff| state {diff[:n].max().item():.3g}, "
          f"rest {diff[n:].max().item():.3g} (state max {ref_packed[:n].abs().max().item():.3g})")

    # 3. gradient checks.
    with torch.no_grad():
        z_bg = model.decode_state(model.advance(state0, steps=hours), only=["z500"])["z500"]
    gen = torch.Generator().manual_seed(0)
    r = torch.randn(model.n_points, generator=gen).to(model.device).reshape(model._nlat, model._nlon)
    r = torch.nn.functional.avg_pool2d(r[None, None], 9, 1, 4, count_include_pad=False)[0, 0].reshape(-1)
    r = r / torch.sqrt((w * r ** 2).mean())

    def loss_fn(inc):
        st = model.advance(state0, steps=hours, latent_increment=inc)
        z = model.decode_state(st, only=["z500"])["z500"]
        return (w * (z - z_bg) * r).mean() / neuralgcm_model.GRAV

    inc = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
    t = time.time()
    loss_fn(inc).backward()
    torch.cuda.synchronize()
    print(f"forward+backward {hours} steps (incl. compile) {time.time() - t:.1f} s")
    inc2 = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
    t = time.time()
    loss_fn(inc2).backward()
    torch.cuda.synchronize()
    print(f"forward+backward {hours} steps {time.time() - t:.2f} s")
    g = inc.grad
    print(f"grad finite: {bool(torch.isfinite(g).all())}")
    ls = torch.arange(65, device=model.device)
    x0 = state0.state[0, :n]
    with torch.no_grad():
        for family, cols in model._family_channels.items():
            bg = x0[cols][..., 1:]
            amplitude = 0.05 * float(torch.sqrt((bg ** 2).mean()))
            d = torch.zeros(model.latent_shape)
            d[cols] = torch.randn(len(cols), *model.latent_shape[1:], generator=gen)
            d = d.to(model.device)
            d[cols] = d[cols] * (ls <= 21)
            d[cols, 127:] = 0.0
            phys = d[cols] * model._channel_scale[cols]
            d = d * (amplitude / float(torch.sqrt((phys[phys != 0] ** 2).mean())))
            analytic = float((g * d).sum())
            line = f"  {family:26s} (amp {amplitude:.2e}) analytic {analytic:+.5g}; FD"
            for eps in (1.0, 0.1, 0.01):
                fd = (loss_fn(eps * d) - loss_fn(-eps * d)).item() / (2 * eps)
                line += f" x{eps:g}: {fd:+.5g}"
            print(line)


if __name__ == "__main__":
    main()
