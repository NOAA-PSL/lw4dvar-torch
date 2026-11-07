"""Smoke test of the NGCM JAX backend behind the torch interface.

1. The bridged per-step rollout (2-entry forcing windows, time rebuilt from
   the date) matches a plain JAX rollout with the full forcing window.
2. Hydrostatically derived z500 / t850 vs ERA5 (the store's pressure.zarr)
   at the initial time and after the rollout.
3. The latent-increment gradient of a ps-based loss through `steps` steps
   matches a finite difference along a random direction.
4. Per-step forward / forward+backward timing (and no per-step retracing).

Run on a GPU node (run_ngcm.sh sets up the env):
  sbatch run_ngcm.sh backends/ngcm/smoke_test_ngcm.py EXP_DIR STEP DATA_DIR [T0] [STEPS]
"""

import datetime
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ngcm_model  # noqa: E402

import coordax as cx  # noqa: E402
import jax  # noqa: E402
import jax_datetime as jdt  # noqa: E402

NGCM_TRAIN = os.environ.get("NGCM_TRAIN", "/scratch4/BMC/gsienkf/whitaker/python/ngcm-train")


DEVICE = os.environ.get("NGCM_DEVICE", "cuda")


def rms(x):
    return float(torch.sqrt(torch.mean(x ** 2)))


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def main():
    exp, step, data = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    t0 = datetime.datetime.strptime(sys.argv[4] if len(sys.argv) > 4 else "2021-01-01T00", "%Y-%m-%dT%H")
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    model = ngcm_model.NGCMModel(exp, step, data, NGCM_TRAIN, device=DEVICE)
    print(f"restored {exp} step {step}; latent {model.latent_shape}; {model.n_points} points; t0 {t0}")

    state0 = model.prepare_initial_state(model.read_inputs(t0), t0)
    # magnitude of each packed family (modal: over l >= 1), for judging the
    # state_scales preconditioning.
    x = state0.state[0]
    for family, key in ngcm_model.CHANNEL_FAMILIES.items():
        cols = model.resolve_columns(family)
        f = x[cols]
        f = f[..., 1:] if key not in ngcm_model.TRACER_KEYS else f[..., :-1]
        scale = float(model._channel_scale[cols[0]])
        print(f"  {family:26s} rms {rms(f):.3e}  max {f.abs().max().item():.3e}  increment scale {scale:g}")
    w = torch.as_tensor(model.area_weights, device=model.device)

    # 2a. derivation accuracy at t0 (ERA5 hybrid -> pressure vs ERA5 pressure).
    def verify(state, label):
        d = model.decode_state(state)
        truth = model._pressure.sel(time=ngcm_model._np_datetime(state.date))
        for var, name, lev, unit, scale in (("geopotential", "z", 500, "m", 1 / ngcm_model.GRAV),
                                            ("temperature", "t", 850, "K", 1.0)):
            i = int(np.argmin(np.abs(model._plevs - lev)))
            y = torch.as_tensor(truth[var].sel(pressure=lev).transpose("latitude", "longitude").values.reshape(-1),
                                dtype=torch.float32, device=model.device)
            e = (d[name][i] - y) * scale
            bias = float((w * e).mean())
            print(f"  {label}: {name}{lev} vs ERA5 rms {float(torch.sqrt((w * e ** 2).mean())):.2f} {unit}, "
                  f"bias {bias:+.2f} {unit}")
        return d

    verify(state0, "t0 (hybrid->pressure derivation)")

    # 1. bridged rollout vs plain JAX with the full forcing window.
    with torch.no_grad():
        t = time.time()
        s = model.advance(state0, steps=1)
        sync()
        print(f"first step (incl. compile) {time.time() - t:.1f} s")
        t = time.time()
        s = model.advance(s, steps=steps - 1)
        sync()
        print(f"{steps - 1} more steps {time.time() - t:.2f} s ({(time.time() - t) / max(steps - 1, 1):.3f} s/step)")
    verify(s, f"+{steps * 6} h")

    end = ngcm_model._np_datetime(t0) + steps * np.timedelta64(model.timestep)
    window = model._surface[["sea_surface_temperature", "sea_ice_cover"]].sel(
        time=slice(ngcm_model._np_datetime(t0), end)).load()
    td = model._coordinates.TimeDelta((window.time.values - window.time.values[0]).astype("timedelta64[s]"))
    key = next(iter(model._dynamic_spec))
    dynamic = {key: {k: cx.field(window[k].transpose("time", "longitude", "latitude").values, td, model._grid)
                     for k in window}}
    dynamic[key]["time"] = cx.field(jdt.to_datetime(window.time.values), td)
    ref = model._assimilate(model.read_inputs(t0), dynamic)
    advance = jax.jit(model._inference.advance)
    for _ in range(steps):
        ref = advance(ref, dynamic)
    ref_packed = torch.from_dlpack(model._pack_jit(ref))
    diff = (s.state[0] - ref_packed).abs().max().item()
    print(f"bridged per-step rollout vs plain JAX (full window): max |diff| {diff:.3g} "
          f"(state max {ref_packed.abs().max().item():.3g})")

    # 3. latent gradient vs finite differences, per family. The loss is a
    # fixed smooth random weighting of the ps change from the background
    # (linear near 0, so float32 rounding of the forecast averages out),
    # and each family is perturbed along a smooth random direction (modal:
    # l <= 21; tracers: smoothed in longitude) of physical rms AMPLITUDE,
    # then 10x and 100x smaller.
    AMPLITUDE = {"vorticity": 1e-6, "divergence": 3e-7, "air_temperature": 0.1,
                 "log_surface_pressure": 1e-4, "hybrid_specific_humidity": 1e-5,
                 "specific_total_condensate": 1e-6}
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        bg = model.advance(state0, steps=steps)
        sp_bg = model.decode_state(bg, only=["sp"])["sp"]
    r = torch.randn(model.n_points, generator=gen).to(model.device).reshape(model._nlat, model._nlon)
    r = torch.nn.functional.avg_pool2d(r[None, None], 9, 1, 4, count_include_pad=False)[0, 0].reshape(-1)
    r = r / torch.sqrt((w * r ** 2).mean())

    def loss_fn(inc):
        st = model.advance(state0, steps=steps, latent_increment=inc)
        sp = model.decode_state(st, only=["sp"])["sp"]
        return (w * (sp - sp_bg) * r).mean() / 100.0

    inc = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
    t = time.time()
    loss = loss_fn(inc)
    loss.backward()
    sync()
    print(f"forward+backward {steps} steps (incl. backward compile) {time.time() - t:.1f} s; loss at 0 {loss.item():.3g}")
    t = time.time()
    inc2 = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
    loss_fn(inc2).backward()
    sync()
    print(f"forward+backward {steps} steps {time.time() - t:.2f} s")
    g = inc.grad
    print(f"grad finite: {bool(torch.isfinite(g).all())}, |grad| {g.norm().item():.4g}")
    ls = torch.arange(model.latent_shape[2], device=model.device)
    with torch.no_grad():
        for family, key in ngcm_model.CHANNEL_FAMILIES.items():
            cols = model.resolve_columns(family)
            d = torch.zeros(model.latent_shape)
            d[cols] = torch.randn(len(cols), *model.latent_shape[1:], generator=gen)
            d = d.to(model.device)
            if key in ngcm_model.TRACER_KEYS:
                d[cols] = torch.nn.functional.avg_pool1d(
                    d[cols].transpose(1, 2).reshape(-1, 1, model.latent_shape[1]), 15, 1, 7,
                    count_include_pad=False).reshape(len(cols), -1, model.latent_shape[1]).transpose(1, 2)
                d[cols, :, -1] = 0.0
            else:
                d[cols] = d[cols] * (ls <= 21)
            # control units -> physical: x += inc * scale; set the physical rms.
            phys = d[cols] * model._channel_scale[cols]
            d = d * (AMPLITUDE[family] / float(torch.sqrt((phys[phys != 0] ** 2).mean())))
            analytic = float((g * d).sum())
            line = f"  {family:26s} analytic {analytic:+.5g}; FD"
            for eps in (1.0, 0.1, 0.01):
                fd = (loss_fn(eps * d) - loss_fn(-eps * d)).item() / (2 * eps)
                line += f" x{eps:g}: {fd:+.5g}"
            print(line)

if __name__ == "__main__":
    main()
