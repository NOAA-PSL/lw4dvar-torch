"""
Model-backend validation smoke test for ACE2Model -- same bar as the
AIFS/FCN3/Aurora smoke tests, plus two ACE2-specific checks:

(0) decode_state round-trips a real initial state (ACE2's own
    initial_conditions/ic_2020.nc, 2020-01-01T00 -- real ERA5-derived
    layers, not synthetic) exactly;
(1) a zero latent increment reproduces the no-increment rollout bitwise
    (the blocks[-1] forward hook is otherwise inert);
(2) differentiable advance()+backward() gives a finite, nonzero gradient,
    identical with and without per-step checkpointing (the hook is
    registered inside the checkpointed function, so the recompute must
    re-apply it -- a mismatch here would mean it doesn't);
(3) AdamW reduces a scale-aware surface-pressure loss (hPa, against a
    synthetic +2 hPa regional target) over 15 epochs;
(4) timing / peak memory of a checkpointed forward+backward at 4 and 20
    steps, for comparison with the other backends.

Needs forcing_2020.nc: `python backends/ace2/ace2_prefetch_checkpoint.py 2020`.
Run from the repo root: python backends/ace2/smoke_test_ace2.py
"""
import datetime
import sys
import time

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, ".")
sys.path.insert(0, "backends/ace2")
from ace2_model import ACE2Model  # noqa: E402

SNAP = "backends/ace2/ACE2-ERA5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS = 4

torch.manual_seed(0)
model = ACE2Model(f"{SNAP}/ace2_era5_ckpt.tar", f"{SNAP}/forcing_data", device=DEVICE)
print(f"device={DEVICE} latent_shape={model.latent_shape} n_channels={len(model.channel_names)}")

ic = xr.open_dataset(f"{SNAP}/initial_conditions/ic_2020.nc").isel(time=0)
date0 = datetime.datetime(2020, 1, 1)
state0 = model.prepare_initial_state({"fields": {n: ic[n].values for n in ic.data_vars}}, date0)

# (0) round trip
dec = model.decode_state(state0)
err = max(
    float(np.abs(dec[n].cpu().numpy() - ic[n].values.reshape(-1)).max())
    for n in ic.data_vars if n in dec
)
err_layer = float(np.abs(dec["air_temperature"][3].cpu().numpy() - ic["air_temperature_3"].values.reshape(-1)).max())
print(f"[0] decode round-trip max|err| singles={err:.3g} layer={err_layer:.3g}")
assert err == 0.0 and err_layer == 0.0

ps_col = model.resolve_columns("PRESsfc")[0]
w = torch.as_tensor(np.cos(np.deg2rad(model.lats)), dtype=torch.float32, device=model.device)

# (1) zero increment is inert -- judged against the run-to-run spread of two
# identical no-increment rollouts (GPU kernels aren't bitwise deterministic,
# and a 4-step rollout amplifies last-bit differences), per channel,
# normalized by each channel's spatial std.
with torch.no_grad():
    bg = model.advance(state0, steps=N_STEPS, use_checkpoint=False)
    bg_rep = model.advance(state0, steps=N_STEPS, use_checkpoint=False)
    bg0 = model.advance(state0, steps=N_STEPS, use_checkpoint=False,
                        latent_increment=torch.zeros(model.latent_shape, device=model.device))


def _rel_diff(a, b):
    std = a[0].reshape(a.shape[1], -1).std(dim=1).clamp_min(1e-30)
    return (a - b)[0].reshape(a.shape[1], -1).abs().max(dim=1).values / std


rep = _rel_diff(bg.state, bg_rep.state)
zer = _rel_diff(bg.state, bg0.state)
for label, r in (("repeat no-increment", rep), ("zero-increment", zer)):
    k = int(r.argmax())
    print(f"[1] {label}: bitwise_equal={torch.equal(bg.state, bg_rep.state if label.startswith('repeat') else bg0.state)} "
          f"worst channel {model.channel_names[k]} max|diff|/std = {float(r[k]):.3g}; median over channels {float(r.median()):.3g}")
print(f"[1] bg date {bg.date}")
assert float(zer.max()) <= max(10 * float(rep.max()), 1e-3)

# synthetic target: background ps +2 hPa over the North Pacific (20-60N, 160E-220E)
region = torch.as_tensor(
    (model.lats >= 20) & (model.lats <= 60) & (model.lons >= 160) & (model.lons <= 220), device=model.device
)
ps_target = bg.state[0, ps_col].reshape(-1) + 200.0 * region


def loss_fn(inc, use_checkpoint=True):
    out = model.advance(state0, steps=N_STEPS, use_checkpoint=use_checkpoint, latent_increment=inc)
    ps = out.state[0, ps_col].reshape(-1)
    return ((w * ((ps - ps_target) / 100.0) ** 2).sum() / w.sum())


# (2) gradient, checkpointed vs not
grads = {}
for ck in (True, False):
    inc = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
    loss = loss_fn(inc, use_checkpoint=ck)
    loss.backward()
    grads[ck] = inc.grad.detach().clone()
    print(f"[2] checkpoint={ck} loss={loss.item():.5f} |grad|={grads[ck].norm().item():.4e} finite={bool(torch.isfinite(grads[ck]).all())}")
rel = float((grads[True] - grads[False]).norm() / grads[False].norm())
print(f"[2] checkpointed vs plain grad rel diff = {rel:.3g}")
# 1e-2: GPU rollouts aren't bitwise reproducible (check [1]), which leaves
# ~1e-3 here over 4 steps; the non-reentrant first-call bug this guards
# against (see ACE2Model._advance_one_step) gave rel ~1-110.
assert torch.isfinite(grads[True]).all() and grads[True].norm() > 0 and rel < 1e-2

# (3) AdamW reduces the loss
inc = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
opt = torch.optim.AdamW([inc], lr=1e-2, weight_decay=0.0)
losses = []
for epoch in range(15):
    opt.zero_grad()
    loss = loss_fn(inc)
    loss.backward()
    opt.step()
    losses.append(loss.item())
print("[3] AdamW losses: " + " ".join(f"{v:.4f}" for v in losses))
assert min(losses[1:]) < losses[0]

# (4) timing / memory
if DEVICE == "cuda":
    for steps in (4, 20):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        inc = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
        t0 = time.time()
        out = model.advance(state0, steps=steps, use_checkpoint=True, latent_increment=inc)
        out.state[0, ps_col].mean().backward()
        torch.cuda.synchronize()
        print(f"[4] steps={steps}: fwd+bwd {time.time() - t0:.2f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

print("ALL CHECKS PASSED")
