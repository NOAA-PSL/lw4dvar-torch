"""
Model-backend smoke test for SFNOModel (same bar as smoke_test_ace2.py), from a
real ERA5 state (2015-01-01T00, via sfno_ic / the FCN3 ERA5 cache):

(0) decode_state round-trips the initial state exactly;
(1) a zero latent increment is inert (vs the run-to-run spread of two
    identical no-increment rollouts, per channel / spatial std);
(2) the FIRST checkpointed latent gradient after a no-grad rollout matches the
    plain (non-checkpointed) gradient -- the sequence that exposed ACE2's
    non-reentrant-checkpoint bug; plus a second checkpointed call as baseline;
(3) AdamW reduces a scale-aware surface-pressure loss (hPa, synthetic +2 hPa
    North Pacific target) over 10 epochs;
(4) timing / peak memory of a checkpointed forward+backward at 1, 2, 4, 8 steps.

Run from the repo root in the fcstnet3 env (GPU): python backends/sfno/smoke_test_sfno.py
"""
import datetime
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "backends/sfno")
import sfno_ic  # noqa: E402
from sfno_model import SFNOModel  # noqa: E402

CACHE = "/scratch4/BMC/gsienkf/Jeffrey.Whitaker/long-window-4dvar-fcstnetv3/ic_cache/"
N_STEPS = 2
model = SFNOModel(device="cuda")
print(f"latent_shape={model.latent_shape} n_channels={len(model._channel_names)} timestep={model.timestep}")
d0 = datetime.datetime(2015, 1, 1, tzinfo=datetime.timezone.utc)
inp = sfno_ic.build_input_state(d0, CACHE)
state0 = model.prepare_initial_state(inp, d0)

# (0)
dec = model.decode_state(state0)
err = max(float(np.abs(dec["z"][list(model.pressure_levels('z')).index(500)].cpu().numpy() - inp["fields"]["z500"].reshape(-1)).max()),
          float(np.abs(dec["sp"].cpu().numpy() - inp["fields"]["sp"].reshape(-1)).max()))
print(f"[0] decode round-trip max|err| {err:.3g}; surface_pressure alias present: {'surface_pressure' in dec}")
assert err == 0.0 and "surface_pressure" in dec

sp_col = model.resolve_columns("sp")[0]
w = torch.as_tensor(model.area_weights, dtype=torch.float32, device=model.device)

# (1)
with torch.no_grad():
    bg = model.advance(state0, steps=N_STEPS, use_checkpoint=False)
    bg_rep = model.advance(state0, steps=N_STEPS, use_checkpoint=False)
    bg0 = model.advance(state0, steps=N_STEPS, use_checkpoint=False,
                        latent_increment=torch.zeros(model.latent_shape, device=model.device))


def _rel(a, b):
    std = a[0].reshape(a.shape[1], -1).std(dim=1).clamp_min(1e-30)
    return (a - b)[0].reshape(a.shape[1], -1).abs().max(dim=1).values / std


rep, zer = _rel(bg.state, bg_rep.state), _rel(bg.state, bg0.state)
print(f"[1] repeat: bitwise={torch.equal(bg.state, bg_rep.state)} max rel {float(rep.max()):.3g} | "
      f"zero-inc: bitwise={torch.equal(bg.state, bg0.state)} max rel {float(zer.max()):.3g}")
assert float(zer.max()) <= max(10 * float(rep.max()), 1e-3)

region = torch.as_tensor((model.lats >= 20) & (model.lats <= 60) & (model.lons >= 160) & (model.lons <= 220), device=model.device)
ps_target = bg.state[0, sp_col].reshape(-1) + 200.0 * region


def loss_fn(inc, ck):
    out = model.advance(state0, steps=N_STEPS, use_checkpoint=ck, latent_increment=inc)
    return (w * ((out.state[0, sp_col].reshape(-1) - ps_target) / 100.0) ** 2).sum() / w.sum()


def grad(ck):
    inc = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
    loss = loss_fn(inc, ck); loss.backward()
    return loss.item(), inc.grad.detach().clone()


# (2) first checkpointed gradient (after the no-grad rollouts above), second, then plain
l1, g1 = grad(True)
l2, g2 = grad(True)
lp, gp = grad(False)
cmp = lambda g: (float((g * gp).sum() / (g.norm() * gp.norm())), float((g - gp).norm() / gp.norm()))
(c1, r1), (c2, r2) = cmp(g1), cmp(g2)
print(f"[2] loss {l1:.5f}/{l2:.5f}/{lp:.5f}; |g| {g1.norm():.4e} {g2.norm():.4e} plain {gp.norm():.4e}; "
      f"ck#1 cos {c1:.6f} rel {r1:.3g} | ck#2 cos {c2:.6f} rel {r2:.3g}")
assert torch.isfinite(g1).all() and g1.norm() > 0 and r1 < max(10 * r2, 1e-2)

# (3)
inc = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
opt = torch.optim.AdamW([inc], lr=1e-2, weight_decay=0.0)
losses = []
for _ in range(10):
    opt.zero_grad(); loss = loss_fn(inc, True); loss.backward(); opt.step(); losses.append(loss.item())
print("[3] AdamW losses: " + " ".join(f"{v:.4f}" for v in losses))
assert min(losses[1:]) < losses[0]

# (4)
for steps in (1, 2, 4, 8):
    try:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        inc = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
        t0 = time.time()
        out = model.advance(state0, steps=steps, use_checkpoint=True, latent_increment=inc)
        out.state[0, sp_col].mean().backward()
        torch.cuda.synchronize()
        print(f"[4] steps={steps}: fwd+bwd {time.time() - t0:.2f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    except torch.OutOfMemoryError:
        print(f"[4] steps={steps}: OOM"); break
print("ALL CHECKS PASSED")
