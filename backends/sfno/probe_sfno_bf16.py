"""
Validate sfno_autocast_dtype: bfloat16 against fp32 before using it:
(1) 6h forecast from ERA5 2015-01-01T00 vs ERA5 06Z, fp32 vs bf16;
(2) latent gradient of a 2-step ps loss: bf16 vs fp32 (cosine, norm ratio);
(3) peak memory / time: checkpointed fwd+bwd at 1,2,4,8 steps (fp32 vs bf16),
    then a 20-step window with the solver's per-step checkpoint_stride logic
    (use_ckpt = s % stride == 0) at stride 1, 2, 4 in bf16.
Run from the repo root in the fcstnet3 env (GPU).
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
d0 = datetime.datetime(2015, 1, 1, tzinfo=datetime.timezone.utc)
models = {"fp32": SFNOModel(device="cuda"), "bf16": SFNOModel(device="cuda", autocast_dtype="bf16")}
m32 = models["fp32"]
inp = sfno_ic.build_input_state(d0, CACHE)
truth = sfno_ic.read_single_date_fields(d0 + datetime.timedelta(hours=6), CACHE)
state0 = m32.prepare_initial_state(inp, d0)
w = torch.as_tensor(m32.area_weights, dtype=torch.float32, device="cuda"); w = w / w.sum()
names = m32._channel_names

# (1)
for tag, m in models.items():
    with torch.no_grad():
        y = m.advance(state0, steps=1, use_checkpoint=False).state[0]
    line = f"[1] {tag} 6h RMS vs ERA5:"
    for n, sc, u in [("z500", 1 / 9.80665, "m"), ("t850", 1, "K"), ("sp", 0.01, "hPa"), ("u250", 1, "m/s")]:
        d = (y[names.index(n)] - torch.as_tensor(truth[n], device="cuda")).reshape(-1) * sc
        line += f"  {n} {float(torch.sqrt((w * d**2).sum())):.3f} {u}"
    print(line)

# (2)
sp = m32.resolve_columns("sp")[0]
with torch.no_grad():
    bg = m32.advance(state0, steps=2, use_checkpoint=False)
region = torch.as_tensor((m32.lats >= 20) & (m32.lats <= 60) & (m32.lons >= 160) & (m32.lons <= 220), device="cuda")
target = bg.state[0, sp].reshape(-1) + 200.0 * region
g = {}
for tag, m in models.items():
    inc = torch.zeros(m.latent_shape, device="cuda", requires_grad=True)
    out = m.advance(state0, steps=2, use_checkpoint=True, latent_increment=inc)
    loss = (w * ((out.state[0, sp].reshape(-1) - target) / 100.0) ** 2).sum()
    loss.backward(); g[tag] = inc.grad.detach().float()
    print(f"[2] {tag}: loss {loss.item():.6f} |grad| {g[tag].norm():.4e}")
cos = float((g["fp32"] * g["bf16"]).sum() / (g["fp32"].norm() * g["bf16"].norm()))
print(f"[2] bf16 vs fp32 gradient: cos {cos:.5f}, norm ratio {float(g['bf16'].norm() / g['fp32'].norm()):.4f}")


def run(m, steps, stride):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    inc = torch.zeros(m.latent_shape, device="cuda", requires_grad=True)
    t0 = time.time()
    s_ = state0
    for s in range(1, steps + 1):
        use_ckpt = stride > 0 and s % stride == 0
        s_ = m.advance(s_, steps=1, use_checkpoint=use_ckpt, latent_increment=inc if s == 1 else None)
    s_.state[0, sp].mean().backward()
    torch.cuda.synchronize()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 2**30


# (3)
for tag, m in models.items():
    for steps in (1, 2, 4, 8):
        try:
            t, mem = run(m, steps, 1)
            print(f"[3] {tag} steps={steps} stride=1: {t:.2f}s peak {mem:.2f} GiB")
        except torch.OutOfMemoryError:
            print(f"[3] {tag} steps={steps}: OOM"); torch.cuda.empty_cache()
for stride in (1, 2, 4):
    try:
        t, mem = run(models["bf16"], 20, stride)
        print(f"[3] bf16 steps=20 stride={stride}: {t:.2f}s peak {mem:.2f} GiB")
    except torch.OutOfMemoryError:
        print(f"[3] bf16 steps=20 stride={stride}: OOM"); torch.cuda.empty_cache()
try:
    t, mem = run(models["fp32"], 20, 1)
    print(f"[3] fp32 steps=20 stride=1: {t:.2f}s peak {mem:.2f} GiB")
except torch.OutOfMemoryError:
    print("[3] fp32 steps=20 stride=1: OOM")
print("DONE")
