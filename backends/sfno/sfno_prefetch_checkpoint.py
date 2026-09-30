"""
Fetch NVIDIA's SFNO-73ch-small checkpoint package from NGC
(nvidia/modulus/sfno_73ch_small, version 0.1.0) into backends/sfno/ngc_cache/
and convert it so makani 0.2.0 (the fcstnet3 env) loads it with its normal,
safe (weights_only) loader.

The package was written by makani v0.1.x (April 2024). Four adjustments, each
checked against makani's source rather than guessed:

1. config.json gains `in_channels`/`out_channels` = range(73): makani 0.2.0
   slices the normalization stats with these. global_means/stds.npy hold 75
   entries whose FIRST 73 match the 73 channel names in order (checked by
   magnitude: sp ~9.7e4 Pa, msl ~1.0e5, z500 ~5.4e4, t2m ~278 K, ...); the
   last two are unused channels.
2. config.json gains `global_means_path`/`global_stds_path` placeholders:
   makani only substitutes the package's own stats files when these keys exist.
3. config.json gains `dhours: 1`: makani's timestep is dt * dhours; the training
   data are hourly ("snapshots at every hour") with dt = 6 -> 6 h.
4. training_checkpoints/best_ckpt_mp0.tar is rewritten weights-only:
   - only `model_state` is kept (drops optimizer/scheduler state and pickled
     training params -- 6.9 GB -> 2.1 GB). The original pickle references
     ruamel.yaml objects that torch's weights_only unpickler cannot rebuild,
     so it is loaded ONCE with weights_only=False, after a static check (no
     code executed) that it only references torch storage/rebuild, OrderedDict,
     makani YParams and ruamel.yaml containers;
   - the DistributedDataParallel `module.` key prefix is dropped;
   - the 8 spectral-filter (dhconv) weights get a leading group axis of 1:
     v0.1 contracted "bixy,iox->boxy" with weight [in, out, l]; v0.2 contracts
     "bgixy,giox->bgoxy" with weight [groups, in, out, l] -- identical for 1
     group.
   The original is kept as best_ckpt_mp0.tar.orig unless --delete-original.

Idempotent: skips files already downloaded/converted. Must run from a LOGIN
node (download), in the fcstnet3 env (torch + makani):
    python backends/sfno/sfno_prefetch_checkpoint.py [--delete-original]
"""

import argparse
import collections
import io
import json
import os
import pickletools
import shutil
import subprocess
import zipfile

URL = "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/sfno_73ch_small/versions/0.1.0/files/sfno_73ch_small"
PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngc_cache", "sfno_73ch_small")
FILES = ["config.json", "metadata.json", "global_means.npy", "global_stds.npy", "land_mask.nc", "orography.nc",
         "training_checkpoints/best_ckpt_mp0.tar"]
CKPT_SIZE = 6870262710
ALLOWED_GLOBALS = {"collections.OrderedDict", "makani.utils.YParams.YParams", "torch._utils._rebuild_tensor_v2",
                   "torch.FloatStorage", "torch.ComplexFloatStorage"}


def _download():
    for f in FILES:
        dst = os.path.join(PKG, f)
        done = os.path.exists(dst) or (f.endswith("best_ckpt_mp0.tar") and os.path.exists(dst + ".orig"))
        if f == "config.json" and os.path.exists(dst + ".orig"):
            done = True
        if done:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"downloading {f}")
        subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "5", "-o", dst, f"{URL}/{f}"], check=True)
    ck = os.path.join(PKG, "training_checkpoints", "best_ckpt_mp0.tar")
    if not os.path.exists(ck + ".orig") and os.path.getsize(ck) != CKPT_SIZE:
        raise RuntimeError(f"{ck} is {os.path.getsize(ck)} bytes, expected {CKPT_SIZE} -- rerun to resume")


def _patch_config():
    cfg, orig = os.path.join(PKG, "config.json"), os.path.join(PKG, "config.json.orig")
    if not os.path.exists(orig):
        shutil.copy(cfg, orig)
    import numpy as np

    c = json.load(open(orig))
    n = len(c["channel_names"])
    means = np.load(os.path.join(PKG, "global_means.npy")).ravel()
    assert means.size >= n, (means.size, n)
    # sanity: the first n stats are the model's channels (surface pressure and MSL in Pa, z500 in m^2/s^2)
    idx = {name: i for i, name in enumerate(c["channel_names"])}
    assert 8e4 < means[idx["sp"]] < 1.1e5 and 9.5e4 < means[idx["msl"]] < 1.05e5 and 4e4 < means[idx["z500"]] < 7e4
    c["in_channels"] = list(range(n))
    c["out_channels"] = list(range(n))
    c["global_means_path"] = "global_means.npy"
    c["global_stds_path"] = "global_stds.npy"
    c["dhours"] = 1
    json.dump(c, open(cfg, "w"), indent=2)
    print("patched config.json (original kept as config.json.orig)")


def _pickle_globals(path):
    z = zipfile.ZipFile(path)
    (pk,) = [n for n in z.namelist() if n.endswith("data.pkl")]
    found = collections.Counter()
    for op, arg, _ in pickletools.genops(io.BytesIO(z.read(pk))):
        if op.name == "GLOBAL":
            found[arg.replace(" ", ".")] += 1
    return found


def _convert_checkpoint(delete_original):
    import torch

    ck = os.path.join(PKG, "training_checkpoints", "best_ckpt_mp0.tar")
    orig = ck + ".orig"
    if os.path.exists(orig) and os.path.exists(ck):
        sd = torch.load(ck, map_location="cpu", weights_only=True)["model_state"]
        if not any(k.startswith("module.") for k in sd) and sd["model.blocks.0.filter.filter.weight"].dim() == 4:
            print("checkpoint already converted")
            if delete_original:
                os.remove(orig)
            return
    if not os.path.exists(orig):
        os.rename(ck, orig)
    found = _pickle_globals(orig)
    bad = [g for g in found if g not in ALLOWED_GLOBALS and not g.startswith("ruamel.yaml.")]
    if bad:
        raise RuntimeError(f"unexpected classes in checkpoint pickle, refusing to load it: {bad}")
    print("checkpoint pickle references only:", sorted(found))
    full = torch.load(orig, map_location="cpu", weights_only=False)  # checked above
    sd = {}
    for k, v in full["model_state"].items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k.endswith("filter.filter.weight") and v.dim() == 3:
            v = v.unsqueeze(0)  # [in, out, l] -> [groups=1, in, out, l]
        sd[k] = v
    torch.save({"model_state": sd}, ck)
    print(f"wrote weights-only {ck} ({len(sd)} tensors)")
    if delete_original:
        os.remove(orig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-original", action="store_true", help="remove the 6.9 GB original checkpoint after converting")
    args = ap.parse_args()
    _download()
    _patch_config()
    _convert_checkpoint(args.delete_original)
    print("done:", PKG)
