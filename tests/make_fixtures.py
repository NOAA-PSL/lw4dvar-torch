"""Rebuild the test fixtures under tests/data (see tests/README.md).

Fixtures are small frozen inputs, committed in plain git; anything too large
for that (a model checkpoint) is a GitHub release asset pinned by sha256 in a
manifest committed next to the data. Regenerating a fixture is a deliberate
change: rerun this script, check what moved, and commit the result.

mini_ace2 -- inputs for the mini-ACE2 smoke test (tests/ace2/test_ace2_mini.py):
one 24 h window from 2015-01-02T00 with a 6 h spin-up, on the 2.5 deg
16-layer CRTM+land dataset the mini ACE2 was trained on (2015 held out).
  data/mini_ace2/forcing_data/forcing_2015.nc  forcings at the 6 times the run
      steps through (2015-01-01T18 .. 2015-01-03T00), ACE2Model's layout
  data/mini_ace2/ic_cache/ace2_ic_2015010118.nc  the initial state
  data/mini_ace2/ic_cache/ace2_verif_*.nc  h500/TMP850 truth (the dataset's own)
  data/mini_ace2/psobs/psobs1_*.txt  2000 ps obs per time (fixed seed)
  data/mini_ace2/manifest.json  provenance and the checkpoint asset's sha256
--weights_out writes the weights-only checkpoint (the `stepper` config,
dataset_info and step state; no training history). torch.save is not
byte-reproducible, so a rebuilt file never matches the published asset's
sha256: the manifest keeps the published asset's entry unless --new_asset,
which records the new file's sha256 -- then upload that file as the asset of
a NEW release (ci-fixtures-vN) and update MINI_ACE2['release'] / ['url'].

Ursa only (reads the training dataset, the checkpoint and the obs archive), in
the ace2 env:
  python tests/make_fixtures.py mini_ace2 --weights_out /tmp/mini_ace2.tar
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

MINI_ACE2 = {
    'checkpoint': '/scratch4/BMC/gsienkf/whitaker/ace2_training/mini-ace2-crtm-land-16layer-2p5deg-e64-l8/'
                  'training_checkpoints/best_inference_ckpt.tar',
    'dataset': '/scratch4/BMC/gsienkf/whitaker/ace2_era5_2p5deg_16layer',
    'obspath': '/scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs',
    'asset_name': 'mini-ace2-crtm-land-16layer-2p5deg-e64-l8.tar',
    'release': 'ci-fixtures-v1',
    'url': 'https://github.com/NOAA-PSL/lw4dvar-torch/releases/download/ci-fixtures-v1/'
           'mini-ace2-crtm-land-16layer-2p5deg-e64-l8.tar',
    'sdate': '2015-01-02T00',
    'spinup_h': 6,
    'window_h': 24,
    'nobs': 2000,
    'seed': 0,
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def _zlib(ds):
    return {v: {'zlib': True, 'complevel': 9} for v in ds.data_vars if ds[v].ndim > 0}


def strip_checkpoint(src, dst):
    """Weights-only copy: what ACE2Model reads (`stepper`: config,
    dataset_info, step)."""
    import torch

    st = torch.load(src, map_location='cpu', weights_only=False)['stepper']
    torch.save({'stepper': {k: st[k] for k in ('config', 'dataset_info', 'step')}}, dst)


def mini_ace2(args):
    sys.path.insert(0, os.path.join(REPO_ROOT, 'backends', 'ace2'))
    import ace2_inference_files as aif

    c = dict(MINI_ACE2)
    out = os.path.join(HERE, 'data', 'mini_ace2')
    fdir, idir, odir = (os.path.join(out, d) for d in ('forcing_data', 'ic_cache', 'psobs'))
    for d in (fdir, idir, odir):
        os.makedirs(d, exist_ok=True)

    t0 = datetime.datetime.fromisoformat(c['sdate'])
    t_ic = t0 - datetime.timedelta(hours=c['spinup_h'])
    t_end = t0 + datetime.timedelta(hours=c['window_h'])
    steps = [t_ic + datetime.timedelta(hours=6 * k) for k in range((t_end - t_ic) // datetime.timedelta(hours=6) + 1)]
    window = [t for t in steps if t >= t0]
    if len({t.year for t in steps}) != 1:
        raise ValueError('the fixture window must not cross a year boundary (one forcing file)')
    files = sorted({os.path.join(c['dataset'], f"{t:%Y%m}0100.nc") for t in steps})
    ds = xr.concat([xr.open_dataset(p) for p in files], dim='time', data_vars='minimal', coords='minimal',
                   compat='override') if len(files) > 1 else xr.open_dataset(files[0])

    forcing, prognostic = aif.checkpoint_names(c['checkpoint'])
    forcing += sorted((n for n in ds.data_vars if n.startswith(('ak_', 'bk_'))), key=lambda n: (n[:2], int(n[3:])))
    f = ds[forcing].sel(time=[np.datetime64(t) for t in steps]).load()
    for v in aif.STATIC:
        if 'time' in f[v].dims:
            f[v] = f[v].isel(time=0, drop=True)
    f.time.encoding = {'units': 'hours since 1940-01-01T12:00:00', 'calendar': 'proleptic_gregorian', 'dtype': 'int64'}
    f.attrs = {'source': f"{c['dataset']} (tests/make_fixtures.py mini_ace2)"}
    f.to_netcdf(os.path.join(fdir, f'forcing_{t0.year}.nc'), encoding=_zlib(f))

    ic = ds[prognostic].sel(time=np.datetime64(t_ic)).drop_vars('time').load()
    ic.attrs = {'source': f"{c['dataset']} (tests/make_fixtures.py mini_ace2)",
                'valid_time': f'{t_ic:%Y-%m-%dT%H:%M:%S}', 'presfc_reduced_to_hgtsfc': 0}
    ic.to_netcdf(os.path.join(idir, f'ace2_ic_{t_ic:%Y%m%d%H}.nc'), encoding=_zlib(ic))
    for t in window:
        v = ds[['h500', 'TMP850']].sel(time=np.datetime64(t)).drop_vars('time').load()
        v.attrs = {'source': f"{c['dataset']} (tests/make_fixtures.py mini_ace2)", 'valid_time': f'{t:%Y-%m-%dT%H:%M:%S}'}
        v.to_netcdf(os.path.join(idir, f'ace2_verif_{t:%Y%m%d%H}.nc'), encoding=_zlib(v))

    rng = np.random.default_rng(c['seed'])
    for t in window:
        with open(os.path.join(c['obspath'], f'psobs1_{t:%Y%m%d%H}.txt')) as fh:
            lines = fh.readlines()
        keep = np.sort(rng.choice(len(lines), min(c['nobs'], len(lines)), replace=False))
        with open(os.path.join(odir, f'psobs1_{t:%Y%m%d%H}.txt'), 'w') as fh:
            fh.writelines(lines[i] for i in keep)

    manifest = {k: c[k] for k in ('sdate', 'spinup_h', 'window_h', 'nobs', 'seed', 'dataset', 'obspath')}
    manifest['checkpoint'] = {'source': c['checkpoint'], 'release': c['release'], 'name': c['asset_name'], 'url': c['url']}
    if args.weights_out:
        strip_checkpoint(c['checkpoint'], args.weights_out)
        print(f'wrote {args.weights_out} (sha256 {_sha256(args.weights_out)})')
    old = os.path.join(out, 'manifest.json')
    if args.new_asset:
        if not args.weights_out:
            raise SystemExit('--new_asset needs --weights_out')
        manifest['checkpoint'].update(sha256=_sha256(args.weights_out), size=os.path.getsize(args.weights_out))
        print(f"recorded the new checkpoint: upload {args.weights_out} as {c['asset_name']} to release {c['release']}")
    elif os.path.exists(old):
        with open(old) as fh:
            manifest['checkpoint'] = json.load(fh)['checkpoint']
    else:
        raise SystemExit('no manifest.json yet: run with --weights_out ... --new_asset')
    with open(os.path.join(out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
        fh.write('\n')
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fixture', choices=['mini_ace2'])
    ap.add_argument('--weights_out', help='mini_ace2: also write the weights-only checkpoint here')
    ap.add_argument('--new_asset', action='store_true',
                    help='mini_ace2: record --weights_out as a new release asset in the manifest')
    args = ap.parse_args()
    {'mini_ace2': mini_ace2}[args.fixture](args)


if __name__ == '__main__':
    main()
