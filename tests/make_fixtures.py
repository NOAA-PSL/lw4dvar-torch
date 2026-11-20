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
  data/mini_ace2/psobs/<YYYY-MM-DD>/<HH>/{reports,levels}/{ADPSFC,SFCSHP}/
      part-00000.parquet  a synthetic normalized-PREPBUFR-parquet archive
      (see src/psobs_parquet.py/docs/integrate_multiple_parquetobs.md for the schema),
      ~2000 ps obs per cycle, globally scattered, fixed seed -- entirely
      generated in-process, no real obs archive needed to rebuild this fixture
  data/mini_ace2/raobs/<YYYY-MM-DD>/<HH>/{reports,levels,events}/ADPUPA/
      part-00000.parquet  a radiosonde (ADPUPA) archive (see
      src/raobs_parquet.py), 50 globally-scattered fixed-seed stations per
      cycle at ACE2's own 13 pressure levels (MINI_ACE2_RAOBS_PLEVS) --
      unlike psobs's pure synthetic noise, t/u/v are the REAL mini-ACE2
      checkpoint's own background (sampled at each station's nearest
      native grid point, see _ace2_background_profiles) plus a small
      Gaussian perturbation, so there's a real, reducible O-B signal for
      tests/ace2/test_ace2_mini.py's cost-drop check to find (needs the
      full ace2 env -- fme/torch -- not just pyarrow, unlike psobs)
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

golden -- the golden outputs (layer 4; tests/golden/, tests/ace2/):
  data/golden/toy_<config>.json  the toy-model minimization for each config in
      tests/golden/toy_golden.py (float64; anywhere, in the CI env)
  data/golden/toy_obs.json  the toy runs' frozen obs values: written only
      with --toy-obs (or if missing), never to absorb a solver change
  data/golden/mini_ace2.json  with --ace2: the mini-ACE2 run's summary
      (tests/ace2/mini_ace2.py; needs fme and the checkpoint, as
      `make check-ace2` does)
Regenerating is a deliberate change: run it in the CI env
(envs/requirements-ci.txt, plus envs/requirements-ci-ace2.txt for --ace2), look at the
diff, and say in the PR what moved and why.
  python tests/make_fixtures.py golden [--ace2] [--toy-obs]
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
    'asset_name': 'mini-ace2-crtm-land-16layer-2p5deg-e64-l8.tar',
    'release': 'ci-fixtures-v1',
    'url': 'https://github.com/NOAA-PSL/lw4dvar-torch/releases/download/ci-fixtures-v1/'
           'mini-ace2-crtm-land-16layer-2p5deg-e64-l8.tar',
    'sdate': '2015-01-02T00',
    'spinup_h': 6,
    'window_h': 24,
    'nobs': 2000,
    'nobs_raobs': 50,
    'seed': 0,
}

# ACE2Model's own pressure levels (backends/ace2/ace2_model.py's
# DEFAULT_PLEVS) -- the mandatory-level match in raobs_parquet.py needs the
# synthetic reports' pressure_mb to land exactly on these, since ACE2
# returns the same 13-level axis for every derived base ('t'/'u'/'v'/'q').
# If ACE2Model's own levels ever change, this fixture needs regenerating.
MINI_ACE2_RAOBS_PLEVS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)


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


def _write_psobs_parquet(root, cycle_times, rng, nobs):
    """Write a synthetic normalized-PREPBUFR-parquet archive (the schema
    src/psobs_parquet.py/src/prepbufr_parquet_common.py read) under `root`, one
    cycle directory per entry in `cycle_times` (each must land exactly on a
    6-hourly boundary -- every synthetic report's observation_time equals
    the cycle time exactly, so get_psobs's default 0.5h time_tolerance_hours
    never needs a neighboring cycle directory). `nobs` globally-scattered,
    fixed-seed reports per cycle, split evenly across the two classes
    get_psobs reads by default (ADPSFC report_type 181, SFCSHP 180) -- only
    the columns src/psobs_parquet.py actually requests are written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    classes = {'ADPSFC': 181, 'SFCSHP': 180}
    n_each = nobs // len(classes)
    for t in cycle_times:
        cycle_dir = os.path.join(root, f'{t:%Y-%m-%d}', f'{t:%H}')
        for cls, report_type in classes.items():
            report_id = np.arange(n_each, dtype=np.int64)
            reports = pa.table({
                'report_id': pa.array(report_id),
                'observation_time': pa.array([t] * n_each, type=pa.timestamp('s')),
                'station_id': pa.array([f'{cls[:2]}{i:06d}' for i in report_id]),
                'latitude': pa.array(rng.uniform(-80., 80., n_each)),
                'longitude': pa.array(rng.uniform(0., 360., n_each)),
                'elevation_m': pa.array(rng.uniform(0., 500., n_each)),
                'report_type': pa.array(np.full(n_each, report_type, dtype=np.int32)),
            })
            levels = pa.table({
                'report_id': pa.array(report_id),
                'level_index': pa.array(np.zeros(n_each, dtype=np.int32)),
                'pressure_mb': pa.array(1013. + rng.normal(0., 4., n_each)),
                'pressure_quality': pa.array(rng.integers(0, 4, n_each).astype(np.float64)),
                'pressure_error_mb': pa.array(np.clip(rng.normal(0.5, 0.15, n_each), 0.2, None)),
            })
            for table_name, table in (('reports', reports), ('levels', levels)):
                out_path = os.path.join(cycle_dir, table_name, cls, 'part-00000.parquet')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                pq.write_table(table, out_path)


def _ace2_background_profiles(checkpoint, forcing_dir, ic_cache_dir, t_ic, spinup_h, window, plevs, station_lat, station_lon):
    """Sample the REAL mini-ACE2 background trajectory's own 't'/'u'/'v' at
    `plevs` (its own pressure levels -- see MINI_ACE2_RAOBS_PLEVS) and at
    each synthetic station's nearest native grid point, at every time in
    `window`. Runs the real checkpoint forward from the same IC/forcing
    files `mini_ace2` already wrote (`ace2_ic.build_input_state`, same as
    `get_input`'s own ACE2 branch), under `torch.no_grad()` -- this is
    sampling the background, not training.

    Returns `{time: {'t': (n_levels, n_stations), 'u': ..., 'v': ...}}`
    (K, m/s, m/s) -- a small Gaussian perturbation around THIS, rather than
    independent standard-atmosphere climatology, is what makes the
    synthetic raobs mismatch something a few epochs of latent-increment
    optimization can actually reduce (see docs/integrate_multiple_parquetobs.md's
    "Adding radiosonde (raobs) observations" section -- the same lesson the
    psobs fixture's own noise-tightening already drew).
    """
    import torch

    sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))  # ace2_model imports forecast_model flatly
    sys.path.insert(0, os.path.join(REPO_ROOT, 'backends', 'ace2'))
    import ace2_ic
    import ace2_model

    model = ace2_model.ACE2Model(checkpoint_path=checkpoint, forcing_dir=forcing_dir, device='cpu')
    input_state = ace2_ic.build_input_state(t_ic, ic_cache_dir)
    with torch.no_grad():
        state = model.prepare_initial_state(input_state, t_ic)
        spinup_steps = int(round(spinup_h / (model.timestep.total_seconds() / 3600)))
        state = model.advance(state, steps=spinup_steps, use_checkpoint=False)

        lat_idx = np.argmin(
            (model.lats[None, :] - station_lat[:, None]) ** 2
            + (np.mod(model.lons[None, :] - station_lon[:, None] + 180., 360.) - 180.) ** 2,
            axis=1,
        )
        result = {}
        for i, t in enumerate(window):
            decoded = model.decode_state(state, only=['t', 'u', 'v'])
            result[t] = {
                base: decoded[base].detach().cpu().numpy()[:, lat_idx] for base in ('t', 'u', 'v')
            }
            if i < len(window) - 1:  # no forcing past the last window time
                state = model.advance(state, steps=1, use_checkpoint=False)
    return result


def _write_raobs_parquet(root, cycle_times, rng, station_lat, station_lon, background, plevs=MINI_ACE2_RAOBS_PLEVS, noise_std=None):
    """Write a synthetic normalized-PREPBUFR-parquet ADPUPA archive (the
    schema src/raobs_parquet.py/src/prepbufr_parquet_common.py read) under
    `root`, one cycle directory per entry in `cycle_times`, one profile per
    `station_lat`/`station_lon` entry, reporting temperature/wind at every
    pressure in `plevs` (one `levels` row per (station, level)) --
    `background[t]['t'/'u'/'v']` (from `_ace2_background_profiles`, shape
    `(len(plevs), n_stations)`) plus a small Gaussian perturbation
    (`noise_std`), so there's a real, reducible O-B signal (see that
    function's docstring). Humidity still uses independent standard-
    atmosphere climatology (no equally cheap real-q background to sample
    here) -- fine since `tests/ace2/test_ace2_mini.py` doesn't enable it.
    Heights follow the barometric formula so they
    increase monotonically with altitude (required by raobs_parquet.py's
    non-monotonic-height QC) and land in a physically sane range. `events`
    is written empty -- common.sensible_temperature then falls back to the
    `temperature_c` column unchanged, which is fine here since this fixture
    has no virtual-temperature distinction to make.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    noise_std = noise_std or {'t': 1.0, 'u': 2.0, 'v': 2.0}
    plevs = np.asarray(plevs, dtype=np.float64)
    n_levels = plevs.size
    nstations = station_lat.size
    # Standard-atmosphere barometric height (m) for each level -- only
    # needs to be monotonically increasing as pressure drops, not exact.
    height_by_level = 44331.0 * (1.0 - (plevs / 1013.25) ** 0.1903)

    for t in cycle_times:
        cycle_dir = os.path.join(root, f'{t:%Y-%m-%d}', f'{t:%H}')
        report_id = np.arange(nstations, dtype=np.int64)
        station_elev = rng.uniform(0., 300., nstations)
        reports = pa.table({
            'report_id': pa.array(report_id),
            'observation_time': pa.array([t] * nstations, type=pa.timestamp('s')),
            'station_id': pa.array([f'RB{i:06d}' for i in report_id]),
            'latitude': pa.array(station_lat),
            'longitude': pa.array(station_lon),
            'elevation_m': pa.array(station_elev),
            'report_type': pa.array(np.full(nstations, 120, dtype=np.int32)),
        })

        level_report_id = np.repeat(report_id, n_levels)
        level_index = np.tile(np.arange(n_levels, dtype=np.int32), nstations)
        pressure = np.tile(plevs, nstations)
        height = np.tile(height_by_level, nstations) + rng.normal(0., 10., nstations * n_levels)
        # background[t][base] is (n_levels, n_stations); flatten to match
        # level_report_id's (station-major, then level) ordering, i.e.
        # transpose to (n_stations, n_levels) before ravel.
        temperature_c = (
            background[t]['t'].T.ravel() - 273.15 + rng.normal(0., noise_std['t'], nstations * n_levels)
        )
        u = background[t]['u'].T.ravel() + rng.normal(0., noise_std['u'], nstations * n_levels)
        v = background[t]['v'].T.ravel() + rng.normal(0., noise_std['v'], nstations * n_levels)
        q_mg_kg = np.clip(
            12000. * np.exp(-np.tile(height_by_level, nstations) / 2500.)
            + rng.normal(0., 200., nstations * n_levels),
            50., None,
        )
        levels = pa.table({
            'report_id': pa.array(level_report_id),
            'level_index': pa.array(level_index),
            'category': pa.array(np.ones(nstations * n_levels, dtype=np.int16)),
            'pressure_mb': pa.array(pressure),
            'height_m': pa.array(height),
            'height_quality': pa.array(np.ones(nstations * n_levels)),
            'temperature_c': pa.array(temperature_c),
            'temperature_quality': pa.array(np.ones(nstations * n_levels)),
            'u_wind_ms': pa.array(u),
            'wind_quality': pa.array(np.ones(nstations * n_levels)),
            'v_wind_ms': pa.array(v),
            'specific_humidity_mg_kg': pa.array(q_mg_kg),
            'humidity_quality': pa.array(np.ones(nstations * n_levels)),
        })
        events = pa.table({
            'report_id': pa.array([], type=pa.int64()),
            'level_index': pa.array([], type=pa.int64()),
            'value_mnemonic': pa.array([], type=pa.string()),
            'event_index': pa.array([], type=pa.int64()),
            'value': pa.array([], type=pa.float64()),
            'program_code': pa.array([], type=pa.int64()),
        })
        for table_name, table in (('reports', reports), ('levels', levels), ('events', events)):
            out_path = os.path.join(cycle_dir, table_name, 'ADPUPA', 'part-00000.parquet')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pq.write_table(table, out_path)


def mini_ace2(args):
    sys.path.insert(0, os.path.join(REPO_ROOT, 'backends', 'ace2'))
    import ace2_inference_files as aif

    c = dict(MINI_ACE2)
    out = os.path.join(HERE, 'data', 'mini_ace2')
    fdir, idir, odir, odir_raobs = (
        os.path.join(out, d) for d in ('forcing_data', 'ic_cache', 'psobs', 'raobs')
    )
    for d in (fdir, idir, odir, odir_raobs):
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
    _write_psobs_parquet(odir, window, rng, c['nobs'])
    station_lat = rng.uniform(-80., 80., c['nobs_raobs'])
    station_lon = rng.uniform(0., 360., c['nobs_raobs'])
    background = _ace2_background_profiles(
        c['checkpoint'], fdir, idir, t_ic, c['spinup_h'], window, MINI_ACE2_RAOBS_PLEVS, station_lat, station_lon,
    )
    _write_raobs_parquet(odir_raobs, window, rng, station_lat, station_lon, background)

    manifest = {k: c[k] for k in ('sdate', 'spinup_h', 'window_h', 'nobs', 'nobs_raobs', 'seed', 'dataset')}
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


def golden(args):
    sys.path[:0] = [os.path.join(REPO_ROOT, 'src'), REPO_ROOT, os.path.join(HERE, 'golden'), os.path.join(HERE, 'ace2')]
    import toy_golden
    if args.toy_obs or not os.path.exists(toy_golden.OBS_PATH):
        print(f'wrote {toy_golden.write_obs()}')
    for name in toy_golden.CONFIGS:
        print(f'wrote {toy_golden.write(name)}')
    if args.ace2:
        import tempfile
        import mini_ace2
        with tempfile.TemporaryDirectory() as tmp:
            rc, log, exp, windows, out = mini_ace2.run(tmp, mini_ace2.checkpoint_path())
            if rc != 0:
                sys.exit(f'mini-ACE2 run failed:\n{log[-4000:]}')
            summary = mini_ace2.summarize(exp, windows, out)
        os.makedirs(os.path.dirname(mini_ace2.GOLDEN), exist_ok=True)
        with open(mini_ace2.GOLDEN, 'w') as fh:
            json.dump(summary, fh, indent=1)
            fh.write('\n')
        print(f'wrote {mini_ace2.GOLDEN}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fixture', choices=['mini_ace2', 'golden'])
    ap.add_argument('--weights_out', help='mini_ace2: also write the weights-only checkpoint here')
    ap.add_argument('--new_asset', action='store_true',
                    help='mini_ace2: record --weights_out as a new release asset in the manifest')
    ap.add_argument('--ace2', action='store_true', help='golden: also the mini-ACE2 golden output')
    ap.add_argument('--toy-obs', action='store_true',
                    help='golden: also rewrite the toy runs\' frozen obs (only if the toy problem changed)')
    args = ap.parse_args()
    {'mini_ace2': mini_ace2, 'golden': golden}[args.fixture](args)


if __name__ == '__main__':
    main()
