"""
2015-native counterpart of build_aircraft_temperature_bias_table.py -- same
pressure-bin x region O-B table, same methodology, but built from real
GSI operational `gdas1.t??z.cnvstat` files for January 2015 instead of the
2022 netCDF ncdiag archive. See that script's docstring and
integrate_aircraft_obs.md's "Static aircraft temperature bias correction"
section for why this table exists and why it isn't per-tail.

Source format is GSI's LEGACY Fortran-unformatted-sequential `diag_conv`
binary (ncdiag didn't exist yet in Jan 2015), one `diag_conv_ges.<date>.gz`
member per `.cnvstat` tarball -- NOT the netCDF ncdiag format the 2022
script reads. Parsed here by hand (no existing Python reader for this
format was found in any available repo -- see integrate_aircraft_obs.md):
record structure and column layout (itype=rdiag[:,0], lat=rdiag[:,2],
pressure=rdiag[:,5], observation=rdiag[:,16], obs-ges adjusted=rdiag[:,17],
obs-ges UNADJUSTED=rdiag[:,18]) were cross-checked directly against GSI's
own `setupt.f90` (`rdiagbuf(N,ii) = ...` assignments) in the
`global-workflow/sorc/gsi_enkf.fd/src/gsi/` source tree, and the
unadjusted/adjusted equality check (100% identical, confirming aircraft_t_bc
was OFF for this 2015 operational configuration -- no coefficient file to
borrow, only raw O-B) was verified directly against the first of these
files before trusting the parser on the rest.

Confirmed 2015-01-20T00 (`2015012000`) is missing from the source directory
(user-reported) -- this script just skips whatever `_2015_CNVSTAT_DIRS`
glob doesn't find, same "skip what's missing" posture as every other
cycle-scanning script in plots/.

Usage:
    python build_aircraft_temperature_bias_table_2015.py
(reads /scratch4/.../procGSIBC/GSIBC_GDAS/201501*/gdas1.t??z.cnvstat;
writes aircraft_temperature_bias_table_2015.json next to this script --
aircraft_temperature_bias_table.json, the 2022-based table, is left alone
so both exist side by side for comparison.)
"""

import glob
import gzip
import json
import os
import struct
import tarfile

import numpy as np

from build_aircraft_temperature_bias_table import PRESSURE_BIN_EDGES_HPA, REGIONS, _pbin_label, _region

_2015_CNVSTAT_GLOB = (
    '/scratch4/BMC/gsienkf/Bo.Huang/expRuns/CycleRun/TestJEDIEnKFOnly/'
    'procGSIBC/GSIBC_GDAS/201501????/gdas1.t??z.cnvstat'
)
_AIRCRAFT_ITYPES = (130, 131, 133, 134, 135)


def _read_record(f):
    """One Fortran unformatted-sequential record: 4-byte big-endian length,
    payload, matching 4-byte trailer. Returns None at EOF."""
    hdr = f.read(4)
    if len(hdr) < 4:
        return None
    (n,) = struct.unpack('>I', hdr)
    payload = f.read(n)
    (n2,) = struct.unpack('>I', f.read(4))
    assert n == n2, f'Fortran record length mismatch: header {n} != trailer {n2}'
    return payload


def read_aircraft_t_from_cnvstat(cnvstat_path, cycle_date):
    """(itype, lat, pressure_hpa, omb_unadjusted) arrays for the '  t'
    diag block's aircraft rows (itype in _AIRCRAFT_ITYPES), from the
    `diag_conv_ges.<cycle_date>.gz` member of one `.cnvstat` tarball.
    """
    with tarfile.open(cnvstat_path) as tf:
        member = tf.getmember(f'diag_conv_ges.{cycle_date}.gz')
        raw = gzip.decompress(tf.extractfile(member).read())

    import io
    f = io.BytesIO(raw)
    _read_record(f)  # idate -- not needed here

    itype_chunks, lat_chunks, pressure_chunks, omb_chunks = [], [], [], []
    while True:
        header = _read_record(f)
        if header is None:
            break
        dtype = header[0:3].decode('latin1')
        _nchar, nreal, ii, _mype = struct.unpack('>4i', header[3:19])
        data = _read_record(f)
        if dtype != '  t' or ii == 0:
            continue
        # cdiag (ii*8 bytes, station ids -- unused here) then rdiag
        # (nreal x ii, row-major per obs), big-endian float32.
        rdiag = np.frombuffer(data[ii * 8:], dtype='>f4').reshape(ii, nreal)
        itype_chunks.append(rdiag[:, 0])
        lat_chunks.append(rdiag[:, 2])
        pressure_chunks.append(rdiag[:, 5])
        omb_chunks.append(rdiag[:, 18])  # obs-ges UNADJUSTED (OmF) -- see module docstring

    if not itype_chunks:
        return np.array([]), np.array([]), np.array([]), np.array([])
    itype = np.concatenate(itype_chunks).astype(np.int64)
    lat = np.concatenate(lat_chunks)
    pressure = np.concatenate(pressure_chunks)
    omb = np.concatenate(omb_chunks)
    mask = np.isin(itype, _AIRCRAFT_ITYPES)
    return itype[mask], lat[mask], pressure[mask], omb[mask]


def build_table():
    files = sorted(glob.glob(_2015_CNVSTAT_GLOB))
    if not files:
        raise FileNotFoundError(f'no cnvstat files found matching {_2015_CNVSTAT_GLOB}')
    print(f'reading {len(files)} real 2015-01 GSI cnvstat cycles ...')

    lat_chunks, pressure_chunks, omb_chunks = [], [], []
    n_cycles_used = 0
    for path in files:
        cycle_date = os.path.basename(os.path.dirname(path))  # e.g. '2015010106'
        try:
            _itype, lat, pressure, omb = read_aircraft_t_from_cnvstat(path, cycle_date)
        except KeyError:
            print(f'  warning: no diag_conv_ges.{cycle_date}.gz member in {path}, skipping')
            continue
        if lat.size == 0:
            continue
        lat_chunks.append(lat)
        pressure_chunks.append(pressure)
        omb_chunks.append(omb)
        n_cycles_used += 1

    lat = np.concatenate(lat_chunks)
    pressure = np.concatenate(pressure_chunks)
    omb = np.concatenate(omb_chunks)
    region = _region(lat)
    print(f'{len(omb)} aircraft temperature obs (pooled across {n_cycles_used} of {len(files)} cycles found)')

    edges = PRESSURE_BIN_EDGES_HPA
    table = {}
    for i in range(len(edges) - 1):
        hi, lo = edges[i], edges[i + 1]
        label = _pbin_label(lo, hi)
        in_bin = (pressure <= hi) & (pressure > lo)
        table[label] = {}
        for r in REGIONS:
            sel = in_bin & (region == r)
            n = int(sel.sum())
            table[label][r] = {
                'n': n,
                'mean_omb_k': float(omb[sel].mean()) if n > 0 else None,
            }
    return table, n_cycles_used, len(files)


if __name__ == '__main__':
    table, n_cycles_used, n_cycles_found = build_table()
    out = {
        'description': (
            'Static aircraft temperature bias-correction table (K), pressure bin x region, '
            'mean O-B (observation minus background/first-guess, NOT bias-corrected -- confirmed '
            'aircraft_t_bc was off in this 2015 operational configuration, adjusted==unadjusted '
            'exactly) from real 2015-01 GSI operational cnvstat files. See '
            'build_aircraft_temperature_bias_table_2015.py and integrate_aircraft_obs.md for how/why. '
            '2015-native counterpart of aircraft_temperature_bias_table.json (built from 2022 data) -- '
            'kept side by side for comparison, neither overwrites the other.'
        ),
        'source': f'2015-01 real GDAS diag_conv_ges (legacy binary) -- {n_cycles_used} of {n_cycles_found} '
                   f'cycles found (2015012000 confirmed missing), aircraft itype 130/131/133/134/135',
        'innovation_convention': 'observation_minus_background (OmF, unadjusted -- bias correction was off for this archive)',
        'pressure_bin_edges_hpa': PRESSURE_BIN_EDGES_HPA,
        'table': table,
    }
    ofile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aircraft_temperature_bias_table_2015.json')
    with open(ofile, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote ' + ofile)
    for label, regions in table.items():
        parts = ', '.join(f"{r}={regions[r]['mean_omb_k']:+.3f}K(n={regions[r]['n']})"
                           if regions[r]['n'] else f'{r}=NA' for r in REGIONS)
        print(f'  {label:>10s} hPa: {parts}')
