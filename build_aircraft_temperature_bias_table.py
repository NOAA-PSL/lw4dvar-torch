"""
Builds `aircraft_temperature_bias_table.json` -- a static, pressure-level x
region lookup table of aircraft temperature O-B (observation minus
background/first-guess, i.e. GSI's "OmF"), used to bias-correct aircraft
temperature observations in `get_aircraft()` (long_window_4dvar_utils.py)
BEFORE they're assimilated.

Why this exists (see integrate_aircraft_obs.md for the full writeup): the
ACE2 ps+raobs+aircraft observation-ablation run showed z500/T850 skill
regressing vs ps+raobs alone, traced to a real, physically-expected aircraft
temperature bias (pitot-tube adiabatic-heating correction error, largest
near cruise level -- see Zhu et al. 2015, MWR 143(9):3774-3803, "Variational
Correction of Aircraft Temperature Bias in the NCEP's GSI Analysis System").
That paper's own per-tail, ascent/descent-rate-dependent (quadratic, w/w^2)
correction can't be ported here: this repo's PREPBUFR archive has
`instantaneous_altitude_rate_ms` (the w predictor) 100% null (confirmed
directly, see integrate_aircraft_obs.md), and per-tail bias tables can't be
borrowed from any external GSI run either -- checked directly against a real
2015-01-01 00Z operational cnvstat AND a 2022 abias_air file: tail-ID string
matching has ZERO overlap with this repo's own archive in both cases (every
PREPBUFR extraction apparently re-anonymizes tail numbers independently, not
just a fleet-turnover effect -- confirmed by the zero overlap holding even
for the exact same cycle date/hour).

So this is the ceiling of what's buildable from available data: not
per-tail, just a population-level (pressure x region) climatological
correction, estimated from a real, independent GSI operational archive
(2022 GDAS, NOT this repo's own ACE2 cycling -- deliberately, since ACE2's
own cycling has its own known issues and estimating a correction from it
would risk correcting for ACE2's background error rather than the aircraft
sensor's own bias). Uses O-B (OmF), not O-A -- O-A is contaminated by that
same cycle's own partial fit to the observation (self-sensitivity), so O-B
is the standard choice for bias estimation (matches GSI's own variational
scheme, which fits bias predictor coefficients against the background-state
innovation).

Source data: real `diag_conv_t_ges.*_ensmean.nc4` GSI conventional
diagnostic files, 2022-01-01 through 2022-01-19 (76 six-hourly cycles,
~7.3M aircraft temperature obs after QC), from
`/scratch4/BMC/gsienkf/Bo.Huang/expRuns/CycleRun/TestJEDIEnKFOnly/procGSINCDiag/GSINCDiag/`.
Cross-validated against one real 2015-01-01 00Z cycle (parsed from the
legacy Fortran-binary `diag_conv_ges` format inside `gdas1.t00z.cnvstat`,
since that predates the netCDF ncdiag format) -- the two agree well at
cruise levels (150-250 hPa) including the SH signal, less well in the
mid-troposphere where the single 2015 cycle is too small a sample to trust
on its own. This table should be rebuilt once more 2015-native cycles are
available (the user is downloading more `gdas1.t*z.cnvstat` files) --
rerun this script once `_2015_CNVSTAT_DIR` below has enough dates, or merge
the two sources.

Usage:
    python build_aircraft_temperature_bias_table.py
(reads the 2022 ncdiag archive directly from the path above; writes
aircraft_temperature_bias_table.json next to this script)
"""

import glob
import json
import os

import numpy as np
import xarray as xr

_2022_NCDIAG_DIR = '/scratch4/BMC/gsienkf/Bo.Huang/expRuns/CycleRun/TestJEDIEnKFOnly/procGSINCDiag/GSINCDiag'
_AIRCRAFT_ITYPES = (130, 131, 133, 134, 135)

# Same bin edges used throughout the exploratory analysis (see
# integrate_aircraft_obs.md) -- high hPa (surface) first.
PRESSURE_BIN_EDGES_HPA = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 50]
REGIONS = ('NH', 'Tropics', 'SH')


def _region(lat):
    return np.where(lat > 20., 'NH', np.where(lat < -20., 'SH', 'Tropics'))


def _pbin_label(lo, hi):
    return f'{lo}-{hi}'


def build_table(ncdiag_dir=_2022_NCDIAG_DIR):
    files = sorted(glob.glob(os.path.join(ncdiag_dir, '2022*', 'diag_conv_t_ges.*_ensmean.nc4')))
    if not files:
        raise FileNotFoundError(f'no diag_conv_t_ges ncdiag files found under {ncdiag_dir}')
    print(f'reading {len(files)} real GSI ncdiag cycles from {ncdiag_dir} ...')

    lat_chunks, pressure_chunks, omb_chunks = [], [], []
    for f in files:
        with xr.open_dataset(f) as ds:
            itype = ds['Observation_Type'].values
            mask = np.isin(itype, _AIRCRAFT_ITYPES) & (ds['Analysis_Use_Flag'].values == 1)
            if not mask.any():
                continue
            lat_chunks.append(ds['Latitude'].values[mask])
            pressure_chunks.append(ds['Pressure'].values[mask])
            # O-B (OmF), NOT O-A -- see module docstring.
            omb_chunks.append(ds['Obs_Minus_Forecast_unadjusted'].values[mask])

    lat = np.concatenate(lat_chunks)
    pressure = np.concatenate(pressure_chunks)
    omb = np.concatenate(omb_chunks)
    region = _region(lat)
    print(f'{len(omb)} aircraft temperature obs (QC-passed, pooled across {len(files)} cycles)')

    edges = PRESSURE_BIN_EDGES_HPA
    table = {}
    for i in range(len(edges) - 1):
        hi, lo = edges[i], edges[i + 1]  # hi > lo (e.g. 1000 down to 850)
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
    return table


if __name__ == '__main__':
    table = build_table()
    out = {
        'description': (
            'Static aircraft temperature bias-correction table (K), pressure bin x region, '
            'mean O-B (observation minus background/first-guess, NOT bias-corrected) from a real, '
            'independent GSI operational archive -- see build_aircraft_temperature_bias_table.py '
            'and integrate_aircraft_obs.md for how/why. Subtract this value from the raw observed '
            'temperature before assimilation (observation = raw_observation - table_value).'
        ),
        'source': '2022-01-01 to 2022-01-19 real GDAS diag_conv_t_ges ncdiag (76 cycles, aircraft itype 130/131/133/134/135)',
        'innovation_convention': 'observation_minus_background (OmF, unadjusted)',
        'pressure_bin_edges_hpa': PRESSURE_BIN_EDGES_HPA,
        'table': table,
    }
    ofile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aircraft_temperature_bias_table.json')
    with open(ofile, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote ' + ofile)
    for label, regions in table.items():
        parts = ', '.join(f"{r}={regions[r]['mean_omb_k']:+.3f}K(n={regions[r]['n']})"
                           if regions[r]['n'] else f'{r}=NA' for r in REGIONS)
        print(f'  {label:>10s} hPa: {parts}')
