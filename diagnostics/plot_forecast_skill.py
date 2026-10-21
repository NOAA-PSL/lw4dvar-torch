"""Plot diagnostics/forecast_skill.py output: RMSE vs forecast lead, columns Global / NH /
Tropics / SH, one line per run, over the init dates common to every run (see
forecast_skill.md). ERA5 scores are averaged over dates (RMSE: root of the mean square);
obs scores pool O - F over dates.

  fcst_era5_upper.png  vs ERA5: z500, q850, t850, 250 hPa wind speed
  fcst_era5_sfc.png    vs ERA5: 10 m wind speed, 2 m temperature, 2 m specific humidity
  fcst_obs_upper.png   vs obs:  surface pressure, raob q850, t850, t500, 250 hPa wind speed
  fcst_obs_sfc.png     vs surface obs: 10 m wind speed (land + marine stations; ASCAT),
                       2 m temperature, 2 m specific humidity (land + marine stations)
  fcst_obs_profile.png vs upper-air conventional obs (radiosondes, aircraft, profilers; no
                       satellite winds), at lead --profile-lead (default 12 h): RMS(O - F)
                       (solid) and mean O - F (dashed) against pressure, T, q, vector wind
  fcst_obs_profile_platform.png  the same, Global, one column per platform group
  forecast_skill.csv   every lead-time curve above, plus the per-network surface scores
  forecast_skill_profile.csv  the profiles, one column per pressure layer

Usage: python diagnostics/plot_forecast_skill.py OUTDIR LABEL[=legend] ... [--start T] [--end T]
                                                 [--skill-dir output/forecast_skill] [--profile-lead 12]
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('out')
ap.add_argument('labels', nargs='+', help='LABEL or LABEL=legend')
ap.add_argument('--start', default='')
ap.add_argument('--end', default='9999')
ap.add_argument('--skill-dir', default='output/forecast_skill')
ap.add_argument('--profile-lead', type=int, default=12, help='lead (h) of the profile figures (default 12)')
args = ap.parse_args()
labels = [(lab.split('=', 1)[0], lab.split('=', 1)[-1]) for lab in args.labels]
os.makedirs(args.out, exist_ok=True)
REGIONS = ['Global', 'NH', 'Tropics', 'SH']
COLORS = ['k', 'tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:gray']
STATIONS = ('land', 'marine')

# (title, kind, spec): kind 'era5' -> field name; kind 'obs' -> (obs names pooled, level or None)
FIGURES = {
    'fcst_era5_upper.png': ('Forecast RMSE vs ERA5', [
        ('z500 (m)', 'era5', 'z500'), ('q850 (g/kg)', 'era5', 'q850'), ('t850 (K)', 'era5', 't850'),
        ('250 hPa wind speed (m/s)', 'era5', 'ws250')]),
    'fcst_era5_sfc.png': ('Forecast RMSE vs ERA5, near surface', [
        ('10 m wind speed (m/s)', 'era5', 'ws10'), ('2 m temperature (K)', 'era5', 't2m'),
        ('2 m specific humidity (g/kg)', 'era5', 'q2m')]),
    'fcst_obs_upper.png': ('Forecast RMS(O - F) vs observations', [
        ('surface pressure (hPa)', 'obs', (['ps'], None)), ('raob q850 (g/kg)', 'obs', (['raob_q'], 850.)),
        ('raob t850 (K)', 'obs', (['raob_t'], 850.)), ('raob t500 (K)', 'obs', (['raob_t'], 500.)),
        ('raob 250 hPa wind speed (m/s)', 'obs', (['raob_ws'], 250.))]),
    'fcst_obs_sfc.png': ('Forecast RMS(O - F) vs surface observations', [
        ('10 m wind speed, land + marine stations (m/s)', 'obs', ([f'sfc_{n}_ws' for n in STATIONS], None)),
        ('10 m wind speed, ASCAT (m/s)', 'obs', (['sfc_scat_ws'], None)),
        ('2 m temperature, land + marine stations (K)', 'obs', ([f'sfc_{n}_t' for n in STATIONS], None)),
        ('2 m specific humidity, land + marine stations (g/kg)', 'obs', ([f'sfc_{n}_q' for n in STATIONS], None))]),
}
EXTRA_CSV = [(f'sfc_{n}_{v}', 'obs', ([f'sfc_{n}_{v}'], None)) for n in ('land', 'marine', 'scat')
             for v in ('t', 'q', 'ws', 'uv') if not (n == 'scat' and v in 'tq')]
EXTRA_CSV += [('vw250', 'era5', 'vw250'), ('vw10', 'era5', 'vw10'), ('msl', 'era5', 'msl'),
              ('raob_uv250', 'obs', (['raob_uv'], 250.))]

files = {lab: {os.path.basename(f)[:13]: f for f in glob.glob(f'{args.skill_dir}/{lab}/*.nc')} for lab, _ in labels}
for lab, f in files.items():
    assert f, f'no forecast_skill output in {args.skill_dir}/{lab}'
dates = sorted(d for d in set.intersection(*[set(v) for v in files.values()]) if args.start <= d <= args.end)
assert dates, 'no common init dates'
print(f'{len(dates)} common init dates, {dates[0]} .. {dates[-1]}')
data = {lab: [xr.open_dataset(files[lab][d]) for d in dates] for lab, _ in labels}
lead_days = data[labels[0][0]][0].lead_hours.values / 24.


def curve(lab, kind, spec):
    """(rmse, n per verification) as (lead, region) arrays; n is None for ERA5"""
    dss = data[lab]
    if kind == 'era5':
        return np.sqrt(np.mean([ds[f'rmse_{spec}'].values ** 2 for ds in dss], 0)), None
    names, level = spec
    n = s2 = 0.
    for name in names:
        for ds in dss:
            a, b = ds[f'obs_{name}_n'], ds[f'obs_{name}_s2']
            if level is not None:
                dim = [d for d in a.dims if d.startswith('level_')][0]
                a, b = a.sel({dim: level}), b.sel({dim: level})
            n, s2 = n + a.values, s2 + b.values
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt(s2 / n), n / len(dss)


for fname, (title, rows) in FIGURES.items():
    fig, axes = plt.subplots(len(rows), len(REGIONS), figsize=(16, 3.0 * len(rows)), sharex=True, squeeze=False)
    for i, (ylabel, kind, spec) in enumerate(rows):
        for j, region in enumerate(REGIONS):
            ax = axes[i, j]
            nobs = None
            for (lab, legend), color in zip(labels, COLORS):
                y, n = curve(lab, kind, spec)
                ax.plot(lead_days, y[:, j], color=color, label=legend, marker='o', ms=2.5)
                nobs = n if n is not None else nobs
            ax.set_title(f'{region}: {ylabel}', fontsize=9)
            if nobs is not None:
                ax.text(0.02, 0.97, f'n per time ~{np.nanmean(nobs[:, j]):,.0f}', transform=ax.transAxes,
                        fontsize=7, va='top', color='0.35')
            ax.grid(alpha=0.3)
            if i == len(rows) - 1:
                ax.set_xlabel('forecast lead (days)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{title}: {len(dates)} daily forecasts, {dates[0][:10]} .. {dates[-1][:10]}', y=1.0)
    fig.tight_layout()
    fig.savefig(f'{args.out}/{fname}', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', f'{args.out}/{fname}')

with open(f'{args.out}/forecast_skill.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['label', 'kind', 'var', 'region'] + [f'{int(h)}h' for h in lead_days * 24])
    specs = [(lab_y, kind, spec) for _, rows in FIGURES.values() for lab_y, kind, spec in rows] + EXTRA_CSV
    for lab, legend in labels:
        for name, kind, spec in specs:
            y, n = curve(lab, kind, spec)
            for j, region in enumerate(REGIONS):
                w.writerow([legend, kind, name, region] + [f'{x:.4f}' for x in y[:, j]])
            if n is not None:
                w.writerow([legend, 'n_obs', name, 'Global'] + [f'{x:.0f}' for x in n[:, 0]])
PROFILE_VARS = [('t', 'temperature (K)'), ('q', 'specific humidity (g/kg)'), ('uv', 'vector wind (m/s)')]


def profile(lab, v, region, groups=None):
    """(rms, mean, n per verification) against pressure at --profile-lead, pooled over dates
    and the platform groups `groups` (default all)"""
    tot = 0.
    for ds in data[lab]:
        x = ds[[f'conv_{v}_n', f'conv_{v}_s1', f'conv_{v}_s2']].sel(lead_hours=args.profile_lead, region=region)
        if groups is not None:
            x = x.sel(conv_group=groups)
        tot = tot + x.sum('conv_group').to_array().values
    n, s1, s2 = tot
    with np.errstate(invalid='ignore', divide='ignore'):
        rms, mean = np.sqrt(s2 / n), s1 / n
    ok = n >= 30
    return np.where(ok, rms, np.nan), np.where(ok & (v != 'uv'), mean, np.nan), n / len(data[lab])


ds0 = data[labels[0][0]][0]
plev, groups = ds0.conv_level.values, list(ds0.conv_group.values)
for fname, cols, title in (('fcst_obs_profile.png', [(r, None) for r in REGIONS], 'all upper-air conventional obs'),
                           ('fcst_obs_profile_platform.png', [('Global', [g]) for g in groups], 'Global, by platform')):
    fig, axes = plt.subplots(len(PROFILE_VARS), len(cols), figsize=(4 * len(cols), 4.2 * len(PROFILE_VARS)),
                             sharey='row', squeeze=False)
    for i, (v, xlabel) in enumerate(PROFILE_VARS):
        for j, (region, grp) in enumerate(cols):
            ax = axes[i, j]
            for (lab, legend), color in zip(labels, COLORS):
                rms, mean, n = profile(lab, v, region, grp)
                ax.plot(rms, plev, color=color, marker='o', ms=3, label=legend)
                ax.plot(mean, plev, color=color, ls='--', lw=1)
            ax.axvline(0, color='0.6', lw=0.8)
            ax.set_yscale('log')
            top = 180 if v == 'q' else 40  # humidity: 200 hPa and below
            ticks = [p for p in (1000, 850, 700, 500, 300, 200, 100, 50) if p > top]
            ax.set_ylim(1050, top)
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(p) for p in ticks])
            ax.minorticks_off()
            ax.set_title(f'{region if grp is None else grp[0]}: {xlabel}', fontsize=9)
            ax.text(0.98, 0.02, f'n per time ~{np.nansum(n):,.0f}', transform=ax.transAxes, fontsize=7,
                    ha='right', color='0.35')
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel('pressure (hPa)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{args.profile_lead} h forecast fit to {title}: RMS(O - F) solid, mean(O - F) dashed; '
                 f'{len(dates)} daily forecasts {dates[0][:10]} .. {dates[-1][:10]}', y=1.0)
    fig.tight_layout()
    fig.savefig(f'{args.out}/{fname}', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', f'{args.out}/{fname}')

print('wrote', f'{args.out}/forecast_skill.csv')

with open(f'{args.out}/forecast_skill_profile.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['label', 'kind', 'var', 'region/group'] + [f'{p:.0f}hPa' for p in plev])
    for lab, legend in labels:
        for v, _ in PROFILE_VARS:
            for region, grp in [(r, None) for r in REGIONS] + [('Global', [g]) for g in groups]:
                rms, mean, n = profile(lab, v, region, grp)
                name = region if grp is None else grp[0]
                for kind, y in ((f'profile_rms_{args.profile_lead}h', rms), (f'profile_mean_{args.profile_lead}h', mean),
                                (f'profile_n_{args.profile_lead}h', n)):
                    w.writerow([legend, kind, v, name] + [f'{x:.4f}' for x in y])
print('wrote', f'{args.out}/forecast_skill_profile.csv')
