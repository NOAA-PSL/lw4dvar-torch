"""ATMS weighting functions for the deck (notes/integrate_atms_obs.md):

  left   clear-sky nadir weighting functions dtau/dln(p) (normalized) from
         pyrtlib (Rosenkranz line-by-line absorption) on the US standard
         atmosphere, averaged over each channel's passband (sidebands sampled
         across their bandwidths); channels 1-15 and 16-22

Writes figures/atms_e1/fig_weighting_functions.png. Needs pyrtlib on PYTHONPATH.
"""

import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pyrtlib.climatology import AtmosphericProfiles as atmp  # noqa: E402
from pyrtlib.utils import ppmv2gkg, mr2rh  # noqa: E402
from pyrtlib.weighting_functions import WeightingFunctions  # noqa: E402


OUT = 'figures/atms_e1/fig_weighting_functions.png'
F9 = 57.290344
# channel -> list of (center GHz, bandwidth GHz) sub-passbands
ATMS = {1: [(23.8, .27)], 2: [(31.4, .18)], 3: [(50.3, .18)], 4: [(51.76, .4)],
        5: [(53.596 - .115, .17), (53.596 + .115, .17)], 6: [(54.4, .4)], 7: [(54.94, .4)], 8: [(55.5, .33)],
        9: [(F9, .33)], 10: [(F9 - .217, .078), (F9 + .217, .078)],
        11: [(F9 + s1 * .3222 + s2 * .048, .036) for s1 in (-1, 1) for s2 in (-1, 1)],
        12: [(F9 + s1 * .3222 + s2 * .022, .016) for s1 in (-1, 1) for s2 in (-1, 1)],
        13: [(F9 + s1 * .3222 + s2 * .010, .008) for s1 in (-1, 1) for s2 in (-1, 1)],
        14: [(F9 + s1 * .3222 + s2 * .0045, .003) for s1 in (-1, 1) for s2 in (-1, 1)],
        16: [(88.2, 2.)], 17: [(165.5, 3.)],
        18: [(183.31 - 7, 2.), (183.31 + 7, 2.)], 19: [(183.31 - 4.5, 2.), (183.31 + 4.5, 2.)],
        20: [(183.31 - 3, 1.), (183.31 + 3, 1.)], 21: [(183.31 - 1.8, 1.), (183.31 + 1.8, 1.)],
        22: [(183.31 - 1, .5), (183.31 + 1, .5)]}
NSUB = 5  # frequencies sampled across each sub-passband

z, p, d, t, md = atmp.gl_atm(atmp.US_STANDARD)
gkg = ppmv2gkg(md[:, atmp.H2O], atmp.H2O)
rh = mr2rh(p, t, gkg)[0] / 100
# fine profile, pressure interpolated in ln(p) (pyrtlib interpolates p linearly in z, which kinks dlnp)
zf = np.arange(0, z.max() + 0.125, 0.125)
pf, tf, rhf = np.exp(np.interp(zf, z, np.log(p))), np.interp(zf, z, t), np.interp(zf, z, rh)
wf = WeightingFunctions(zf, pf, tf, rhf, value_prof_interpolation=0.125)
freqs, owner = [], []
for c, bands in ATMS.items():
    for f0, bw in bands:
        for f in f0 + bw * (np.arange(NSUB) + 0.5 - NSUB / 2) / NSUB:
            freqs.append(f)
            owner.append(c)
wf.frequencies = np.array(freqs)
w = wf.generate_wf()  # (n_freq, n_layer), dtau/dz-like weights on wf.zinterp
pz = wf.pinterp[:w.shape[1]]
owner = np.array(owner)
# per channel: mean over its frequencies, then per unit ln(p), normalized to max 1
lnp = np.log(pz)
dlnp = -np.gradient(lnp)
wch = {}
for c in ATMS:
    m = w[owner == c].mean(0) / dlnp
    wch[c] = m / np.nanmax(m)
    print(f'ch {c:2d}: peak {pz[np.nanargmax(m)]:7.1f} hPa', flush=True)

plt.rcParams.update({'font.size': 11})
fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1]))
style = {c: ('tab:red', 2.2) if c in (5, 6, 7) else ('tab:orange', 1.6) if c in (8, 9, 10, 11) else ('0.6', 1.0)
         for c in ATMS}
for c in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
    col, lw = style[c]
    axes[0].plot(wch[c], pz, color=col, lw=lw)
    i = np.nanargmax(wch[c])
    axes[0].text(min(wch[c][i] + 0.02, 1.02), pz[i], str(c), color=col, fontsize=10, va='center')
for c in [16, 17, 18, 19, 20, 21, 22]:
    axes[1].plot(wch[c], pz, color='tab:blue', lw=1.2)
    i = np.nanargmax(wch[c])
    axes[1].text(min(wch[c][i] + 0.02, 1.02), pz[i], str(c), color='tab:blue', fontsize=10, va='center')
axes[0].set_title('Temperature channels 1-14 (physics, pyrtlib)')
axes[1].set_title('Window and humidity channels 16-22')
for ax in axes:
    ax.set_yscale('log')
    ax.set_ylim(1050, 1)
    ax.set_yticks([1000, 700, 500, 300, 200, 100, 50, 20, 10, 5, 2, 1])
    ax.set_yticklabels([1000, 700, 500, 300, 200, 100, 50, 20, 10, 5, 2, 1])
    ax.grid(alpha=0.3)
    ax.set_xlabel('normalized weight per ln(p)')
axes[0].set_ylabel('pressure (hPa)')
axes[0].plot([], [], color='tab:red', lw=2.2, label='assimilated (5-7)')
axes[0].plot([], [], color='tab:orange', lw=1.6, label='monitored (8-11)')
axes[0].plot([], [], color='0.6', lw=1, label='not used')
axes[0].legend(loc='lower right', fontsize=9)
fig.suptitle('ATMS clear-sky weighting functions, nadir, US standard atmosphere', y=1.0)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print('wrote', OUT)
