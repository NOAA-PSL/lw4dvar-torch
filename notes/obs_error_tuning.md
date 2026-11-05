# Tuning observation errors

Started 2026-10-20. Tool: `obs_error_diag.py RUN_DIR OUT_PREFIX [FIRST_CYCLE] [LAST_CYCLE]`
(untracked; per group: assigned sigma, O-B / O-A rms, Desroziers sigma, share beyond the Huber
threshold, normalized O-B / O-A variance; plus ATMS inter-channel Desroziers matrices).
Output: `figures/obs_error/` (`winoc_cld.csv`, `accept_a_c2_10.csv`).

## First diagnosis: `winoc_cld`, cycles 2-10

**Desroziers doesn't work as-is here, because every ob is reused.** With a 5-day window and
12-h cycling, an ob is assimilated in up to 10 overlapping windows. Its O-B in a later window is
against a background that has already fitted it, so O-B is nearly flat with lead time
(ATMS ch 5 clear ocean: 0.36, 0.36, 0.38, 0.44 K for leads 0-24, 24-48, 48-96, 96-120 h), and
O-A ~ O-B everywhere. Desroziers then returns ~rms(O-B), and normalized O-B variances below 1
don't prove sigma is too large on their own.

**Fresh innovations are the clean signal.** Only the newest 12 h of each window (leads 108-120 h)
holds obs no earlier window has seen. Their O-B variance bounds sigma_o^2 from above (it is
sigma_o^2 + HBH^T). 96-120 h O-B rms vs the assigned sigma:

| group | assigned sigma | O-B rms, 96-120 h | verdict |
|---|---|---|---|
| ATMS ch 5 clear / land / cloudy ocean | 0.87 / 1.83 / 1.67 | 0.44 / 1.16 / 0.83 | sigma at least 2x too large |
| ATMS ch 6 | 0.39 / 0.77 / 0.64 | 0.25 / 0.53 / 0.40 | ~1.5x too large |
| ATMS ch 7 | 0.20 / 0.24 / 0.23 | 0.19 / 0.23 / 0.23 | consistent |
| ATMS ch 18 | 3.60 / 6.75 / 10.2 | 1.91 / 3.11 / 4.16 | 2-2.5x too large |
| ATMS ch 19 | 2.76 / 4.38 / 6.19 | 1.98 / 2.59 / 3.41 | ~1.5-1.8x too large |
| ATMS ch 22 | 1.87 / 1.86 / 2.43 | 2.29 / 2.34 / 2.72 | consistent |
| ATMS window ch 1, 2, 3 (state on, ocean) | 4.5-19 | 2.2-7.2 | 1.5-3x too large |
| ps | 1.64 hPa | 1.23 | >= 1.3x too large |
| raob t 850 / 500 / 250 | 1.0 K | 1.44 / 1.01 / 1.17 | plausible, can't tell |
| raob u 850 / 500 / 250 | 2.0 m/s | 2.94 / 3.17 / 3.56 | can't tell (background error likely large) |

- **Huber isn't switching obs off.** Beyond 1.5 sigma at x_b, ATMS ch 5, 6, 18, 19 and the window
  channels have 0.5-6% (Gaussian: 13.4%), and ch 7, 20-22 have 6-15%. Huber only applies to ATMS;
  raobs and ps are quadratic with a background check.
- **Why ATMS sigma is too large isn't known yet.** The table came from `atms_cloud_error.py`
  on top of the decoder bundle's sigma_o (ch 5: 0.70 K), which is itself larger than our fresh
  O-B (0.44 K). Check what the bundle sigma_o was measured against (inputs, bias correction).
- The ATMS inter-channel Desroziers correlations (`*_atms_corr.png`) are affected by the reuse
  too, so they aren't trusted yet.

## Plan

1. **Outer objective = fresh innovations.** Score sigma changes on O-B at 108-120 h (obs no
   window has used) plus ERA5 RMSE. This is also the held-out loss for learning sigma by backprop
   later.
2. **10-cycle sweep from `winoc_cld`:** ATMS sigma x 0.5 for ch 5, 6, 18, 19 and the window
   channels (ch 7, 20-22 unchanged); ps sigma x 0.75.
3. Redo Desroziers on fresh obs only, using the O-A from the window that first assimilates them.

## Sweep 1: sigma x 0.5 (2026-10-20, H100, 10 cycles from `winoc_cld`)

Arms: `winoc_cld_h100` (control), `winoc_cld_sig05` (ATMS sigma x 0.5 for ch 1-6, 16-19),
`winoc_cld_sig05_ps075` (plus `psobs.error_scale: 0.75`). Scores in `figures/sigsweep_10cyc/`.
The MI300X `winoc_cld` vs the H100 control gives the platform noise: <= 0.4% on the window means.

| window-mean RMSE vs ERA5, global | ctl | sig05 | sig05_ps075 |
|---|---|---|---|
| q850 (g/kg) | 1.014 | **0.986** (-2.8%; tropics -3.5%) | 0.986 |
| z500 (m) | **6.000** | 6.218 (+3.6%; tropics +5.8%) | 6.191 |
| ws250 (m/s) | **2.807** | 2.838 (+1.1%) | 2.862 |

Fresh O-B (leads 108-120 h, obs the control used):
- the down-weighted-sigma channels improve 1-3% over ocean (ch 1, 3, 5, 6 clear), and are flat over land;
- ch 7, 20, 22 get 2-4% worse, since they now have less relative weight;
- raobs and ps are neutral to 1% worse.

The smaller sigma also rejects about 1.6% more ch 1 and ch 19 obs at the background check.

Reading:
- Tightening the humidity and window channels helps moisture but costs mass and wind.
- ch 5 and 6 are lower-tropospheric temperature channels, the likeliest source of the z500
  cost.
- ps x 0.75 doesn't help: drop it.

Next: split the scale, humidity and window channels (1-4, 16-19) x 0.5 with ch 5, 6 unchanged.

## Sweep 2: humidity + window channels only (`winoc_cld_sig05hw`, 2026-10-20)

sigma x 0.5 for ch 1-3, 16-19; ch 4-7, 20-22 unchanged. `figures/sigsweep2_10cyc/`.

| global window-mean change vs H100 control | sig05 (ch 1-6, 16-19) | sig05hw (ch 1-3, 16-19) |
|---|---|---|
| q850 | -2.8% | **-3.0%** (tropics -3.6%) |
| z500 | +3.6% | +3.2% (tropics +0.1%, SH +4.6%) |
| ws250 | +1.1% | +1.5% |

- The hypothesis that ch 4-6 cause the z500 cost is only half right. Leaving them alone removes
  the tropical z500 cost, but the extratropical cost stays, so it comes from the humidity and
  window channels.
- Fresh raob O-B is where it shows: SH t250 +7.5%, SH u250 +4.0%, SH t/u/v at other levels
  +1.5-2.4%, while NH is within 1% (u500 NH -2.7%). The SH has few raobs and the ATMS weight
  dominates there.
- Fresh ATMS O-B: ch 1 and 18 clear ocean improve 2-4%, ch 22 is 3% worse.

Reading: giving the humidity channels more weight through a shared latent control pulls the
mass/wind field in the SH. Halving sigma on channels whose errors are likely correlated
(18-22, the decoder) double-counts them. Options:
1. A middle scale, x 0.75, for the same channels.
2. A correlated R for 18-22 instead of a smaller diagonal: first needs a clean
   (fresh-obs) estimate of the inter-channel correlations.
3. Longer runs before deciding: 10 cycles with few SH raobs is a small sample.

## Inter-channel correlation from fresh obs (2026-10-20, `winoc_cld_h100`, cycles 2-10)

Obs at leads 108-120 h, used in all of ch 5-7, 18-22. Matrices in
`figures/obs_error/fresh_atms_cov_h100.npz`.
- **Fresh Desroziers doesn't separate R from B either.** At the window end the analysis barely
  fits the obs (O-A ~ 0.9 O-B), so the fresh Desroziers sigma is 0.85-0.92 x the fresh O-B std,
  and the two correlation matrices agree to within 0.03.
- **Humidity channels 18-22 are strongly correlated in O-B**, about the same in every scene:
  adjacent pairs 0.93-0.96, ch 18-20 0.79-0.85, ch 18-22 0.43-0.52. ch 5-7 vs 18-22 are near 0
  over ocean and 0.1-0.5 over land.
- Background humidity errors are vertically correlated too, so this can't yet be put on R.
  The decoder's own error correlation, measured in radianced on held-out obs with ERA5 or replay
  inputs, would settle it.
- Either way, 18-22 carry about two independent pieces of information, not five. A diagonal R
  with smaller sigma counts them five times, which fits the SH mass/wind cost in sweep 2.

## Decoder error on ERA5 pairs (`atms_decoder_error.py`, 2026-10-20)

120 cycles, Jan-Apr 2015 (every 2nd cycle), radianced ERA5-paired S-NPP files. The decoder is run
as in the 4D-Var: Obs_Time 0, z fitted to the window channels by Gauss-Newton over land, snow and
sea ice, z = 0 over ocean. d = (y_o - BC) - H(x_ERA5), trimmed at 5 robust std.
Output: `figures/obs_error/decoder_era5.{json,log,_corr.png}`. The replay GSI O-F on the same
obs (CRTM on the 6-h replay forecast) is computed alongside for comparison.

| clear ocean, K | ch 5 | 6 | 7 | 18 | 19 | 20 | 21 | 22 |
|---|---|---|---|---|---|---|---|---|
| std(y - H(ERA5)) | 0.31 | 0.20 | 0.18 | 1.02 | 1.12 | 1.26 | 1.42 | 1.55 |
| replay GSI O-F std | 0.35 | 0.15 | 0.12 | 1.08 | 1.10 | 1.18 | 1.32 | 1.45 |
| assigned in `winoc_cld` | 0.87 | 0.39 | 0.20 | 3.62 | 2.76 | 2.23 | 1.95 | 1.87 |
| fresh 4D-Var O-B (AIFS) | 0.45 | 0.25 | 0.20 | 1.92 | 1.98 | 2.08 | 2.23 | 2.40 |

- **Why the assigned sigma is too large.** The bundle sigma_o (ch 5: 0.70 K, ch 18: 3.08 K) is
  the decoder's validation RMSE over all scenes, with land, snow and cloud mixed in.
  `atms_cloud_error.py` multiplies it by ratios relative to clear ocean, so clear-ocean obs get
  the all-scene error. The clear-ocean base should be the clear-ocean std: 2.8x smaller for ch 5
  and 3.5x for ch 18. The ratio table itself is self-consistent.
- **The decoder is as good as CRTM on the replay.** It matches the replay GSI O-F within ~10% in
  every scene for ch 18-22 and ch 5 over ocean. Over land and snow it is better (z fit; GSI
  doesn't use those obs).
- **The correlation is the same for both operators.** Adjacent humidity channels correlate
  0.91-0.95 (ch 18 vs 22: 0.45) in every ocean scene, and the replay GSI O-F gives the same to
  within 0.02. Two different operators against two different backgrounds giving the same
  structure means it isn't decoder error. It is common to both: obs-side representativeness
  (footprint, the 3-h time offset) and the vertically correlated humidity error that ERA5 and the
  replay share. The first belongs in our R. The second doesn't, but it is part of what the
  4D-Var sees anyway. ch 5-6 correlate 0.56 over ocean and 0.8-0.86 over land and snow (skt and
  emissivity).
- **Practical R:** sigma = the clear-ocean std above, times the scene and cloud ratios already
  in the table. Use the scene correlation matrix for 18-22 (and 5-7), reconditioned to a capped
  condition number before inversion, as the operational centres do.

## Sweep 3: the middle scale (`winoc_cld_sig075hw`, 2026-10-21)

sigma x 0.75 for ch 1-3, 16-19 (ch 4-7, 20-22 unchanged), 10 cycles on H100.
`figures/sigsweep3_10cyc/`.

| window-mean change vs H100 control | sig05hw (x 0.5) | sig075hw (x 0.75) |
|---|---|---|
| q850 Global | -3.0% | -1.6% (tropics -1.7%) |
| z500 Global | +3.2% | +1.3% (SH +2.4%, NH +0.7%, tropics -0.5%) |
| ws250 Global | +1.5% | +0.4% |

- Fresh raob O-B in the SH is mostly back to the control's: t250 +1.3% (vs +7.5% at x 0.5),
  u250 +0.2% (vs +4.0%). The rest of SH t/u/v is within +1.4%.
- Fresh ATMS O-B: ch 22 is still 2% worse everywhere. ch 1 and 18 have lost most of their gain.
  ps fresh O-B is 1-2% worse.

Reading: moving along the diagonal scale only slides the trade-off. Halfway in sigma gives about
half the q850 gain and 40% of the z500 cost. No diagonal scale of these channels gains moisture
without costing SH mass. That points to the correlated-R test (see "Decoder error on ERA5 pairs"
above) rather than more diagonal tuning.

## Sweep 4: decoder sigma base, diagonal vs correlated R (2026-10-21)

New option `observations.atms.corr_r` (+ `corr_cond_max`, default 50): per-scene inter-channel
correlation of the assimilated channels from `decoder_era5.json`, reconditioned by
C' = (C + dI)/(1 + d) so cond <= 50 (raw cond 430-760). Obs used in all assimilated channels (99%)
are whitened, w = L^-1 (d / sigma) with C' = L L^T, and Huber acts on w; the rest keep a diagonal R.
A uniform 1-sigma departure costs Jo 2.75 instead of 8 in clear ocean. Window channels stay diagonal.

Both arms set the clear-ocean sigma base to the decoder std on ERA5 pairs for all 14 channels
(e.g. ch 5 0.31, ch 18 1.02, ch 1 2.05), keeping the scene and cloud ratios:
- `winoc_cld_decsig` (24266289): diagonal R.
- `winoc_cld_decsig_corr` (24266290): correlated R.

Expected: decsig overweights 18-22 (bigger z500 cost than sweep 2); decsig_corr keeps the q850 gain
without it. Caveat: sigma is now about half the fresh O-B, so Huber c = 1.5 and bg_check 4 act on more obs.

Results (`figures/sigsweep4_10cyc/`), global window-mean change vs the H100 control:

| | sig05hw | decsig (diag R) | decsig_corr (corr R) |
|---|---|---|---|
| q850 | -3.0% | -2.1% | **-3.1%** (NH -3.2, tropics -3.5, SH -2.4) |
| z500 | +3.2% | +3.6% | **+1.0%** (NH -0.6, SH +1.6, tropics +5.6) |
| ws250 | +1.5% | +1.4% | **+0.2%** |

- Correlated R does what it should outside the tropics: the best q850 of any arm, and the NH
  z500 and fresh NH raob O-B now improve (u500 -2.6%, t500 -1.5%).
- Tropical z500 is +4.6% (diag) / +5.6% (corr) in both arms; sweep 1 shows the same (+5.8%) when ch 5-6
  sigma shrinks. It's the lower-tropospheric temperature channels: their base sigma drops 0.87 -> 0.31
  (8x weight), and their correlation with 18-22 is weak, so corr R doesn't help there.
- SH: fresh raob t850 +4.5%, t250 +3.4%, v250 +4.5% in both arms; ps fresh O-B 1-2% worse.
- The share of obs with |O-B|/sigma > 1.5 goes from 1-13% (ctl) to 13-19%, close to the Gaussian
  13.4%: the new sigma is consistent in scale. 1-2% fewer obs pass the 4-sigma bg check.

Next: decsig_corr with ch 5-6 (and 7) back at their old sigma, to test whether ch 5-6 cause the
tropical z500 cost.

## Sweep 5: correlated R with ch 5-7 at the bundle sigma (`winoc_cld_decsig_corr_t0`, 2026-10-21)

`figures/sigsweep5_10cyc/`. Change vs the H100 control, window means:

| | decsig_corr | decsig_corr_t0 (ch 5-7 bundle sigma) |
|---|---|---|
| q850 Global | -3.1% | -3.4% |
| z500 Global | +1.0% | +1.3% |
| z500 Tropics / SH / NH | +5.6% / +1.6% / -0.6% | **-1.1%** / +3.9% / -0.4% |
| ws250 Global | +0.2% | +0.7% |

- Tight ch 5-6 cause the tropical z500 cost, as sweep 1 suggested, but they also help SH z500.
  The two arms trade about 0.15-0.2 m in each region, within 10-cycle noise for regional z500.
- The fresh SH raob cost (t850 +4.5-4.9%, v250 +3-4.5%, t250 +3%) is the same in both, so it comes
  from the humidity or window channels, not ch 5-7. The likeliest suspect is the 531k ocean window
  radiances per window on the state, still with a diagonal R and a sigma now about half the old one.

Decision: the month run uses the full decoder sigma (decsig_corr, all sigma from one source) on
WC24 q075 + all-sky window channels: `1mo_wc24_q075_corr` (24292278-81, chained).
Next 10-cycle candidate: window channels back to their old sigma, or correlated R for them too.

_Run outputs of the superseded experiments (3mo_ps_atms, early 1mo ATMS variants, wcsweep/wc*_a, wc24_q025, win4d-10d, prior/cprior, early 10-cycle raob tests, aborted 6mo non-tsens runs) were deleted on 2026-10-21; their scores remain in this note, figures/ and output/forecast_skill/._

## Sweep 6: correlated R with the window channels at the bundle sigma (`winoc_cld_decsig_corr_w0`, 2026-10-22)

`figures/sigsweep6_10cyc/`. Change vs the H100 control, window means vs ERA5:

| | decsig_corr | corr_t0 (ch 5-7 old) | corr_w0 (window old) |
|---|---|---|---|
| q850 Global | -3.1% | -3.4% | -0.7% |
| z500 Global | +1.0% | +1.3% | **-1.1%** |
| z500 NH / SH / Tropics | -0.6 / +1.6 / +5.6% | -0.4 / +3.9 / -1.1% | **-2.5 / -2.7** / +9.8% |
| ws250 Global | +0.2% | +0.7% | **-1.1%** (SH and tropics -1.5%) |

- The window channels carry most of the q850-vs-ERA5 gain, and their halved sigma causes the
  extratropical mass/wind cost. With the old window sigma, corr R is the first arm that improves
  z500 and ws250 on the control.
- Fresh raob O-B agrees: NH t, u, v improve 1-3%, SH v850 -1.3%, ps neutral. Fresh raob q850/q500
  is neutral to better in w0 but worse in decsig_corr (SH q850 +1.7%, NH q500 +2.2%). So the window
  channels' q850 gain vs ERA5 isn't confirmed by raobs, which are mostly over land.
- Tight ch 5-6 still cost tropical z500 (+9.8%). SH raob t250 +3.6% and u250 +3.3% remain.

Next: `winoc_cld_decsig_corr_w0t0` (24307365): ch 5-7 and the window channels at the bundle sigma,
so only 18-22 use the decoder sigma with corr R. The month run `1mo_wc24_q075_corr` (full decoder
sigma) had already started and was left running.

## Sweep 7: decoder sigma for ch 18-22 only (`winoc_cld_decsig_corr_w0t0`, 2026-10-22) -- leading config

ch 18-22 clear-ocean sigma base = decoder std (1.02, 1.12, 1.26, 1.42, 1.55 K); ch 1-7, 16, 17
at the bundle sigma; per-scene correlated R for ch 5-7, 18-22 (cond 50); scene and cloud ratios
from oberr_table_w.json. `figures/sigsweep7_10cyc/`. Change vs the H100 control, vs ERA5:

| | corr_w0 | **corr_w0t0** |
|---|---|---|
| q850 Global / NH / SH / Tropics | -0.7 / -0.9 / -0.9 / -0.6% | -0.9 / -0.6 / -1.1 / -0.8% |
| z500 Global / NH / SH / Tropics | -1.1 / -2.5 / -2.7 / +9.8% | -1.1 / -1.0 / -1.1 / **-1.3%** |
| ws250 Global / NH / SH / Tropics | -1.1 / -0.1 / -1.5 / -1.5% | -1.4 / -0.6 / -1.4 / -2.0% |

- The first arm that improves every variable in every region on the control.
- Fresh raob O-B: none worse than +1.3%. The SH t250 / u250 cost of the earlier arms is gone
  (-1.5% / +1.2%), NH u/v 1-2% better, raob q850 and SH/tropics q500 about 1% better, ps neutral.
- ch 22 fresh O-B +2.4% (more weight on 18-21 through the correlated R).

Month run: `1mo_wc24_q075_corrhum` (24313740-43, chained), WC24 q075 + all-sky window
channels + this ATMS block. The full-decoder-sigma month run `1mo_wc24_q075_corr` was cancelled
after chunk 1 (2015-01-01 .. 01-08 in `output/lwatms_1mo_wc24_q075_corr/`).

### Month run `1mo_wc24_q075_corrhum` vs `1mo_wc24_q075_wincld` (2026-10-22)

60 cycles, 2015-01-01 .. 01-30. Analyses vs ERA5 (window mean, `figures/corrhum_1mo/`): z500
-2.2% (NH -1.9, SH -2.4, tropics -1.0), ws250 -1.1%, q850 -0.6%; at the window end z500 -1.9%.

Forecast skill (30 daily 5-day forecasts, 01-06 .. 02-04, `figures/fcst_skill_corrhum/`), change vs wincld:

| | 0 h | 24 h | 48 h | 72 h | 120 h |
|---|---|---|---|---|---|
| z500 Global | -1.9% | -0.1% | +0.4% | -0.1% | -3.1% |
| z500 NH / SH | -2.4 / -1.9% | -1.7 / +0.8% | -1.2 / +1.5% | -2.6 / +1.8% | -5.1 / -1.4% |
| q850 Global | -0.8% | -0.9% | -0.5% | -0.3% | -0.2% |
| t850 Global | -1.4% | -1.0% | -0.5% | +0.3% | -1.3% |
| raob t500 / uv250 speed | -0.6 / +0.1% | -0.3 / -0.9% | -0.9 / -1.1% | -1.7 / -1.7% | +0.1 / +1.3% |

The analysis gain is clear and consistent; in the forecasts it is small, mostly neutral to
1-2% better (NH z500 better at every lead, SH slightly worse at 1-3 days). 30 forecasts don't
resolve differences of about 1%. Adopted as the new ATMS ob-error baseline.

## Best experiment: `lwatms_1mo_wc24_q075_corrhum` (recorded 2026-10-22)

The current best configuration of the whole system, superseding `1mo_wc24_q075_wincld`:
weak constraint (jumps every 24 h, Q/B 0.75) + ps + raobs (fixed q error) + ATMS arm A with
the window channels 1-4, 16, 17 acting on the state over clear and cloudy ocean, decoder
sigma for ch 18-22 and per-scene correlated R for ch 5-7, 18-22 (cond 50). Configs:
`config_lwatms_1mo_wc24_q075_corrhum{,_c2..c12}.yml`.

Global forecast RMSE vs ERA5 (30 forecasts, 01-06 .. 02-04), 0 h / 24 h / 120 h:

| run | z500 (m) | t850 (K) | q850 (g/kg) | ws250 (m/s) |
|---|---|---|---|---|
| WC clear-ocean (`wc24_q075`) | 8.14 / 11.54 / 46.32 | 0.86 / 0.97 / 2.36 | 1.13 / 1.24 / 1.86 | 2.82 / 3.59 / 8.48 |
| WC wincld | 8.17 / 11.39 / 46.56 | 0.81 / 0.94 / 2.36 | 0.96 / 1.11 / 1.84 | 2.81 / 3.56 / 8.38 |
| **WC wincld + corr R (best)** | **8.01 / 11.38 / 45.13** | **0.80 / 0.93 / 2.33** | **0.95 / 1.10 / 1.84** | **2.77 / 3.55 / 8.30** |
| ERA5 IC | 0 / 3.57 / 27.44 | 0 / 0.55 / 1.64 | 0 / 0.73 / 1.57 | 0 / 2.03 / 6.23 |

**3-month extension** (submitted 2026-10-22): chunks c5-c12 continue the same run directory
from the 2015-01-30T12 analysis through 2015-03-31T12 (120 cycles, 15 per chunk, afterok
chain), jobs 24367154-24367161. Score with `NCYC=180` and forecast skill over Jan 6 - Mar 27.

## Sweep 8: correlated R for the window channels (`winoc_cld_decsig_corr_w0t0_wcorr`, 2026-10-22)

New option `observations.atms.corr_r_window: True`: a separate per-scene correlated-R block for
the emis_channels 1-4, 16, 17 (same json, same cond-50 reconditioning), applied only to
window_state obs (clear and cloudy ocean, clw <= cloud_max) used in all six; z-only obs keep the
diagonal R. No cross-block correlation, so ch 5-7, 18-22 are unchanged from w0t0.
Matrices: `figures/obs_error/corr_r_matrices.png`. Over ocean, ch 1-4, 16 correlate 0.7-0.96 and
the six window channels have ~1.9 effective dof (Jo of a uniform 1 sigma departure 1.9 clear /
2.4 cloudy instead of 6), so this cuts their weight about 3x. Ch 17 is weakly tied to the rest.
Raw cond (window block) is in the run log.

Job 24368256 (10 cycles, H100). Compare against w0t0. Expected: the q850 gain from the window
channels shrinks somewhat; z500 / ws250 neutral or better (less overfitting of correlated
window departures).

Result (2026-10-22, `figures/wcorr_10cyc/`). Change vs the H100 control (ERA5, window mean):

| | w0t0 | **wcorr** | wcorr vs w0t0 |
|---|---|---|---|
| z500 Global / NH / SH / Tropics | -1.1 / -1.0 / -1.1 / -1.3% | -2.3 / -2.1 / -2.8 / -0.6% | -1.2 / -1.1 / -1.8 / +0.7% |
| t500 Global / NH / SH / Tropics | -0.9 / -0.3 / -1.0 / -1.8% | -1.6 / -1.0 / -2.0 / -2.0% | -0.7 / -0.7 / -1.1 / -0.2% |
| q850 Global / NH / SH / Tropics | -0.9 / -0.6 / -1.1 / -0.8% | -0.3 / -0.6 / -0.3 / -0.2% | +0.5 / 0.0 / +0.8 / +0.6% |
| ws250 Global / NH / SH / Tropics | -1.4 / -0.6 / -1.4 / -2.0% | -1.4 / -0.6 / -1.9 / -1.7% | 0.0 / 0.0 / -0.5 / +0.4% |

- As predicted: the q850 gain mostly goes (+0.5% vs w0t0), z500 / t500 improve 1-2% in the
  extratropics, tropics z500 +0.7%.
- Fresh O-B (last 12 h of the window, cycles 2-10) vs w0t0: raob winds 0.5-3.4% better
  (NH v250 -3.4%, SH u/v 500 -1.4 to -1.7%), t 500 hPa -0.7 to -1.0%, ps -0.6 to -1.7%; worse
  only SH t250 +0.8%, SH q500 +1.2%, tropics v850 / q850 / t250 +0.3-0.5%. ATMS ch 1-16 +0.1-0.3%,
  ch 22 -0.6%.
- Better than w0t0 on dynamics and on the independent obs; a small q850 cost vs ERA5 (raob q850
  NH / SH still 0.5-0.6% better). Candidate for the next month run on top of corrhum.
