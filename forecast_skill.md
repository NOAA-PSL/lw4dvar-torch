# Forecast skill: 5-day forecasts from the 4D-Var analyses vs from ERA5

Tools: `diagnostics/forecast_skill.py` (forecasts and scores, GPU),
`diagnostics/plot_forecast_skill.py` (figures and CSV, CPU),
`diagnostics/run_forecast_skill.sh` (Slurm launcher), with the readers
`sfcobs_parquet.py` (surface stations, scatterometer) and `gsi_convinfo.py`
(GSI use flags and gross-error limits).

## Protocol

- **Init dates:** daily 00Z (`--every 24`). Each init date T gets a 5-day forecast
  (`--leads 20` × 6 h), scored every 6 h.
- **Run IC:** the analysis of the 4D-Var window that ends at T. The script reads
  `<T - window>_optimal_inputs_<window>h_*it.pkl`, which is valid at window
  start + 6 h, and advances it to T. Then it resets skt over ocean from ERA5,
  as the cycling does. Lead 0 is the window's last analysis time, so its obs
  were assimilated.
- **Reference:** a forecast from ERA5 at T, built from the checkpoint's lagged
  pair (T − 6 h, T). The `ic_cache` needs `era5_<T>_lagged.grib`.
- **Scores vs ERA5:** area-weighted (`model.area_weights`) RMSE and mean error
  in four regions: Global, NH (> 20N), Tropics, SH (< 20S).
  - z500, t850, q850
  - 250 hPa wind: speed and vector
  - msl
  - 10 m wind: speed and vector
  - t2m
  - q2m, computed from 2 m dewpoint and surface pressure
- **Scores vs obs:** n, Σ(O − F) and Σ(O − F)², per lead, region (by station
  latitude) and level.
  - Surface pressure, using the solver's operator.
  - Radiosonde t, q, u and v on the model levels, plus wind speed and vector
    wind. Sonde temperature is sensible T: the reader (`raobs_parquet.py`)
    takes it from the first non-VIRTMP event, because the archive's current
    TOB is virtual temperature wherever humidity was reported. This was fixed
    2026-10-17. Before the fix, O − ERA5 was +1.0 K at 925–1000 hPa and
    +0.75 K at 850 hPa; see integrate_prepbufr_obs.md.
  - Surface obs (`sfcobs_parquet.py`), in three networks:
    - land: ADPSFC 181/183 and 281/284, synoptic and METAR. US mesonet
      187/287 is excluded.
    - marine: SFCSHP 180/183 and 280/282/284, ships, buoys and C-MAN.
    - scat: ASCATW 290, ASCAT on MetOp-A and MetOp-B.
  - Surface variables:
    - 2 m T: sensible temperature, taken from the event stack because the
      current TOB is virtual. The model value is lapse-rate adjusted at
      6.5 K/km to the station elevation.
    - 2 m q.
    - 10 m wind: speed and vector.
  - Surface obs within ±1 h of each 6-h valid time (`--sfc-tolerance-hours`).
  - Upper-air profiles (`convobs_parquet.py`, `conv_<v>` variables):
    - Platforms: radiosondes at every level (with balloon drift positions),
      aircraft (AIRCAR, AIRCFT) and wind profilers / VAD winds. Satellite
      winds are left out by choice; the reader can still read them.
    - Variables: T (sensible), q and vector wind, at the reported pressure.
    - Model values are interpolated horizontally, then linearly in ln p
      between model levels. Obs outside the model's levels or below its
      surface are dropped, and q is used only at p ≥ 300 hPa.
    - Time window: ±1.5 h (`--conv-tolerance-hours`).
    - Binned into 13 pressure layers centred on 1000 … 50 hPa, per platform
      group.
    - QC: quality mark ≤ 3, plus the convinfo use flag and gross check
      against ERA5 using the archive's obs errors.

## QC

The same obs are used for every run and lead, because the background for the
gross check is ERA5 at the valid time, not the forecast.

1. **Read-time QC:**
   - PREPBUFR quality mark ≤ 3, or 9. 9 means good but monitored; that is
     what GDAS gives land T, q and wind, which it does not assimilate.
   - Physical bounds.
   - One report per station per slot.
   - For raobs, the solver's raob read QC.
   - For ps, the solver's orography screen (`zthresh`).
2. **Elevation screen:** land and marine stations need
   |station elevation − model orography| ≤ `--zthresh` (default 200 m).
3. **GSI convinfo:** uses the dated `global_convinfo.txt.<date>` from
   [NOAA-PSL/build_gsinfo](https://github.com/NOAA-PSL/build_gsinfo)
   (`--gsinfo`, the checkout). The file valid at T is chosen; for 2015 it is
   `2014041400`.
   - The obs type (and subtype; for ASCAT, the satellite id) must be in the
     file with iuse 1 (used) or −1 (monitored).
   - Gross check: reject when |O − H(ERA5)| > gross × clip(err, ermin, ermax).
   - err:
     - ps: the solver's effective error (including the orography term).
     - raob t/u/v: the solver's errors.
     - surface t: 2.5 K.
     - wind: 2.5 m/s for stations, 3.5 m/s for ASCAT. Winds are checked on
       the vector departure, as GSI does.
     - q: checked as relative humidity, |Δq| / qsat > gross × clip(0.2,
       ermin, ermax) / 100. This is a loose check, as in GSI.
   - The counts of rejected obs are saved (`obs_<name>_nrej`). For surface
     obs the count includes the elevation screen: about 25% of land
     stations fail it at 200 m.

## Running

```
CONFIG=config_fcst_skill.yml sbatch diagnostics/run_forecast_skill.sh \
    era5 era5 2015-01-06T00 2015-02-04T00 \
    ps_atms output/lwatms_1mo_ps_atms_h2o 2015-01-06T00 2015-02-04T00
python diagnostics/plot_forecast_skill.py figures/forecast_skill era5=ERA5_IC ps_atms=ps+ATMS --end 2015-02-04T00
```

- **CONFIG:** any lwatms config with raobs enabled. It supplies the model,
  `ic_cache` and `observations`.
- **Skipping:** a date whose output file already exists is skipped, so a job
  can be resubmitted.
- **Cost:** about 1–2 min per date on one MI300X.
- **Run order:** the first date for a run needs that run's analysis for the
  window ending there. For 12-h cycling from 2015-01-01T00 with a 5-day
  window, that means 2015-01-06T00 onward.

## Figures (`plot_forecast_skill.py`)

RMSE vs lead in four columns (Global, NH, Tropics, SH), one line per run,
over the init dates common to all runs. Obs counts are printed in each panel.

| file | rows |
|---|---|
| `fcst_era5_upper.png` | z500, q850, t850, ws250 vs ERA5 |
| `fcst_era5_sfc.png` | ws10, t2m, q2m vs ERA5 |
| `fcst_obs_upper.png` | ps, raob q850, t850, t500, ws250 vs obs |
| `fcst_obs_sfc.png` | ws10 at stations, ws10 from ASCAT, t2m and q2m at stations |
| `fcst_obs_profile.png` | profiles at `--profile-lead` (default 12 h) vs upper-air conventional obs: RMS(O − F) solid, mean dashed; T, q, vector wind × 4 regions |
| `fcst_obs_profile_platform.png` | the same, Global, one column per platform (raob, aircraft, profiler) |

`forecast_skill.csv` has every lead-time curve, plus per-network surface
scores and obs counts; `forecast_skill_profile.csv` has the profiles.

## Caveats

- The forecasts are not bit-reproducible across GPU nodes. ERA5-started
  forecasts agree to about 0.01 m z500 RMSE at 24 h. A run's forecast first
  advances its analysis 114 h, so the differences grow: the same 2015-01-10
  ps + ATMS forecast gave z500 RMSE of 11.62 vs 11.70 m at lead 0 and 16.9
  vs 17.7 m at 24 h on two nodes. Compare runs over many dates, not one.
- Output files carry `skill_version`. Files from an older version are
  recomputed and replaced, not skipped.
- Profile T at 50–100 hPa is limited by the model's sparse stratospheric
  levels: ln-p interpolation across 10–50 hPa and 50–100 hPa.

- ERA5 assimilated most of these obs, and the ERA5 forecast's lead 0 is ERA5
  itself. So against obs, the ERA5 line at short leads is an analysis-fit
  number, not an independent forecast score.
- 6-hourly verification with a ±1 h window uses about a third of the ASCAT
  swath data.
