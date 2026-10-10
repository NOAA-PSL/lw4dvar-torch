# Training our own ACE2 (`ace2-training.md`)

Goal: train a new ACE2 checkpoint ourselves, to (a) improve its behaviour in
cycled DA and (b) later add vertical layers so satellite radiances can be
assimilated (8 layers are too coarse for the temperature/humidity structure a
radiance operator needs). Started 2026-10-09 on ALCF Polaris.

## Decisions (2026-10-09)

- **Train from scratch**, not fine-tune ACE2-ERA5. Our data follows the
  *current* Ai2 pipeline, while ACE2-ERA5 was trained on its predecessor's
  data (see "Differences" below), so fine-tuning would start from a model
  trained on a slightly different distribution.
- **Model top ~1 Pa** (upstream's current `ak[0] = (ak[0]+ak[1])/2`), not the
  0 Pa recorded in the ACE2-ERA5 checkpoint. `ace2_dataset.py` defaults to
  it (`--model_top_zero` gives 0 Pa). NB `ace2_ic.py` still defaults to 0 Pa
  (it makes ICs for the existing checkpoint): ICs for a model trained on this
  data need `ace2_ic.py --top_interface_midpoint`.
- **Validate the pipeline at 8 layers first** (ACE2's
  `0 48 67 79 90 100 109 119 137` L137 interface indices), then move to more
  layers.

## Why we build the dataset ourselves

Ai2's ACE2-ERA5 training set is in a requester-pays bucket we can't use
(`gs://ai2cm-public-requester-pays/2024-11-13-ai2-climate-emulator-v2-amip/...`).
But Ai2's own pipeline, `../ace/scripts/era5/pipeline/xr-beam-pipeline.py`
(Apache Beam on Google Dataflow), reads only Google's PUBLIC ARCO-ERA5
stores -- except CO2, from a private Ai2 bucket. `ace2_ic.py` was already a
partial vendoring of it (model-level stream only).

## The pipeline: `backends/ace2/ace2_dataset.py`

The per-time-step logic of the upstream pipeline (vendored as of
ai2cm/ace e9d7fc227, 2026-08-14) without Beam; reuses `ace2_ic.py`'s
vendored helpers.

- All five upstream streams: model level (8 ERA5 L137 fields
  pressure-weighted into layers at 0.25 deg, then conservatively regridded to
  F90), 6-h mean fluxes, 6-h surface means, surface analysis (sea ice/ocean
  fraction, SST, soil, snow), pressure-level diagnostics (h500, TMP850, ...);
  plus static fields (HGTsfc, land_fraction, soil type fractions) and
  `global_mean_co2`. 155 fields per time at 8 layers (pressure levels
  extended 2026-10-09 to the 13 that ACE2.1-ERA5's secondary pressure-level
  decoder is trained on, 50-1000 hPa, plus 10 hPa; +24 fields, no extra reads
  since ARCO chunks all 37 levels together).
- Output is `fme`'s training layout directly: monthly `YYYYMM0100.nc`,
  6-hourly, dims `(time, latitude, longitude)`, time units
  `hours since 1940-01-01T12:00:00` (as Ai2's), static fields and
  `ak_N`/`bk_N` in every file. Written to `.tmp` and renamed; existing months
  are skipped, so jobs over disjoint month ranges can share one OUTDIR.
- `--layer_indices` (any increasing L137 interface list from 0 to 137) and
  `--extra_layer_vars` (`specific_humidity`, the four condensates,
  `ozone_mass_mixing_ratio`, `fraction_of_cloud_cover`, `vertical_velocity`)
  for the radiance work.
- Shared per-OUTDIR files: `regrid_weights_0p25deg_to_F90.nc` (xESMF weights,
  ~3 min to compute once), `invariant.nc`, `co2.nc`.
- **CO2**: `ace2_dataset.py co2 OUT.nc 1940 2022` pulls ACE2-ERA5's own
  6-hourly `global_mean_co2` out of the public HF `forcing_YYYY.nc` files via
  HTTP range reads (~3 s/year, no full download). Verified bit-identical to
  the local `forcing_2020.nc`. HF has 1940-2022 only; later years need
  another source (not done). Runs in the `ace2` env (`ace2ic` lacks
  `h5netcdf`).
- Memory: a family summing several ERA5 fields (total water) is built as the
  sum of each field's coarsened layers (the layer mean is linear, so this
  equals coarsening the sum), and dp is built one coarse layer at a time:
  one 3D float32 field in memory at a time, ~2.2 GB peak per time step
  (first version: ~10 GB).
- **PRESsfc valid at HGTsfc (2026-10-10).** HGTsfc is the conservative
  regrid of ERA5's orography, the same regrid as PRESsfc -- so `ace2_ic.py`'s
  reduction (from that regridded orography to Ai2's HGTsfc) would be a no-op
  here. The real inconsistency is sub-grid: the plain regrid averages
  pressures valid at different heights, and since ps falls off ~exponentially
  with height that average exceeds ps at the cell-mean height. Now each
  0.25-deg ps is reduced to its F90 cell's HGTsfc *before* averaging
  (`_presfc_at_hgtsfc`; `ace2_ic._reduce_surface_pressure`, 6.5 K/km from the
  2-m virtual temperature), for PRESsfc and each hour of PRESsfc_mean. Effect
  (1985-01-01T00 and 1985-07-01T12): global mean -6 Pa, rms 31-35 Pa; by
  sub-grid height std: <100 m (89% of the globe) ~1 Pa, 100-300 m -19 Pa
  mean, 300-600 m -90 Pa, >600 m (Andes, Himalaya, Antarctic margin) -320
  to -370 Pa, extremes -1.2 to -1.3 hPa. Q2m/Q2m_mean still use the plain
  average (as `ace2_ic.py` does, where DPT2m is valid). Files carry
  `presfc_reduced_to_hgtsfc=1`; `build --no_reduce_presfc` restores the
  plain regrid. Validated: a full `process_time` changes only PRESsfc and
  PRESsfc_mean vs a file built before (other 153 fields bit-identical).
- **`fix_presfc` mode** patches PRESsfc/PRESsfc_mean in place in months
  built before 2026-10-10 (no attribute): reads only surface ps/T2m/Td2m
  (~2 s/time/worker), first checks its plain-regrid recomputation against
  the file (< 1 Pa, measured 0.018) and refuses otherwise; idempotent
  (skips months with the attribute). Validated on a 3-time copy of 1985-01:
  patched = direct computation exactly, nothing else changed.

## Validation (2026-10-09, 2020-01-01T00, 8 layers, model top 0 Pa)

- vs `ace2_ic.py`'s cache `ace2_ic_2020010100.nc`: layer T/total water/u,
  TMP2m, Q2m **bit-identical** (rms 0) -- the low-memory coarsening matches.
- vs Ai2's HF files (old pipeline; expected small differences): PRESsfc
  431 Pa rms (same as documented in `ace2_ic.py`, steep terrain),
  air_temperature_7 0.26 K rms (std 14 K), surface_temperature 0.52 K rms,
  DSWRFtoa 0.03 W/m2 rms, HGTsfc 41 m rms, land/ocean/sea-ice fraction
  0.02-0.03 rms. All mean differences ~0.

## Differences from Ai2's ACE2-ERA5 training data

ACE2-ERA5's 2024-11-13 dataset was made by the PREVIOUS upstream pipeline
(native-grid ERA5 from ARCO `co/` stores, MIR point-sampling regridding,
vertical coarsening after regridding, model top 0 Pa). Ours follows the
current one (0.25 deg ARCO `ar/` stores, conservative xESMF, coarsening at
0.25 deg, model top ~1 Pa). Normalization statistics must be recomputed from
our data. HGTsfc comes from our own regrid (41 m rms from Ai2's), so a model
trained here needs forcing files and a ps observation operator using OUR
HGTsfc, not the HF forcing files.

## Polaris pitfalls found

- **Login nodes cap each user at 8 GB RAM / 8 CPUs** (cgroup
  `/sys/fs/cgroup/users/$USER/memory.max`; `memory.high` 7.5 GB). Above
  `memory.high` processes are throttled, not killed -- the 10 GB first
  version looked like a hang. Test there with 1 worker; build on compute
  nodes. One time step took 55 s on the login node (vs ~12 s of reads).
- Compute nodes reach the internet only via `proxy.alcf.anl.gov:3128`;
  aiohttp (gcsfs) ignores proxy env vars unless `trust_env=True`, which
  `ace2_dataset.STORAGE_OPTIONS` sets.
- Someone's `/tmp/inspect.py` on the login node shadows the stdlib module
  for any Python started with cwd `/tmp`: don't run Python from `/tmp`.
- Python `-I` drops the script's directory from `sys.path`, which breaks
  `import ace2_ic` -- the launcher runs without it.

## Environment

`ace2ic` built on Polaris 2026-10-09 at `/home/jwhitaker/.conda/envs/ace2ic`
from `ace2ic-spec.txt` (`module use /soft/modulefiles; module load conda;
conda create --name ace2ic --file ace2ic-spec.txt`).

On Ursa (NOAA RDHPCS) the existing `ace2ic` env
(`/scratch4/BMC/gsienkf/whitaker/conda/envs/ace2ic`, same package versions
as the spec) runs `ace2_dataset.py build` unchanged; `co2` mode runs in the
`ace2` env there too (`ace2ic` lacks `h5netcdf`). No new env was needed.

## Ursa (2026-10-10)

- Ursa compute nodes have no internet; the `u1-service` partition
  (ufe05-14: 384 CPUs, 755 GB; 1 node, 24 h per job) reaches GCS directly,
  no proxy. Launcher: `run_ace2_dataset_ursa.sh` (account `gsienkf`, 32 CPUs,
  128 GB, WORKERS=16 default).
- Default OUTDIR `/scratch4/BMC/gsienkf/Jeffrey.Whitaker/ace2_era5_1deg_8layer`;
  `co2.nc` (1940-2022) built there on a login node in 3.4 min.
- One-month test (job 23427483, 2020-01, 16 workers): exit 0, 124 times in
  10.7 min, **~4-5 s/time steady** (vs ~20 s on Polaris -- ~4x faster),
  ~30 GB RSS total (~1.9 GB/worker). 5.0 GB/month with the 13+1 pressure
  levels (184 vars, 156 time-varying). At this rate 1979-2022 (~64k times)
  is ~80 h on one node, i.e. ~4 concurrent 24-h jobs over disjoint ranges
  (e.g. one decade each) finish in about a day, if u1-service throughput
  scales -- not yet tested, nor WORKERS>16.
- Checked 2020-01-01T00 vs `ic_cache_ace2/ace2_ic_2020010100.nc`: layer
  T/total water/u (layers 1-7), TMP2m, Q2m, surface_temperature
  bit-identical; air_temperature_0 0.008 K rms (model top ~1 Pa here vs
  0 Pa in the IC cache); PRESsfc 412 Pa rms because the IC cache reduces
  PRESsfc to Ai2's HGTsfc and the dataset does not (by design; this check
  predates the dataset's own reduction to HGTsfc, below). No
  non-finite values except `sea_surface_temperature` over land (NaN by
  design; `merged_sea_surface_and_skin_temperature` fills it).

## Cost estimates

ARCO model-level chunks are (1 h, 18 levels, full grid); the 8 fields are
~10 s of reads per output time per process, the rest ~2-3 s more. 1979-2022
is ~64k times, ~150 TB read from Google's free public bucket. Output
~28 MB/time at 8 layers: ~3.5 GB/month, ~1.8 TB for 1979-2022. Training
period in Ai2's ACE2.1 config: train 1979-2008, validate 2009-2014; stats
over 1990-2019 in their ERA5 stats config.

## Running

```
# login node, ace2 env: CO2 series
python backends/ace2/ace2_dataset.py co2 $OUTDIR/co2.nc 1940 2022
# compute node(s)
qsub -v START=1979-01,END=1989-12 run_ace2_dataset_polaris.sh
# Ursa: u1-service partition
sbatch --export=ALL,START=1979-01,END=1989-12 run_ace2_dataset_ursa.sh
# Ursa: patch PRESsfc in months built before 2026-10-10 (own log files)
sbatch -o ace2_fix_presfc.out -e ace2_fix_presfc.err \
    --export=ALL,MODE=fix_presfc,START=1979-01,END=1988-12 run_ace2_dataset_ursa.sh
```
Default OUTDIR on Polaris `/lus/eagle/projects/moonshot-reanalysis/jwhitaker/ace2_era5_1deg_8layer`,
on Ursa `/scratch4/BMC/gsienkf/Jeffrey.Whitaker/ace2_era5_1deg_8layer`
(co2.nc staged in both).

## Status / next steps

- [x] One-month compute-node test (PBS job 7731559, debug queue, 2020-01,
      16 workers): exit 0, wrote `2020010100.nc` (124 times, 4.2 GB) in
      47.3 min (~23 s/time wall; ~20.6 s/time steady, ~330 s per time per
      worker vs ~55 s on the login node -> looks network/proxy-bound,
      ~110 MB/s). Full 1979-2022 at this rate: ~420 node-hours, ~2.2 TB.
      Not yet checked: file contents vs the earlier login-node validation;
      whether WORKERS=32 or several concurrent nodes raise total throughput
      (i.e. whether the proxy is the cap).
- [ ] Throughput tuning, 2020-01 on one debug node (s/time = wall per output
      time over times 21-40; startup dominates the first 20):
      16 workers, default threads, 18:40 UTC: 20.6 (job 7731559, old levels);
      32 workers, default threads, 21:00: 33 (7731799, killed);
      16 workers, THREADS=1, 21:35: 42 (7731860, killed).
      Confounded by time of day (shared proxy/network load?) -- baseline
      rerun (16 workers, default threads) as job 7731906 to separate them.
- [ ] Full 8-layer build, Ursa (2026-10-10): job 23429181 (1979-1988,
      running; 1979-1985 done by 15:20, ~10 min/month) built WITHOUT the
      PRESsfc reduction; queued 23431267 (1989-1998) and 23431519
      (1999-2008) will import the new code and build with it. Patch job
      23465227 (`fix_presfc` 1979-1988, after 23429181 ends) brings the
      first decade in line. 2009-2022 not yet submitted.
- [ ] Normalization stats (centering, scaling-full-field, scaling-residual,
      time-mean; cf. `../ace/scripts/data_process/get_stats.py`). Upstream
      takes PLAIN (not area-weighted) mean/std over (time, lat, lon);
      residual = std of `diff("time")`; drops ak/bk/pressure_thickness;
      time_means = mean over time only. Match that, not our area-weight rule.
- [ ] Training config from scratch (start from
      `../ACE2.1-ERA5-AIMIP/configs/ace-train-config.yaml` + the ACE2-ERA5
      checkpoint's in/out names, which add global_mean_co2, TMP2m, Q2m,
      UGRD10m, VGRD10m, h500, TMP850).
- [ ] Yearly forcing files from our dataset for inference in lw4dvar.
- [ ] Higher vertical resolution + extra layer fields for radiances.
- [ ] CO2 after 2022.
