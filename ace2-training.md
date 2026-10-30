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
- `--layer_indices` (any increasing L137 interface list from 0 to 137),
  `--layer_families` (`ace2`, the default, or `crtm`; see "CRTM layer
  variables" below) and `--extra_layer_vars` (`specific_humidity`, the four
  condensates, `ozone_mass_mixing_ratio`, `fraction_of_cloud_cover`,
  `vertical_velocity`) for the radiance work.
- Shared per-OUTDIR files: `regrid_weights_0p25deg_to_F90.nc` (xESMF weights,
  ~3 min to compute once), `invariant.nc`, `co2.nc`.
- **CO2**: `ace2_dataset.py co2 OUT.nc 1940 2022` pulls ACE2-ERA5's own
  6-hourly `global_mean_co2` out of the public HF `forcing_YYYY.nc` files via
  HTTP range reads (~3 s/year, no full download). Verified bit-identical to
  the local `forcing_2020.nc`. HF has 1940-2022 only. Runs in the `ace2` env (`ace2ic` lacks
  `h5netcdf`).
- **CO2 after 2022** (2026-10-18): ACE2-ERA5's series is, from ~2017 on,
  NOAA GML global annual means (`co2_annmean_gl.txt`) linearly interpolated
  between knots at July 1 00Z (Jul-1 values within 0.01 ppm of NOAA's; slope
  breaks exactly at Jul 1 00Z; 1979-2016 annual means differ by up to
  0.45 ppm, an older NOAA vintage). `ace2_dataset.py co2_noaa IN.nc OUT.nc
  2025` appends the same construction (input kept bit-identical; the 2022
  knot taken from the input; past the last NOAA year the last slope is
  extrapolated). With NOAA's 2023-2025 (419.35, 422.79, 425.62) there is a
  0.02 ppm step at 2023-01-01 (vs ~0.0016 per 6 h), from NOAA's revision of
  2023 since Ai2's series was made. H2 2025 is extrapolated until NOAA
  publishes 2026. Installed as `co2.nc` (1940-2025) in both the 8- and
  32-layer dirs; the original is kept as `co2_1940-2022.nc`.
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
- **aws-ofi-nccl plugin segfaults on 1-node jobs since 2026-10** (driver
  580.65.06 / CUDA 13.0, libfabric 2.3.1; found 2026-10-27): every rank dies
  in the first collective (fme's `dist.barrier` in `prepare_directory`)
  after `NET/OFI Couldn't open a fabric access domain. RC: -38` -- with the
  v1.9.1-aws and the libfabric-1.22 builds alike. Without the plugin, 1-node
  NVLink works. On 2 nodes the plugin works (job 7789746: 0.46 GB
  all-reduce 0.022 s vs 0.23 s with `NCCL_NET=Socket`). So
  `run_ace2_train_polaris.sh` sets the NCCL/plugin env only when NNODES > 1.
  Diagnostics: untracked `nccl_diag/`.
- **DataLoader workers hang with `OSError: AF_UNIX path too long`**
  (2026-10-28, 16-layer smoke job 7789826, stuck 35 min after the
  inference dataset loaded): torch shares tensors between processes over
  AF_UNIX sockets under TMPDIR (limit 108 chars); the PBS job TMPDIR is
  `/var/tmp/pbs.<jobid>.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov` (63
  chars). `run_ace2_train_polaris.sh` now prints it and sets `TMPDIR=/tmp`;
  the next run went through.

## Environment

`ace2ic` built on Polaris 2026-10-09 at `/home/jwhitaker/.conda/envs/ace2ic`
from `ace2ic-spec.txt` (`module use /soft/modulefiles; module load conda;
conda create --name ace2ic --file ace2ic-spec.txt`).

On Ursa (NOAA RDHPCS) the existing `ace2ic` env
(`/scratch4/BMC/gsienkf/whitaker/conda/envs/ace2ic`, same package versions
as the spec) runs `ace2_dataset.py build` unchanged; `co2` mode runs in the
`ace2` env there too (`ace2ic` lacks `h5netcdf`). No new env was needed.

On Hercules (MSU) `ace2ic` was built 2026-10-10 at
`/work2/noaa/gsienkf/whitaker/conda/envs/ace2ic` from `ace2ic-spec.txt`
plus `h5netcdf` (conda-forge), so `co2` mode runs in it too. Login nodes
have direct internet. Activate it (or set
`ESMFMKFILE=$ENV/lib/esmf.mk`) before running: a stale `ESMFMKFILE` from
another conda install gives ESMPy's "ESMF installation version ... differs"
error. `run_ace2_dataset_hercules.sh` runs on the `service` partition
(hercules-dtn-[1-4], internet verified 2026-10-10); it exports ESMFMKFILE
itself because the `.bashrc` conda init re-activates base on the node.
The default `batch` QOS caps jobs at 8 h. Measured (1979-01, 16 workers,
8 layers): ~5 s/time, 10.1 min/month, 38 GB peak RSS, 5.0 GB/month output
(~2.6 TB for 1979-2022) -- unchanged with 4 jobs on 4 DTN nodes at once.
1979-2022 was submitted as 22 two-year jobs in 4 `--dependency=singleton`
chains (`-J ace2ds_chain<N>`) into `/work2/noaa/gsienkf/whitaker/ace2-training`.

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

## Normalization statistics (2026-10-12)

`ace2_dataset.py stats OUTDIR STATSDIR [1990-01 2019-12]` (Polaris:
`qsub -q debug -l walltime=01:00:00 -o ace2_stats.out -e ace2_stats.err -v
MODE=stats run_ace2_dataset_polaris.sh`) writes `centering.nc`,
`scaling-full-field.nc`, `scaling-residual.nc`, `time-mean.nc` as upstream's
`scripts/data_process/get_stats.py` does on one contiguous series: PLAIN (not
area-weighted) NaN-skipping mean/std (ddof 0) over (time, lat, lon), residual
= std of `diff("time")` (including the diffs across month boundaries),
time-mean per grid point; static fields get their spatial mean/std, their
spatial std as the residual scale and themselves as the time mean (what xarray
does there); ak/bk dropped. One streaming pass, a month per worker, float64
accumulation (Chan's pairwise mean/M2), output cast to each variable's dtype.
It refuses months without `presfc_reduced_to_hgtsfc=1` or a gap in the 6-h
series, and logs per-variable min/max/non-finite counts.

- Validated against upstream's xarray calls (float64) on a 16-time,
  month-boundary-crossing subset: max relative error 6e-8 (float32 rounding;
  upstream's own float32 computation is off by up to 6e-7), same 166
  variables, same NaN pattern.
- 1990-2019 (43828 times = 30 x 1460 + 7 leap days x 4): Polaris debug node,
  16 workers, 23 min (~4 s/month). Only non-finite values: SST over land.
- vs the ACE2-ERA5 checkpoint's normalizer (56 shared variables; its
  `loss_normalizer` stds are the residual stds for prognostic variables and the
  full stds otherwise): means agree to ~4 significant figures, full and
  residual stds to within ~1% (e.g. PRESsfc 9464 vs 9485, residual 250.4 vs
  250.9) -- consistent with the pipeline differences above.
- ERA5 oddities, not pipeline bugs: specific_total_water_1 max 4.9 g/kg and a
  residual std above its full std (the checkpoint's stats show the same);
  negative Q100/Q150 (min ~ -1e-4); and one bad-wind event, the only one in
  1979-2022 with |VGRD10m| or |VGRD1000| > 60 m/s at 1 deg: 2004-07-11T18 at
  40.4N 87.5E (Tarim Basin), VGRD10m 115.6 / VGRD1000 119.4 m/s (-0.9 and
  -0.3 m/s 6 h before/after). It is in ARCO-ERA5 itself (0.25 deg: 10-m v
  154 m/s, 30 points > 50 m/s; 73 m/s at 19Z, < 3 m/s at 17Z; u and ps
  normal), so Ai2's data has it too. Left in; negligible for the stats.

## Training (2026-10-12)

Config `backends/ace2/training/ace2-era5-vars-train-config.yaml`, launcher
`run_ace2_train_polaris.sh` (4 Polaris nodes, 4 x 40 GB A100 each, `torchrun`
through `backends/ace2/training/fme_train.py`, `ace2` env, fme 2026.5.1):

- **Stepper = the ACE2-ERA5 checkpoint's**, extracted from
  `ace2_era5_ckpt.tar`: SphericalFourierNeuralOperatorNet (embed_dim 384, 8
  layers, dhconv, 3 spectral layers), in/out names incl. global_mean_co2,
  TMP2m/Q2m/UGRD10m/VGRD10m (prognostic) and TMP850/h500 (diagnostic), its
  loss weights and corrector -- so results compare directly with ACE2-ERA5
  in lw4dvar. Training settings from ACE2.1-ERA5: 120 epochs, batch 16,
  n_forward_steps 2, AdamW lr 1e-4 wd 0.01, EMA 0.999, validate with EMA.
- **Data split (decided 2026-10-12): train 1979-2014 + 2016-2022, 2015
  held out** (validation, step 3; inline inference). Subsets select sample
  start times, so stops at T06 keep 3-time samples inside each range. Inline
  inference: 4 ICs (fme needs a multiple of the 4 ranks), 2015-01-01..04
  00Z, 1440 steps (all inside 2015). NB lw4dvar tests from 2020 ICs fall in
  the training period; 2015 ones do not.
- `file_pattern` `[12][0-9][0-9][0-9][01][0-9]0100.nc` keeps fme off
  `co2.nc`, `invariant.nc` and the regrid weights in the data dir.
- **Memory:** 4 samples/GPU does not fit a 40 GB A100 (OOM in the first
  backward, 37.5 GB allocated). `checkpointing: 3` (gradient checkpointing
  of every SFNO block) fits; same math, more compute. Batch 16 and fp32 kept.
- **fme bug, worked around:** the modulus SFNO behind the
  `SphericalFourierNeuralOperatorNet` builder calls
  `torch.utils.checkpoint.checkpoint` without `use_reentrant` (reentrant in
  torch 2.x); checkpointing the encoder on the raw input then gives its
  weights no gradient and DDP fails ("Expected to have finished reduction
  ... Parameter indices which did not receive grad: 1 2 3"). `fme_train.py`
  rebinds those modules' `checkpoint` to `use_reentrant=False` (gradients
  bit-identical to no checkpointing, checked on a small network). Upstream
  fixed only conditional_sfno (#1493); reported as ai2cm/ace#1577. Drop the
  wrapper once the env's fme has the fix.
- **Smoke test** (job 7740494, debug, 2 epochs on 3 months, 80-step
  inference): exit 0 in 7.5 min; loss 22 -> 13; checkpoints saved
  (`ckpt.tar` 7.3 GB with optimizer state, best/best_inference 1.8 GB).
  Steady 2.83 s/batch (1.42 samples/s/GPU); inference ~0.26 s/step.
- **Resume test** (job 7740554, `max_epochs=3` on the smoke run): resumed
  from `ckpt.tar` at epoch 3 / step 45, same loss and rate. Chaining works.
- **Multi-node** (`run_ace2_train_polaris.sh`: `mpiexec` one `torchrun`
  per node, c10d rendezvous on the first node; NCCL over Slingshot via
  `/soft/libraries/aws-ofi-nccl/v1.9.1-aws` with ALCF's settings). Global
  batch stays 16, so 1-4 nodes train identically. Scaling test (job
  7740610, debug-scaling, smoke config):

  | nodes | samples/GPU | checkpointing | s/batch | node-s/batch |
  |---|---|---|---|---|
  | 1 | 4 | 3 | 2.83 | 2.83 |
  | 2 | 2 | 3 | 1.43 | 2.86 |
  | 2 | 2 | 0 | 1.09 | 2.18 |
  | 4 | 1 | 3 | 0.76 | 3.04 |
  | 4 | 1 | 0 | 0.60 | 2.38 |

  Inter-node all-reduce is cheap (2 nodes scale perfectly; 4 nodes lose
  ~7%). Without checkpointing 1-2 samples/GPU fit the 40 GB A100s, and
  dropping its recompute more than pays for the communication: **4 nodes,
  no checkpointing is ~4.8x faster than 1 node AND uses fewer node-hours.**
  The config is set up for that (`checkpointing: 0`; 1 node needs
  `--override stepper.step.config.builder.config.checkpointing=3`).
  Inline inference: 16 ICs (a multiple of the GPU count), ~0.22 s/step at
  4 nodes.
- **Cost (4 nodes):** ~62.8k training samples = ~3930 batches x 0.60 s =
  ~39 min/epoch, plus validation (a few min) and the 1440-step inference
  (~5-6 min): ~48 min/epoch, **~4 days for 120 epochs** (~385 node-h) --
  one 168-h `capacity` job. ACE2-ERA5's checkpoint is from epoch 58 (best
  inference error).
- Earlier smoke-test failures, for the record: 1 inference IC with 4 ranks;
  validation batch 128 (32/rank) larger than a one-month subset.

## Training on Ursa (2026-10-13)

The dataset is also on Ursa at `/scratch4/BMC/gsienkf/whitaker/ace2_era5_1deg_8layer`
(checked 2026-10-13: 528 months, 64284 6-h times, every file
`presfc_reduced_to_hgtsfc=1`, `top_interface_midpoint=1`). Its normalization
stats were computed there too (`run_ace2_dataset_ursa.sh` `MODE=stats`).

- Config `backends/ace2/training/ace2-era5-vars-train-config-ursa.yaml`: the
  Polaris config with Ursa paths only (keep the two in sync); experiment dir
  `/scratch4/BMC/gsienkf/whitaker/ace2_training/ace2-era5-vars-1979-2022-no2015`.
- Launcher `run_ace2_train_ursa.sh`: `u1-h100` (2 x 80 GB H100 per node),
  account `gpu-ai4wp`, QOS `gpu` (<= 20 GPUs = 10 nodes per job, <= 7 days);
  `srun` one `torchrun --nproc_per_node 2` per node, c10d rendezvous on the
  first node, NCCL over InfiniBand with its defaults. Default 4 nodes (2
  samples/GPU, ~25 GB by the memory probe); `-N 2` is 4 samples/GPU (est.
  ~42 GB + DDP), `-N 8` is 1 sample/GPU. 1 node needs `checkpointing=3`.
- **`--cpus-per-task=192` is essential.** Without it the job, even with
  `--exclusive`, is bound to ONE core (`Cpus_allowed_list: 0`), shared by
  both ranks and all 16 data-loader workers: 4.1-4.2 s/batch on 2 and on 4
  nodes alike, loader-bound, and 24 loader workers per rank did not help.
  (`nproc` reports 1 either way because it honours `OMP_NUM_THREADS`.)
- Stats on Ursa: the first `MODE=stats` job hung -- 16 workers at ~14 GB
  each against the 128 GB default, 259 OOM kills, and `Pool.imap` never
  returns a killed worker's task. u1-service caps a user at 64 CPUs and 3.9
  GB/CPU, so it runs with `-c 64 --mem=240g` and 12 workers: 19 min, ~3
  s/month. Result identical to Polaris's (166 variables, 43828 times,
  PRESsfc mean 96697, std 9464.1, residual std 250.43).
- **Smoke test** (job 23664478, 2 nodes, `ace2-smoke-ursa.yaml`: 2 epochs on
  2016-01..03, validation 2015-01..02 at batch 16, 80-step inference; run
  with the 1-core binding): exit 0; train loss 23.09 -> 13.98, valid 22.92
  -> 12.96, inference error 0.577 -> 0.456; ~49 GB/GPU at 4 samples/GPU.
  Rerunning it gave bit-identical losses.
- **Throughput** (one epoch on 2016-2017, 182-183 batches, 8 loader workers
  per rank, with 192 CPUs; includes the first-batch warm-up):

  | nodes | GPUs | samples/GPU | s/batch | node-s/batch |
  |---|---|---|---|---|
  | 2 | 4 | 4 | 1.30 | 2.6 |
  | 4 | 8 | 2 | 0.97 | 3.9 |

  Full run (~3930 batches/epoch plus validation and the 1440-step
  inference): 4 nodes ~70 min/epoch, ~6 days for 120 epochs (one 7-day
  job); 2 nodes ~95 min/epoch, ~8 days (two chained jobs, fewer
  node-hours). Untested: more loader workers now that the CPUs are there;
  8 nodes.

## Memory probe: layers vs embed_dim (2026-10-13)

Job 7741018 (debug, 1 node; one-off untracked
`backends/ace2/training/memory_probe.py` + `memory-probe.sh`): fme's SFNO
(training-config architecture) built with 12+4L inputs and 18+4L outputs
(4 layer variables per layer, as now), synthetic F90 data, fme's training
step on one GPU (2 forward steps, backward after each with the state
detached, fused AdamW, EMA, non-reentrant checkpointing). No DDP, which adds
one gradient-sized buffer (+1.7/3.0 GB at embed 384/512). Peak allocated
GB, s/batch:

| embed_dim | params | ckpt | 1 sample/GPU | 2 | 4 |
|---|---|---|---|---|---|
| 384 | 456M | 0 | 16.7-17.0, 0.50 s | 24.9-25.4, 1.06 s | OOM |
| 384 | 456M | 3 | 10.5-10.8, 0.66 s | 12.7-13.2, 1.40 s | 16.7-17.8, 2.86 s |
| 512 | 799M | 0 | 25.7-26.0, 0.80 s | 36.6-37.2, 1.68 s | OOM |
| 512 | 799M | 3 | 17.5-17.8, 1.07 s | 20.3-20.9, 2.22 s | 25.7-26.8, 4.45 s |
| 768 | 1773M | 0 | OOM | - | - |
| 768 | 1773M | 3 | 37.2-37.4, 2.14 s | OOM | - |

Ranges are over L = 8, 16, 24, 32, 48: **layers are nearly free** (8 -> 48
layers: +0.3 GB at 1 sample/GPU, +1.1 GB at 4; time unchanged). Channels
only enter the encoder/decoder; the 8 SFNO blocks run at embed_dim.
**embed_dim is what costs**: the dhconv spectral weights make params
~quadratic in it, and params x 5 (weights, grads, 2 AdamW moments, EMA)
= 8.5 / 15 / 33 GB at 384 / 512 / 768 before activations. Calibration
against the real runs: embed 384, L=8, 4 samples/GPU, no checkpointing
OOMs as in smoke test 3; with checkpointing 2.86 s vs 2.83 s measured; 1
sample/GPU 0.50 s vs 0.60 s in the 4-node DDP run (data + all-reduce).

So with the 40 GB A100s and global batch 16:
- **embed 384, up to 48 layers:** as now (4 nodes, no checkpointing,
  ~4 days for 120 epochs; 1 node fits with checkpointing).
- **embed 512:** 4 nodes no checkpointing (26 GB + DDP): ~1.6x slower,
  ~6.5 days -- just inside one 168-h job; 1 node with checkpointing fits
  (27 GB) but ~4.5 s/batch, ~4 weeks.
- **embed 768:** only 1 sample/GPU with checkpointing, at 37 GB before DDP's
  6.6 GB buffer -- does not fit in practice without sharding the
  optimizer state (ZeRO/FSDP) or bf16 AMP.
Not covered: data loading (bytes per sample grow ~linearly with layers;
8 loader workers per GPU read from Eagle) and whether embed 384 is wide
enough to carry 4-6x more channels -- that is a skill question for a
training run, not a memory one.

## 32-layer vertical grid (2026-10-14)

Where the 8 layers come from: Ai2's ERA5 pipeline picks the L137 interfaces
nearest ACE's (Watt-Meyer et al. 2023, arXiv:2310.02074, App. A, Table 2:
FV3GFS 63-layer interfaces 0 18 26 31 36 41 47 53 63), which were chosen by
hand to match the SPEEDY model's 8 sigma levels (half levels ~0, .05, .14,
.26, .42, .60, .77, .90, 1); top moved up to ERA5's. Not a formula: not
even in index, pressure or log p.

Chosen 32 layers: **nested** -- each of the 8 layers split into 4, evenly in
L137 index, so all 9 original interfaces are kept (each 8-layer field is the
dp-weighted mean of its 4 sub-layers: the 8-layer dataset checks the
32-layer one, and models at both resolutions compare on the same 8 layers)
and the sub-layers inherit L137's own resolution distribution:

```
--layer_indices 0 12 24 36 48 53 58 62 67 70 73 76 79 82 84 87 90 \
                92 95 98 100 102 104 107 109 112 114 116 119 124 128 132 137
```

Interfaces at ps = 1013.25 hPa: 0.01, 0.7, 5.6, 20.8, 51.2 | 69, 91, 112,
144 | 167, 193, 222, 254 | 290, 316, 359, 407 | 442, 498, 558, 600 | 642,
683, 742, 778 | 828, 857, 883, 916 | 958, 982, 999, 1013 hPa; dp 15-60 hPa
in the troposphere, 14-24 hPa in the lowest 4 layers. Rejected: splitting
evenly in pressure (nothing above 13 hPa -- bad for stratospheric channels,
e.g. AMSU-A 12-14) and in log p (3 interfaces above 7 hPa, a 44 hPa layer
below). Open question: the top two layers (0-0.7, 0.7-5.6 hPa) are where
ERA5 is least reliable; `0 20 32 41 48` (~3, 13, 32 hPa) is a milder top
split if they cause trouble.

Test month (Hercules `service`, job 10114455, 2020-01, 16 workers, into
`/work2/noaa/gsienkf/whitaker/ace2_era5_1deg_32layer_test`): 124 times in
16.9 min (8.3 s/time vs ~5 at 8 layers), 52 GB peak RSS, 8.07 GB/month
(vs 4.99; ~4.3 TB for 1979-2022), 328 vars (vs 184). Checked against the
8-layer 2020-01 file (first 8 times): the 9 original interfaces are
`ak_{4k}`/`bk_{4k}`; every non-layer variable bit-identical; the dp-weighted
mean of each 4 sub-layers (dp from the file's PRESsfc) matches the 8-layer
field to rms <= 5e-4 K / 6.5e-4 m/s / 1.8e-7 kg/kg -- not exact because
coarsening uses 0.25 deg ERA5 surface pressure before regridding, and
PRESsfc is reduced to HGTsfc.

## Total water and its components (2026-10-14)

`specific_total_water_<i>` is vapor plus all four ERA5 condensates:
`specific_humidity` + `specific_cloud_liquid_water_content` +
`specific_cloud_ice_water_content` + `specific_rain_water_content` +
`specific_snow_water_content` (`WATER_VARS` in `ace2_ic.py`, imported by
`ace2_dataset.py` through `LAYER_VARS`), each coarsened to layers and then
summed (as Ai2's). Both the 8- and 32-layer builds store **only the total**
(checked 32-layer `1979010100.nc`: no `specific_humidity_<i>` or condensate
fields).

For radiances this matters: vapor, cloud liquid and cloud ice have different
radiative properties. Vapor dominates clear-sky absorption, liquid emits in
microwave window channels, and ice (and snow/rain) scatter at high microwave
frequencies and in the IR. Clear-sky operators (CRTM/RTTOV) need q; all-sky
operators also need the hydrometeor profiles. Options:

- **Prognostic split**: rebuild with `--extra_layer_vars specific_humidity
  specific_cloud_liquid_water_content specific_cloud_ice_water_content`
  (+ rain/snow if all-sky MW) and train on the components instead of, or
  besides, the total. Costs more channels; the condensates are intermittent
  and skewed, so they probably need a transformed normalization.
- **Diagnostic secondary decoder** (like ACE2.1-ERA5's pressure-level
  decoder): reconstruct q / liquid / ice from total water and temperature
  (plus layer pressure, and possibly the rest of the column), keeping total
  water as the conserved prognostic. The training data still needs the
  components, as output-only fields. A physical baseline to beat:
  saturation adjustment (q = min(qt, qsat(T, p)), the excess split into
  liquid and ice by a temperature ramp). This will underestimate cloud at
  1-deg layer means, where clouds exist below grid-mean saturation.

Either way the dataset has to be rebuilt (or extended) with the component
fields.

## CRTM layer variables (2026-10-26)

Decided: for CRTM radiances the model carries vapor and ozone as
prognostics, and total water is replaced by the total condensate.
`build --layer_families crtm` writes six layer families:
`air_temperature`, `specific_humidity`, `specific_total_condensate` (cloud
liquid + cloud ice + rain + snow, each coarsened and then summed, like total
water), `ozone_mass_mixing_ratio`, `eastward_wind`, `northward_wind` -- no
`specific_total_water_<i>` (it is `specific_humidity_<i>` +
`specific_total_condensate_<i>`). The default `--layer_families ace2` is
unchanged, so the existing 8/16/32-layer datasets rebuild identically. Files
carry the `layer_families` attribute. One more ERA5 model-level field read
per time (ozone: 9 vs 8). Can be combined with `--extra_layer_vars` (e.g.
the individual condensates for all-sky work); repeating a family there is an
error.

- Validated (Hercules login node, 2023-01-01T00, 8 layers, one
  `process_time`, ~80 s): vs the 8-layer `2023010100.nc`, vapor + condensate
  = total water to float32 rounding (relative max error <= 1e-7, layer 0
  exactly); the other 147 fields bit-identical; condensate >= 0 (max 3 g/kg,
  layer 4; 0 in layer 0); ozone layer-mean 7.8e-6 kg/kg in layer 0 down to
  6e-8 in layer 7.
- **`add_layer_families` mode** adds a set's missing families to existing
  months instead of rebuilding (`ace2_dataset.py add_layer_families OUTDIR
  START END [--layer_families crtm]`; Hercules launcher `MODE=add_layer_families`).
  Why not rebuild: per time step at 16 layers a full `build` takes ~80-110 s
  per worker, while each ERA5 model-level field reads in 0.5-2 s and coarsens
  in <1 s -- most of a build is the other streams and the regridding. The
  patch reads only vapor, ozone, the four condensates and surface pressure
  (the condensates are read, not taken as total water - vapor, so empty
  cells stay exactly 0 and the result equals a fresh `crtm` build). It uses
  the files' own layer indices and model top, keeps total water (so the
  files serve both sets; fme ignores unused variables), refuses a month
  unless vapor + condensate matches its total water within 1e-7 kg/kg, and
  writes a .tmp copy renamed over the original; done months carry
  `layer_families = "ace2 crtm"` and are skipped. Validated on a 3-time copy
  of the 16-layer 2020-01: the 48 added fields bit-identical (values and
  attributes) to `process_time` with `crtm`, existing fields and attributes
  unchanged, same (1, 180, 360) chunking, total water check 2.2e-9 kg/kg.
  +48 fields at 16 layers = +1.5 GB/month (~0.9 TB for 1979-2025).
- For training (not done): the ACE2-ERA5 stepper config's corrector
  (`conserve_dry_air`, `moisture_budget_correction`) and `force_positive_names`
  refer to `specific_total_water_<i>`, which these files lack: check how fme
  2026.5's `ClimateData` finds total water before reusing it, and add the new
  families to `force_positive_names`. The condensate is intermittent and
  skewed (zero in most cells), so its normalization may need a transform.

## Training: 16 layers, CRTM variables (2026-10-27)

Config `backends/ace2/training/ace2-crtm-16layer-train-config.yaml`
(generated from the 8-layer ACE2-ERA5-vars config; same architecture,
forcings, diagnostics, training settings, corrector), data
`ace2_era5_1deg_16layer` on Eagle (564 months 1979-01..2025-12, crtm
families added). Prognostic layer families: `air_temperature`,
`specific_humidity`, `specific_total_condensate`, `ozone_mass_mixing_ratio`,
`eastward_wind`, `northward_wind` x 16 layers -- 108 input / 114 output
channels (44/50 at 8 layers). Train 1979-2014 + 2016-2025, 2015 held out.
Experiment dir `ace2_training/ace2-crtm-16layer-1979-2025-no2015` (never
run: before job 7794813 started the config gained the land variables, next
section).

- **Total water for the corrector.** fme 2026.5 finds total water only by
  the `specific_total_water_` prefix (`ATMOSPHERE_FIELD_NAME_PREFIXES` in
  `fme/core/atmosphere_data.py`). `fme_train.py` now also replaces
  `AtmosphereData.specific_total_water` with one that falls back to
  `specific_humidity_<k>` + `specific_total_condensate_<k>` when the total is
  absent (data with the total are unaffected). The correctors
  (`conserve_dry_air`, `moisture_budget_correction:
  advection_and_precipitation`) and the derived total water path only read
  total water, so that is all they need: dry air is conserved by adjusting
  PRESsfc with total water = q + c, and the column budget closes on the
  TWP of q + c. Checked on CPU with synthetic 16-layer data: total water
  equals q + c exactly; dry-air global-mean imbalance 178 Pa -> 0.001 Pa;
  column budget residual 7e-12; no water fields at all still raises
  KeyError. `force_positive_names` covers all three water/ozone families.
  **Inference (lw4dvar's `ace2_model.py`) will need the same patch** to run
  a CRTM checkpoint's corrector.
- **Condensate normalization floor.** In the 16-layer stats
  (`stats-1990-01-2019-12`, 1990-2019) condensate std is 8.6e-12 kg/kg in
  layer 0 (0.01-5.6 hPa), 3.6e-9 in layer 1 (5.6-51 hPa), 2.6e-7 in layer 2
  (51-91 hPa), then 4.7e-6 and up. Layer 0/1 condensate is ERA5
  stratospheric noise: zero in all but ~1e-5 / 1e-2 of cells, with rare
  values 130-580 sigma (layer 2-3 tropopause cirrus is heavy-tailed too, up
  to ~400 sigma, but physical). Training uses
  `stats-1990-01-2019-12-condensate-floor`: a copy with the full-field and
  residual std of `specific_total_condensate_<k>` floored at 1e-6 kg/kg
  (layers 0-2 changed; recorded in the files' `history`). Plain linear
  normalization otherwise (fme has no transform; a log/sqrt transform would
  break q + c = total water); watch the condensate channels.
- **Loss weights:** fme multiplies prediction and target by the weight
  before the MSE (`VariableWeightingLoss`), so a channel's share of the loss
  goes as w^2. To keep each family's 8-layer share when a layer splits into
  n sub-layers, each gets w8 / sqrt(n) (fixed 2026-10-28; the first version
  copied w8 to both halves, doubling the 3-D share, and the 32-layer Ursa
  config's w8 / 4 cut it to a quarter). 16 layers (8-layer k = 2k, 2k+1):
  air_temperature_0-3, winds 0-1: 0.354, rest 0.707. Vapor and condensate
  share one set (user's choice, 2026-10-28): layers 0-3 (top-144 hPa;
  global-mean layer pressures 2.8, 28, 71, 117 hPa) 0.1 (first 0.01, like
  the 32-layer config's stratospheric total water; raised to 0.1 before the
  run started) -- not ACE2-ERA5's non-monotonic 0.5/0.25 for its top two
  total-water layers; 4-5 (144-252 hPa) 0.354;
  6-15 0.707. Ozone (stratosphere-dominated; user's choice, 2026-10-28):
  layers 0-5 (top-252 hPa) 0.707, 6-7 (252-401 hPa) 0.354, 8-15 (below
  ~401 hPa) 0.1 (untuned). Loss shares: 2-D fields ~83%, each 3-D family 2-4%
  (8-layer: 87%, 3-4%).
- Memory: like the 8-layer model (probe: 4L = 96 layer channels changes
  peak memory by <0.3 GB at 1 sample/GPU) -- 4 nodes, no checkpointing.

## Training: 16 layers, CRTM + land (2026-10-28)

For satellite surface emissivity, `ace2-crtm-16layer-train-config.yaml` (and
the untracked `smoke-crtm16.yaml`) were extended, while job 7794813 was still
queued, with four prognostics: `soil_moisture_0` (ERA5 swvl1, 0-7 cm), `soil_temperature_0`
(stl1), `surface_snow_amount_capped` and `surface_snow_area_fraction` --
112 input / 118 output channels. Experiment dir
`ace2_training/ace2-crtm-land-16layer-1979-2025-no2015`. CRTM's emissivity models also want 10 m
wind, skin temperature and the surface-type fractions, which the model
already has; what it lacked was snow and land state (sea ice is a forcing).
Only the top soil layer (the user's choice: it's the one emissivity sees).

- **Data, 16-layer files (2015-01/07, stats 1990-2019):** no NaNs. Over
  ocean soil moisture and snow are 0, soil temperature ~SST. Residual/full
  std: soil_moisture_0 0.029, soil_temperature_0 0.11, snow fraction 0.026,
  snow amount 0.0002 (deeper soil layers 0.001-0.02). soil_temperature_3's
  residual std (0.40 K) is larger than layer 2's (0.09) -- unexplained, not
  used.
- **Snow amount capped at 1000 kg/m2 (1 m water equivalent):** ERA5 pins
  snow at 1e4 kg/m2 on the ice sheets (6717 cells, 2.75% of the globe, every
  month of 2015) and partly glaciated cells hold thousands (snowy non-cap
  cells: median 41, 90th percentile 3990 kg/m2), so the std is 1720 (562
  without cap cells) and seasonal snow gets a tiny normalized range.
  Capped at 1000: mean ~40, std ~190 (area-weighted, 2015). Emissivity
  doesn't see deeper snow. The 1990-2019 normalization stats (unweighted
  over grid cells, like every other variable, so polar ice counts more):
  uncapped mean 1139 / std 3123 / residual std 0.62; capped mean 127 / std
  326 / residual std 0.36 kg/m2.
- **Sidecar dataset:** `ace2_dataset.py capped_snow OUTDIR SIDECAR START END
  [--cap 1000]` writes `ace2_era5_1deg_16layer/snow-capped-1000/YYYYMM0100.nc`
  (just `surface_snow_amount_capped`, OUTDIR's time/lat/lon/attributes,
  ~5 s/month on the login node), merged with the main files by fme
  (`merge:` in every loader; inference uses MergeNoConcatDatasetConfig). The
  training files are untouched, so the queued CRTM run is unaffected.
- **Normalization:** `stats-1990-01-2019-12-condensate-floor-land` = the
  condensate-floor stats + the sidecar's own (`stats` mode on the sidecar).
- **Corrector:** soil moisture and both snow fields force_positive; snow
  fraction can exceed 1 (fme has no upper bound), clip downstream. Soil and
  snow water are outside fme's moisture budget (atmosphere only).
- **Loss weights:** `surface_snow_amount_capped` 0.1, the other three 1.
  Smoke test at weight 1 (job 7795087, `smoke-crtm-land16.yaml`, 2 epochs):
  valid loss 317 -> 142 and inference error 0.72 -> 0.71, vs 20 -> 12 and
  0.61 -> 0.48 without land. The loss normalizes by the residual std, the
  network predicts full fields, so a field-sized error costs
  (w * full/residual std)^2: snow amount's ratio is 899 (326 / 0.36 kg/m2;
  the residual std is a global mean dominated by cells where snow never
  changes), next largest snow fraction 38, PRESsfc 38, soil moisture 34,
  h500 14 (weight 10). No single weight balances both regimes: 0.04 puts
  field-sized errors at PRESsfc's level (~1300) but tendency-sized errors at
  w^2 = 0.0016; 0.1 (chosen) gives ~8000 (between TMP850 2500 and h500
  19000) and w^2 = 0.01. Judge it by snow in a free run. Smoke test at 0.1
  (job 7795267, `smoke-crtm16.yaml`): valid loss 47.7 -> 21.6, inference
  error 0.69 -> 0.55, no NaN, 13 min like the others; still above the
  no-land run (the new channels start with large errors) but falling faster.
- **Risk:** ACE2 predicts the full field each step (`residual_prediction:
  false`, all channels), and soil memory is slow (top soil moisture changes
  ~3% of its std per 6 h); check snow cover and soil moisture drift in a
  30-60 day free run, not just the 6-h loss.

## Training on Ursa MI300X (2026-10-28)

Ursa's `u1-mi300x` partition: 5 nodes (e01g[01-02], e02g[01-03]), each 8 x
192 GB AMD MI300X, 96 CPUs, 2 TB RAM, one mlx5 InfiniBand HCA, 1 TB tmpfs
`/tmp`, ROCm 7.2.4 / amdgpu 6.16 driver, no internet. Account `gpu-ai4wp`,
QOS `gpuwf`, the only one allowed there: at most 8 MI300X per user (so
**one node**; multi-node is refused), 11520 GPU-min running per account (=
one node for **24 h**), and **one submitted job per user** (no
`--dependency` chains, no test jobs while a run is queued).

- **Env `ace2-rocm`** (`/scratch4/BMC/gsienkf/whitaker/conda/envs/ace2-rocm`,
  `ace2-rocm-requirements.txt`): the `ace2` env rebuilt on a login node with
  only torch swapped for the same version's ROCm wheel --
  `conda create --prefix $P --file ace2-spec.txt`, then
  `pip install torch==2.7.1+rocm6.3 torchvision==0.22.1+rocm6.3 --index-url
  https://download.pytorch.org/whl/rocm6.3`, then the `ace2` lock minus
  torch/torchvision/triton/nvidia-*. The wheel bundles its own ROCm and RCCL
  (no `module load rocm`); torch-harmonics is pure Python. fme needs no
  changes: `torch.cuda` and `backend="nccl"` map to HIP and RCCL.
  `~/.local` site-packages leak into conda envs (pip took `requests` from
  there), so the launcher sets `PYTHONNOUSERSITE=1`.
- **Launcher `run_ace2_train_mi300x.sh`**: one `torchrun --nproc_per_node 8`
  per node; default 1 node = 2 samples/GPU at batch 16, no checkpointing
  needed. `expandable_segments` is unsupported in this torch build (dropped).
- **Per-rank MIOpen caches (`backends/ace2/training/rocm_rank_env.sh`,
  started by torchrun with `--no-python`, then runs `fme_train.py`).** fme
  sets `torch.backends.cudnn.benchmark = True`, which on ROCm is MIOpen's
  exhaustive convolution search; with all 8 ranks sharing one MIOpen user DB
  (sqlite) in `/tmp`, rank 3 segfaulted 4 min in, still tuning (job
  24889236). With a DB per rank both the exhaustive and the
  `MIOPEN_FIND_MODE=FAST` runs work, with identical losses (to ~1e-8) and
  speed, so the default is kept.
- **Smoke test** (`ace2-smoke-mi300x.yaml`, untracked = `ace2-smoke-ursa.yaml`
  with its own experiment dir; jobs 24889407/8): valid loss 22.04 -> 12.61,
  inference error 0.569 -> 0.452 (H100: 22.92 -> 12.96, 0.577 -> 0.456).
- **Throughput** (`ace2-rate-mi300x.yaml`, untracked = the Ursa rate test;
  2 epochs on 2016-2017, jobs 24890813/4), steady (epoch 2): 183 batches in
  142 s = **0.78 s/batch on 1 node vs 0.97 on 4 u1-h100 nodes**; whole epoch
  incl. validation and 80-step inference 259 s (4 H100 nodes 272 s). Inline
  inference ~0.33 s/step. Every job first spends ~5-6 min warming up before
  step 1 at the same rate in both MIOpen modes (so not conv tuning; cause not
  found) -- not seen when the 16-layer run resumed (job 25001695). Estimate for a full run (~3930
  batches/epoch): ~51 min training + ~3 min validation + ~8 min 1440-step
  inference, ~1 h/epoch, **~5 days for 120 epochs** on one node, if 16 layers
  run like 8 (the memory probe says they do).
- **16-layer CRTM + land on Ursa:** `ace2-crtm-16layer-train-config-ursa.yaml`
  = `ace2-crtm-16layer-train-config.yaml` with Eagle paths -> `/scratch4/BMC/gsienkf/whitaker`
  and **seed 1** (Polaris: 0), a second realization (different init and
  shuffling); regenerate it from the Polaris config, don't edit it. Data
  `ace2_era5_1deg_16layer` (+ `snow-capped-1000/`,
  `stats-1990-01-2019-12-condensate-floor-land/`) copied from Polaris,
  staged via `/scratch3/data_untrusted`.
- **Resubmission (`resubmit_ace2_mi300x.sh`, cron on a login node every 15
  min):** since jobs can't be chained, it submits the next 24-h job when none
  is queued or running and the last one ended in TIMEOUT/NODE_FAIL/PREEMPTED;
  any other end state writes `ace2_crtm_land16_mi300x.STOP` and it stops
  (fix, delete STOP). It also stops at "Time taken for epoch 120" (then
  remove the crontab line). It passes the loader settings below as an fme
  `--override` (since 2026-10-30 also in the Polaris config, so redundant
  once the Ursa config is regenerated from it).
- **16-layer data loading was the bottleneck, fixed with 16 loader workers
  per rank (`train_loader.num_data_workers=16 prefetch_factor=2`).** With 8
  workers (prefetch 4, then the Polaris setting) the first job (24908976) went
  from 1.2 s/batch (epoch 1) to 1.4-2.0 (epochs 2-10), slower at busy
  daytime hours, with some GPUs ~20-30% busy while others were ~85%; epochs
  took 107-164 min. fme reads each variable separately and the files have
  one chunk per variable per time (180x360 float32, ~260 KB), so a sample (3
  times x 123 variables, ~93 MB) is 369 small reads, 1.5-8 s from one
  process on a login node: latency-bound, not bandwidth-bound. Twice the
  reads in flight (same 32 batches buffered per rank): **0.71-0.78 s/batch**
  from the start of job 25001695 (overnight), all GPUs 88-99% busy. If it
  holds, ~70 min/epoch with validation (~5 min) and inline inference (~9.5
  min). Next steps if not: rewrite with ~8-time chunks; run inline
  inference every Nth epoch (`inference.epochs`, only affects
  `best_inference_ckpt`).
- **Polaris uses the same loader settings** (2026-10-30, before job 7794813
  started): `train_loader` 16 workers, prefetch 2 in
  `ace2-crtm-16layer-train-config.yaml` itself, since a queued PBS job reads
  the config at launch but keeps the launcher script it was submitted with.
  Same 32 samples (~3 GB) buffered per rank; 64 loader processes per node on
  32 cores and 512 GB (they mostly wait on Lustre). Untested on Polaris:
  check s/batch and GPU utilization early in the run.

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
# Polaris: normalization stats (debug node, ~25 min)
qsub -q debug -l walltime=01:00:00 -o ace2_stats.out -e ace2_stats.err -v MODE=stats run_ace2_dataset_polaris.sh
# Polaris: training (24-h jobs; resubmit/chain until max_epochs -- fme resumes)
qsub run_ace2_train_polaris.sh     # 4 nodes, <= 168 h
# Ursa: normalization stats (u1-service), then training (u1-h100, <= 7 days)
sbatch -t 02:00:00 -c 64 --mem=240g -o ace2_stats.out -e ace2_stats.err --export=ALL,MODE=stats,WORKERS=12 run_ace2_dataset_ursa.sh
sbatch run_ace2_train_ursa.sh      # 4 nodes; -N 2..8
# Ursa: training on u1-mi300x (1 node = 8 MI300X, <= 7 days)
sbatch --export=ALL,CONFIG=backends/ace2/training/ace2-crtm-16layer-train-config-ursa.yaml run_ace2_train_mi300x.sh
# Ursa: patch PRESsfc in months built before 2026-10-10 (own log files)
sbatch -o ace2_fix_presfc.out -e ace2_fix_presfc.err \
    --export=ALL,MODE=fix_presfc,START=1979-01,END=1988-12 run_ace2_dataset_ursa.sh
```
Default OUTDIR on Polaris `/lus/eagle/projects/moonshot-reanalysis/jwhitaker/ace2_era5_1deg_8layer`,
on Ursa `/scratch4/BMC/gsienkf/Jeffrey.Whitaker/ace2_era5_1deg_8layer`
(co2.nc staged in both).

## Running our checkpoint in lw4dvar (2026-10-20)

Ursa run 23736094 finished all 120 epochs (5 d 4 h, 471,120 batches): train
loss 0.0992, valid loss 0.1490 (lowest at epoch 120, so `best_ckpt.tar` =
epoch 120), inline inference error 0.0629 (best 0.0584, epoch ~100,
`best_inference_ckpt.tar`; flat at 0.062-0.068 from epoch ~20 on). With
`validate_using_ema: true` fme saves both best checkpoints with the EMA
weights.

- **Checkpoint layout:** fme 2026.5 writes the stepper state as
  `config.step.config` + `dataset_info` (grid, `vertical_coordinate`,
  `timestep`); the released ACE2-ERA5 checkpoint has the legacy flat layout
  (`sigma_coordinates`, `encoded_timestep`, `img_shape`). `ACE2Model` reads
  both; `Stepper.from_state` takes either.
- **Forcings and ICs must come from our dataset**
  (`backends/ace2/ace2_inference_files.py OUT --years ... --ic ...
  --ic_template ... --link_verif ic_cache_ace2`): our HGTsfc differs from
  Ai2's by 41 m rms and our model top is ~1 Pa; the network takes HGTsfc as
  input and the ps operator uses it. Our 2014-12-30T00 IC differs from the
  ace2_ic.py (HF-HGTsfc) one by 413 Pa rms in PRESsfc, identical elsewhere.
  ERA5 h500/TMP850 verification is linked, so z500 scores compare directly.
  Files for 2014-2015: `/scratch4/BMC/gsienkf/whitaker/ace2_training/inference_ours/`;
  on Polaris `/lus/eagle/projects/moonshot-reanalysis/jwhitaker/ace2_training/inference_ours/`
  (built 2026-10-28 on a debug node, ~1 min; the login node's 8 GB cap is too
  tight). Without `exp.forcing_dir` lw4dvar looks in `path_model/forcing_data/`
  and stops with "no fetched forcing_*.nc".
- **Free rollout 2015-01-01..07-13** (`free_rollout: True`, 390 x 12 h,
  same setup as the official-checkpoint and AIFS rollouts of issue #12):
  48-h z500 rms 7.6 m vs 8.9 m for the official checkpoint. Bias change from
  Jan 1-10 to Jun 14-Jul 13 (ours / official / AIFS): NH t250 -0.8 / -0.2 /
  +13.6 K, SH t250 +0.4 / +1.3 / +11.3 K, GL z500 -4 / -0.5 / +22 m, GL z250
  -12 / -2 / +55 m, GL z50 **-40** / -9 / +138 m, GL t850 0.0 / +0.2 / -0.5 K,
  GL sp 0 / 0 / -2.5 hPa. As stable as the official checkpoint in the
  troposphere; a slow stratospheric cooling from ~May (z50 -40 m) is the one
  new signal -- untested leads: our ~1 Pa model top, or the seed (upstream
  picks 1 of 4 by multi-year evaluation).
- **Cycled ps 4D-Var:** `config_ace2ours_6h_50it_5day_lr1e-3_jc25_obden_{a,b}.yml`
  (= the official-checkpoint run test_ace2_6h_50it_5day_lr1e-3_jc25_diurnal_obden
  minus `diurnal_clim`, a no-op at dt_obs 6), jobs 24136025 -> 24136026.

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
- [x] Full 8-layer build, 1979-01..2022-12 (finished 2026-10-12, mostly on
      Hercules; copied to Eagle `ace2_era5_1deg_8layer/`, 2.4 TB): 528
      months, 64284 contiguous 6-h times, one variable set (187 vars),
      every file `presfc_reduced_to_hgtsfc=1`, `top_interface_midpoint=1`,
      layers 0 48 67 79 90 100 109 119 137. File sizes exact per month
      length.
- [x] Normalization stats: `ace2_era5_1deg_8layer/stats-1990-01-2019-12/`
      (see "Normalization statistics" above).
- [ ] Training (see "Training" above): config, launcher, smoke, resume and
      scaling tests done; full 120-epoch run on 4 nodes submitted
      2026-10-12 (job 7741004, capacity queue, seed 0). Upstream trains 4
      seeds and picks one by multi-year evaluation (time-mean climate and
      trends) -- a follow-on step if we need it.
- [x] Ursa run (see "Training on Ursa"): config, launcher, stats, smoke and
      throughput tests done; full 120-epoch run on 4 nodes, job 23736094
      (seed 0), finished 2026-10-20 in 5 d 4 h, experiment dir
      `/scratch4/BMC/gsienkf/whitaker/ace2_training/ace2-era5-vars-1979-2022-no2015`
      (results: "Running our checkpoint in lw4dvar").
- [x] Yearly forcing files from our dataset for inference in lw4dvar
      (`ace2_inference_files.py`; see "Running our checkpoint in lw4dvar").
- [ ] 32-layer build, 1979-01..2022-12 (see "32-layer vertical grid"):
      submitted 2026-10-14 on Hercules as 44 one-year jobs (10114650-93;
      ~3.4 h each, 8-h QOS cap) in 4 `--dependency=singleton` chains
      (`-J ace2ds32_chain<N>`, years round-robin), logs in
      `logs_ace2ds32/YYYY.{out,err}`, into
      `/work2/noaa/gsienkf/whitaker/ace2_era5_1deg_32layer` (2020-01 from
      the test reused). Then: normalization stats, copy to Ursa/Eagle.
- [ ] Higher vertical resolution + extra layer fields for radiances. Memory
      probe done (see "Memory probe" above): up to 48 layers fit at embed
      384 on the current setup; embed 512 fits on 4 nodes; 768 needs
      ZeRO/FSDP or AMP.
- [ ] Split total water into vapor / liquid / ice for radiances: decided
      2026-10-26 to make vapor, total condensate and ozone prognostic
      (`--layer_families crtm`; see "CRTM layer variables"). Builder done
      and validated on one time; dataset build, stats and training config
      not yet.
- [x] CO2 after 2022 (NOAA GML; see "CO2 after 2022" above).
- [ ] 2023-01..2025-12 for both datasets (2026-10-18, Hercules `service`):
      32-layer one job per year (10159808-10, chains `ace2ds_new_chain1-3`)
      into `ace2_era5_1deg_32layer`; 8-layer 2023/2024/2025 chained
      (10159811-13, `ace2ds_new_chain4`) into a recreated
      `/work2/noaa/gsienkf/whitaker/ace2_era5_1deg_8layer` holding only
      2023-2025 (its 1979-2022 files were deleted on Hercules after backup).
      Logs `logs_ace2ds_2023-2025/{8L,32L}_YYYY.{out,err}`.
- [ ] 16-layer build, 1979-01..2025-12 (2026-10-20, Hercules `service`):
      nested, every other 32-layer interface (each 8-layer layer split in
      two, so it contains all 9 original interfaces and is itself contained
      in the 32-layer grid):
      `--layer_indices 0 24 48 58 67 73 79 84 90 95 100 104 109 114 119 128 137`.
      47 one-year jobs (10180627-73) in 4 `--dependency=singleton` chains
      (`-J ace2ds16_chain<N>`, years round-robin), logs
      `logs_ace2ds16/YYYY.{out,err}`, into
      `/work2/noaa/gsienkf/whitaker/ace2_era5_1deg_16layer` (`co2.nc`
      1940-2025, `co2_1940-2022.nc`, `invariant.nc` and regrid weights
      copied from the 32-layer dir). Then: check vs 8/32-layer, stats.
- [ ] CRTM families added to the 16-layer dataset (2026-10-26, Hercules
      `service`, `MODE=add_layer_families`): 8 jobs of ~6 years
      (10278388-95: 1979-84, 85-90, 91-96, 97-02, 03-08, 09-14, 15-20,
      21-25) in 4 `--dependency=singleton` chains (`-J ace2crtm16_chain<N>`),
      logs `logs_ace2ds16_crtm/YYYY-YYYY.{out,err}`. Then: stats with the new
      fields, copy to Ursa/Polaris.
      First months: 4.3-4.4 min/month (16 workers), total water check
      ~2.4e-9 kg/kg; ~5.3 h per 6-year job.
- [ ] CRTM families added to the 32-layer dataset (2026-10-26, same mode):
      16 jobs of 3 years (10278432-47: 1979-81, ..., 2021-23, 2024-25) in 4
      singleton chains (`-J ace2crtm32_chain<N>`), each also `afterok` on
      all eight 16-layer jobs; logs `logs_ace2ds32_crtm/YYYY-YYYY.{out,err}`.
      +96 fields = +3 GB/month (~1.7 TB). If a 16-layer job fails these stay
      pending (DependencyNeverSatisfied): fix, rerun it, resubmit them.
- [ ] 16-layer normalization stats incl. the crtm families (2026-10-26):
      job 10280347 (`MODE=stats` in `run_ace2_dataset_hercules.sh`, 8
      workers, 100 GB -- fits beside a 32-layer job on a DTN node),
      `afterok` on the 16-layer jobs covering 1985-2020, into
      `ace2_era5_1deg_16layer/stats-1990-01-2019-12/`; log
      `logs_ace2ds16_crtm/stats.{out,err}`. Fixed first: `_month_stats` kept
      each variable's whole month array alive through the `first`/`last`
      views (23 GB per worker at 16 layers, 236 vars; now 8.4 GB, results
      identical) -- likely also the cause of the ~14 GB/worker OOMs on Ursa.
- [ ] 16-layer CRTM training on Polaris (see "Training: 16 layers, CRTM
      variables"): config `ace2-crtm-16layer-train-config.yaml`, total-water
      patch in `fme_train.py`, condensate-floor stats, loss weights (see
      that section). 1-node smoke test passed 2026-10-28 (job 7790856,
      `smoke-crtm16.yaml`, untracked; 2 epochs, 1.19 samples/s/GPU like 8
      layers, inference error 0.61 -> 0.48, no NaN), after fixing three
      launch problems: the NCCL plugin (above), TMPDIR (below), and inline
      inference OOM at forward_steps_in_memory 40 (now 20). Full 120-epoch
      run on 4 nodes submitted 2026-10-28 (job 7794813, capacity queue,
      seed 0, logs `ace2_crtm16.{out,err}`); the config gained the land
      variables before it started ("Training: 16 layers, CRTM + land"),
      experiment dir
      `/lus/eagle/projects/moonshot-reanalysis/jwhitaker/ace2_training/ace2-crtm-land-16layer-1979-2025-no2015`.
- [ ] 16-layer CRTM + land training on Ursa MI300X (see "Training on Ursa
      MI300X"): env, launcher, 8-layer smoke and throughput tests done
      2026-10-28; `ace2-crtm-16layer-train-config-ursa.yaml` (seed 1)
      running since 2026-10-29 (job 24908976, then 25001695 with 16 loader
      workers), resubmitted every 24 h by cron (`resubmit_ace2_mi300x.sh`,
      logs `ace2_crtm_land16_mi300x_<job>.{out,err}`); 11 epochs done by
      2026-10-30, valid loss 0.564 -> 0.199.
