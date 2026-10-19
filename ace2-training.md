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
- [ ] Ursa run (see "Training on Ursa"): config, launcher, stats, smoke and
      throughput tests done; full run to be submitted after the Ursa
      maintenance starting 2026-10-14T12:00 (`sbatch run_ace2_train_ursa.sh`).
- [ ] Yearly forcing files from our dataset for inference in lw4dvar.
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
- [ ] CO2 after 2022.
