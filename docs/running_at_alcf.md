# Running at ALCF (Polaris)

What it took to port `lw4dvar-torch` from the NOAA RDHPCS Slurm systems
(`/scratch4/...` envs, H100 nodes) to ALCF Polaris (PBS, 4 x 40 GB A100 per
node). So far only the **ACE2** backend has been ported; the other backends
would need their own envs built the same way.

## Locations

| What | Path |
|---|---|
| repo | `/lus/eagle/projects/moonshot-reanalysis/jwhitaker/lw4dvar-torch` |
| `ace2` conda env | `/home/jwhitaker/.conda/envs/ace2` |
| ps obs | `/lus/eagle/projects/moonshot-reanalysis/jwhitaker/psobs` |
| ACE2 IC / verification cache | `/lus/eagle/projects/moonshot-reanalysis/jwhitaker/ic_cache_ace2` |
| ACE2 checkpoint + forcing | `backends/ace2/ACE2-ERA5/` (checkpoint; forcing 2014, 2015, 2020, 2021 fetched) |
| PBS project | `moonshot-reanalysis` |

## Building the `ace2` env (login node)

ALCF conda: `module use /soft/modulefiles; module load conda; conda activate base`.

1. Conda layer (conda-forge Python 3.12.14 + pip):
   `conda create --name ace2 --file ace2-spec.txt`
2. torch from the PyTorch cu128 index. **`pip install -r
   ace2-requirements.txt` alone fails**: the `+cu128` pins are not on PyPI.
   ```
   pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 triton==3.3.1 \
       --index-url https://download.pytorch.org/whl/cu128
   ```
3. torch-harmonics 0.8.0 from source with its CUDA kernels. The login node
   has no GPU, so `setup.py` silently skips the CUDA extensions unless forced:
   ```
   module load cudatoolkit-standalone/12.8.1     # gcc-native 14 is the host compiler
   export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.8.1
   export FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST="8.0"   # A100
   pip download --no-deps --no-binary :all: torch_harmonics==0.8.0
   pip install --no-build-isolation --no-deps torch_harmonics-0.8.0.tar.gz
   ```
   Takes >10 min (no ninja, so the nvcc compiles run serially). Check with
   `python -c "import torch, disco_cuda_extension, attention_cuda_extension"`
   (torch first -- the extensions need torch's libraries already loaded).
4. Everything else, with torch held fixed:
   ```
   printf "torch==2.7.1+cu128\ntorchvision==0.22.1+cu128\ntriton==3.3.1\ntorch_harmonics==0.8.0\n" > constraints.txt
   pip install -r ace2-requirements.txt -c constraints.txt \
       --extra-index-url https://download.pytorch.org/whl/cu128
   pip check
   ```
   `ace2-requirements.txt` had a `packaging @ file:///home/conda/feedstock_root/...`
   line (a `pip freeze` artifact from the conda build) that cannot install
   anywhere; it was removed -- conda's `packaging` 26.3 from `ace2-spec.txt`
   provides it. The result matches `ace2-requirements.txt` exactly.

## Launcher: `run_ace2_polaris.sh`

```
qsub run_ace2_polaris.sh                                   # config_test_ace2_polaris.yml
qsub -v CONFIG=my_config.yml run_ace2_polaris.sh
qsub -q debug -l walltime=00:30:00 run_ace2_polaris.sh     # short test
```

- **Queues**: `capacity` (1-4 nodes, <= 168 h) by default; `debug` (1-2
  nodes, <= 1 h) for tests. `prod` is for >= 10 nodes. One run uses one of
  the node's four GPUs.
- **PBS does not pass script arguments**, so the config comes in via
  `-v CONFIG=...` instead of `$1`; the script `cd`s to `$PBS_O_WORKDIR`.
- **Module order matters**: `module load conda` puts CUDA 12.9.1's
  libraries on `LD_LIBRARY_PATH`, so `cudatoolkit-standalone/12.8.1` must be
  loaded **after** it to match the build and torch's cu128 runtime. (With no
  CUDA module at all, the extensions use the CUDA 12.8 runtime bundled with
  torch, which is also fine.)

## Config changes: `config_test_ace2_polaris.yml`

A copy of `config_test_ace2.yml` (left unchanged for the NOAA systems) with:

- `obspath` and `ic_cache` pointing at the Eagle paths above.
- `n_init: 2` and `cycle: True` -- `load_config` rejects `n_init > 1` with
  `cycle: False`.
- **`checkpoint_stride: 1`** (was 2): stride 2 runs out of memory on a
  40 GB A100 for this 20-step (5-day) window -- it only fits on the 80 GB
  H100. Stride 1 checkpoints every step and uses the least memory.

## Validation run (job 7720785, 2026-10-06)

`debug` queue, 1 A100, 2 cycles x 50 epochs, 8.5 min wall, exit 0.
~4.5 s/epoch for the 20-step window.

| Cycle | Loss, epoch 1 -> 50 | z500 err at +6h (bg -> analysis) |
|---|---|---|
| 2015-01-01T00 | 502271 -> 341387 | 14.00 -> 13.74 |
| 2015-01-01T06 | 343586 -> 309785 | 14.87 -> 14.80 |

The second cycle's background error at t0 equals the first cycle's
analysis error, so cycling works.

Logged each cycle, and harmless for ACE2: `control_variables contains none of
the forward-operator fields [p_lowest, q_lowest, sp, t_lowest, z]`. The check
compares decoded field names with ACE2's native variable names (`PRESsfc`,
`air_temperature`, ...), which never match; the nonzero first-step `sp`
increment shows the first-6h obs do get a gradient.

## Sophia (not working yet)

Sophia (DGX A100 nodes, its own PBS server) cannot be reached from Polaris:
`qstat @sophia-pbs-01...` fails MUNGE authentication and ssh needs
interactive MFA. Submitting needs a Sophia login, which we do not have yet.
