# Layer-model (L2/L3/L3W/L3N/L3S) evaluation

Portable copies (2026-11-25) of Sergey's untracked evaluation scripts. Run from **your own lwatms checkout**: every
output (logs/, output/, figures/, eval/layer_models/cfg/) goes there.

| Variable | Default | Meaning |
|---|---|---|
| `LWATMS` | the repo these scripts sit in | your writable checkout |
| `RUNROOT` | `/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/long-window/EAGLE/run` | dir with `aifs-r2-<arm>/training` (checkpoints) |
| `CONDA_SH` | Sergey's `eagle/latent/EAGLE/conda/etc/profile.d/conda.sh` | provides `anemoi-rocm` (GPU) and `anemoi-dev` (CPU) |
| `ACCOUNT` | `gsienkf` | Slurm account |
| `DRYRUN` | `0` | `1` prints the sbatch lines instead of submitting |
| `REF_LWATMS` | Sergey's lwatms | where `gate_summary.py` finds the T1 / L3 reference rows if your tree lacks them |

```bash
DRYRUN=1 bash eval/layer_models/l3_variant_eval.sh l3s      # check first
bash eval/layer_models/l3_variant_eval.sh l3s               # 6 free runs + 71-start forecast bias + gate table
cat output/freerun_diag/gates_l3s.txt
```

- `r2_freeruns.sh <arm>`: 90-day fake-cycled free runs from 2015-01-01 (final + epochs 4–20), one MI300X each, 3 h. Uses `fr_base.yml`.
- `fcst_bias_maps.py`: forecasts from ERA5, 71 starts. Config built from `cycle_gold_ice.yml` with the layer backend.
- `drift.py`, `runaway.py`, `gate_summary.py`: drift statistics, Southern Ocean runaway diagnostic, G2/G3 gate table.
- Data paths in the configs (ERA5 zarrs, obs) point at Sergey's read-only copies.
