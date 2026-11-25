# Decoder bias / observation-error tables (H2, H2b, H2i, H2iR)

Portable copies (2026-11-25) of the scripts that turn an exported decoder bundle into the tables lwatms reads.
The full recipe, from decoder training to the cycling config, is in
`eagle/long-window/HANDOFF_layer_models.md`, section "Decoders for H2i / H2iR".

- `dec_tables.sh` (sbatch, ~4 min on 96 CPUs): `TAG=<tag> BA=<atms bundle> BM=<amsua bundle> BI=<iasi bundle>`.
  Writes `figures/dec_<tag>/`:
  - `oberr_table_w_<tag>.json` (scene/cloud R);
  - `decoder_era5_<tag>.json` (ATMS sigma + inter-channel corr);
  - `atms_<tag>_era5_2015_ch10-13.json` (extra_bc);
  - the AMSU-A and IASI extra_bc tables.
- `run_fit.sh SENSOR BUNDLE PRE obsbc OUT_PREFIX`: per-scene bias table (`fit_scene_bc.py`). Env `SCBC_CLW_CAP=0.6` gives
  the H2b/H2i/H2iR variant. Then `restrict_channels.py IN.json OUT.json 5,6,7`.
- `mw_ref.py`, `make_extra_bc.py`: IASI 77-channel extra_bc (`CHANNELS=1-175`).
- Env: `LWATMS` (your checkout), `VENV` (default Sergey's `envs/lwaifs2-rocm`).
