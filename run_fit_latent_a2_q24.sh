#!/bin/bash
# A2 latent Q24 samples (24-h forecast error from ERA5 in the latent control space, L-BFGS 60 as the A0 accept_a samples)
# for latent_eof.py: ~70 00Z dates every 37 days over 2015-2021 (all seasons; 2022, the clean PITA year, excluded).
# Usage: sbatch [...] run_fit_latent_a2_q24.sh FIRST LAST
cd /scratch4/BMC/gsienkf/Sergey.Frolov/lwatms
source /scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/etc/profile.d/conda.sh; conda activate anemoi-rocm
python -u fit_latent_to_era5.py a37tmp/pita/config_pita_a2_0104_fg24.yml output/pita_a2_0104_fg24 \
    output/latent_fit_a2_q24_${1:0:10}.npz $1 $2 --kinds Q24 --optimizer lbfgs --iters 60 --lr 0.05 --every 888 \
    --save output/latent_samples_a2_q24
