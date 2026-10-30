#!/bin/bash
#SBATCH -J lw4dvar_ace2_ps_raobs_aircraft_amv_gpsro
#SBATCH -o logs/lw4dvar_ace2_ps_raobs_aircraft_amv_gpsro_%j.out
#SBATCH -e logs/lw4dvar_ace2_ps_raobs_aircraft_amv_gpsro_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH -t 08:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 explicitly (matches the torch==2.7.1+cu128 every env pins).
module load cuda/12.8.1
# Own user-owned 'lwace2' env, not the shared
# /scratch4/BMC/gsienkf/whitaker/conda/envs/ace2 -- see run_ace2.sh.
source /scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/bin/activate lwace2

# Observation-ablation series (ACE2, see integrate_prepbufr_obs.md,
# integrate_aircraft_obs.md, integrate_amv_satwnd.md, integrate_gpsro.md):
# ps + raobs + aircraft + amv + gpsro -- the full series, and the FIRST
# real GPU run of GPSRO on the ace2 backend (see integrate_gpsro.md's
# "Plan: extend GPSRO to the ACE2 backend" -- steps 0-4 were offline/CPU-
# only; this is step 5). -t 08:00:00 matches
# run_aifs2-1.0deg-v1.0_ps_raobs_amv_gpsro.sh's budget for the same
# gpsro-added-cost reason. Own exp_name/rpath/job-name/log-path from the
# ps_raobs_aircraft_amv sibling -- see CLAUDE.md's "give every concurrent
# run its own exp_name" rule.
#
# ACE2-specific gpsro caveats going into this run (see
# config_test_ace2_ps_raobs_aircraft_amv_gpsro.yml's own comments and
# integrate_gpsro.md): impact_height_range_km narrowed to [6.,30.] (ACE2's
# measured ~20.4 km top is much lower than AIFS's ~31 km); ACE2's 'q'
# (specific_total_water) is a confirmed, non-negligible bias source in the
# bending-angle calculation (15-70% of ERA5's own spatial std, 850-250 hPa)
# with no fix available, accepted as a documented limitation.

# Compute nodes have no internet: the ACE2 checkpoint + yearly forcing files
# (backends/ace2/ace2_prefetch_checkpoint.py, ace2 env) and the ICs /
# verification (backends/ace2/ace2_ic.py [--verif], ace2ic env) must already
# be fetched from a login node.
python -u long_window_4dvar.py config_test_ace2_ps_raobs_aircraft_amv_gpsro.yml
