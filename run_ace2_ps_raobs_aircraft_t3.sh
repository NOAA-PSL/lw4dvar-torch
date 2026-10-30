#!/bin/bash
#SBATCH -J lw4dvar_ace2_ps_raobs_aircraft_t3
#SBATCH -o logs/lw4dvar_ace2_ps_raobs_aircraft_t3_%j.out
#SBATCH -e logs/lw4dvar_ace2_ps_raobs_aircraft_t3_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH -t 04:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 explicitly (matches the torch==2.7.1+cu128 every env pins).
module load cuda/12.8.1
# Own user-owned 'lwace2' env, not the shared
# /scratch4/BMC/gsienkf/whitaker/conda/envs/ace2 -- see run_ace2.sh.
source /scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/bin/activate lwace2

# Sensitivity test t3 (see sensitiveTest.record): same as t2 (static
# pressure bin x NH/Tropics/SH region aircraft temperature bias
# correction), but using the 2015-native bias table
# (aircraft_temperature_bias_table_2015.json, built from 123 real 2015-01
# GSI cnvstat cycles) instead of t2's 2022-based one -- see
# integrate_aircraft_obs.md's "Static aircraft temperature bias
# correction" section and build_aircraft_temperature_bias_table_2015.py.
# Own exp_name/rpath/job-name/log-path from the ps_raobs_aircraft baseline
# -- see CLAUDE.md's "give every concurrent run its own exp_name" rule.

# Compute nodes have no internet: the ACE2 checkpoint + yearly forcing files
# (backends/ace2/ace2_prefetch_checkpoint.py, ace2 env) and the ICs /
# verification (backends/ace2/ace2_ic.py [--verif], ace2ic env) must already
# be fetched from a login node.
python -u long_window_4dvar.py config_test_ace2_ps_raobs_aircraft_t3.yml
