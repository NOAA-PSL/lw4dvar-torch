#!/bin/bash
#SBATCH -J lw4dvar_ace2_ps_raobs
#SBATCH -o logs/lw4dvar_ace2_ps_raobs_%j.out
#SBATCH -e logs/lw4dvar_ace2_ps_raobs_%j.out
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

# Observation-ablation series (ACE2, see integrate_prepbufr_obs.md): ps +
# raobs. Own exp_name/rpath/job-name/log-path from config_test_ace2.yml's
# ps-only baseline (and from the ps+raobs+amv sibling) -- see CLAUDE.md's
# "give every concurrent run its own exp_name" rule.

# Compute nodes have no internet: the ACE2 checkpoint + yearly forcing files
# (backends/ace2/ace2_prefetch_checkpoint.py, ace2 env) and the ICs /
# verification (backends/ace2/ace2_ic.py [--verif], ace2ic env) must already
# be fetched from a login node.
python -u long_window_4dvar.py config_test_ace2_ps_raobs.yml
