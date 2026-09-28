# lw4dvar-torch

Prototype long-window 4dvar solver (no background or model error terms in loss).

Compute optimal initial conditions for torch-based AI forecast models. 
Options include AIFS-single-1.1, AIFS-single-2.0, ACE2-ERA5, FourCastNet3 and Microsoft Aurora
(via the `model_backend={aifs,fcn3,ace2,aurora}` yaml config parameter).

Currently only surface pressure observations are assimilated.

## Getting the code

The model checkpoints are git submodules pointing at their Hugging Face
repositories, stored with git-lfs (`git-lfs` must be installed). The ACE2-ERA5
repository also holds ~75 GB of yearly forcing files and training data, most
of which is never needed, so clone with LFS downloads switched off and then
fetch only what you need:

```
GIT_LFS_SKIP_SMUDGE=1 git clone --recursive https://github.com/NOAA-PSL/lw4dvar-torch.git
cd lw4dvar-torch

# AIFS and FCN3 checkpoints (~1 GB per AIFS checkpoint, 2.8 GB for FCN3) --
# GIT_LFS_SKIP_SMUDGE applied to every submodule, so pull their LFS content now
for d in backends/aifs/aifs-single-2.0 backends/aifs/aifs-single-1.1 backends/fcn3/fourcastnet3; do
    git -C $d lfs pull
done

# ACE2: just the checkpoint and the forcing files for the years you will run
python backends/ace2/ace2_prefetch_checkpoint.py 2014 2015

# Aurora is not a submodule (its Hugging Face repo bundles many checkpoints):
python backends/aurora/aurora_prefetch_checkpoint.py
```

If you already cloned without `--recursive`, run
`GIT_LFS_SKIP_SMUDGE=1 git submodule update --init` in the repository and then
the same `lfs pull` / prefetch steps. Skip any backend you don't plan to use.
Without `GIT_LFS_SKIP_SMUDGE=1`, the clone downloads all of ACE2-ERA5's LFS
content. The fetch steps need internet access, so on a cluster run them on a
login node.
