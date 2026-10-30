#!/bin/bash
# torchrun --no-python entry point for ROCm runs (run_ace2_train_mi300x.sh):
# gives each rank its own MIOpen user DB and kernel cache on node-local /tmp,
# then runs "python $@". fme turns on cudnn.benchmark, i.e. MIOpen's
# exhaustive convolution search, and all ranks writing one shared (sqlite)
# DB segfaulted a rank (job 24889236).
D=/tmp/miopen-${USER}-${SLURM_JOB_ID:-$$}-rank${LOCAL_RANK}
mkdir -p ${D}
export MIOPEN_USER_DB_PATH=${D}
export MIOPEN_CUSTOM_CACHE_DIR=${D}
export ROCFFT_RTC_CACHE_PATH=${D}/rocfft_kernel_cache.db
exec python "$@"
