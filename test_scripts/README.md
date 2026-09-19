# test_scripts

`sbatch` wrappers for the one-off diagnostic/probe/profiling scripts under
`backends/aurora/` (and any future backend-specific diagnostics). These
are not part of the production pipeline (`run_aifs.sh`/`run_fcn3.sh`/
`run_aurora.sh`, which remain at the repo root) -- they're throwaway or
semi-permanent tooling used to investigate a specific question, documented
in CLAUDE.md.

**Must be submitted from the repo root**, not from inside this directory:

```
sbatch test_scripts/run_probe_aurora_checkpoint.sh
```

None of these scripts `cd` anywhere first -- they invoke their Python
script by a path like `backends/aurora/probe_aurora_checkpoint.py`,
relative to `sbatch`'s own working directory (the directory you ran
`sbatch` from), not the directory the `.sh` file lives in. `sbatch
test_scripts/run_probe_aurora_checkpoint.sh` from the repo root works;
`cd test_scripts && sbatch run_probe_aurora_checkpoint.sh` does not.
