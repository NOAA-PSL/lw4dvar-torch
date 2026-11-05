# GitHub PR strategy: carving feature/lwatms into main

Draft, 2026-11-01. Follows `notes/merge_plan.md` (one `make check` gate, PRs of about 400 lines, agents open and
humans merge, provenance for result claims, `configs/candidate.yml` + `CAPABILITIES.md` decide what merges).
Read that first; this note only turns it into an ordered PR list for this branch.

## Where things stand (checked 2026-11-01)

| Item | State |
|---|---|
| `feature/lwatms` | Local only (Ursa). 4 commits on top of main at `9d01fcf` (PR #3, sfno): `26e332c` add raobs, `067edac` AIFS sliding-window attention, `d0b9c7e` forecast-skill verification, `712fc88` ATMS/AMSU-A, weak constraint, ob-error tools. Committed diff: 56 files, +11,089 / −76. |
| Uncommitted on top | 11 modified files, +337 / −67 (mostly `long_window_4dvar_utils.py` +149: IASI, `corr_r_masked`, `fg_cache`; ob-error tools; notes), plus untracked code that runs today's experiments: `backends/aifs/aifs37_model.py`, `backends/aifs/aifs_inference_aifs37.yaml`, `iasi_decoder_error.py`, about 70 `config_*.yml`, `a37tmp/`, `figures/`. |
| `origin/main` | 47 commits ahead of our base: PR #6 AIFS v2 1° backend (Bo), #7 ALCF port, #8 `aifs3` backend (multi-dataset anemoi), #9 dry-mass fixer, #10 **our** sliding-window attention, #11 ACE2 training, #16 latent static-ensemble Jb (`jb_samples`, `latent_b.py`). Open: #13 diurnal_clim, #18 ACE2 CRTM vars. |
| Already upstream | `067edac` is the same change as PR #10 (`d9a990d`): drop it when rebasing. |
| Raobs / PREPBUFR parquet | `26e332c` lives on Bo's `origin/add-raobs-parquet` (+2,770 lines vs main), which already took our PRs #14 and #15 (sensible T, per-level q error, q background check). Not on main. |
| Test merge with origin/main | `git merge-tree` gives 6 conflicts: `long_window_4dvar_utils.py`, `integrate_multiple_backends.md`, and add/add on `backends/aifs/aifs_inference_aifs2-1.0deg-v1.0.yaml`, `config_test_aifs2-1.0deg-v1.0.yml`, `run_aifs2-1.0deg-v1.0.sh`, `test_scripts/run_prefetch_aifs2-1.0deg-v1.0.sh` (the last four: our copies of what PR #6 merged). |
| Solver size | `long_window_4dvar_utils.py` is 3,472 lines here vs 2,107 on main; our +1,584 lines touch raobs (≈180 lines), ATMS/MW (≈230 + scenes, emissivity, Huber, correlated R), weak constraint (jumps, carry-over of controls), diagnostics savers and `compute_optimal`. They are interleaved in the cost function, so the split is by feature flag, not by file. |
| Overlaps to settle | (a) our `control_prior` vs main's `jb_samples` / `jb_weight` (both are a prior on the latent control); (b) our `aifs37_model.py` (A1/A2, 37-level anemoi) vs main's `aifs3` backend; (c) our copies of the AIFS v2 1° files vs PR #6's. |

## Principles for the carve

1. **Rebuild, don't cherry-pick.** Each PR is a fresh branch off current `origin/main`, with the final version of
   just the functions it needs, behind a config flag that defaults to off. Never merge `feature/lwatms` whole.
2. **One equivalence check across the stack.** Before PR 3, freeze a golden run on `feature/lwatms`: one corrhum
   window (A0, 5 days, 20 iterations, fixed seed) and one A2 PITA window, saving J per iteration and the window-end
   state. After each PR that touches the solver, the main-based branch run with the same config reproduces what
   that feature contributes (J trace to 1e-6 relative; at the end of the stack, bit-for-bit or a stated reason).
   This is a layer-5 check (GPU, outside CI), recorded in the experiment note the PR links.
3. **Math gates go in with the feature.** Every solver/operator PR adds its layer-2 tests (gradcheck, Taylor,
   adjoint dot-product, FD of the cost) on the CPU toy backend from PR 1, with a planted-bug twin (merge-plan rule 5).
4. **Results claims link runs.** PRs that change behaviour cite the experiment (commit, config, run dir) and the
   baseline it was checked against (rule 6). Today these live in `notes/*.md`; they move to the experiments repo.
5. **Only what the candidate or a named capability uses** (rule 8). Everything else stays on archive branches.

## Phase 0 — housekeeping (no PRs, needs a human for two items)

- [ ] Commit today's uncommitted work on `feature/lwatms` in topic-sized commits (IASI, `corr_r_masked`, `fg_cache`,
      ob-error tools, notes), so nothing exists only in the working tree.
- [ ] Push `feature/lwatms` to GitHub as a draft branch (merge-plan checklist item 1). It is a reference for the carve, never a PR.
- [ ] Ask an NOAA-PSL admin to create the private `lw4dvar-experiments` repo; move `notes/*` result sections,
      `figures/`, `a37tmp/` and the experiment configs there as `experiments/YYYY-MM-DD_owner_slug/README.md` folders.
- [ ] Decide the three overlaps above (owners: Sergey + Jeff for (a) and (b), Bo for (c)).
- [ ] Freeze the golden runs (principle 2).

## The PR sequence

Sizes are real-diff estimates. "Gate" = tests the PR adds to `make check`.

| # | PR | Contents | Size | Gate | Depends on | Owner |
|---|---|---|---|---|---|---|
| 1 | Test foundation | CPU toy backend implementing `forecast_model.py`; pytest; first layer-1/2 tests on the existing solver (gradcheck of J, Taylor, adjoint dot-product); `make check`; CPU GitHub Actions workflow | ~400 | defines it | — | Sergey (agent) |
| 2 | Rules | Merge-plan rules into `CLAUDE.md`; `notes/merge_plan.md`; `submit.sh` prediction wrapper if confirmed; `CAPABILITIES.md` skeleton | ~150 | — | 1 | Sergey |
| 3 | PREPBUFR parquet obs + raobs | Land Bo's `add-raobs-parquet` (with our #14/#15) on main, rebased, split if over ~600 lines: (3a) parquet readers `prepbufr_parquet_common.py`, `psobs_parquet.py`; (3b) `raobs_parquet.py` + `get_raobs` + raob operator in the cost + `observations:` schema | 3a ~450, 3b ~650 | reader unit tests on a tiny fixture (`tests/data/`); raob operator TL/adjoint on the toy model | 1 | Bo (Sergey reviews) |
| 4 | Verification tools | `diagnostics/forecast_skill.py`, `plot_forecast_skill.py`, `gsi_convinfo.py`, `convobs_parquet.py`, `sfcobs_parquet.py`, `analysis_error_ts.py` (fix the silent 14-run limit), `forecast_skill.md` | ~1,100, split 4a obs readers / 4b scoring / 4c plots | scoring unit tests (pooled RMS, QC screens) on synthetic data | 3a | Sergey |
| 5 | Weak constraint | `exp.weak_constraint`: jumps every N h, `advance_with_jumps`, carry-over of jump controls between cycles, `save_window_end_state`; `notes/weak_constraint_4dvar.md` | ~450 | FD of the full weak-constraint J; zero jumps == strong constraint (golden); planted bug: jump applied one step late | 1 | Sergey |
| 6 | Control prior / latent Jb | Reconcile `control_prior` with main's `jb_samples` (one prior API for window-start and jump controls). This is also the regularization the iteration study asks for. | ~250 | gradient of Jb; prior off == today | 5, decision (a) | Sergey + Jeff |
| 7 | MW operator plumbing | `obs_atms_operator.py` (two-head bundle loader, any channel count), `export_mw_bundle.py`, `build_mw_obs.py`, `get_mwobs` reader | ~450 | a tiny synthetic bundle (fixed seed, generated at test time): operator == reference forward; Taylor/adjoint of H; planted bug | 1 | Sergey |
| 8 | ATMS in the cost | `observations.atms`: scenes, background check, per-ob emissivity control z, window channels over ocean, `detach_q_above_hpa`, Huber; ATMS diagnostics saver | ~550 | FD of J with ATMS on the toy model; z gradient; single-ob analysis moves toward the ob (layer 3) | 7 | Sergey |
| 9 | ATMS observation errors | `scene_cloud` error model, correlated R (`corr_r`, `corr_cond_max`, `corr_r_window`, `corr_r_masked`); ob-error tools `atms_decoder_error.py`, `atms_cloud_error.py`, `obs_error_diag.py`; the JSON tables the candidate uses (`decoder_era5_a2q70.json`, `oberr_table_w_a2q70.json`) under `data/obs_error/` | ~600, split 9a error model / 9b tools | whitened cost == rᵀC⁻¹r (exact); masked == full when every channel is used; conditioning bound | 8 | Sergey |
| 10 | AIFS 37-level backend | A1/A2 support: either fold into main's `aifs3` backend or add `aifs37_model.py` + inference yaml (decision (b)); ICs and truth from the training zarr | ~400 | layer-5 smoke (GPU): zero increment inert, one window J falls | 1, decision (b) | Sergey |
| 11 | Candidate config | `configs/candidate.yml` = today's best full system, `CAPABILITIES.md` entries. Today that is corrhum obs on A2, run as independent 5-day windows from a 24 h ERA5 forecast (PITA), 200 iterations; links the PITA experiment and deck | ~150 | candidate parses and its smoke passes on the toy backend | 3–10 | Sergey |
| 12 | IASI | `MW_SENSORS 'iasi'`, IASI channel config, `iasi_decoder_error.py` | ~200 | masked-R exactness for >30 channels | 9 | Sergey, **only if** the IASI PITA test helps (then via 11) or IASI is named a capability |
| 13 | First-guess cache | `fg_cache` (first guess from another GRIB dir, for HealDA) | ~40 | config test | 11 | HealDA work, when it is used |

PRs 3, 5 and 7 are independent once PR 1 is in, so three can be open at once. The solver PRs (5, 6, 8, 9) touch
the same cost function, so they merge in order and each rebases on the last.

## What does not go to main

Kept on archive branches (`archive/<slug>`) and recorded in the experiments repo:

- **PIT / multiple shooting** (`a37tmp/pit_dev/`): no gain without adjoint coupling; held.
- **AMSU-A** (in `712fc88`): a side project per the ATMS decision; merges only if named a capability.
- **One-off analysis scripts**: `atms_figures*.py`, `atms_omb_maps.py`, `atms_vs_replay_oma.py`,
  `atms_weighting_functions.py`, `compare_obs_10cyc.py`, `e0_atms_check.py`, `e1_summary.py`, `fcst_skill_plot.py`,
  `bench_attention.py`, `estimate_latent_errors.py`, and all `a37tmp/` drivers.
- **Experiment configs** (`config_lwatms_*`, `config_a37_*`, `config_pita_*`): one goes to main as the candidate;
  the rest become experiment folders.
- **Results prose** in `notes/integrate_atms_obs.md`, `notes/obs_error_tuning.md`,
  `notes/integrate_aifs37_models.md`: the method parts stay as task docs; the results move to the experiments repo.

## Open questions

1. Will Bo open the `add-raobs-parquet` → main PR (3), or should we carve it from our branch?
2. `control_prior` vs `jb_samples`: one API, or two priors that can coexist?
3. A1/A2 through `aifs3` or a separate `aifs37` backend?
4. Candidate: independent windows (PITA, better on every score we have) or cycled corrhum (what the deck series used)?
   PITA is the measured best; cycling needs a fix before it can come back.
5. Second AI reviewer, and the `submit.sh` wrapper (merge-plan checklist).
