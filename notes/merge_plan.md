# lw4dvar-torch merge plan and agent rules

Source: Claude Docs "lw4dvar-torch merge plan and agent rules", https://claude.ai/artifact/Bfs3vBwUrQpdq8jeaja51A (revision 22, exported 2026-11-01). Edit the doc, then re-export.

Oct 30, 2026 · Sergey Frolov

## Agent rules for merging to main

Work reaches main of NOAA-PSL/lw4dvar-torch only as small PRs that pass one check command, opened by agents and merged by humans. These rules extend the existing 63-line CLAUDE.md rather than replacing it.

1. **One gate.** `make check` runs lint, unit tests, gradcheck, the adjoint dot-product test and a CPU smoke run. It must pass locally before a PR is opened, and CI runs the same command. No skipping or loosening a test to get green. These permanent gates check only math and code: gradients, adjoints, and that the code runs. Science acceptance criteria keep changing as we learn, so they live in experiment notes, not in CI.
2. **Small PRs off main.** One idea per PR, with about 400 lines of real diff as a target, not a hard limit. A larger PR states why it can't be split. feature/lwatms is carved into a sequence of PRs, never merged whole.
3. **Agents open PRs, humans merge.** Any human on the team can merge. No agent pushes to main or merges.
4. **No hidden science in defaults.** A default that changes results must be explicit, and the PR shows the diff of the resolved config.
5. **Operator changes carry a planted bug.** A change to the obs operator, TL/adjoint or model shows that the gate catches a deliberately broken version.
6. **Results claims need provenance.** A public PR shows the headline result that justifies the change and links its commit, config and run. Intermediate results stay in the private experiments repo. It also states which baseline the result was checked against (for ATMS today, the GSI/NNJA O-B), because the right baseline will change as the science does.
7. **Predict before burning GPUs.** CI is CPU-only, so this rule applies when a job is submitted on Ursa or Polaris/Aurora. Before any job over 3 GPU-hours (nodes × GPUs × walltime), the agent writes the Prediction section of the experiment's `README.md` (see below) with the purpose, expected numbers and what would count as a surprise. After the run it fills in the actual result. PRs quoting that run link the note. Proposed enforcement: a `submit.sh` wrapper that refuses such jobs when the Prediction section is missing (not yet confirmed).
8. **Main defines the best candidate config.** Main holds one best candidate config, `configs/candidate.yml`, which is our current best full system. Changing it is a reviewed PR that links the experiment showing the change helps. Code merges to main if the candidate uses it, or if an active experiment uses it and the PR links that experiment, or if a person on the team selects it as a supported capability (for example another backend or an observation type we want ready). Selected capabilities are listed in `CAPABILITIES.md` with an owner and a one-line reason, and they need tests in make check like everything else. Side investigations stay on their own branches, and their results are recorded in the private experiments repo. About every three months, an agent opens a PR to remove options that neither the candidate, any active experiment nor CAPABILITIES.md uses or lists. At that review, each capability owner confirms the entry is still wanted, or it comes off the list, and owners can object.

## Experiment notes and results

Experiment records live in a private companion repo, for example NOAA-PSL/lw4dvar-experiments, because lw4dvar-torch is public. Side and archived branches are pushed there too, as a second git remote, and someone with NOAA-PSL admin rights creates it. Experiments are a flat set of folders, one per person per question. Several people can then work on several questions at once without editing each other's files or a shared list.

- **Folder:** `experiments/YYYY-MM-DD_owner_slug/README.md`. Only its owner edits it, so there are no merge conflicts.
- **Front matter:** owner, question, free-form tags (for example atms, weak-constraint, aifs), status (active, adopted, abandoned or inconclusive), branch or archive tag, and related experiments. The body holds the prediction, setup, results and verdict.
- **Questions are tags, not folders:** a new question needs no new structure, and one experiment can serve several questions.
- **No hand-edited index:** `make index` generates a table from the front matter, filterable by tag, owner and status. It isn't committed, so it never conflicts.
- **Baseline is the candidate:** every experiment states the candidate commit it starts from, so results from different people are comparable. An adopted experiment becomes a PR to the candidate config.
- **Abandoned code:** the branch is tagged `archive/<slug>` before deletion, and the README records the tag.
- **Large outputs** stay on Ursa and Eagle, and the README gives their paths.
- **Agent duties:** create the README before submitting a job, and set the status and verdict before abandoning a branch or citing a result in a PR.
- **Existing notes:** the results in `integrate_multiple_backends.md` migrate into experiments over time.

## Unit test strategy

Tests come in five layers: layers 1 to 4 run in CI on CPU in a few minutes, and layer 5 runs on GPUs outside CI.

| Layer | What it checks | Where it runs |
| --- | --- | --- |
| 1. Plain unit tests | Config parsing, ATMS reading and QC, bias correction, obs-operator pieces, data loaders | CI, CPU, seconds |
| 2. Math gates (float64, small grids) | gradcheck per component; Taylor test of the TL (error shrinks like ε²); adjoint dot-product ⟨M dx, y⟩ = ⟨dx, Mᵀy⟩ to about 1e-12; finite-difference check of the full weak-constraint cost gradient. Each gate has a planted-bug twin it must catch | CI, CPU |
| 3. Smoke run | One window, a few ATMS obs, toy or coarsened model: cost goes down, a single-obs analysis moves toward the obs | CI, CPU, under 1 minute |
| 4. Golden regression | Small frozen inputs with stored outputs; a change must regenerate them explicitly and explain why | CI, CPU |
| 5. Science validation | GPU runs with real checkpoints, such as reproducing GSI/NNJA ATMS O-B; marked `gpu`/`slow`, excluded from `make check`, tracked through prediction notes | Ursa, Polaris/Aurora |

Two standing rules: every bug fix comes with a test that failed before the fix, and loosening a tolerance or skipping a test needs a stated reason in the PR.

## Where test fixtures live

Proposal: small fixtures live in the repo under `tests/data/` in plain git, not LFS, and are rebuilt by a committed script.

- **Size budget:** under about 1 MB per file and about 10 MB in total, so a fresh clone and CI stay fast.
- **Real-data samples:** a few hundred NNJA ATMS observations and a coarse ERA5 or UFS replay state for one window, as small netCDF or parquet files. A committed `tests/data/make_fixtures.py` cuts them from the source archives on Ursa and records the source path and date, so anyone can regenerate them.
- **Synthetic inputs:** toy-model weights, random states and synthetic obs are generated at test time from fixed seeds, not stored.
- **Golden outputs:** small `.npz` files under `tests/golden/`, updated only through an explicit regenerate command that the PR justifies.
- **Not in the repo:** real checkpoints and full-size windows stay on Ursa and Eagle and are used only by layer 5.
- **Why not LFS:** the repo already keeps the ACE2 submodule pointer-only because of LFS size, and anonymous or CI clones can't always fetch LFS objects.

## Current state of main

As of 2026-10-30, main has no CPU test suite and no CI, so the foundation comes first.

- **Tests that exist:** GPU smoke scripts for 3 of 5 backends (`backends/sfno`, `backends/aurora`, `backends/ace2`), submitted with sbatch. They check the decode round-trip, that a zero increment is inert, that checkpointed and plain gradients agree, and that the loss decreases. They need real checkpoints and a GPU, so they become layer 5.
- **Missing:** pytest, `.github/workflows`, a Makefile, any tests of the solver (`long_window_4dvar_utils.py`, about 2,100 lines), finite-difference gradient checks and adjoint tests. aifs and fcn3 have no smoke test.
- **Agent guidance:** a 63-line CLAUDE.md already exists, with an index of task documents. The merge rules extend it.
- **Work to merge:** feature/lwatms is not on GitHub, only on Ursa.

## Merge plan and next steps

PR 1 builds the test foundation, and the carving of feature/lwatms starts once that branch is on GitHub.

1. **PR 1, test foundation:** a tiny CPU toy backend that implements the `forecast_model.py` interface, pytest, the first layer 1 and 2 tests on the existing solver, `make check`, and a CPU CI workflow.
2. **PR 2, rules:** the agent rules above added to CLAUDE.md, plus the `submit.sh` prediction wrapper if confirmed.
3. **Then:** split feature/lwatms into ordered PRs, each with its own tests under these rules.

- [ ] Push feature/lwatms to GitHub, even as a draft branch, so it can be planned into PRs
- [ ] Confirm the `submit.sh` wrapper for the 3 GPU-hour rule
- [ ] Confirm the fixture layout and size budget
- [ ] Settle the remaining open questions from the earlier review: the ATMS baseline, whether to publish an NNJA skill, and whether to use a second AI reviewer

Background: [agent-rules review of legoESM and ECMWF](https://claude.ai/artifact/1dCBn21BS6SiGp5AtPAzpm).
