# JEDI UFO observation handling and crtm-torch radiances

Plan for using JEDI UFO observation processing (QC, thinning, observation
errors) in lw4dvar-torch, and for assimilating satellite radiances with
`crtm-torch`, a PyTorch implementation of CRTM. Discussion: issue #25.
Branch: `jediufo`. Started 2026-11-13.

## Design

Each piece of work runs where it fits best.

- **Inside the minimization, on the GPU, differentiable:**
  - the observation operators in torch: the existing ps operator and
    `crtm-torch` for radiances;
  - variational bias correction (VarBC) in torch, with its coefficients in
    the control vector.

  Autograd provides the adjoint, so there is no hand-written TL/AD code and
  no data moves between GPU and CPU during the AdamW iterations.
- **Outside the minimization, on the CPU, once per cycle:** UFO does
  - observation selection and thinning (`obs pre filters`, which do not need
    H(x));
  - QC that depends on H(x): background check, cloud screening, and
    inflation of the observation errors;
  - computation of the final observation errors.

  UFO runs on H(x) computed by our torch operators on the first-guess
  trajectory. Its QC flags and errors stay fixed for the minimization, which
  matches GSI's practice of doing QC at the start of each outer loop.

This keeps lw4dvar-torch's single autograd minimization and reuses UFO's
GSI-derived QC configurations without running OOPS. Calling UFO's own
operators (Fortran CRTM) inside the loop was considered and rejected. It
would need a forward and an adjoint call on the CPU at every one of the
roughly 100 epochs, or a Jacobian refreshed every few epochs (see the
comments on #25).

## Inventory (checked 2026-11-13)

| Item | Status |
|---|---|
| `crtm-torch` 2.0.0, `/scratch4/BMC/gsienkf/whitaker/python/crtm-torch` | Provides `forward`, `jacobian` (K-matrix) and `vjp`, on CPU or CUDA, in float64 or float32, and `forward` keeps the autograd graph. Its documentation reports brightness temperatures within 2.4e-8 K of Fortran CRTM and an A100 40–185 times faster than Fortran on one core. It requires **torch >= 2.8**, but the `ace2` env has 2.7.1. |
| crtm-torch sensors | The pinned manifest covers `amsua_n19`, `atms_npp`/`n20`, `cris399_npp`, `cris-fsr_npp`, `iasi_metop-a`/`b`, `airs_aqua`, `abi_g16`/`g18` and `ahi_himawari8`. Sensors are found by whatever is in `SpcCoeff/`. The full upstream archive (`fix_REL-3.1.2.0.tgz`, 8.75 GB, kept in `/scratch4/BMC/gsienkf/whitaker/crtm/`) also has AMSU-A N15/N18/MetOp-A/B, MHS and HIRS, but crtm-torch has not validated those sensors. |
| Model top | crtm-torch extends each profile with a climatological atmosphere up to 0.005 hPa (`CRTM_NATIVE_MAX_ADDED_LAYERS`), so a low model top does not break the calculation. It still limits which channels are worth assimilating. |
| Model output | The 16-layer CRTM+land ACE2 checkpoint predicts T, q, condensate and ozone on every layer, plus surface temperature. That is everything CRTM needs. |
| JEDI build | `/scratch4/BMC/gsienkf/whitaker/i-jedi` contains `libufo.so`, `libioda.so` and `liboops.so`, built with spack-stack 1.9.2 and oneAPI 2024.2.1, **without CRTM and without UFO's test executables**. spack-stack 2.2.0 (`ue-gcc-12.4.0`) has OOPS 1.13, IODA 2.12 with `pyioda` (Python 3.13), `bufr-query` 0.0.5 with Python bindings, and Fortran CRTM 3.1.5, but no UFO. |
| Raw observations | NNJA archive, `s3://noaa-reanalyses-pds/observations/reanalysis/<type>/<subtype>/YYYY/MM/bufr/gdas.YYYYMMDD.tHHz.<subtype>.tm00.bufr_d`. Covers `amsua/1bamua`, `atms/atms`, `cris`, `iasi`, `airs`, `mhs`, `gps`, `conv`, and more. |
| Converters | GDASApp `parm/ioda/bufr2ioda` has YAML converters for ATMS (with `atms_beamwidth.txt` remapping), CrIS and IASI, but **none for AMSU-A**. The nearest one is `dump/scripts/atmosphere/radiance_amsua.py` in NOAA-EMC/spoc. |
| GSI reference | The GEFSv13 replay, `s3://noaa-ufs-gefsv13replay-pds/YYYY/MM/YYYYMMDDHH/gsi/`, has `diag_<sensor>_{ges,anl}` netCDF diag files. They contain the observations after thinning, `QC_Flag`, `Inverse_Observation_Error`, background minus observation before and after bias correction, every bias predictor and its contribution, and the surface and geometry fields. It also has `satinfo`, `convinfo` and the bias coefficients `gdas.tHHz.abias{,_pc}`. Radiance sensors in 2015: AMSU-A (N15, N18, N19, MetOp-A/B), ATMS NPP, MHS (x4), IASI (x2), AIRS, HIRS-4 and SEVIRI, but no CrIS. There are also conventional diags (ps, t, q, uv) and GPS RO. One cycle (2015010100: AMSU-A N19, ATMS, ps, info files) is in `/scratch4/BMC/gsienkf/whitaker/radiance_data/gefsv13replay/2015010100/gsi/`. |

The ps diag for 2015010100 holds 64,996 observations with types 120, 180,
181 and 187; 62,794 of them have `Analysis_Use_Flag` = 1. The AMSU-A N19 ges
diag has 117,377 observation-channel rows (13 channels), of which 93,465 have
`QC_Flag` = 0.

## Phases

### Phase 0: environment, coefficients and data

1. Create a new `ace2crtm` env: `ace2` plus torch >= 2.8 plus crtm-torch.
   This follows the one-env-per-backend rule, and `ace2` itself is left
   unchanged. Check that an ACE2 rollout gives the same result under the
   newer torch.
2. Build the float64 coefficient store from the downloaded archive
   (`scripts/fetch_coefficients.py --archive`), then run
   `verify_coefficients.py` and the crtm-torch tests.
3. Benchmark on one H100 at our sizes, about 1e4–1e5 profiles per 6-h slot:
   AMSU-A/ATMS, forward plus `vjp`, float32 against float64, memory, and the
   one-time compilation cost.
4. **Validate crtm-torch against GSI.** Interpolate the replay background
   (`sfg_*` in the replay directory) to the diag locations and run
   crtm-torch. Compare with `Observation − Obs_Minus_Forecast_unadjusted`
   for AMSU-A N19 and ATMS NPP. Remaining differences should come only from
   interpolation and the CRTM version.
5. Get the observations: check Ursa for existing copies, otherwise download
   from NNJA on a login node. Convert one cycle with the GDASApp ATMS YAML
   using spack-stack 2.2.0's `bufr-query` and `pyioda`, adapt the spoc AMSU-A
   script, and check the results with `ioda-validate.x`.

### Phase 1: test the UFO interface with surface pressure only

Before any radiances, exercise the whole interface (IODA input and output,
GeoVaLs exported from torch, the UFO filter driver, and QC flags and errors
returned to torch) on the observation type we already assimilate. Each part
then has a known answer.

1. **Observations in IODA.** Convert GSI's `diag_conv_ps` files (or NNJA
   `conv` prepbufr with GDASApp `bufr2ioda_conventional_prepbufr_ps.json`)
   to IODA.
2. **GeoVaLs from torch.** Export the model's surface pressure, surface
   geopotential height, and lowest-level T, q and height at the observation
   locations, under UFO's GeoVaLs variable names.
3. **Operator check.** Run UFO's `SfcPCorrected` operator (`da_psfc_scheme:
   GSI`) on those GeoVaLs, and compare its H(x) with ours (`preduce` /
   `compute_ps_observation_hx`). This checks the GeoVaLs export and
   documents any difference between the two height-correction methods. UFO's
   `SfcPCorrected` has no linear version, and in the minimization we keep
   our torch operator either way.
4. **Filter driver.** Write a small C++ program against the UFO build that
   - reads an IODA file, GeoVaLs, and H(x) written by Python (UFO's filter
     tests work this way);
   - runs the pre, prior and post filters from a YAML file;
   - writes the QC flags and the final observation errors.

   This also needs a minimal UFO operator, about 200 lines, that copies the
   externally computed H(x) (and, for radiances, extra outputs) from an IODA
   group into the ObsSpace. The Python side writes and reads IODA with
   `pyioda` and calls the program once per cycle.
5. **QC check.**
   - Run a GSI-equivalent ps QC YAML (as in GDASApp or UFO's test inputs)
     with H(x) from the replay background.
   - Compare UFO's accepted observations and `Errinv_Final` with
     `Analysis_Use_Flag` and `Errinv_Final` from the GSI diag.
   - Then repeat with our model's H(x), and compare with the current
     dynamic background check (`bg_check`, `zthresh`, `zconst`).
6. **Cycling check.** Run a short cycle (about 120 cycles, like the
   `noobden` test) with UFO-QC'd ps observations, and compare z500 and O-B
   with the existing ps-only run on the same model. Decide whether the UFO
   QC replaces or supplements the in-loop background check.

Where the JEDI build should live is open. The `i-jedi` build is enough for
ps (no CRTM). For radiance QC we supply our own H(x) and the extra outputs
the filters need, so that build may still be enough. The alternative is a
UFO-only build against spack-stack 2.2.0.

### Phase 2: GeoVaLs for radiances (torch)

Write a differentiable function, one per backend, that turns the model state
into crtm-torch `AtmosphericState` profiles at the observation locations:

- Reuse `grid_interp` (horizontal) and `_interp_decoded` (time).
- Compute layer and interface pressures from ak/bk and ps.
- Convert units: q to g/kg, ozone mass mixing ratio to ppmv, and condensate
  to liquid or ice water path, split by temperature.
- Surface inputs: skin temperature, land/water/ice/snow fractions from the
  forcing, and 10-m wind.
- Geometry from the observation metadata.
- Use UFO's GeoVaLs names, so the profiles can also be written for UFO or
  the Fortran CRTM.

### Phase 3: radiance operator and the first experiments with GSI QC

1. Write a `RadianceOperator` module that calls `crtm_torch.forward` inside
   `compute_loss_4dvar`, batched by time slot and sensor, on the selected
   channels. Load each sensor once per process (the first call compiles
   kernels), and add gradient checkpointing per slot if memory requires it.
   Check the adjoint with dot-product and finite-difference tests.
2. **First radiance experiments without UFO.**
   - Use the GSI diags directly: rows with `QC_Flag` == 0, observation error
     1/`Inverse_Observation_Error`, and observations with GSI's bias
     correction removed (subtract `Obs_Minus_Forecast_unadjusted −
     Obs_Minus_Forecast_adjusted`).
   - Start with AMSU-A N19 and ATMS NPP, clear-sky channels that peak below
     the model's usable top, and the 16-layer CRTM ACE2 model.
   - Compare with the ps-only run: z500, temperature against ERA5, and O-B.
   - Limitation: GSI's QC and bias correction were done against the replay
     background, not our model.

### Phase 4: UFO radiance QC

Run UFO's GSI-derived radiance filter YAMLs on the NNJA-converted
observations through the Phase 1 driver, with H(x) from crtm-torch.
Validate against the replay diags: thinning counts, QC flags, and final
errors, using H(x) from the replay background. Then switch the experiments
from GSI QC to UFO QC against our own background.

### Phase 5: VarBC in torch

Implement GSI's predictors in torch: constant, scan-angle polynomial, lapse
rate and its square, cloud liquid water, emissivity, and cos(lat)·node and
sin(lat). The predictor values in the diags (`BCPred_*`) serve as a check.
Put the coefficients in the control vector with a prior from the previous
cycle, and initialise them from the replay's `gdas.tHHz.abias` files.

### Phase 6: extensions

More sensors (other AMSU-A satellites and MHS once their crtm-torch
coefficients are checked, then IASI/AIRS channel subsets), the 32-layer ACE2
and NeuralGCM models, GPS RO, and all-sky radiances.

## Risks and open questions

- **Model top and vertical resolution.** The 16-layer ACE2 model has thick
  upper layers, so stratospheric channels (for example AMSU-A 9–14) may need
  to be left out.
- **Autograd memory** across 50 epochs times the window's radiance count.
  Gradient checkpointing per slot is the fallback.
- **Derivative smoothness.** crtm-torch does not claim smooth derivatives at
  interpolation end points, clamps, or category changes. This is not
  expected to matter for clear-sky microwave, but needs watching for
  all-sky.
- **First-call compilation** takes 1.5–7.5 minutes per sensor and sky type
  (crtm-torch on an A100). It is paid once per job.
- **Building the UFO driver** in C++ against the existing build.
- **Observations past 2015-06:** not yet checked whether the replay diags
  and NNJA extend over every period we may want.
