# SOCRATES Calibration Migration Guide: Architecture & Implementation Plan

**Status:** 43/43 tests passing. Full serial calibration confirmed running end-to-end. Distributed path pending validation (next planned task).
## Latest Update: Observation Normalization — Port of CalibrateEDMF `pooled_nonzero_mean_to_value` (2026-03-14)

### What was wrong
- `experiment_config.yml` contained `log_vars: [clw]` and hardcoded `norm_factors_by_var` (mean/std z-score statistics in log-space).  Neither existed in the original CalibrateEDMF SOCRATES calibration, which used `normalization_type=:pooled_nonzero_mean_to_value`.
- The log transform caused `DomainError` when `clw` was slightly negative (numerical noise from interpolation).
- The z-score normalization with hardcoded log-space statistics was inconsistent and not defensible scientifically.

### What was ported from CalibrateEDMF
CalibrateEDMF's `:pooled_nonzero_mean_to_value` normalization:
- For each variable, compute `char_scale = mean(abs(nonzero elements of var_data in time window))`.
- Normalize: `y / char_scale`.  After this, the mean of the non-zero elements is ≈ 1; the scale is physically meaningful rather than statistical.
- Applied identically to both the observation vector `y` (LES reference) **and** the forward-model vector `G` (ClimaAtmos output), ensuring the EKI sees consistent units.

### Files changed
- **`run_calibration.jl`** — Added `compute_pooled_norms(cfg, z_model_by_case)` which reads each LES reference file over the calibration time window and computes per-variable characteristic scales averaged over all cases.  Scales are passed to `build_observations` (for `y`) and saved to `obs_metadata.jld2` (for `G`).
- **`observation_map.jl`** — `process_profile_variable` and `process_observation_variable` now accept `pooled_norms_dict` and divide by the characteristic scale.  `observation_map()` loads `pooled_norms_by_var` from `obs_metadata.jld2` and injects it into the config dict consumed by `process_member_data`.
- **`experiment_config.yml`** — Removed `log_vars`, removed `norm_factors_by_var`.  `const_noise_by_var` values are retained (they are dimensionless noise levels relative to the normalized scale ≈ 1).

### Consequence for existing output directories
Any `output/simple_socrates/` run started before this change used the old log+z-score normalization.  The stored `eki_file.jld2` observation vectors are incompatible.  Delete the output directory and restart calibration to pick up the new normalization.


## Latest Update: Native Debug-State Reporting + Probe Command Normalization (2026-03-13)

### What changed in the NaN forensics path
- Integrated ClimaAtmos native state debugger output into the NaN-stage dump pipeline.
- On first non-finite detection, dumps now include:
    - flat vectors (`*_state.txt`, `*_tendency.txt`, optional `*_tendency_lim.txt`),
    - structured range maps (`*_state_structure.txt`, `*_tendency_structure.txt`, optional `*_tendency_lim_structure.txt`),
    - native recursive debug report (`*_debug_state_report.txt`) generated from the existing debug-state traversal utilities.
- This avoids relying only on custom traversal logic and keeps reporting aligned with ClimaAtmos internals.

### Command-equivalence finding (important)
- For this SOCRATES directory, the following are equivalent:
    - `julia --project ...`
    - `julia --project=. ...`
    Both activate the same project file:
    `.../calibration/experiments/SOCRATES/Project.toml`.
- Restart checkpoint path forms are also equivalent here:
    - local relative path under `output/debug_crash_probe/.../day0.3090.hdf5`
    - absolute `/net/sampo/data1/.../day0.3090.hdf5`
    They resolve to the same inode and file metadata in this environment.

### Recommended canonical probe command
- Keep one canonical form for reproducibility and reduced ambiguity:

```bash
cd /home/jbenjami/Research_Schneider/CliMA/ClimaAtmos.jl/calibration/experiments/SOCRATES && \
LOG=output/debug_crash_probe/logs/restart3090_dt10_full_integration_debug.log && \
RESTART=/net/sampo/data1/jbenjami/Research_Schneider/CliMA/ClimaAtmos.jl/calibration/experiments/SOCRATES/output/debug_crash_probe/iter_000_member_003_case_1_dt10_perstep_clean/output_0000/day0.3090.hdf5 && \
CLIMA_NAN_TENDENCY_DEBUG=1 CLIMA_NAN_TENDENCY_DEBUG_FULL=1 \
julia --project run_debug_crash_probe.jl 0 3 1 3110secs "$RESTART" 10secs 10secs 10secs > "$LOG" 2>&1; \
echo "EXIT_CODE=$?"; echo "LOG=$LOG"
```


## Latest Update: Runtime Forcing Validation and Crash-Window Isolation (2026-03-14)

### Operational policy note (tooling discipline)
- In ClimaAtmos.jl and ClimaCalibrate.jl workflows, do not pin Julia versions in commands unless explicitly requested by the user in that prompt.
- Use the active/default Julia and the existing Julia REPL session for iterative debugging.
- For this crash investigation, per-step checkpointing saved model state (`Y`) every step; it did not save per-process tendency terms.
- Therefore, `day0.3090.hdf5` (finite) and `day0.3100.hdf5` (non-finite) precisely localize the failing timestep window, but not the first offending tendency term.

### What was fixed
- Removed hardcoded dry-air constants from the SOCRATES forcing conversion path and now use ClimaAtmos parameter-set values for `R_d` and `grav`.
- Widened forcing-time parsing helpers to accept `AbstractString` inputs so regex-derived `SubString` values do not fail during unit parsing.
- Fixed `validate_forcing_time_support` to inspect raw numeric NetCDF time values via `ds["time"].var[:]` instead of auto-decoded CF-time values.
- Removed stale `cfsite_number` plumbing from the SOCRATES experiment config and case-configuration path.
- Localized the EDMFX TOML reference so SOCRATES now uses `scm_tomls/prognostic_edmfx.toml` inside the experiment rather than an inherited path from `gcm_driven_scm`.

### What is now tested
- The forcing conversion tests now cover duration parsing, DateTime parsing, forcing time-unit parsing, pressure-to-height conversion with ClimaAtmos parameters, forcing support validation, z-interpolation sanity, and evaluation of `TimeVaryingInput` objects on simulation-style times.
- The test suite includes a real SSCF forcing integration test that builds the same type of converted forcing used at runtime and explicitly evaluates the resulting time-varying forcing across the ClimaAtmos simulation-time domain.
- Result: forcing-conversion/runtime-interpolation coverage passes end to end in `test/forcing_conversion_tests.jl`.

### Crash-probe findings
- A dedicated single-case debug probe was added in `run_debug_crash_probe.jl` with `check_nan_every = 1`, dense diagnostics, and frequent HDF5 checkpoints.
- For the known failing case `(iteration=0, member=3, case=1)`, the run still crashes near `t = 3100 s`.
- The checkpoint at `day0.3000.hdf5` is fully finite.
- The checkpoint at `day0.3100.hdf5` already contains NaNs in essentially all major prognostic state components, including density, momentum, total energy, moisture species, TKE, EDMFX SGS state, and face vertical velocity.
- This means the failure is a real model-state blow-up in the 100 s window between 3000 and 3100 s, not just a bad diagnostic write.
- Model instantiation in the debug probe reported `AtmosRadiation{Nothing, IdealizedInsolation}`, so full radiation is not currently active in this failing configuration.

### Per-step checkpoint confirmation at unchanged crashing dt
- A clean rerun was executed with the original crashing timestep unchanged: `dt = 10secs`.
- NaN checks were enforced every integrator step: `check_nan_every = 1`.
- HDF5 checkpoint saves were enforced every integrator step for this dt via `dt_save_state_to_disk = 10secs`.
- Result: the run still crashed, and the first non-finite checkpoint was pinned exactly:
  - `day0.3090.hdf5` is fully finite.
  - `day0.3100.hdf5` is the first checkpoint with NaNs.
- This directly confirms the NaN onset occurs in the single 10-second step from `t = 3090 s` to `t = 3100 s` under the original timestep.

### Full integration instrumentation refinement (where NaNs start)
- Added full-path non-finite instrumentation across cache updates, implicit tendencies, implicit SGS-u3 subproblem, and explicit tendencies.
- Added full dump artifacts for first-failure snapshots (`state`, `tendency`, and when available `tendency_lim`) under `nan_stage_dumps/`.
- Key result from restart at `day0.3090.hdf5` with `dt = 10secs`:
    - First non-finite appears during `implicit_tendency!`, stage `implicit_vertical_advection`, at `t = 3094.358665215085`.
    - At first failure: `state_nonfinite = 0/24328` and `tendency_nonfinite = 8/24328`.
    - Therefore, NaNs start in the computed tendency and later propagate into the state; this is not consistent with a missing-write artifact.
- The first non-finite tendency indices were localized to the density tendency storage region (flat storage indices mapped under `Y.c.ρ` in the contiguous state backing array).
- This refines earlier wording: the checkpoint at `day0.3100.hdf5` is globally non-finite, but the origin is sparse and earlier within the step.

### Current debugging direction
- Use the pinned boundary (`3090 -> 3100 s`) to inspect process-level tendencies for that single step.
- Keep dt fixed at `10secs` for root-cause localization runs where the goal is reproducing the exact failure path.

## Latest Update: Input-Forcing Crash Debugging (2026-03-13)

### What failed
- Calibration aborted with `Iteration 0 had a 100% failure rate` in `ClimaCalibrate` worker loop.
- Worker logs showed repeated `DomainError` in thermodynamics (`log` of negative value) during saturation-adjustment calls.

### Key findings from direct forcing inspection
- Raw SSCF forcing file for RF01/obs_data looked physically reasonable:
    - `T` min/max: `228.10 K` to `276.21 K`
    - `q` min/max: `2.58e-6` to `3.71e-3`
    - `lev` min/max: `100 Pa` to `97500 Pa`
    - `T units = K`
- Converted ClimaAtmos forcing file was found in an inconsistent state before forced regeneration:
    - observed `z` range was about `5.65 km` to `55.0 km` (suspicious for this setup)
- After deleting and regenerating the converted file, range became:
    - `z` min/max: `75.5 m` to `49.5 km`
- Conclusion: stale/previously converted forcing files can persist and silently affect runs; forcing files should be inspected/regenerated during debugging.

### New instrumentation added (implemented)
- Added automatic per-member, per-case forcing debug dumps in `model_interface.jl` during `configure_member_case(...)`.
- New artifacts are written to:
    - `<member_output>/config_<case_idx>/debug_inputs/forcing_inputs_on_model_z.nc`
    - `<member_output>/config_<case_idx>/debug_inputs/forcing_inputs_summary.txt`
- `forcing_inputs_on_model_z.nc` contains forcing values interpolated to the actual model vertical centers for all forcing times.
- `forcing_inputs_summary.txt` contains min/max checks for key forcing variables and z-ranges.
- Toggle via environment variable:
    - `SOCRATES_DEBUG_DUMP_INPUTS=1` (default on)
    - set `SOCRATES_DEBUG_DUMP_INPUTS=0` to disable.

### Variables exported in `forcing_inputs_on_model_z.nc`
- Profile variables (z,time): `ta`, `hus`, `ua`, `va`, `rho`, `wa`, `wap`, `tntha`, `tnhusha`, `tntva`, `tnhusva`
- Surface/time-series variables (time): `ts`, `hfls`, `hfss`, `coszen`, `rsdt`

### Additional unresolved clue from run logs (important)
- Worker logs still showed only `NetCDFWriter: ["thetaa_10m_inst", "hus_10m_inst", "clw_10m_inst"]` and `EquilibriumMicrophysics0M` in model printouts.
- This is inconsistent with `model_config.yml` edits that requested broader diagnostics and `microphysics_model: 1M`.
- Implication: some configuration overrides may still not be applied as intended during worker runs; this remains an active debugging target.

---

## 0. Source File Map — Where Everything Lives

This section is the single authoritative reference for what file does what, where it comes from,
and how it maps to the new ClimaAtmos experiment. **Read this before searching the workspace.**

### 0.1 Legacy System (CalibrateEDMF + TurbulenceConvection)

| Role | File | Notes |
|------|------|-------|
| **Calibration config entry point** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/config_calibrate_template_body.jl` | Sets `nonequilibrium_moisture_scheme`, `flight_numbers`, `forcing_types`, `calibration_vars`, `dt_min`, `dt_max`, `adapt_dt`. Derives `dz_min` from CFL or from `calibration_setup` name (e.g. `__cfl_dt_min_5.0` suffix). Sets `N_ens=100`, `N_iter=50`, `EKI_termination_T`, `normalization_type=:pooled_nonzero_mean_to_value`, `alt_scheduler=DataMisfitController(on_terminate="continue")`. Strips noneq-only params when using equilibrium setup. Scales LWP/IWP observation variance by `1/O_nz²` so column integrals count as much as a full profile. Conditionally disables `reweight_processes_for_grid` and `conservative_interp` when `dz_min ≤ 100m` (native Atlas LES resolution). Includes header then footer. |
| **Calibration config header** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/config_calibrate_template_header.jl` | Defines everything *shared across all calibration runs*. Contains: all prior distributions and constraints per parameter in `default_calibration_parameters` (dynamics: `tke_ed_coeff`, `tke_diss_coeff`, `base_detrainment_rate_inv_s`...), `autoconversion_calibration_parameters` (`τ_acnv_rai`, `τ_acnv_sno`, `q_liq_threshold`...), `sedimentation_parameters` (`liq_sedimentation_scaling_factor`, `χv_rai`, `ice_sedimentation_scaling_factor`, `χv_sno`...) — each with `prior_mean`, `constraints` (e.g. `bounded(lo, hi)`), `unconstrained_σ`, `CLIMAParameters_longname`; `default_namelist_args` (the full TC namelist override list — sets entrainment type, area limiters, spinup dt, tendency limiters, microphysics scaling factors, `("grid", "dz_min", "cfl")` sentinel, `conservative_interp_kwargs_in/out`, `reweight_processes_for_grid`, τ limits, N limits); `conservative_interp_kwargs_in` (conservative regridding for forcing input onto model grid) / `conservative_interp_kwargs_out` (for output onto obs grid — `integrate_method`, `enforce_positivity`, `nnls_alg`); `Costa_SOTA` parameter dict (posterior from Costa calibrations, used as prior means for dynamics params); N_CCN defaults per flight number (RF01: 75e6/cm³, RF09: 190e6/cm³, RF10: 55e6/cm³, etc.); `default_obs_var_scaling` dict. |
| **Calibration config footer** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/config_calibrate_template_footer.jl` | Defines `get_config()` assembling all sub-configs. Key sub-configs: `get_regularization_config()` — `normalization_type`, `variance_loss`, `perform_PCA`, `obs_var_scaling`, `obs_var_additional_uncertainty_factor`, `additive_inflation`, `tikhonov_mode/noise`, `dim_scaling`; `get_process_config()` — `N_iter`, `N_ens`, `algorithm="Inversion"`, `scheduler` (`DataMisfitController` or default), `localizer` (`SECNice` or `NoLocalization`), `T_stops`, `failure_handler="sample_succ_gauss"`; `get_reference_config(SOCRATES_Train)` — reads `SOCRATES_summary.nc` to set per-flight `t_start`/`t_end` (obs_data: 10h–12h; ERA5_data: from Atlas Table 2 relative to 12h reference), `time_shift` (12h for obs, 14h for ERA5), `y_names=calibration_vars`, `y_reference_type=LES()`, `Σ_t_start/end`; merges `default_namelist_args` + `local_namelist` and `default_calibration_parameters` + per-experiment `calibration_parameters`; calls `process_SOCRATES_reference()` (atlas LES or flight obs path); `default_user_args` dict (sedimentation scheme, terminal velocity scheme, moisture limiter type, `truncated_basic_limiter_factor=5`). |
| **SOCRATES case dispatch (TC)** | `~/Research_Schneider/CliMA/TurbulenceConvection.jl/driver/Cases.jl` | `struct SOCRATES`, `initialize_profiles`, `initialize_forcing`, `initialize_radiation`, `surface_params`, `update_forcing`, `update_radiation`. Calls SSCF `process_case`. |
| **TC grid construction** | `~/Research_Schneider/CliMA/TurbulenceConvection.jl/driver/common_spaces.jl` | `construct_mesh(namelist)` for SOCRATES case: reads Atlas LES z-grid via `SSCF.get_default_new_z(flight_number)`, converts cell-centers to faces via `zf[i] = 2*zc[i] - zf[i-1]`, then enforces `dz_min` by skipping faces too close to the previous one. The SOCRATES grid is therefore the **Atlas LES native grid** (possibly coarsened by `dz_min`), NOT a user-specified `nz/dz_surf` grid. Non-SOCRATES cases use `GeneralizedExponentialStretching(dz_surf, dz_toa)` and `TCMeshFromGCMMesh`. |
| **TC dycore/timestepper** | `~/Research_Schneider/CliMA/TurbulenceConvection.jl/driver/dycore.jl` | Adaptive dt loop, calls `update_forcing`/`update_radiation` each step, single-column 1D grid. |
| **SSCF core** | `~/Research_Schneider/CliMA/SOCRATESSingleColumnForcings.jl/src/` | `process_case(flight_number, forcing_type; initial_condition, surface)`. Returns time-dependent forcing functions `f[k](t)` for each height level. |
| **SSCF Atlas LES inputs** | `~/Research_Schneider/CliMA/SOCRATESSingleColumnForcings.jl/Data/Atlas_LES_Profiles/` | Raw Atlas LES forcing files per flight. Accessed via `open_atlas_les_input(flight_number, forcing_type)`. |
| **SSCF Atlas LES outputs** | `~/Research_Schneider/CliMA/SOCRATESSingleColumnForcings.jl/Data/Observed_Profiles/` or artifact root | Reference LES simulation outputs. Accessed via `open_atlas_les_output(flight_number, forcing_type)`. |
| **Inferred microphysics params** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/Reference/Output_Inferred_Data/SOCRATES_Atlas_LES_inferred_parameters.nc` | Pre-fit parameter values used to seed priors. Loaded in config_calibrate_template_header.jl. |
| **CalibrateEDMF Pipeline** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/src/Pipeline.jl` | Init/run EKP loop calling TC as forward model. Defines `get_reference_config` structure. |
| **CalibrateEDMF ReferenceStats** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/src/ReferenceStats.jl` | Reads TC output NetCDF, computes observation vector. Conservative regridding, time windowing. |
| **CalibrateEDMF SOCRATESUtils** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/src/SOCRATESUtils.jl` | SOCRATES-specific helper utilities for the reference data pipeline. |
| **Reference processing script** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/process_SOCRATES_reference.jl` | Converts raw Atlas LES outputs into CalibrateEDMF reference format. Called by footer config. |
| **Postprocessing storage** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES_postprocess_runs_storage/` | Stored calibration runs under: `subexperiments/SOCRATES_<scheme>/Calibrate_and_Run/<calibration_setup>/<dt_string>/<vars_str>/` |
| **Best-particle output example** | `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES_postprocess_runs_storage/subexperiments/SOCRATES_exponential_T_scaling_ice/Calibrate_and_Run/tau_autoconv_noneq/adapt_dt__dt_min_2.0__dt_max_4.0/iwp_mean__lwp_mean__qi_mean__qip_mean__ql_mean__qr_mean/postprocessing/output/Atlas_LES/RFAll_obs/best_particle_final/` | Reference output for comparison. Contains TC output NC files. |

### 0.2 New System (ClimaAtmos + ClimaCalibrate)

| Role | File | Notes |
|------|------|-------|
| **Calibration entry point** | `calibration/experiments/SOCRATES/run_calibration.jl` | `main()` sets up and runs EKP loop via `ClimaCalibrate`. Guard: `if abspath(PROGRAM_FILE) == @__FILE__`. |
| **Experiment config** | `calibration/experiments/SOCRATES/experiment_config.yml` | `y_var_names`, `flight_numbers`+`forcing_types` per case, `ensemble_size`, `n_iter`, `batch_size`, `z_max=4000`, time windows, prior path, model config path. |
| **Model config** | `calibration/experiments/SOCRATES/model_config.yml` | ClimaAtmos YAML: `z_max=40km, z_elem=60, z_stretch=true, dz_bottom=30m, dt=10s, t_end=14h`. |
| **Prior distribution** | `calibration/experiments/SOCRATES/prior.toml` | `[tau_autoconv_ice]` and `[tau_autoconv_liq]` only (current scope). |
| **Observation map** | `calibration/experiments/SOCRATES/observation_map.jl` | `observation_map(iteration)` → reads `output_active` NC files, interpolates to `z_model`, concatenates per-case per-variable profiles. |
| **Reference preprocessing** | `calibration/experiments/SOCRATES/preprocess_reference.jl` | `preprocess_socrates_reference(...)`: converts raw Atlas LES outputs to ClimaAtmos-compatible NC format. Writes to `Reference/Atlas_LES/<flight>_<forcing>/stats/`. |
| **Calibration model interface** | `calibration/model_interface.jl` | ClimaCalibrate hook: `build_model_interface()`. Not SOCRATES-specific. |
| **Helper functions** | `calibration/experiments/SOCRATES/helper_funcs.jl` | `fetch_interpolate_transform`, `nc_fetch`, `window`, `average_time`, `slice`. Adapted from `CalibrateEDMF/experiments/gcm_driven_scm/helper_funcs.jl`. |
| **Forcing conversion** | `calibration/experiments/SOCRATES/forcing_conversion.jl` | `convert_socrates_forcing_file()`, `pressure_to_height()`, `is_valid_converted_forcing_file()`. Converts SSCF-format forcing to ClimaAtmos `ReanalysisTimeVarying` NetCDF format. |
| **Preprocessed reference data** | `calibration/experiments/SOCRATES/Reference/Atlas_LES/<RF##_<forcing>/stats/<RF##_<forcing>.nc` | Output of `preprocess_reference.jl`. ClimaAtmos diagnostic-named variables (e.g., `thetaa`, `hus`, `clw`). |
| **Converted forcing files** | `calibration/experiments/SOCRATES/Forcing/ClimaAtmos/<RF##_<forcing>.nc` | Output of `forcing_conversion.jl`. Used as `external_forcing_file` in ClimaAtmos config. |
| **Test suite** | `calibration/experiments/SOCRATES/test/runtests.jl` | Includes all 4 test files. Run via `include("test/runtests.jl")` with project activated. |
| **Offline pipeline test** | `calibration/experiments/SOCRATES/test/calibration_pipeline_offline_tests.jl` | Exercises preprocess→build_observations→observation_map pipeline using stored iteration_000. |
| **EKP failure gating test** | `calibration/experiments/SOCRATES/test/ekp_failure_gating_tests.jl` | Synthetic test for partial/full NaN G handling by EKP update_ensemble. |
| **Observation map tests** | `calibration/experiments/SOCRATES/test/observation_map_tests.jl` | Unit tests for `resolve_dims_per_var!`, `resolve_member_simdir`, `resolve_z_model!`, `interpolate_profile_to_target_grid`. |
| **Forcing conversion tests** | `calibration/experiments/SOCRATES/test/forcing_conversion_tests.jl` | Unit tests for `pressure_to_height`, `is_valid_converted_forcing_file`, `convert_socrates_forcing_file`. |

---

## 0.3 Grid, Timestep, and CFL: TC vs ClimaAtmos Comparison

This is the answer to "why didn't we follow the CalibrateEDMF pipeline for adjusting the grid for each timestep?"

### What TC/CalibrateEDMF did

TC uses an **adaptive 1D grid derived from the Atlas LES native z-grid** with **adaptive timestepping**.

**Source of truth: `~/Research_Schneider/CliMA/TurbulenceConvection.jl/driver/common_spaces.jl`**

For the SOCRATES case, `construct_mesh(namelist)` does NOT use `nz/dz_surf/z_toa` namelist entries. Instead:
1. Calls `SSCF.get_default_new_z(flight_number)` → Atlas LES cell-center heights (`zc`).
2. Converts cell-centers to face heights: `zf[i] = 2*zc[i] - zf[i-1]` (exact for the Atlas LES grid layout).
3. Enforces hard minimum `dz_min` by skipping faces that are closer than `dz_min` to the previous face.
4. Builds `CC.Meshes.IntervalMesh` from the resulting face array.

The SOCRATES TC grid is therefore **the Atlas LES native z-grid**, possibly coarsened by `dz_min`. There are no `nz/dz_surf/z_toa` knobs for the SOCRATES case — only `dz_min` matters.

**What `dz_min` controls**: set via `namelist["grid"]["dz_min"]`. In `config_calibrate_template_header.jl` the entry is the sentinel `("grid", "dz_min", "cfl")`, which gets replaced at runtime by `config_calibrate_template_body.jl`.

**Time-stepping** (from `namelist["time_stepping"]` — these ARE namelist-driven):
```
adapt_dt: true
dt_min: 1.0 s     (default; calibration scripts override this)
dt_max: 12.0 s    (default; calibration scripts override this)
cfl_limit: 0.5
t_max: 50400.0 s  (14 hours, overridden per forcing_type in calibration scripts)
allow_spinup_adapt_dt: true  (set in default_namelist_args)
```

The **calibration scripts** (`config_calibrate_template_body.jl`) then override dt and derive `dz_min` from CFL:

```julia
dt_min = 0.5    # more conservative in calibration runs
dt_max = 2.0    # more conservative in calibration runs

# CFL-derived dz_min: ensures minimum cell size is CFL-stable
# for the assumed maximum hydrometeor fall speed of ~5 m/s
dz_min = 5.0 * (2 * dt_min / cfl_limit)   # = 5.0 * (2*0.5 / 0.5) = 10.0 m
```

The directory name `adapt_dt__dt_min_2.0__dt_max_4.0` in postprocessing paths encodes these choices.

Additionally, TC has **conservative regridding** (`conservative_interp_kwargs`) for both:
- **Input**: interpolating forcing data onto the model grid
- **Output**: regridding model output onto the observation grid (important when dz varies)

And `reweight_processes_for_grid = true`: process rates (e.g., autoconversion) are weighted by grid cell volume when dz varies, so thin cloud layers don't vanish at coarse resolution.

### What ClimaAtmos does instead

ClimaAtmos uses a **fixed, stretched pressure-based spectral element grid**:

```yaml
# From model_config.yml (current settings):
z_max: 40e3         # 40 km domain top  (TC uses 45 km)
z_elem: 60          # vertical elements (TC uses 55 levels)
z_stretch: true     # Marchuk-Skamarock stretch
dz_bottom: 30       # surface spacing — matches TC's dz_surf=30m ✓
dt: 10secs          # FIXED timestep — TC uses adapt_dt (dt range: 0.5-12 s)
dt_rad: 30mins
t_end: 14hours      # matches TC's t_max=50400s ✓
```

**Key differences and their implications:**

| Aspect | TC (CalibrateEDMF) | ClimaAtmos (current) | Risk |
|--------|-------------------|---------------------|------|
| **Timestep** | Adaptive: dt_min=0.5-1s, dt_max=2-12s | Fixed: dt=10s | ⚠️ 10s may be too coarse for non-eq micro; hydrometeor fall speeds of 5 m/s over dz=30m require dt < 3s for CFL=0.5 |
| **Grid adaptivity** | Cell size adjusted to maintain CFL | Fixed grid | ✓ ClimaAtmos handles CFL differently via ODE solver |
| **Conservative regridding** | Yes, for observation-map output | No (uses linear interpolation) | Moderate: thin cloud layers may be under-weighted |
| **Process reweighting** | `reweight_processes_for_grid=true` | N/A | Moderate: process rates at coarse levels may be over-weighted |
| **Domain top** | 45 km | 40 km | Minor for SOCRATES (boundary layer + low cloud focus) |

**Why we didn't replicate TC's adaptive-dt/CFL grid approach:**
- ClimaAtmos does not have TC's grid-adaptive architecture. TC is a 1D finite-volume SCM that can rebuild its grid between runs; ClimaAtmos is a spectral-element GCM with a fixed grid defined at model init.
- ClimaAtmos uses its own ODE integrator (IMEX splitting) which handles stiff terms implicitly, reducing the explicit CFL constraint compared to TC's explicit timestepper.
- The `dt=10s` fixed timestep was set as an initial working value. It has not been validated for CFL stability with non-equilibrium microphysics at `dz_bottom=30m`.

**Action items / known risks to address:**
1. **CFL check for precipitation**: with `dz_bottom=30m` and assumed max fall speed ~5 m/s, `dt_cfl = 0.5 * 30/5 = 3s` → current `dt=10s` likely violates CFL for rain/snow at the bottom levels. Consider reducing dt or checking whether ClimaAtmos's limiter handles this.
2. **Consider using ClimaAtmos adaptive timestepping**: ClimaAtmos supports `adaptive_dt: true` in the YAML config. This is the ClimaAtmos equivalent of TC's `adapt_dt`. Should be evaluated for calibration runs.
3. **Conservative regridding**: the current observation map uses linear interpolation via `interpolate_profile_to_target_grid`. TC used conservative regridding to preserve integrated quantities (LWP, IWP) across grid resolutions. This matters most when `y_var_names` includes column integrals.

---

## 0.4 Calibrated Parameters: What Was Calibrated Before vs Now

### CalibrateEDMF (TC) — full parameter space

The header config defines priors for multiple non-equilibrium microphysics parameterizations:

**Autoconversion/accretion schemes** (selected per `nonequilibrium_moisture_scheme`):
- `geometric_liq_c_{1,2,3,4}` — geometric liquid autoconversion parameters
- `geometric_ice_c_{1,2,3,4}` — geometric ice autoconversion parameters
- `exponential_T_scaling_ice_c_{1,2}` — ice nucleation T-scaling (Cooper-like)
- `powerlaw_T_scaling_ice_c_{1,2}` — power-law ice nucleation
- `tau_autoconv_ice`, `tau_autoconv_liq` — relaxation timescales for autoconversion (calibration_setup: `tau_autoconv_noneq`)
- `E_liq_ice` — liquid-ice collection efficiency
- `ice_dep_acnv_scaling_factor` — depositional accretion factor
- `r_ice_snow_threshold_scaling_factor` — ice→snow threshold radius
- `τ_acnv_sno_threshold`, `massflux_N_i_boost_factor`, `sedimentation_N_i_boost_factor`

For the **`tau_autoconv_noneq`** calibration (the relevant one for the ClimaAtmos migration), only:
- `tau_autoconv_ice`
- `tau_autoconv_liq`
are calibrated (plus potentially scheme-specific parameters).

**Priors seeded from** `Reference/Output_Inferred_Data/SOCRATES_Atlas_LES_inferred_parameters.nc`.

### ClimaAtmos (current) — minimal parameter space
```toml
# prior.toml
[tau_autoconv_ice]
# ...
[tau_autoconv_liq]
# ...
```

Currently only `tau_autoconv_ice` and `tau_autoconv_liq` — consistent with `tau_autoconv_noneq` setup.
Prior bounds and mean need to be cross-checked against CalibrateEDMF header `constraints` dict.

---

## 0.5 Observation Variables: What Was Calibrated Before vs Now

### CalibrateEDMF — calibration variable sets

Variables are concatenated per-flight, per-variable, across the vertical grid. From postprocessing paths:

**Used in the `tau_autoconv_noneq` experiment:**
```
iwp_mean, lwp_mean, qi_mean, qip_mean, ql_mean, qr_mean
```

Variable names are **TC diagnostic names** (from TC `NetCDFIO_Diags`):
| TC name | Physical meaning | ClimaAtmos equivalent |
|---------|-----------------|----------------------|
| `lwp_mean` | Column LWP (kg/m²) | Computed from `clw` profile |
| `iwp_mean` | Column IWP (kg/m²) | Computed from `cli` profile |
| `ql_mean` | Cloud liquid water profile (g/kg) | `clw` (with unit conversion) |
| `qi_mean` | Cloud ice profile (g/kg) | `cli` |
| `qip_mean` | Ice precipitation profile (precip ice; g/kg) | Not yet in ClimaAtmos output |
| `qr_mean` | Rain water profile (g/kg) | Output as `qr` in ClimaAtmos |

**Time averaging window** (calibration template body):
```julia
t_bnds.obs_data = missing  # full window: 0 to 12h
t_bnds.ERA5_data = missing  # full window: 0 to 14h
# or in some runs:
t_bnds.obs_data = (10h, 12h)  # last 2 hours only
```

**ClimaAtmos current window** (experiment_config.yml):
```yaml
y_t_start_sec: 36000.0   # 10h
y_t_end_sec: 43200.0     # 12h
g_t_start_sec: 36000.0
g_t_end_sec: 43200.0
```
→ Only the final 2 hours (10h–12h). Consistent with the `t_bnds` trimmed option.

### ClimaAtmos — current calibration variable set
```yaml
y_var_names: [thetaa, hus, clw]
```
Observed with `reduction: 10m_inst`.

**Gaps vs CalibrateEDMF:**
- `clw` ≈ `ql_mean` ✓ (direct cloud liquid water)
- `thetaa` ≈ `THETAL`-equivalent temperature ✓ (but not identical; thetaa is actual potential T, not liquid-ice)
- `hus` ≈ total water ✓ (but TC uses `qt_mean = ql+qi+qv`)
- Missing: `iwp_mean`, `lwp_mean`, `qi_mean`, `qip_mean`, `qr_mean`

The current ClimaAtmos calibration is therefore calibrating to a different observation vector than CalibrateEDMF used. Ice microphysics is not yet reflected in the observation variables.

---

## 0.6 Flight Cases: Which Flights Were Used

### CalibrateEDMF flight defaults (from header config)

```julia
const N_CCNs_default = Dict(
    1  => 75e6,   # RF01 — 75/cm³
    9  => 190e6,  # RF09 — 190/cm³
    10 => 55e6,   # RF10 — 55/cm³
    11 => 115e6,  # RF11 — 115/cm³
    12 => 210e6,  # RF12 — 210/cm³
    13 => 180e6,  # RF13 — 180/cm³
)
```

All-flights string = `"All"` → uses `[1, 9, 10, 11, 12, 13]` (obs_data) or `[1, 9, 10, 12, 13]` (obs_data only).

`tau_autoconv_noneq` runs in postprocessing storage used: **RF01 + RF09**, both obs_data and ERA5_data.

### ClimaAtmos current experiment_config.yml
```yaml
cases:
  - {flight_number: 1,  forcing_type: obs_data}
  - {flight_number: 9,  forcing_type: obs_data}
  - {flight_number: 1,  forcing_type: ERA5_data}
  - {flight_number: 9,  forcing_type: ERA5_data}
```
→ RF01 + RF09, both forcing types. Consistent with the relevant CalibrateEDMF runs. ✓

---

## 0.7 Calibration Run Validation: Iteration 000 Diagnostic Evidence

This section records what was verified when the first full serial calibration run completed iteration_000.
It is the reference baseline for "we know this is working."

### G_ensemble.jld2 structure

File: `calibration/experiments/SOCRATES/output/simple_socrates/iteration_000/G_ensemble.jld2`

```
Keys: ["single_stored_object"]
Size: (348, 12)
  - 348 = observation vector length (y_var_names × cases × z_model levels)
  - 12  = ensemble_size members
NaN cols: 0
Finite cols: 12  ← all members produced valid output
member_001 through member_012: all OK
```

Verified with:
```julia
using JLD2
d = JLD2.load("output/simple_socrates/iteration_000/G_ensemble.jld2")
G = d["single_stored_object"]  # (348, 12) matrix
```

### Simulation completion check

Each member runs ClimaAtmos simulations across all 4 cases. All 12 members ran to completion:
- 85 time steps
- `t_end = 50400.0 s` (14 hours)

Verified by reading `thetaa_10m_inst.nc` from each member's `output_active` output directory.

### "Ended too early" guard in observation_map.jl

`observation_map.jl` line ~109 contains:
```julia
if sim_t_end < 0.95 * t_end
    throw("Simulation ended too early")
end
```

Parameters from `experiment_config.yml`:
- `g_t_end_sec: 43200.0` (12 h) → this is the `t_end` in the guard
- Threshold: `0.95 * 43200 = 41040 s`
- Actual `sim_t_end = 50400 s` (14 h from model_config.yml)

**→ The guard `50400 < 41040` never triggers.** The "ended too early" warnings seen in offline test output come from the test reading the `output_active` symlink (which points to `output_0008`), NOT from a real failure.

### output_active symlink behavior

Each member's output directory looks like:
```
member_001/
  config_1/
    output_active -> output_0008    ← symlink to latest completed sub-run
    output_0001/ ... output_0008/   ← intermediate sub-run checkpoints
```

`output_active` always points to the latest completed sub-run. `resolve_member_simdir()` in `observation_map.jl` prefers this symlink and falls back to `output_0000` for older runs.

### What the 348-element observation vector contains

With `y_var_names: [thetaa, hus, clw]`, 4 cases (RF01+RF09 × obs_data+ERA5_data), and `z_model` levels:
```
obs_len = len(y_var_names) × n_cases × n_z
348 = 3 × 4 × 29      (approximately; depends on z_max=4000 and z_elem=60 grid)
```
The exact `z_model` comes from reading the model metadata NC file via `resolve_z_model!()`.

---

## 0.8 Distributed Path: Next Steps and Technical Watchpoints

The serial path is validated. The next planned task is running with `Distributed.jl` workers. The infrastructure is partially in place (see Section 6.8, items 3–5) but has not been end-to-end tested.

### What is already implemented

- `run_calibration.jl` selects `WorkerBackend()` when `nworkers() > 1`, `JuliaBackend()` otherwise.
- Worker bootstrap via `Distributed.remotecall_eval(Main, wid, quote ... end)` to avoid toplevel-expression errors.
- `bootstrap_workers()` in `run_calibration.jl` handles `@everywhere include("observation_map.jl")` equivalently on each worker.

### What still needs end-to-end validation

1. **`output_active` symlink resolution on workers**: `resolve_member_simdir()` must resolve symlinks correctly from each worker's filesystem view (NFS paths should match since all workers see same `/net/sampo/data1/...` mount).

2. **Worker project environment**: Workers must be started with the correct project:
   ```julia
   addprocs(n; exeflags=["--project=/path/to/calibration/experiments/SOCRATES"])
   ```
   Otherwise package loading fails silently with wrong method dispatches.

3. **Coordinator-only `CAL.save_G_ensemble` call**: Only the coordinator (rank 1 / `myid() == 1`) should write `G_ensemble.jld2`. If workers also call this, JLD2 write contention occurs. Verify that `ClimaCalibrate`'s `WorkerBackend` already enforces this, or add an explicit `myid() == 1` guard.

4. **JLD2 write contention**: Multiple workers simultaneously writing calibration artifacts (e.g., per-iteration checkpoint files, `eki_file.jld2`) can conflict. The serial path is safe because writes are sequential. With workers, check whether `ClimaCalibrate` serializes these writes through the coordinator.

5. **Forcing file conversion races**: The per-case forcing file conversion in `run_calibration.jl` is done once before workers launch. Confirm this conversion completes fully before any worker tries to re-read the same forcing files.

### Practical startup recipe (to validate)

```julia
using Distributed
addprocs(12; exeflags=["--project=$(pwd())"])  # from SOCRATES dir
@everywhere begin
    import Pkg; Pkg.activate(".")
    include("run_calibration.jl")
end
main()  # or however the distributed run is kicked off
```

---
- Core ClimaAtmos was kept generic. No SOCRATES-specific forcing mode was added in `src/`.
- `ReanalysisTimeVarying` now accepts an optional `external_forcing_file` override when provided by the experiment.
- The SOCRATES experiment now resolves case forcing files via SSCF and injects them through that generic hook.
- This path is syntax-checked, but still needs end-to-end runtime validation.
- Observation-map vertical-grid mismatch is fixed by interpolating model profiles onto `z_model` before concatenation.
- Offline end-to-end observation-map validation now runs against saved calibration artifacts in `output/simple_socrates/iteration_000`.
- Latest offline smoke result (single member): `G_size=(348, 1)`, `obs_len=348`, `nan_frac=0.0`.
- SSCF has been upgraded as a breaking `0.14` line for modern Thermodynamics (`Thermodynamics = "1"`, `ForwardDiff = "1"`, `NCDatasets = "0.14"`).
- The temporary Thermodynamics compatibility branch in SSCF (`ThermodynamicStateCompat`) was removed in favor of one direct Thermodynamics-1.0 code path.
- Atlas LES input/output retrieval in SSCF is now artifact-backed: `download_atlas_les_inputs` and `download_atlas_les_outputs` create/update Julia Artifacts, and `open_atlas_les_input` / `open_atlas_les_output` read from artifact roots.
- NCDatasets migration hardening is in progress for v0.13+ / v0.14+: SSCF now avoids direct `NC.CFVariable` type assumptions, avoids unsafe `[:]` full-variable reads for multi-dimensional data, and adds regression tests for shape-preserving reads and insertion-index broadcasting.

---

## NCDatasets v0.13/v0.14 Migration Notes (SSCF + SOCRATES)

These release changes were directly relevant to SOCRATES forcing regressions.

### What changed upstream

- In NCDatasets v0.13 (DiskArrays integration), `ncvar[:]` for multi-dimensional variables now follows base array semantics and flattens to a vector.
- In NCDatasets v0.14, `CFVariable` moved from `NCDatasets.CFVariable` to `CommonDataModel.CFVariable`; code that dispatches on the old concrete type is brittle.
- NCDatasets variables are more commonly lazy/labeled wrappers, so shape- and label-sensitive code must avoid assumptions about concrete array types.

### Why this affected SOCRATES

- Several SSCF paths implicitly relied on pre-v0.13 behavior where `[:]` was used as a full-read shorthand without flattening consequences.
- Multiple utilities assumed `NC.CFVariable` concrete typing for dimension-label logic.
- Insertion and concatenation logic around air/ground profile merging became sensitive to shape mismatches once data arrived as DiskArray/CommonDataModel wrappers.

### Fix pattern adopted

- Replace concrete-type checks (`isa(..., NC.CFVariable)`) with capability checks (for example, `hasmethod(NC.dimnames, Tuple{typeof(x)})`).
- Use dimension-preserving full reads for multi-dimensional variables (`var[:, :, ...]` semantics via helper), not `var[:]`.
- Keep one-dimensional explicit vectorization intentional and centralized; SSCF now routes those reads through dedicated helpers such as `read_vector`, `read_profile_at_time`, and `read_profiles_over_time` so raw `vec(selectdim(...))` does not silently flatten the wrong axes.
- Validate insertion-index arrays by singleton-expansion rules rather than integer-division repeat factors.
- Prefer small function-level tests for these shape contracts and reserve the full process-case matrix for slower integration coverage.

### Regression tests added

- Unit test that demonstrates `var[:]` flattening on a 2D NetCDF variable and validates SSCF shape-preserving helper behavior.
- Unit test for `combine_air_and_ground_data` insertion-index broadcasting to guard against zero-repeat or mismatched-shape failures.
- Unit tests for vertical-profile extraction and `z x time` slab extraction, including failure cases where lon/lat dimensions are unexpectedly non-singleton.

### Test strategy update

- SSCF test entry now defaults to fast unit coverage.
- The slow process-case combination matrix is opt-in (`Pkg.test(; test_args=["integration"], allow_reresolve=false)` or `SSCF_RUN_INTEGRATION_TESTS=true`).
- This matters for migration work because shape bugs are much easier to isolate in small synthetic fixtures than in the full flight matrix.

### SOCRATES offline full-pipeline test (ClimaAtmos)

- The SOCRATES experiment test suite now includes an offline calibration dry-run in `test/calibration_pipeline_offline_tests.jl` (`"SOCRATES offline calibration dry-run pipeline"`).
- This test exercises the real calibration path end-to-end (excluding simulation launches):
    - reads experiment config,
    - runs reference preprocessing in no-overwrite mode,
    - builds `z_model`, observations, and prior using the same setup code as `run_calibration.jl`,
    - writes/validates `obs_metadata.jld2` and copied config inputs,
    - loads saved `eki_file.jld2` from `output/simple_socrates/iteration_000`,
    - calls `observation_map(0; config_dict=cfg)`,
    - asserts output shape matches EKP observation length for the minibatch and full ensemble,
    - asserts all values are finite.
- The suite also retains a focused observation-map test in `test/observation_map_tests.jl`.
- Run from a Julia REPL with project activated:
    - `import Pkg`
    - `Pkg.activate("calibration/experiments/SOCRATES")`
    - `include("calibration/experiments/SOCRATES/test/runtests.jl")`

This provides the requested offline pipeline coverage without waiting for a full online calibration iteration.

### Remaining recommendation

- Continue auditing SSCF and ClimaAtmos SOCRATES adapters for `[:]` usage where source objects may be NCDatasets/CommonDataModel variables; only retain `[:]` where flattening is explicitly intended.

---

## 1. Executive Summary

This guide documents the migration of SOCRATES calibration workflows from legacy `CalibrateEDMF.jl` to `ClimaAtmos.jl` + `ClimaCalibrate.jl`. **This is NOT a template/file-path migration.** It requires a full architectural port of SOCRATES-specific forcing machinery from TurbulenceConvection into ClimaAtmos.

### The Core Problem

**TurbulenceConvection.jl** (Legacy):
- Has a `::SOCRATES` case type with full forcing/surface/radiation dispatch
- Calls `SOCRATESSingleColumnForcings` (SSCF) to get **time-dependent forcing functions**
- All forcing is **pre-computed by SSCF** (not derived online)
- Evaluates forcing functions each timestep and applies nudging + advection

**ClimaAtmos.jl** (Target):
- Has `GCMForcing`, `ExternalDrivenTVForcing`, `ISDACForcing` dispatch on external data
- All of these read **static or time-varying profiles** from NetCDF files
- Caches interpolated fields, applies nudging toward those fields
- **Does NOT currently know how to handle SSCF flight-number + forcing_type dispatch**

**Decision Required:** How to represent SOCRATES-SSCF data in ClimaAtmos's forcing architecture?

---

## 2. TurbulenceConvection SOCRATES Architecture

### 2.1 Case Definition

```julia
struct SOCRATES <: AbstractCaseType
    flight_number::Int       # e.g., 9, 12, 13 (SOCRATES observational flights)
    forcing_type::Symbol     # :obs_data or :ERA5_data
end
```

### 2.2 SSCF Integration: Three-Phase Setup

**Phase 1: Reference State**
```julia
ref_state = SSCF.process_case(
    flight_number = flight_number,
    forcing_type = forcing_type,
    surface = "ref",  # returns reference ThermodynamicState for hydrostatic balance
)
```

**Phase 2: Initial Conditions**
```julia
ic = SSCF.process_case(
    flight_number = flight_number,
    forcing_type = forcing_type,
    initial_condition = true,  # returns interpolated profiles at model grid
)
# ic contains: H_nudge, qt_nudge, u_nudge, v_nudge, ug_nudge, vg_nudge, dTdt_rad
```

**Phase 3: Time-Dependent Forcing Functions**
```julia
forcing_funcs = SSCF.process_case(
    flight_number = flight_number,
    forcing_type = forcing_type,
    initial_condition = false,  # returns function objects
)
# Returns 11 function arrays: (where k = height index)
#   dTdt_hadv[k](t)        :: T horizontal advection [K/s]
#   dqtdt_hadv[k](t)       :: qt horizontal advection [kg/kg/s]
#   subsidence[k](t)       :: large-scale vertical velocity [1/s]
#   H_nudge[k](t)          :: liquid-ice potential T target [K]
#   qt_nudge[k](t)         :: total water target [kg/kg]
#   u_nudge[k](t)          :: zonal wind target [m/s]
#   v_nudge[k](t)          :: meridional wind target [m/s]
#   ug_nudge[k](t)         :: geostrophic zonal wind [m/s]
#   vg_nudge[k](t)         :: geostrophic meridional wind [m/s]
#   dTdt_rad[k](t)         :: radiative heating [K/s] (PRE-COMPUTED)
#   dTdt_fluc[k](t)        :: T fluctuations (mostly zero for SOCRATES)
```

**Key property:** Each function is `f[k](t::Real) -> scalar`, evaluated **once per timestep** at each height.

### 2.3 Dycore Integration

**Initialization Dispatches:**
- `initialize_profiles(::SOCRATES, ...)` reads Phase 2 ICs, sets Y.θ_liq_ice, Y.q_tot, Y.uₕ
- `initialize_forcing(::SOCRATES, ...)` stores Phase 3 forcing_funcs in mutable array
- `initialize_radiation(::SOCRATES, ...)` stores Phase 3 dTdt_rad functions
- `surface_params(::SOCRATES, ...)` calls SSCF for surface T, q; creates Monin-Obukhov surface
  - Surface conditions **also come from SSCF**, not separately computed

**Timestepping Dispatches:**
- `update_forcing(::SOCRATES, state, t, ...)` evaluates all `forcing_funcs[k](t)`, applies nudging + advection
- `update_radiation(::SOCRATES, state, t, ...)` evaluates all `dTdt_rad[k](t)`, adds to heating

### 2.4 Nudging Strategy

Applied at each timestep via:
```julia
# Scalar nudging
dT/dt_nudge = -inv_τ_scalar * (T - T_nudge[k](t))
dqt/dt_nudge = -inv_τ_scalar * (qt - qt_nudge[k](t))

# Wind nudging
du/dt_nudge = -inv_τ_wind * (u - u_nudge[k](t))
dv/dt_nudge = -inv_τ_wind * (v - v_nudge[k](t))
```

**Timescales vary by forcing_type:**
| Type | wind_τ | scalar_τ |
|------|--------|----------|
| obs_data | 20 min | 20 min |
| ERA5_data | 60 min | ∞ (no nudging) |

### 2.5 Additional Forcing Components

**Subsidence:**
```julia
w_subsidence[k](t) :: vertical velocity anomaly (1/s)
[Applied to scalar transport vertically]
```

**Horizontal Advection (Direct):**
```julia
dTdt_hadv[k](t), dqtdt_hadv[k](t) :: directly added to tendencies
[Pre-computed from LES diagnostics]
```

**Radiation:**
```julia
dTdt_rad[k](t) :: pre-computed from LES (or observational estimate)
[NOT computed online; consistent with forcing data]
```

---

## 3. ClimaAtmos Forcing Architecture

### 3.1 Dispatch Types

```julia
abstract type AbstractForcingType end

struct GCMForcing <: AbstractForcingType
    external_forcing_file::String       # NetCDF path
    cfsite_number::String               # site ID
end

struct ExternalDrivenTVForcing <: AbstractForcingType
    external_forcing_file::String       # NetCDF path (time-varying)
end

struct ISDACForcing <: AbstractForcingType
    # No file; analytic profiles
end

struct Nothing <: AbstractForcingType
    # No forcing
end
```

### 3.2 Cache Structure

```julia
function external_forcing_cache(Y, forcing_type::GCMForcing, params, start_date)
    # Returns NamedTuple with pre-allocated/interpolated fields:
    return (;
        ᶜdTdt_fluc,         # vertical eddy advection component [K/s]
        ᶜdqtdt_fluc,        # [kg/kg/s]
        ᶜdTdt_hadv,         # horizontal T advection [K/s]
        ᶜdqtdt_hadv,        # horizontal qt advection [kg/kg/s]
        ᶜT_nudge,           # GCM temperature profile (nudging target) [K]
        ᶜqt_nudge,          # GCM specific humidity (nudging target) [kg/kg]
        ᶜu_nudge,           # GCM zonal wind (nudging target) [m/s]
        ᶜv_nudge,           # GCM meridional wind (nudging target) [m/s]
        ᶜinv_τ_wind,        # inverse relaxation timescale for wind [1/s]
        ᶜinv_τ_scalar,      # inv. relax. timescale for scalars [1/s] (height-dependent)
        ᶜls_subsidence,     # large-scale subsidence [m/s]
        toa_flux,           # top-of-atmosphere insolation [W/m²]
        cos_zenith,         # cosine of solar zenith angle
    )
end
```

**Key difference:** All cache slots are `Fields.Field` objects (spatially-distributed), not function arrays.

**Important implementation detail:** function-backed `TimeVaryingInput` can already drive field-valued destinations directly, but it does not perform vertical interpolation automatically. If we use analytic/function-backed TVI for SOCRATES, the closure itself must either:
- return values already defined on the ClimaAtmos target grid, or
- perform the profile-to-grid interpolation internally before writing into `dest`.

### 3.3 Tendency Application

```julia
function external_forcing_tendency!(Yₜ, Y, p, t, forcing_type::GCMForcing)
    # 1. Wind relaxation
    Yₜ.c.uₕ -= (Y.c.uₕ - ᶜuₕ_nudge) * ᶜinv_τ_wind
    
    # 2. Scalar relaxation
    dT_nudge = -(T - T_nudge) * inv_τ_scalar
    dqt_nudge = -(qt - qt_nudge) * inv_τ_scalar
    
    # 3. Horizontal advection (pre-computed)
    dT_total = dT_hadv + dT_nudge + dT_fluc
    dqt_total = dqt_hadv + dqt_nudge + dqt_fluc
    
    # 4. Convert to energy/moisture tendencies
    Yₜ.c.ρe_tot += ρ * cv * dT_total + ρ * Lv * dqt_total
    Yₜ.c.ρq_tot += ρ * dqt_total
    
    # 5. Subsidence
    subsidence!(Yₜ.c.ρe_tot, ..., ᶜls_subsidence, ...)
    subsidence!(Yₜ.c.ρq_tot, ..., ᶜls_subsidence, ...)
end
```

---

## 4. Architectural Differences: What Doesn't Match

| Aspect | TurbulenceConvection | ClimaAtmos | Implication |
|--------|-------|-----------|---|
| **Forcing source** | SSCF time-dependent functions | Static/time-varying NetCDF profiles | Need mapping strategy |
| **Forcing indexing** | `funcs[k](t)` (single height) | `ᶜfield[z]` (all heights, spatially) | Need callback to populate fields |
| **Data flow** | functions stored in mutable arrays | fields pre-allocated in cache | Need evaluation step each timestep |
| **Surface** | Part of forcing dispatch | Separate from forcing | Need wiring |
| **Radiation** | Pre-computed from SSCF | Computed online (RRTMGP) | Need decision on source |
| **Grid matching** | SSCF interpolates to model grid | ClimaAtmos expects pre-interpolated data | SSCF already does this OK |
| **Case activation** | `::SOCRATES(flight_num, forcing_type)` | Config file string dispatch | Need translation layer |

Additional mismatches that are easy to forget:

- **Nudging variable mismatch:** SOCRATES nudges `H_nudge`, i.e. liquid-ice potential temperature, while the current ClimaAtmos SCM external-forcing tendency nudges actual temperature `ᶜT`. Reusing the existing tendency path without modification would silently change the physics.
- **Radiation term mismatch:** SSCF provides an explicit `dTdt_rad[k](t)` term. The current `ExternalDrivenTVForcing` cache/callback path has no dedicated radiative-heating slot; ClimaAtmos instead expects online radiation plus `cos_zenith` and `toa_flux` updates.
- **Geostrophic wind mismatch:** SOCRATES provides `ug_nudge` and `vg_nudge`, but the current ClimaAtmos external-forcing tendency path only consumes direct wind nudging targets `u_nudge`, `v_nudge`. If geostrophic forcing matters dynamically, it will need an additional hook rather than being silently dropped.
- **Surface humidity mismatch:** the existing time-varying surface pathway updates surface temperature through `surface_inputs.ts`, but SOCRATES surface conditions also prescribe humidity via `qg`. That does not currently flow through the `ReanalysisTimeVarying` surface path.
- **Callback ordering dependency:** for time-varying forcings, the callback must populate cache fields before `remaining_tendency!` calls `external_forcing_tendency!`. Any SOCRATES adapter that reuses `ExternalDrivenTVForcing` semantics must preserve that ordering.

---

## 5. Work Required: Detailed Implementation Plan

### Step 1: Architectural Decision ✗ *To Do*

**Choose one approach:**

**Option A: New `SOCRATESForcing` Type**
```julia
struct SOCRATESForcing <: AbstractForcingType
    flight_number::Int
    forcing_type::Symbol
    # Internally cache SSCF results
end

# Pros: Clean separation, explicit SOCRATES handling
# Cons: New type adds ~500 LOC, need callbacks for time-dependent eval
```

**Option B: Extend `ExternalDrivenTVForcing` with SSCF Backend**
```julia
# Treat SSCF as "external time-varying forcing source"
# Reuse ClimaAtmos callback/cache plumbing and provide SSCF-backed TimeVaryingInputs
# either as analytic/function-backed inputs or as a custom data handler
# Pros: Reuses existing ClimaAtmos infrastructure, preserves the current forcing callback model,
#       and avoids introducing a parallel forcing stack just for SOCRATES
# Cons: Still requires SOCRATES-specific plumbing for surface state, radiation, and initialization
```

**Option C: Pre-process SOCRATES to NetCDF**
```julia
# Create one-time conversion: SSCF output → NetCDF file with all forcing profiles
# Then use standard GCMForcing/ExternalDrivenTVForcing
# Pros: No code changes to ClimaAtmos, reuses existing paths
# Cons: Extra preprocessing step, loses time-dependency benefits
```

**Updated recommendation:** Prefer **Option B** first.

Rationale:
- `ClimaUtilities.TimeVaryingInputs.TimeVaryingInput` already supports more than file-backed inputs:
  - function-backed analytic inputs
  - direct `times, vals` inputs for scalar series
  - file/data-handler-backed regridded field inputs
- File-backed `TimeVaryingInput` already supports vertical regridding to the model target space, so file support was never the blocker.
- The main gap is not interpolation. The main gap is adapting SSCF's output structure to ClimaAtmos's existing callback/cache pipeline.
- The most direct path is to keep the current `ExternalDrivenTVForcing` style flow and supply SSCF-backed time-varying inputs for the column fields, while separately wiring SOCRATES surface and radiation hooks.

What Option B does **not** mean:
- It does not require writing NetCDF files.
- It does not require losing vertical interpolation support.
- It does not mean the current `ExternalDrivenTVForcing` implementation can be reused unchanged.

What Option B **does** require:
- a ClimaAtmos-facing type or configuration mode that selects SOCRATES-specific setup,
- SSCF-backed field-valued `TimeVaryingInput`s or equivalent data-handler objects,
- a SOCRATES-specific tendency path for the `H_nudge` versus `T_nudge` distinction,
- separate handling for surface humidity and possibly precomputed radiation.

Working interpretation of Option B:
- Use function-backed `TimeVaryingInput` for SSCF-provided field-valued forcings when possible.
- If needed, add a thin custom `TimeVaryingInput`/data-handler path that evaluates SSCF functions into preallocated destination fields.
- Avoid temporary NetCDF output unless a quick prototype is needed.

### Step 2: Implement `external_forcing_cache` for SOCRATES ✗ *To Do*

```julia
function external_forcing_cache(Y, f::ExternalDrivenTVForcing, params, start_date)
    # Phase 1: Call SSCF to get forcing functions and wrap them as TimeVaryingInputs
    # so they can be evaluated through the existing callback pattern.

    # Phase 2: Call SSCF for surface conditions and cache those separately.

    # Phase 3: Allocate ClimaAtmos cache fields (same as current ExternalDrivenTVForcing)
    FT = Spaces.undertype(axes(Y.c))
    ᶜdTdt_hadv = similar(Y.c, FT)
    ᶜdqtdt_hadv = similar(Y.c, FT)
    ᶜT_nudge = similar(Y.c, FT)
    # ... (etc. for all cache slots)

    # Phase 4: Store SSCF-backed time-varying inputs and surface data for later
    return (;
        ᶜdTdt_hadv, ᶜdqtdt_hadv, ...,  # cache fields (will populate each step)
        column_timevaryinginputs = sscf_timevaryinginputs,
        sscf_surface_conds = surface_conds,  # keep for surface setup
    )
end
```

**Challenge:** SSCF returns profiles as arrays of height-wise functions, while ClimaAtmos expects `evaluate!(dest, tv_input, t)` to fill destination fields.

**Solution:** Wrap SSCF output in function-backed or custom field-valued `TimeVaryingInput`s and evaluate them through the existing forcing callback path before `external_forcing_tendency!()` runs.

### Step 3: Implement `external_forcing_tendency!` for SOCRATES ✗ *To Do*

```julia
function external_forcing_tendency!(Yₜ, Y, p, t, ::ExternalDrivenTVForcing)
    # Step 1: Assume the forcing callback has already evaluated the SSCF-backed
    # TimeVaryingInputs into the cached fields.

    # Step 2: Reuse the existing nudging + advection logic once the fields are filled.
end
```

**Nuance:** Timescales are `forcing_type`-dependent. We still need to set `ᶜinv_τ_wind`, `ᶜinv_τ_scalar` in cache based on `obs_data` vs. `ERA5_data`.

**Additional nuance:** This step likely needs a SOCRATES-specific tendency implementation, even if the cache/callback path reuses `ExternalDrivenTVForcing` patterns, because the existing shared tendency implementation nudges actual temperature rather than liquid-ice potential temperature.

### Step 4: Wire Surface Conditions ✗ *To Do*

**Current ClimaAtmos:**
- Surface setup is separate from external_forcing
- Dispatches on `surface_setup_cache()` and `surface_setup_tendency!()`

**Need to:**
1. Define how SSCF surface conditions (T_sfc, q_sfc) map to ClimaAtmos surface model
2. Either:
   - Create `SOCRATESSurface` type, or
   - Modify Monin-Obukhov surface fetch to call SSCF for boundary conditions

**Key question:** Does ClimaAtmos have Monin-Obukhov surface model? (Legacy TurbulenceConvection does.)

### Step 5: Handle Radiation ✗ *To Do*

**Current ClimaAtmos:** Uses online RRTMGP radiation (or placeholder)

**SSCF provides:** Pre-computed `dTdt_rad[k](t)` from LES/observational consistency

**Decision needed:**
- **Option A:** Skip online radiation, use SSCF `dTdt_rad` directly (set as part of forcing tendency)
- **Option B:** Use online radiation, ignore SSCF radiation (inconsistent with forcing data)
- **Option C:** Hybrid: Use online but rescale by SSCF factor

**Recommendation:** **Option A** (use SSCF) — maintains consistency with observational forcing.

### Step 6: Initialize Profiles ✗ *To Do*

```julia
function initialize_profiles(case::SOCRATESForcing, params, start_date, ...)
    # Call SSCF Phase 2: get initial conditions
    ic = SSCF.process_case(
        flight_number = case.flight_number,
        forcing_type = case.forcing_type,
        initial_condition = true,
        # ... grid, thermo params, etc.
    )
    
    # Map to initial state Y
    ᶜz = Fields.coordinate_field(Y.c).z
    
    # Initialize scalars
    # Y.c.ρe_tot[z] ← compute from ic.H_nudge[k] and ic.qt_nudge[k]
    # Y.c.ρq_tot[z] ← ic.qt_nudge[k] * ρ
    
    # Initialize winds
    # Y.c.uₕ[z] ← Geometry.UVVector(ic.u_nudge[k], ic.v_nudge[k])
    
    return Y
end
```

### Step 7: Map Diagnostics ✗ *To Do*

**TurbulenceConvection SOCRATES verifies:**
- `q_tot_mean`, `q_liq_mean`, `q_ice_mean` — water phases
- `thetal_mean` — liquid-ice potential temperature
- `temperature_mean` — actual temperature
- `u_mean`, `v_mean` — horizontal winds
- `tke_mean` — kinetic energy
- `lwp_mean`, `iwp_mean` — integrated water paths

**ClimaAtmos diagnostics available:**
- `Y.c.ρ`, `Y.c.ρq_tot`, `Y.c.ρq_liq`, `Y.c.ρq_ice`, `Y.c.ρe_tot`, `Y.c.uₕ`
- Derived: temperature (from ρe_tot), specific humidity (from ρq_tot), etc.

**To Do:** Map ClimaAtmos output fields to observation-space quantities, update `observation_map.jl` accordingly.

---

## 6.5 Current Scope And Remaining Gaps

What is implemented now:
- Core forcing dispatch remains generic; no `SOCRATESTVForcing`-style type exists in ClimaAtmos core.
- `ReanalysisTimeVarying` in `model_getters.jl` accepts a provided `external_forcing_file`, and otherwise keeps existing ERA5 generation behavior.
- SOCRATES experiment config uses `ReanalysisTimeVarying` for initial condition, external forcing, surface setup, and surface temperature.
- SOCRATES experiment interface resolves per-case forcing files via SSCF (`open_atlas_les_input`) and injects them through `config["external_forcing_file"]`.

What is still intentionally incomplete or approximate:
- **Reference data preprocessing:** Raw SOCRATES Atlas LES output files have variable names and units that differ from what ClimaAtmos expects (e.g., THETAL→theta_mean, QT→qt_mean, weight fraction→specific humidity, grams→kg). A preprocessing pipeline must convert these files into ClimaAtmos-compatible format before observation_map can read them successfully.
- **Surface humidity:** Not wired through the interactive surface path. Current setup relies on existing generic surface pathways.
- **Geostrophic wind inputs:** `ug_nudge`, `vg_nudge` still not used.
- **Runtime validation:** No full SOCRATES forward run has been executed yet.

### 6.7 Regression Postmortem: Collapsed Vertical Coordinate (`z`)

Observed failure mode during calibration migration debugging:
- Converted forcing files in `Forcing/ClimaAtmos/*.nc` were found with a collapsed vertical coordinate (`z` all zeros or effectively non-unique).
- ClimaAtmos then reported duplicated interpolation knots and quickly produced NaNs during callback initialization.

Root cause:
- A previously generated malformed converted forcing file was being reused in subsequent runs.
- Legacy existence-only checks (`isfile(output_file)`) are insufficient, because they do not validate the content quality of the converted file.

Fix implemented in experiment layer:
- Added `is_valid_converted_forcing_file(path)` guard that verifies:
    - file exists,
    - variable `z` exists,
    - `z` values are finite,
    - `z` has more than one unique value.
- Conversion now regenerates output unless the file passes this validity check.
- Conversion uses temporary output files and replacement logic to reduce partial-file reuse risks.

Verification result:
- Current converter produces valid SOCRATES forcing with non-collapsed `z` (e.g., 42 unique levels for RF01 obs_data).

New tests added:
- `test/solver/socrates_forcing_conversion.jl`
    - unit test for monotonic `pressure_to_height`,
    - regression test that collapsed-`z` files fail validity check,
    - synthetic raw-forcing conversion test asserting output has non-collapsed `z`.
- wired into `test/runtests.jl` under the infrastructure test group.

### 6.8 Migration Debug Timeline And Hardening Changes

This section records the concrete issues encountered during migration and the implemented fixes.
It is intentionally detailed so future work can avoid repeating the same failures.

1. Observation map metadata mismatch (`dims_per_var`)
- Symptom: `KeyError: key "dims_per_var" not found` during `observation_map`.
- Cause: backend path reloaded YAML without runtime-populated `dims_per_var`.
- Fix: added metadata fallback in `resolve_dims_per_var!(cfg)` to read
    `output_dir/obs_metadata.jld2` and infer from `z_model` if needed.

2. Output directory resolution mismatch (`output_active` vs `output_0000`)
- Symptom: observation map looked in stale/empty output paths.
- Cause: hardcoded `output_0000` did not match active-link output behavior.
- Fix: added `resolve_member_simdir(member_path, case_idx)` to prefer
    `output_active` and fall back to `output_0000` for older runs.

3. Worker environment bootstrap failure
- Symptom: distributed workers failed with package load errors (`ClimaCalibrate` not found).
- Cause: workers started without the SOCRATES experiment project environment.
- Fix:
    - local workflow guidance: start workers with `addprocs(...; exeflags=["--project=<SOCRATES>"])`.
    - runtime bootstrap in `run_calibration.jl` via `bootstrap_workers()`.

4. `run_calibration.jl` parse/compile failure
- Symptom: include error with "toplevel expression not at top level".
- Cause: worker bootstrap block used local-scope import/macro patterns that are invalid in that context.
- Fix: switched bootstrap loading to `Distributed.remotecall_eval(Main, wid, quote ... end)`
    so imports/includes execute on workers at top level.

5. Optional distributed execution behavior
- Requirement: serial runs must remain supported.
- Fix: backend selection now uses worker count at runtime:
    - `WorkerBackend()` when `nworkers() > 1`
    - `JuliaBackend()` otherwise.

6. Parallel forcing conversion race condition
- Symptom: intermittent NetCDF permission errors (`error code: 13`) when many workers started together.
- Cause: concurrent workers all attempted `rm(output_file)` + recreate same converted forcing file.
- Fix: conversion now writes to per-process temp files and validates/replaces output defensively.

7. NaN-at-initialization failures
- Symptom: NaN callback failure at integrator initialization (`Found NaN`) across distributed and serial modes.
- Investigation result: non-finite values present in initial state, traced to invalid forcing conversion reuse.
- Fix: added converted-file validity guard (`is_valid_converted_forcing_file`) to reject malformed files
    before reuse and force regeneration.

8. Regression testing added for migration-specific logic
- Added SOCRATES experiment-level tests:
    - forcing conversion validity and non-collapsed `z`
    - forcing reader dimensionality helpers
    - observation-map helper behavior (`resolve_dims_per_var!`, `resolve_member_simdir`)
- Test entrypoint: `calibration/experiments/SOCRATES/test/runtests.jl`.

9. Pressure-level orientation bug in `pressure_to_height`
- Symptom: physically inconsistent altitude mapping (e.g., near-surface pressure levels mapped far above surface).
- Root cause: conversion integrated from top-pressure levels and applied a global offset,
    effectively using a top-referenced height frame instead of a surface-referenced one.
- Legacy context: SSCF/TC build `z` with explicit ground insertion and surface-referenced
    logic (`lev_to_z*` family in SSCF), so the migration converter needed to follow that convention.
- Fix:
    - sort pressure levels by descending pressure for integration,
    - initialize the first (highest-pressure) level relative to surface pressure,
    - integrate upward using hypsometric thickness,
    - map heights back to original pressure-level ordering.
- Regression guard: added test asserting a near-surface pressure level remains near-surface in altitude.

10. Why not call SSCF `process_case` directly for z in this adapter?
- Expected path: TC uses SSCF `process_case` and SSCF's own `lev_to_z*` helpers, so in principle
    this migration should also reuse SSCF's vertical conversion directly.
- Current blocker in this environment: direct `SSCF.process_case(...; return_old_z=true)` fails with
    a Thermodynamics API mismatch (`dry_pottemp_given_pressure` undefined in the loaded Thermodynamics API).
- Practical consequence: the SOCRATES ClimaAtmos adapter cannot currently rely on SSCF's direct
    end-to-end conversion path without additional SSCF compatibility work.
- Interim choice (current code): keep a local, surface-referenced hypsometric conversion in
    `model_interface.jl` with regression tests.
- Follow-up action item: once SSCF compatibility is restored for this environment, replace local
    p→z conversion with SSCF-native z mapping and revalidate against TC outputs.

---

## 6.6 Missing Data Pipeline: Reference Preprocessing

The ClimaAtmos SOCRATES integration requires a reference data preprocessing step to convert raw Atlas LES output into ClimaAtmos-compatible format.

**Raw Atlas LES Format:**
- Variable names: `THETAL`, `QT`, `TABS`, `QCL`, `QCI`, `QR`, `QS`, `QG`, `z`, `time`
- Units: Weight fraction (g/kg), grams, Kelvin, days
- Structure: Single flat NetCDF file per flight+case

**Target ClimaAtmos Diagnostic Format:**
- Variable names: `thetal`, `hus` (specific humidity), `ta` (air temperature), `clw`, `cli`, etc.
- Units: Specific humidity (kg/kg), kilograms, Kelvin, seconds
- Structure: Flat NetCDF with no groups; variable names match ClimaAtmos diagnostic output
- **Key difference:** Uses ClimaAtmos diagnostic variable names directly, NOT TurbulenceConvection names

**Transformation steps:**
1. Map variable names from Atlas → ClimaAtmos diagnostic format (dict-based)
2. Apply unit conversions (e.g., weight fraction→specific humidity, grams→kg, days→seconds)
3. Preserve time dimension (converted to seconds from simulation start)
4. Store in flat NetCDF structure (no TC-style group hierarchy)
5. Create case-specific output files accessible to observation_map
6. Use helper_funcs.jl utilities to read and interpolate fields

**Key architectural choice:** Reference files use ClimaAtmos diagnostic variable names so that `observation_map.jl` can read them directly without TC-specific mappings. The helper_funcs.jl utilities already handle vertical interpolation and time averaging for these files.


---

## 7. Implementation Checklist

- [x] **Architecture Choice:** Keep ClimaAtmos core generic and move SOCRATES logic to experiment layer
- [x] **Core Hook:** Allow `ReanalysisTimeVarying` to accept explicit `external_forcing_file`
- [x] **Experiment Wiring:** Resolve SOCRATES forcing files via SSCF and pass them through config
- [x] **Reference Data Preprocessing:** `preprocess_reference.jl` implemented and tested; writes to `Reference/Atlas_LES/<RF##_<forcing>/stats/`. Tests run in no-overwrite mode.
- [x] **Observation Map:** `observation_map.jl` reads `output_active`, interpolates to `z_model`, concatenates per-case per-variable. All 43 tests pass including offline pipeline end-to-end test.
- [x] **Forcing Conversion Guards:** `is_valid_converted_forcing_file()` prevents reuse of collapsed-z files; `pressure_to_height()` uses surface-referenced hypsometric integration.
- [x] **Serial Calibration Validated:** Full serial calibration confirmed running end-to-end. iteration_000 G_ensemble.jld2 verified: (348,12), all finite (see Section 0.7).
- [x] **Tests:** 43/43 passing (forcing conversion, observation map, offline pipeline dry-run).
- [ ] **Distributed Path Validation:** `WorkerBackend` untested end-to-end. See Section 0.8 for watchpoints.
- [ ] **Surface Humidity Path:** Not wired through interactive surface path. SOCRATES surface humidity (`qg`) not flowing through `ReanalysisTimeVarying` surface path.
- [ ] **Geostrophic Wind Use:** `ug_nudge`/`vg_nudge` from SSCF still not consumed. Currently silently dropped.
- [ ] **Runtime Forward Validation:** No full SOCRATES single-member forward run has been profiled for CFL / NaN behavior at `dt=10s` fixed.
- [ ] **CFL/dt Action Item:** With `dz_bottom=30m` and precipitation fall speeds ~5 m/s, `dt_cfl = 0.5*30/5 = 3s` → current `dt=10s` likely violates CFL for rain/snow. Consider `adaptive_dt: true` in `model_config.yml` or reduce dt.
- [x] **Model Output Variable Expansion:** SOCRATES ClimaAtmos model output now includes additional TC-parity diagnostics in `model_config.yml`:
    - Profiles: `thetaa`, `hus`, `clw`, `ta`, `cli`, `husra`, `hussn`, `arup`
    - Column integrals: `lwp`, `clivi`, `rwp`
    - This required switching `microphysics_model` from `0M` to `1M` so rain/snow prognostic fields (`husra`, `hussn`) are available.
- [ ] **Calibration Vector Expansion (Remaining):** `experiment_config.yml` still uses `y_var_names: [thetaa, hus, clw]`. New outputs are available for postprocessing now, but they are not yet all in the calibrated observation vector. `arup` (TC `updraft_area` analog) currently has no direct Atlas LES reference variable in `preprocess_reference.jl` and therefore should not be added to `y_var_names` until an explicit reference-data definition is added.
- [ ] **Prior Cross-Check:** Prior bounds for `tau_autoconv_ice` and `tau_autoconv_liq` in `prior.toml` need explicit cross-check against `constraints` dicts in `config_calibrate_template_header.jl`.

### Note: Deleted Files

- `~/Research_Schneider/CliMA/CalibrateEDMF.jl/namelist_SOCRATES.in` — **deleted**. Was a stray example file, NOT a canonical source. The real SOCRATES TC grid comes from `common_spaces.jl` reading the Atlas LES z-grid via `SSCF.get_default_new_z()` (see Section 0.3).

### Latest Update: TC `io_dictionary_aux_calibrate()` Parity Pass

Reference function inspected: `TurbulenceConvection.jl/src/diagnostics.jl` `io_dictionary_aux_calibrate()`.

TC calibration-oriented outputs in that function:
- `qt_mean`, `ql_mean`, `thetal_mean`, `qr_mean`, `qi_mean`, `qs_mean`, `temperature_mean`, `updraft_area`
- plus derived/combined fields: `ql_all_mean`, `qi_all_mean`, `qip_mean`

Implemented ClimaAtmos-side parity work:
- `model_config.yml`
    - `microphysics_model: 0M -> 1M`
    - diagnostics expanded to include hydrometeor and temperature profiles (`ta`, `cli`, `husra`, `hussn`) and EDMFX updraft area proxy (`arup`)
    - added column integrals (`lwp`, `clivi`, `rwp`) for LWP/IWP/RWP analysis
- `observation_map.jl`
    - `CLIMADIAGNOSTICS_TO_LES_NAME_MAP` expanded so ClimaAtmos rain/snow variables map to preprocessed Atlas LES keys:
        - `husra -> prw` (Atlas `QR`)
        - `hussn -> snw` (Atlas `QS`)
    - retained existing mappings (`thetaa -> thetal`, `hus -> hus`, `clw -> clw`)

Current parity mapping summary (TC -> ClimaAtmos diagnostics):
- `qt_mean` -> `hus`
- `ql_mean` -> `clw`
- `thetal_mean` -> `thetaa` (closest existing calibration variable)
- `temperature_mean` -> `ta`
- `qr_mean` -> `husra`
- `qi_mean` -> `cli`
- `qs_mean` / `qip_mean` -> `hussn`
- `updraft_area` -> `arup`
- `lwp_mean` (TC postprocessed) -> `lwp`
- `iwp_mean` (TC postprocessed) -> `clivi`

Known caveat:
- `arup` has no corresponding Atlas LES reference variable in the current reference preprocessing map, so it is output by the model but not yet suitable as a calibration target without additional reference-data engineering.

---

## 8. References

**See Section 0.1 and 0.2 for the full source file map with exact paths.**

**TurbulenceConvection SOCRATES:**
- Case type & dispatch: `~/Research_Schneider/CliMA/TurbulenceConvection.jl/driver/Cases.jl` — `struct SOCRATES`, `initialize_profiles`, `initialize_forcing`, `initialize_radiation`, `surface_params`, `update_forcing`, `update_radiation`
- Forcing application: `~/Research_Schneider/CliMA/TurbulenceConvection.jl/driver/dycore.jl` — adaptive dt loop, calls `update_forcing`/`update_radiation` each step
- Diagnostics: `~/Research_Schneider/CliMA/TurbulenceConvection.jl/driver/compute_diagnostics.jl` — TC-named variables: `lwp_mean`, `iwp_mean`, `ql_mean`, `qi_mean`, `qip_mean`, `qr_mean`, `thetal_mean`, `qt_mean`
- Non-equilibrium microphysics: `~/Research_Schneider/CliMA/TurbulenceConvection.jl/src/microphysics_coupling.jl` — tau_autoconv_ice/liq implementation, limiter logic
- TC grid construction: `~/Research_Schneider/CliMA/TurbulenceConvection.jl/driver/common_spaces.jl` — SOCRATES case reads Atlas LES z-grid via `SSCF.get_default_new_z()`, enforces `dz_min`; non-SOCRATES uses `GeneralizedExponentialStretching`

**CalibrateEDMF SOCRATES:**
- Config entry point: `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/config_calibrate_template_body.jl`
- Config header (priors, CFL/dz logic): `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/config_calibrate_template_header.jl`
- Config footer (get_config, reference processing): `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/Calibrate_and_Run_scripts/calibrate/config_calibrate_template_footer.jl`
- Prior seed data: `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/Reference/Output_Inferred_Data/SOCRATES_Atlas_LES_inferred_parameters.nc`
- Best calibration output: `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES_postprocess_runs_storage/subexperiments/SOCRATES_exponential_T_scaling_ice/Calibrate_and_Run/tau_autoconv_noneq/adapt_dt__dt_min_2.0__dt_max_4.0/iwp_mean__lwp_mean__qi_mean__qip_mean__ql_mean__qr_mean/`

**ClimaAtmos Forcing:**
- External forcing framework: `~/Research_Schneider/CliMA/ClimaAtmos.jl/src/prognostic_equations/forcing/external_forcing.jl`
- GCMForcing implementation (reference for SOCRATES nudging): lines ~346–550
- Nudging timescales: `compute_gcm_driven_scalar_inv_τ()`, `compute_gcm_driven_momentum_inv_τ()`

**SOCRATESSingleColumnForcings:**
- Root: `~/Research_Schneider/CliMA/SOCRATESSingleColumnForcings.jl/`
- Core source: `src/` — `process_case(flight_number, forcing_type; initial_condition, surface)`, `open_atlas_les_input`, `open_atlas_les_output`, `Atlas_Scripts.jl`
- Atlas LES input data: `Data/Atlas_LES_Profiles/`
- Observed profiles: `Data/Observed_Profiles/`
- Processed cases: `Data/Processed_Cases/`

**ClimaCalibrate:**
- Source: `~/Research_Schneider/CliMA/ClimaCalibrate.jl/src/`
- Key API: `initialize`, `save_G_ensemble`, `update_ensemble`, `path_to_iteration`, `get_prior`

---

## 9. Legacy References (For Context, Not to Copy)

- **CalibrateEDMF SOCRATES:** `~/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/`
  - Time windows, EKI parameters, observation variables, prior constraints — **reference only**
  - **Do NOT copy code directly** — architecture is TC-specific and incompatible with ClimaAtmos
- **SOCRATES Campaign Data:** `/home/jbenjami/Research_Schneider/Projects/Microphysics/Data/SOCRATES/`
  - Contains pretrained NN checkpoints, summary data
  - Not needed for initial ClimaAtmos setup
- **Python postprocessing scripts:** `/home/jbenjami/Research_Schneider/Projects/Microphysics/Python/publication_figures_helper_scripts/` and `/net/sampo/data1/jbenjami/Research_Schneider/Projects/Microphysics/Python/publication_figures_helper_scripts/`
  - Used for comparing calibration results and generating publication figures
  - Not part of the ClimaAtmos calibration pipeline
