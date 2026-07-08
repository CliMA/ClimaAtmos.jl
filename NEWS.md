ClimaAtmos.jl Release Notes
============================

main
----

- [#4653](https://github.com/CliMA/ClimaAtmos.jl/pull/4653) ![][badge-🔥behavioralΔ] Improve the manual implicit Jacobian (the tendencies are unchanged; behavior changes only through the single-Newton-iteration implicit solve).
  - Document that the existing height-form pressure-gradient-force Jacobian blocks are the exact linearization of the Exner-form tendency `cp_d θ_v ∇Π` (stencils unchanged): the thermal-buoyancy and pressure-buoyancy terms cancel via the equation-of-state identity `δρ/ρ = (1-κ_d) δp/p - δθ_v/θ_v`, leaving the acoustic operator `∇δp/ρ` in every column perturbed at fixed density plus a single buoyancy term in the `ρ` column, so sound and gravity waves are both treated fully implicitly.
  - Add the neglected environment feedback `∂χ⁰/∂χʲ = -w·ρaʲ/ρa⁰` to the entrainment-relaxation diagonal, scaling it by `(1 + w·ρaʲ/ρa⁰)` (the exact derivative of the regularized `specific` that diagnoses the environment relaxation target, with `w = sgs_weight_function(ρa⁰/ρ, a_half)`).
  - Include the sedimentation velocity in the continuity-equation (`ρ`-row) Jacobian, matching the SGS-water contribution to the moist-air-mass tendency; this required reordering the scalar solve (condensate masses precede `ρ`) and dropping the always-zero `(tracer, ρ)` diffusion blocks.
  - Enumerate the EDMFX grid-mean tracer diffusion from the grid-mean state instead of the updraft state, so grid-mean microphysics tracers are also diffused under `EDOnlyEDMFX` (which carries no SGS tracers); diffuse passive (non-water) tracers — grid-mean and SGS — with the unscaled `K_h`, matching the boundary-layer diffusion path, and give them the matching implicit-Jacobian diagonals, which were previously missing.
  - Gate the updraft-internal diffusion Jacobian on `edmfx_vertical_diffusion`, mirroring the tendency it linearizes (previously the updraft scalar diagonals could carry diffusion terms with no tendency counterpart when implicit diffusion was enabled without EDMFX vertical diffusion).
  - Document the remaining frozen-coefficient approximations in the Jacobian ledger: the diffusive fluxes' `ρ`-dependence, the `∂l_mix/∂tke` chain term in the TKE dissipation derivative, and the explicitly-treated EDMFX subdomain sedimentation-energy corrections.
- [#4644](https://github.com/CliMA/ClimaAtmos.jl/pull/4644) ![][badge-🔥behavioralΔ]
  - Remove the coherent (mass-flux) buoyancy production from the EDMFX TKE budget; the coherent updraft/environment buoyancy flux belongs to the resolved SGS circulation, not the isotropic turbulence that TKE represents.
  - Decompose the diffusive enthalpy flux into dry static energy + water enthalpy, `F_h = -K_h ∇s_d + Σ_μ h_tot,μ (-K_h ∇q_μ)` with unit turbulent Lewis number, at both diffusion sites; diffusing `h_tot` directly implied a spurious enthalpy flux carried by dry-air diffusion that systematically warmed entrained air across capping inversions. The implicit-Jacobian diffusion factors are updated to match.
  - Damp the relative part of the cloud-fraction floor as the subdomain mean saturates (`D = x/√(1+x²)`, `x = max(-μ_S, 0)/(ε_rel·q_sat)`), so an equilibrated overcast deck is no longer capped below full cloud cover by the patchiness floor. No new parameters.
- [#4638](https://github.com/CliMA/ClimaAtmos.jl/pull/4638) ![][badge-🔥behavioralΔ] Fix the subsidence top boundary condition: the advective-form subsidence operator now uses the zero-boundary-flux divergence (`ᶜadvdivᵥ`), which is equivalent to a zero-gradient inflow condition `χ = χ_top` above the lid, instead of `Extrapolate` (which copied the cell below into the top cell). Remove the two `external_forcing.jl` blocks that hard-zeroed the accumulated top-cell `ρe_tot`/`ρq_tot` tendencies to mask that defect; GCM/ERA5-driven top-cell tendencies (radiation, nudging, subsidence) are no longer discarded.
- [#4637](https://github.com/CliMA/ClimaAtmos.jl/pull/4637) ![][badge-🔥behavioralΔ]
  - Fix the spurious mass flux through the model top over sloped terrain-following coordinates: `set_velocity_at_top!` now cancels the contravariant projection of the horizontal wind (`u₃ = -uₕ³/g³³`, mirroring the surface treatment), and the continuity equation uses the same zero-boundary-flux divergence (`ᶜadvdivᵥ`) as the tracers, which also makes the `ρ`-row Jacobian exact.
  - Transport auto-discovered EDMFX SGS tracers (`q_lcl`, `q_icl`, `q_rai`, `q_sno`, `n_*`) like `mse` and `q_tot`: grid-mean vertical advection plus difference-form SGS corrections `ρᵏaᵏ(u³ᵏ-u³)(χᵏ-χ)`, which vanish for uniform fields, instead of absolute subdomain fluxes. These corrections now use the same `edmfx_sgsflux_upwinding` reconstruction as the `mse`/`q_tot` fluxes (central by default, previously first-order upwind for the tracer fluxes), and their implicit Jacobian uses the matching central, updraft-only linearization (environment contributions, which are `O(aʲ²)`, are dropped).
  - Add the moist-air-mass counterpart of the EDMFX SGS mass flux of `q_tot` to the continuity equation, mirroring the diffusive-flux treatment. A new tracer-mass consistency test asserts the `χ ≡ 1` identity for these transport pathways.

0.41.1
-------

- [#4622](https://github.com/CliMA/ClimaAtmos.jl/pull/4622) ![][badge-🔥behavioralΔ] Move the `ρe_tot`/`ρq_tot` upwind correction into the implicit advection tendency (the Wfact Jacobian is unchanged).
- [#4621](https://github.com/CliMA/ClimaAtmos.jl/pull/4621) ![][badge-🔥behavioralΔ] Clamp the implicit `ρa` stage value to `[0, ρ·a_max]` per-cell in the column sweep for stability, so updraft area cannot exceed `a_max` when the explicit area limiters fail to prevent overshoot.
- [#4623](https://github.com/CliMA/ClimaAtmos.jl/pull/4623) Hotfix: replace a rational exponent (`^(1 // 4)`) with `sqrt(sqrt(...))` in the Smagorinsky–Lilly stratification correction.
- [#4618](https://github.com/CliMA/ClimaAtmos.jl/pull/4618) Add a verbose logging option for `AtmosSimulation`.
- [#4606](https://github.com/CliMA/ClimaAtmos.jl/pull/4606) ![][badge-✨feature/enhancement] Add ARM VARANAL SCM forcing support.
- [#4615](https://github.com/CliMA/ClimaAtmos.jl/pull/4615) Add microphysics 1M tendency plots for the Larcform1 case.
- [#4610](https://github.com/CliMA/ClimaAtmos.jl/pull/4610) Add LES comparison plots for EDMF post-processing.

0.41.0
-------

- [#4607](https://github.com/CliMA/ClimaAtmos.jl/pull/4607) ![][badge-💥breaking] Promote hardcoded EDMF, cloud-fraction, and sponge parameters to calibratable parameters, and update to ClimaParams v1.1. Renames break existing TOML overrides and code: the ClimaParams keys `EDMF_surface_area` → `EDMF_max_surface_area` and `alpha_rayleigh_sgs_tracer` → `alpha_rayleigh_tracer`, and the `TurbulenceConvectionParameters` field `surface_area` → `max_surface_area`.
- [#4608](https://github.com/CliMA/ClimaAtmos.jl/pull/4608) ![][badge-🔥behavioralΔ] SGS saturation moments: use a linearized analytic `μ_S` and accumulate `σ_S²` as `E[(S − μ_S)²]` in a single quadrature pass, avoiding Float32 catastrophic cancellation when `Var[S] ≪ (E[S])²`.
- [#4600](https://github.com/CliMA/ClimaAtmos.jl/pull/4600) ![][badge-🔥behavioralΔ] Add new prognostic-EDMF column diagnostics and change the default `detr_massflux_vertdiv_coeff`.
- [#4596](https://github.com/CliMA/ClimaAtmos.jl/pull/4596) ![][badge-🔥behavioralΔ] PrognosticEDMFX: replace the prescribed surface-area boundary condition with a Deardorff surface mass-flux BC (`a_s = a_s_max·w*³/(w*³ + c_u·u*³)`, `F_surf = a_s·ρ·w*`) that drives `ρa`, `mse`, and `q_tot` at the first cell.
- [#4593](https://github.com/CliMA/ClimaAtmos.jl/pull/4593) ![][badge-✨feature/enhancement] Add an "explicit" Bomex prognostic-EDMF case with a passive tracer A.
- [#4567](https://github.com/CliMA/ClimaAtmos.jl/pull/4567/) ![][badge-✨feature/enhancement] Create the Larcform1 case (WIP).
- [#4516](https://github.com/CliMA/ClimaAtmos.jl/pull/4516) ![][badge-✨feature/enhancement] Add a perfect-model single-column (SCM) calibration experiment.

0.40.0
-------

- [#4594](https://github.com/CliMA/ClimaAtmos.jl/pull/4594) Remove the `diagnostic_edmfx` model configuration.
- [#4590](https://github.com/CliMA/ClimaAtmos.jl/pull/4590) ![][badge-🔥behavioralΔ] Non-dimensionalize the variance floor parameter in the truncated-Gaussian cloud-fraction closure. The hardcoded `σ_S_floor = 1e-6` is replaced with a scale-aware floor `σ_S_floor = sqrt((ε_rel · q_sat)² + σ_abs²)` that tracks local saturation humidity.

v0.39.5
-------

- [#4548](https://github.com/CliMA/ClimaAtmos.jl/pull/4548) ![][badge-✨feature/enhancement] Auto-discover SGS updraft tracers in prognostic EDMF tendency processes via `sgs_tracer_names(Y)`, so adding a passive SGS tracer no longer requires editing each tendency. New documentation in `docs/src/passive_tracers.md`.
- [#4578](https://github.com/CliMA/ClimaAtmos.jl/pull/4578) ![][badge-🔥behavioralΔ] Add the Rayleigh sponge tendency for microphysics tracers.

v0.39.4
-------

- [#4568](https://github.com/CliMA/ClimaAtmos.jl/pull/4568) ![][badge-🔥behavioralΔ] `config: column` simulations now use an actual single-column (FiniteDifference) geometry instead of a minimal 2×2 box.
- [#4558](https://github.com/CliMA/ClimaAtmos.jl/pull/4558) ![][badge-🔥behavioralΔ] Use an analytic area fraction in the implicit stage solve, and refactor entrainment/detrainment.
- [#4569](https://github.com/CliMA/ClimaAtmos.jl/pull/4569) ![][badge-🔥behavioralΔ] Use the same diffusion and hyperdiffusion scaling for condensate and precipitation.
- [#4570](https://github.com/CliMA/ClimaAtmos.jl/pull/4570) ![][badge-🔥behavioralΔ] Change the default microphysics process options.
- [#4556](https://github.com/CliMA/ClimaAtmos.jl/pull/4556) Move EDMF column/box diagnostics into shared common configs (`config/common_configs/diagnostics_column_*.yml`) and update the EDMF post-processing plots.

v0.39.3
-------

- [#4551](https://github.com/CliMA/ClimaAtmos.jl/pull/4551) Remove q_tot diffusion contribution to updraft area tendency and its Jacobian entries
- [#4546](https://github.com/CliMA/ClimaAtmos.jl/pull/4546) Add consistent non-eq microphysics and cloud fraction based on trancated gaussian distribution of condensates
- [#4455](https://github.com/CliMA/ClimaAtmos.jl/pull/4455) Refactor surface conditions
- [#4545](https://github.com/CliMA/ClimaAtmos.jl/pull/4545)  Add microphysics process tendency diagnostics

v0.39.2
-------

- [#4521](https://github.com/CliMA/ClimaAtmos.jl/pull/4521) Make entrainment and detrainment use updated (implicit) velocity
- [#4514](https://github.com/CliMA/ClimaAtmos.jl/pull/4514) Replace deprecated ClimaTimeSteppers backward-compatibility shims with canonical names (`DistributedODEIntegrator` → `TimeStepperIntegrator`, `CallbackSet(continuous_callbacks, discrete_callbacks)` → `CallbackSet(discrete_callbacks...)`).
- [#4512](https://github.com/CliMA/ClimaAtmos.jl/pull/4512) ![][badge-🔥behavioralΔ] Remove SGS explicit/implicit timestepping options; all SGS tendencies (advection, entrainment/detrainment, NH pressure drag, vertical diffusion, mass flux) are now always implicit.

v0.39.1
-------

- ![][badge-🔥behavioralΔ] Unify cloud fraction and microphysics SGS quadrature via a shared `sgs_moments` pre-pass. Replace the Sommeria–Deardorff cloud fraction with a hybrid CDF formula, with variances from quadratures, and introduce a mass-conserving shape-function partition in the `Microphysics1MEvaluator`.

v0.39.0
-------

- [#4486](https://github.com/CliMA/ClimaAtmos.jl/pull/4486) [badge-💥breaking]
  Updated YAML config schema:
  - `vert_diff: true` → `"VerticalDiffusion"`; `false` → `~`.
  - `hyperdiff: false` → `~`.
  - `prognostic_surface: true` → `"SlabOceanSST"`; `false` → `"PrescribedSST"`.
  - Removed keys: `subsidence`, `ls_adv`, `scm_coriolis`.
  - `rad`: dropped `"nothing"` string alias; use `~`.

v0.38.4
-------

v0.38.3
-------

- Added a `Presets` module with convenience constructors for common module/simulation configurations.
- Add a keyword-based constructor `AtmosSimulation(; kwargs)` that always runs with Float32.
- Set the existing keyword-based constructor `AtmosSimulation{FT}(;kwargs)` to always use ITime. Removed the `itime` kwarg.

v0.38.2
-------

v0.38.1
-------

v0.38.0
----

- Refactor names in 1M microphysics q_liq -> q_lcl, q_ice -> q_icl
- Refactor names in 0M microphysics q_liq_rai -> q_liq, q_ice_sno -> q_ice
- Refactor names in 0M microphysics q_tot_safe -> q_tot_nonneg

v0.37
----

- Replace `InitialConditions` module with `Setups` module. Remove `surface_temperature` config key.

- [#4361](https://github.com/CliMA/ClimaAtmos.jl/pull/4361) Refactor reproducibility infrastructure to use rms deviations in prognostic variables and only fail when rms deviation exceeds tolerance.

v0.36.1
-------

v0.36.0
-------

-[#4285](https://github.com/CliMA/ClimaAtmos.jl/pull/4320) Delete moisture_model.
Rename microphysics_models.

v0.35.4
-------

- [#4252](https://github.com/CliMA/ClimaAtmos.jl/pull/4252) Add option to write
  diagnostics in pressure coordinates

- [#4260](https://github.com/CliMA/ClimaAtmos.jl/pull/4260) Add option to use quadrature for
microphysics tendencies. Updated to CloudMicrophysics v0.31, and use the BulkMicrophysicsTendencies
API.

v0.35.2
-------

v0.35.1
-------

- [4276](https://github.com/CliMA/ClimaAtmos.jl/pull/4276) Create IC type to allow initializing AMIP with ERA5 on model levels. Add artifact which corresponds to default AMIP start date (Jan 1, 2010). Also removes unused ᶜspecific in the precomputed cache.

v0.35.0
-------

- [#4225](https://github.com/CliMA/ClimaAtmos.jl/pull/4225) Add an option for vertical water borrowing limiter. Configure via `tracer_nonnegativity_method: "vertical_water_borrowing"`
  and optionally `vertical_water_borrowing_species` (defaults to all tracers if not specified).
  The limiter enforces non-negativity using a single threshold (0.0) that applies uniformly to all selected tracers.

  Rename `limiter` to `sem_quasimonotone_limiter` in cache numerics for clarity.
  The cache now stores the quasi-monotone limiter instance as `sem_quasimonotone_limiter`
  to better distinguish it from other limiter types (e.g., `vertical_water_borrowing_limiter`).

- [#4231](https://github.com/CliMA/ClimaAtmos.jl/pull/4231) [badge-💥breaking] removes grid-scale
thermo state, including ᶜts in p.precomputed.sfc_conditions.

- [#4211](https://github.com/CliMA/ClimaAtmos.jl/pull/4211)
  add experimental methods to remove negative microphysical condensate values

- [#4245](https://github.com/CliMA/ClimaAtmos.jl/pull/4245) Refactor hyperdiffusion to use Prandtl number parameterization. The scalar hyperdiffusion
  coefficient is now computed as `ν₄_scalar = ν₄_vorticity / prandtl_number`, replacing the
  previous direct `ν₄_scalar_coeff` parameter. Configure via `hyperdiffusion_prandtl_number`
  (default: 1.0). The `CAM_SE` hyperdiffusion configuration maintains a 5x ratio between
  vorticity and scalar coefficients by using `prandtl_number = 0.2`.

v0.34.0
-------

- [#4198](https://github.com/CliMA/ClimaAtmos.jl/pull/4198) [badge-💥breaking] modifies surface conditions
to use SurfaceFluxes v0.15.

- [#4220](https://github.com/CliMA/ClimaAtmos.jl/pull/4220) modifies `SphereGrid` to use spacefillingcurve.

v0.33.2
-------

v0.33.1
-------

- PR [#4185](https://github.com/CliMA/ClimaAtmos.jl/pull/4185) adds a new cloud fraction parameterization `MLCloud`. It can be set in the toml under `cloud_model`.

- PR [#4191](https://github.com/CliMA/ClimaAtmos.jl/pull/4191) renames ρatke to ρtke and move it out of sgs⁰.

- PR [#4175](https://github.com/CliMA/ClimaAtmos.jl/pull/4175) adds support for lazy diagnostics, as introduced in [`ClimaDiagnostics.jl`](https://github.com/CliMA/ClimaDiagnostics.jl) v0.2.13.

v0.33.0
-------

- ![#4169](https://github.com/CliMA/ClimaAtmos.jl/pull/4169) [badge-💥breaking]
Remove the options `co2_model` and `prescribe_ozone`. These are both fixed by
default now, and can be set to time-varying by including one or both in the
option `time_varying_trace_gases: ["CO2", "O3]`.

v0.32.0
-------

- PR [#4162](https://github.com/CliMA/ClimaAtmos.jl/pull/4162) adds an option for
reproducible restart. It is set to false by default. This shouldn't affect restart
in the coupler as the coupler save the cache for restarting.

- PR [#4021](https://github.com/CliMA/ClimaAtmos.jl/pull/4021) uses ClimaCore
convenience constructors to create spaces without an AtmosConfig.

v0.31.9
-------

v0.31.8
-------

v0.31.7
-------

v0.31.6
-------

v0.31.5
-------

PR [#3975](https://github.com/CliMA/ClimaAtmos.jl/pull/3975) updates the pressure gradient formulation to subtract a reference state and use the Exner pressure.

v0.31.4
-------

v0.31.3
-------

v0.31.2
-------

### Add RWP diagnostic

PR [#3946](https://github.com/CliMA/ClimaAtmos.jl/pull/3946) adds rainwater path diagnostic variable.

v0.31.1
-------

PR [#3917](https://github.com/CliMA/ClimaAtmos.jl/pull/3917) adds common numerics configs for different resolutions.
These configurations are intended to serve as the default settings. Please modify them only if you are certain of the implications.

v0.31.0
-------

PR [#3886](https://github.com/CliMA/ClimaAtmos.jl/pull/3886) renames `PrognosticSurfaceTemperature` -> `SlabOceanSST`, `PrescribedSurfaceTemperature` -> `PrescribedSST` and deprecates the `prognostic_surface = "Prognostic/PrescribedSurfaceTemperature"` config argument

PR [#3883](https://github.com/CliMA/ClimaAtmos.jl/pull/3883) adds atmosphere initial condition for WeatherQuest from ERA5

PR [#3870](https://github.com/CliMA/ClimaAtmos.jl/pull/3870) adds externally driven SCM models forced by monthly-averaged ERA5
reanalysis which are less data intensive and more stable. The simulation also employs the new implicit solver yielding a several
factor speedup. The associated documentation section, "Single Column Model", was also improved.

v0.30.4
-------

PR [#3856](https://github.com/CliMA/ClimaAtmos.jl/pull/3856) adds number adjustment tendencies to the two-moment microphysics scheme.

v0.30.3
-------

### Add diagnostic for CAPE

PR [#3820](https://github.com/CliMA/ClimaAtmos.jl/pull/3820) adds support for computing convective available potential energy (CAPE), or the vertical integral of the buoyancy differential between a parcel lifted from the surface and the environment. Exemplified in the TRMM deep convection case.

v0.30.2
-------

### Add limiting by max Prandtl number

This is part of a larger refactoring of the Eddy Diffusivity code for EDMF

v0.30.1
-------

v0.30.0
-------

### Add support for reanalysis-driven single column model with time-varying forcing

PR [#3758](https://github.com/CliMA/ClimaAtmos.jl/pull/3758) adds support for driving single-column model (SCM) simulations with time-varying ERA5 reanalysis data. This extends the existing GCM-driven SCM interface to allow site-specific simulations that resolve the diurnal cycle and are suited for calibration against observations. Users can now run reanalysis-driven cases globally using only a date and lat/lon, thanks to integrated data handling via ClimaArtifacts.jl. See the updated “Single Column Model” docs page for details on setup, variable requirements, and how to prepare ERA5 input files.

### Non-orographic gravity wave tendency as a callback

PR[#3761](https://github.com/CliMA/ClimaAtmos.jl/pull/3761) introduces support for intermittent calls to update the computation of non-orographic gravity wave tendencies. This PR closes issue[#3434](https://github.com/CliMA/ClimaAtmos.jl/issues/3434).

### Remove `dt_save_to_sol`

The option to save the solution to the integrator object (`dt_save_to_sol`) was
removed from the configurable options.

v0.29.1
-------

### Remove contribution from condensate, precip diffusion in mass tendency

PR[#3721](https://github.com/CliMA/ClimaAtmos.jl/pull/3721)
Diffusion of condensate (liq, ice) and precip (rai, sno) vars no longer
contributes to the mass tendency terms (updates in vert diffusion boundary layer,
smag-lilly, implicit solver terms)

### Add support for non-zero `t_start`

Passing a non zero `t_start` is useful in conditions where one wants to have a
specific `start_date`, but start the simulation from a different point. This is
used by `ClimaCoupler` to restart simulations.

v0.29.0
-------

### Remove precipitation from cache

And move all the fields into precomputed

v0.28.6
-------

### Features

### Add a flag for disabling surface flux tendency

Surface flux tendency is not controlled by `vert_diff` or `edmfx_sgs_diffusive_flux` anymore.
Instead, it is controlled by the new flag `disable_surface_flux_tendency`. When it is set to
true, no surface flux tendency is applied, no matter what `surface_setup` is.
This flag is set to false by default. PR [3670](https://github.com/CliMA/ClimaAtmos.jl/pull/3670).

### Automatically determine diagnostic resolution based on model resolution

If `netcdf_interpolation_num_points` is not provided, `ClimaAtmos` will
determine it automatically by matching approximately the same number of points
as the model grid.

### Change reconstruction of density on cell faces for stretched grids

PR [3584](https://github.com/CliMA/ClimaAtmos.jl/pull/3584) changes the weighted
interpolation of density from centers to faces so that it uses `ᶜJ` and `ᶠJ`,
rather than `ᶜJ` and `ᶠint(ᶜJ)`. As of ClimaCore v0.14.25, `ᶠJ` is no longer
equivalent to `ᶠint(ᶜJ)` for stretched grids.

v0.28.5
-------

### Features

### Add EDOnlyEDMFX

PR [3622](https://github.com/CliMA/ClimaAtmos.jl/pull/3622) adds a new
simplified EDMF model that only implements the Eddy-Diffusivity part of the
scheme (not the Mass-Flux).

### Update default configuration to use deep-atmosphere eqns, fix diagnostic bug

PR [3422](https://github.com/CliMA/ClimaAtmos.jl/pull/3422)
Updates the `default_config` to set `deep_atmosphere=true`, and updates the
`rv` relative vorticity diagnostic to store the curl of horizontal velocity.

### Allow different sizes of dust and sea salt for radiation

Added functionality to allow five different size bins of dust and sea salt aerosols
for radiation calculation. This feature requires RRTMGP version v0.20.0 or later.
PR [3555](https://github.com/CliMA/ClimaAtmos.jl/pull/3555)

### Maintenance

### Rmove FriersonDiffusion option

The option `FriersonDiffusion` is removed from `vert_diff` config. Use `DecayWithHeightDiffusion` instead.
PR [3592](https://github.com/CliMA/ClimaAtmos.jl/pull/3592)

v0.28.4
-------

### Development

The `.dev` was deprecated. The two utilities in this folder can be replaced with
more established and better developed tools:

- instead of `clima_format`, use `JuliaFormatter`,
- instead of `up_deps`, use `PkgDevTools`.
See the [documentation](https://clima.github.io/ClimaAtmos.jl/dev/contributor_guide/#Formatting) for more information.

`ClimaAtmos` now only support equilibrium moisture + 0-moment microphysics and
nonequilibrium + 1-moment microphysics (No precipitation is still supported too).
PR [3557](https://github.com/CliMA/ClimaAtmos.jl/pull/3557)

### File Logging

`ClimaAtmos` now supports logging to stdout and file simultaneously using
`ClimaComms.FileLogger`. To enable, set the configuration with `log_to_file = false`.
See [ClimaComms documentation](https://clima.github.io/ClimaComms.jl/dev/logging/)
 for more background on logging.

v0.28.3
-------

### Read CO2 from file

`ClimaAtmos` now support using data from the Mauna Loa CO2 measurements to set
CO2 concentration. This is currently only relevant for radiation transfer with
RRTGMP.

### Maintenance

### Remove override_precip_timescale config

![][badge-🔥behavioralΔ] The override_precip_timescale config has been removed.
To recover the previous behavior, set `precipitation_timescale` to `dt` in the
toml. PR [3534](https://github.com/CliMA/ClimaAtmos.jl/pull/3534)

v0.28.2
-------

### Features

### Add van Leer class operator

Added a new vertical transport option `vanleer_limiter` (for tracer and energy
variables) which uses methods described in Lin et al. (1994) to apply
slope-limited upwinding. Adds operator

### Read initial conditions from NetCDF files

Added functionality to allow initial conditions to be overwritten by
interpolated NetCDF datasets.

To use this feature from the YAML interface, just pass the path of the file.
We expect the file to contain the following variables:

- `p`, for pressure,
- `t`, for temperature,
- `q`, for humidity,
- `u, v, w`, for velocity,
- `cswc, crwc` for snow and rain water content (for 1 moment microphysics).

For example, to use the DYAMONDSummer initial condition, set

```
initial_condition: "artifact\"DYAMONDSummer\"/DYAMOND_SUMMER_ICS_p98deg.nc"
```

in your configuration file.

### Write diagnostics to text files

Added functionality to write diagnostics in DictWriter to text files.
This is useful for outputting scalar diagnostics, such as total mass of
the atmosphere. PR [3476](https://github.com/CliMA/ClimaAtmos.jl/pull/3476)

v0.28.0
-------

v0.27.9
-------

### Features

### New option for vertical diffusion

When `vert_diff` is set to `DecayWithHeightDiffusion`, diffusion decays
exponentially with height.
PR [3475](https://github.com/CliMA/ClimaAtmos.jl/pull/3475)

v0.27.8
-------

### Features

### New option for prescribing clouds in radiation

When `prescribe_clouds_in_radiation` is set to true, clouds in radiation
is prescribed from a file (monthly cloud properties in 2010 from ERA5).
PR [3405](https://github.com/CliMA/ClimaAtmos.jl/pull/3405)

### ETOPO2022 60arc-second topography dataset

- Update artifacts to use 60arc-second ETOPO2022 ice-surface topography
  dataset. Update surface smoothing functions to rely only on spectral
  Laplacian operations. Update raw-topo gravity wave parameterization
  dataset. Update interfaces in `make_hybrid_spaces` to support new
  inputs using `SpaceVaryingInput` utility. Include a simple example
  to generate spectra from scalar variables.
  PR [3378](https://github.com/CliMA/ClimaAtmos.jl/pull/3378)

v0.27.7
-------

### Features

### Reproducible restarts for simulations with clouds with RRTMGP

- Reset the RNG seed before calling RRTGMP to a known value (the iteration number).
  When modeling cloud optics, RRTGMP uses a random number generator. Resetting
  the seed every time RRTGMP is called to a deterministic value ensures that the
  simulation is fully reproducible and can be restarted in a reproducible way.
  Disable this option when running production runs.

  Note: Setting this option to `true` is behavior-changing.
  PR [3382](https://github.com/CliMA/ClimaAtmos.jl/pull/3382)

### ![][badge-🐛bugfix] Bug fixes

- Update RRTMGP to v0.19.1, which fixes the sea salt aerosol lookup table.
  Sea salt aerosol is added to the target amip config.
  PR [3374](https://github.com/CliMA/ClimaAtmos.jl/pull/3374)

- Fixed radiation diagnostics conflicting with each other. Prior to this change,
  adding multiple diagnostics associated to the same variable would lead to
  incorrect results when the more diagnostics were output at the same time. PR
  [3365](https://github.com/CliMA/ClimaAtmos.jl/pull/3365)

- ClimaAtmos no longer fails when reading restart files generated with versions
  of ClimaAtmos prior to `0.27.6`. PR
  [3388](https://github.com/CliMA/ClimaAtmos.jl/pull/3388)

v0.27.6
-------

### Features

### Ozone model is now a dispatchable type

The `prescribe_ozone` flag was turned into a type, allowing for prescribing
arbitrary ozone concentrations. The two types that are currently implemented are
`IdealizedOzone` (implementing a static profile from Wing 2018), and
`PrescribedOzone` (reading from CMIP6 forcing files).

### Aerosol and ozone data can now be automatically downloaded

Prescribed aerosol and ozone concentrations require external files. Now, a
low-resolution version of such files is automatically downloaded when a
higher-resolution version is not available. Please, refer to ClimaArtifacts for
more information.

### ![][badge-🐛bugfix] Bug fixes

- Fixed incorrect time/date conversion in diagnostics when restarting a
  simulation. PR [3287](https://github.com/CliMA/ClimaAtmos.jl/pull/3287)

- ![][badge-🔥behavioralΔ] Switch to hyperbolic tangent grid stretching,
  which only requires z_elem and dz_bottom.
  PR [3260](https://github.com/CliMA/ClimaAtmos.jl/pull/3260)

- Fixed restarts with radiation and idealized ozone.

v0.27.5
-------

- Update RRTMGP and allow multiple aerosols for radiation.
  Note: Don't use sea salt as there is an issue with the lookup
  table. PR [#3264](https://github.com/CliMA/ClimaAtmos.jl/pull/3264)

v0.27.4
-------

- Add artifact decoding from YAML
  PR [#3256](https://github.com/CliMA/ClimaAtmos.jl/pull/3256)

v0.27.3
-------

- Add support for monthly calendar diagnostics
  PR [#3235](https://github.com/CliMA/ClimaAtmos.jl/pull/3241)
- Use period filling interpolation for aerosol time series
  PR [#3246] (<https://github.com/CliMA/ClimaAtmos.jl/pull/3246>)
- Add prescribe time and spatially varying ozone
  PR [#3241](https://github.com/CliMA/ClimaAtmos.jl/pull/3241)

v0.27.2
-------

- Use new aerosol artifact and change start date
  PR [#3216](https://github.com/CliMA/ClimaAtmos.jl/pull/3216)
- Add a gpu scaling job with diagnostics
  PR [#2852](https://github.com/CliMA/ClimaAtmos.jl/pull/2852)

v0.27.1
-------

- Allow different aerosol types for radiation.
  PR [#3180](https://github.com/CliMA/ClimaAtmos.jl/pull/3180)
- ![][badge-🔥behavioralΔ] Switch from `Dierckz` to `Interpolations`. `Interpolations`
  is type-stable and GPU compatible. The order of interpolation has decreased to first.
  PR [#3169](https://github.com/CliMA/ClimaAtmos.jl/pull/3169)

v0.27.0
-------

- ![][badge-💥breaking] Change `radiation_model` in the radiation cache to `rrtmgp_model`.
  PR [#3167](https://github.com/CliMA/ClimaAtmos.jl/pull/3167)
- ![][badge-💥breaking] Change the `idealized_insolation` argument to `insolation`,
  and add RCEMIP insolation. PR [#3150](https://github.com/CliMA/ClimaAtmos.jl/pull/3150)
- Add lookup table for aerosols
  PR [#3156](https://github.com/CliMA/ClimaAtmos.jl/pull/3156)

v0.26.3
-------

- Add ClimaCoupler downstream test.
  PR [#3152](https://github.com/CliMA/ClimaAtmos.jl/pull/3152)
- Add an option to use aerosol radiation. This is not fully working yet.
  PR [#3147](https://github.com/CliMA/ClimaAtmos.jl/pull/3147)
- Update to RRTMGP v0.17.0.
  PR [#3131](https://github.com/CliMA/ClimaAtmos.jl/pull/3131)
- Add diagnostic edmf cloud scheme.
  PR [#3126](https://github.com/CliMA/ClimaAtmos.jl/pull/3126)

v0.26.2
-------

- Limit temperature input to RRTMGP within the lookup table range.
  PR [#3124](https://github.com/CliMA/ClimaAtmos.jl/pull/3124)

v0.26.1
-------

- Updated RRTMGP compat from 0.15 to 0.16
  PR [#3114](https://github.com/CliMA/ClimaAtmos.jl/pull/3114)
- ![][badge-🔥behavioralΔ] Removed the filter for shortwave radiative fluxes.
  PR [#3099](https://github.com/CliMA/ClimaAtmos.jl/pull/3099).

v0.26.0
-------

- ![][badge-💥breaking] Add precipitation fluxes to 1M microphysics output.
  Rename col_integrated_rain (and snow) to surface_rain_flux (and snow)
  PR [#3084](https://github.com/CliMA/ClimaAtmos.jl/pull/3084).

v0.25.0
-------

- ![][badge-💥breaking] Remove reference state from the dycore and the
  relevant config. PR [#3074](https://github.com/CliMA/ClimaAtmos.jl/pull/3074).
- Make prognostic and diagnostic EDMF work with 1-moment microphysics on GPU
  PR [#3070](https://github.com/CliMA/ClimaAtmos.jl/pull/3070)
- Add precipitation heating terms for 1-moment microphysics
  PR [#3050](https://github.com/CliMA/ClimaAtmos.jl/pull/3050)

v0.24.2
-------

- ![][badge-🔥behavioralΔ] Fixed incorrect surface fluxes for uh. PR [#3064]
  (<https://github.com/CliMA/ClimaAtmos.jl/pull/3064>).

v0.24.1
-------

v0.24.0
-------

- ![][badge-💥breaking]. CPU/GPU runs can now share the same yaml files. The driver now calls `AtmosConfig` via `(; config_file, job_id) = ClimaAtmos.commandline_kwargs(); config = ClimaAtmos.AtmosConfig(config_file; job_id)`, which recovers the original behavior. PR [#2994](https://github.com/CliMA/ClimaAtmos.jl/pull/2994), issue [#2651](https://github.com/CliMA/ClimaAtmos.jl/issues/2651).
- Move config files for gpu jobs on ci to config/model_configs/.
  PR [#2948](https://github.com/CliMA/ClimaAtmos.jl/pull/2948).

v0.23.0
-------

- ![][badge-✨feature/enhancement]![][badge-💥breaking]. Use
  [ClimaUtilities](https://github.com/CliMA/ClimaUtilities.jl) for
  `TimeVaryingInputs` to read in prescribed aerosol mass concentrations. This PR
  is considered breaking because it changes `AtmosCache` adding a new field,
  `tracers`. PR [#2815](https://github.com/CliMA/ClimaAtmos.jl/pull/2815).

- ![][badge-✨feature/enhancement]![][badge-💥breaking]. Use
    [ClimaUtilities](https://github.com/CliMA/ClimaUtilities.jl) for
    `OutputPathGenerator` to handle where the output of a simulation should be
    saved. Previously, the output was saved to a folder named `$job_id`. Now, it
    is saved to `$job_id/output-active`, where `output-active` is a link that
    points to `$job_id/output-XXXX`, with `XXXX` a counter that increases ever
    time a simulation is run with this output directory. PR
    [#2606](https://github.com/CliMA/ClimaAtmos.jl/pull/2606).

v0.22.1
-------

- ![][badge-🚀performance] Reduced the number of allocations in the NetCDF
  writer. PRs [#2772](https://github.com/CliMA/ClimaAtmos.jl/pull/2772),
  [#2773](https://github.com/CliMA/ClimaAtmos.jl/pull/2773).
- Added a new script, `perf/benchmark_netcdf_io.jl` to test IO performance for
  the NetCDF writer. PR [#2773](https://github.com/CliMA/ClimaAtmos.jl/pull/2773).

<!--

Contributors are welcome to begin the description of changelog items with badge(s) below. Here is a brief description of when to use badges for a particular pull request / set of changes:

 - 🔥behavioralΔ - behavioral changes. For example: a new model is used, yielding more accurate results.
 - 🤖precisionΔ - machine-precision changes. For example, swapping the order of summed arguments can result in machine-precision changes.
 - 💥breaking - breaking changes. For example: removing deprecated functions/types, removing support for functionality, API changes.
 - 🚀performance - performance improvements. For example: improving type inference, reducing allocations, or code hoisting.
 - ✨feature - new feature added. For example: adding support for a cubed-sphere grid
 - 🐛bugfix - bugfix. For example: fixing incorrect logic, resulting in incorrect results, or fixing code that otherwise might give a `MethodError`.

-->

[badge-🔥behavioralΔ]: https://img.shields.io/badge/🔥behavioralΔ-orange.svg
[badge-💥breaking]: https://img.shields.io/badge/💥BREAKING-red.svg
[badge-🚀performance]: https://img.shields.io/badge/🚀performance-green.svg
[badge-✨feature/enhancement]: https://img.shields.io/badge/feature/enhancement-blue.svg
[badge-🐛bugfix]: https://img.shields.io/badge/🐛bugfix-purple.svg
