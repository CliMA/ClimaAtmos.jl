"""
Self-contained CI smoke runscript for the TRMM_LBA column with prognostic EDMF
+ 2M warm-rain + P3 ice microphysics.

The bulk microphysics tendency cache is filled by `microphysics_substep_callback!`
at the start of each step (8 substeps inside the callback). The IMEX RK loop sees
the substep-averaged tendency as a constant explicit forcing.

The run uses a short `t_end` (30 minutes). The post-solve assertions check that
the run reaches `t_end` with finite state; any crash, NaN, or early exit throws
and fails the job.
"""

# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

# Set up environment
import ClimaAtmos as CA
import ClimaCore as CC
import ClimaComms
ClimaComms.@import_required_backends
context = CA.get_comms_context(Dict("device" => "auto"))

### -------------------- ###
### Variable model setup ###
### -------------------- ###
job_id = "prognostic_edmfx_trmm_column_2M"
reference_job_id = Val(:prognostic_edmfx_trmm_column)  # for plotting
output_dir = nothing  # customize if desired
FT = Float32

dt = "5secs"
t_end = "30mins"

### -------------------- ###


## Load model parameters
override_file = joinpath(pkgdir(CA), "toml", "prognostic_edmfx_1M.toml")
toml_dict = CA.CP.create_toml_dict(FT; override_file)
params = CA.ClimaAtmosParameters(toml_dict)

## TRMM_LBA model prescriptions
initial_condition = CA.Setups.TRMM_LBA(;
    prognostic_tke = true,
    thermo_params = params.thermodynamics_params,
)
# Time-varying prescribed surface (shf, lhf ramp over the first 5.25 h), surface
# temperature, and boundary overrides — supplied by the TRMM_LBA setup.
surface = CA.Setups.surface_condition(initial_condition, params)

## Construct the model
model = CA.AtmosModel(;
    # AtmosWater - Moisture, Precipitation & Clouds
    microphysics_model = CA.NonEquilibriumMicrophysics2M(; n_substeps = 8),
    cloud_model = CA.QuadratureCloud(),
    microphysics_tendency_timestepping = CA.Explicit(),  # implicit_microphysics: false
    microphysics_substep_callback = true,
    # Post-step nonneg projection on the 2M+P3 prognostics. Without this, BMT +
    # sedimentation produce single-Δt overshoots that leave `Y.c.ρq_rai` (and
    # friends) negative in the bottom cells once falling rain reaches the surface.
    tracer_nonnegativity_method = CA.TracerNonnegativityVaporConstraint{false}(),

    # AtmosTurbconv - Turbulence & Convection (prognostic_edmfx)
    turbconv_model = CA.PrognosticEDMFX(;
        n_updrafts = 1,
        prognostic_tke = true,
        area_fraction = params.turbconv_params.min_area,
    ),
    edmfx_model = CA.EDMFXModel(;
        entr_model = CA.InvZEntrainment(),             # edmfx_entr_model: "Generalized"
        detr_model = CA.BuoyancyVelocityDetrainment(), # edmfx_detr_model: "Generalized"
        sgs_mass_flux = true,
        sgs_diffusive_flux = true,
        nh_pressure = true,
        vertical_diffusion = true,
        filter = true,
        scale_blending_method = CA.SmoothMinimumBlending(),
    ),

    # AtmosRadiation
    radiation_mode = CA.RadiationTRMM_LBA(FT),

    # AtmosSurface — built directly from the TRMM_LBA `surface_condition`.
    # The current API has no `surface_model`/`PrescribedSST` and no
    # `surface_setup` kwarg on AtmosSimulation; the surface is passed to
    # AtmosModel via these grouped kwargs (`surface_condition` returns
    # `overrides`, which maps to AtmosSurface's `boundary_overrides`).
    flux_scheme = surface.flux_scheme,
    temperature = surface.temperature,
    boundary_overrides = surface.overrides,

    # Numerics
    numerics = CA.AtmosNumerics(;
        tracer_upwinding = :first_order,
        diff_mode = CA.Implicit(),  # implicit_diffusion: true
        hyperdiff = nothing,
    ),
)

## Grid creation
# The YAML path's `config: column` does not use a 1D `ColumnGrid` — it uses
# a minimal `BoxGrid` with `x_elem=y_elem=1, nh_poly=1` so the space is still a
# 2x2 ExtrudedFiniteDifferenceSpace. The 1D FiniteDifferenceSpace produced by
# ColumnGrid breaks the EDMF Jacobian (which trips on the empty horizontal axis).
grid = CA.BoxGrid(FT;
    context,
    nh_poly = 1,
    x_elem = 1, x_max = FT(1e5), periodic_x = true,
    y_elem = 1, y_max = FT(1e5), periodic_y = true,
    z_elem = 82, z_max = FT(16400), z_stretch = false,
    bubble = false,
)

## Discretization
# IMEXAlgorithm(tableau_name, newtons_method); mirrors `ode_configuration` in
# src/config/type_getters.jl. ARS222 + 1-iter Newton (UpdateEvery NewNewtonIteration).
import ClimaTimeSteppers as CTS
update_j = CTS.UpdateEvery(CTS.NewNewtonIteration)
newtons_method = CTS.NewtonsMethod(; max_iters = 1, update_j)
ode_config = CTS.IMEXAlgorithm(CTS.ARS222(), newtons_method)

## Aerosol tracers (prescribed)
aerosol_names =
    ["SO4", "CB1", "OC1", "DST01", "SSLT01", "SSLT02", "SSLT03", "SSLT04", "SSLT05"]

## Output diagnostics
diagnostics = [
    Dict(
        "short_name" => [
            "wa", "ua", "va", "ta", "thetaa", "ha",  # dynamics & thermodynamics
            "hus", "hur", "cl", "clw", "cli",  # liquid
            "pr",  # precipitation
            "ke",  # kinetic energy for spectrum
            # Smagorinsky diagnostics
            "Dh_smag", "strainh_smag",  # horizontal
            "Dv_smag", "strainv_smag",  # vertical
        ],
        "period" => "10mins",
    ),
]
### 1M microphysics
if model.microphysics_model ∈
   (CA.NonEquilibriumMicrophysics1M(), CA.NonEquilibriumMicrophysics2M())
    push!(diagnostics, Dict("short_name" => ["husra", "hussn"], "period" => "10mins"))
end
### 2M microphysics
if model.microphysics_model == CA.NonEquilibriumMicrophysics2M()
    push!(diagnostics, Dict("short_name" => ["cdnc", "ncra"], "period" => "10mins"))
end

## Assemble simulation
simulation = CA.AtmosSimulation{FT}(; job_id,
    model, params, context, grid,
    setup = initial_condition,
    restart_file = nothing,
    dt, t_end,
    ode_config,
    output_dir,
    aerosol_names,
    # Diagnostics
    diagnostics = CA.DiagnosticsConfig(; default = false, additional = diagnostics),
    # Numerics
    jacobian = CA.ManualSparseJacobian(approximate_solve_iters = 2),
    # Misc
    log_to_file = true,
)

## Solve
sol_res = CA.solve_atmos!(simulation)
(; sol) = sol_res

# A crash, non-convergence, callback failure, early exit, or any non-finite
# prognostic must fail the job (throw).
CA.error_if_crashed(sol_res.ret_code)
CA.verify_callbacks(sol.t)

# --> Make ci plots
if ClimaComms.iamroot(context)
    include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))
    make_plots(reference_job_id, simulation.output_dir)
end
# <--

### -------------------- ###
### Smoke assertions (CI check)
### -------------------- ###
if haskey(ENV, "CI")
    @assert Float64(last(sol.t)) == Float64(simulation.t_end) "solve did not reach t_end: last(sol.t)=$(Float64(last(sol.t))) != t_end=$(Float64(simulation.t_end))"

    # Explicit finite check over the final prognostic state (all of Y.c and Y.f).
    let Yend = sol.u[end]
        nonfinite = String[]
        for region in (:c, :f)
            Yr = getproperty(Yend, region)
            for name in propertynames(Yr)
                all(isfinite, parent(getproperty(Yr, name))) ||
                    push!(nonfinite, "Y.$region.$name")
            end
        end
        @assert isempty(nonfinite) "non-finite values in final state: $(nonfinite)"
    end
end
