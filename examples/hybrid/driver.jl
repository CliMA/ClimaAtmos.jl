if !(@isdefined parsed_args)
    include("cli_options.jl")
    (s, parsed_args) = parse_commandline()
end

include("../implicit_solver_debugging_tools.jl")
include("../ordinary_diff_eq_bug_fixes.jl")
include("../common_spaces.jl")
include("classify_case.jl")
include("utilities.jl")
include("nvtx.jl")

parse_arg(pa, key, default) = isnothing(pa[key]) ? default : pa[key]

const FT = parsed_args["FLOAT_TYPE"] == "Float64" ? Float64 : Float32

fps = parsed_args["fps"]
idealized_insolation = parsed_args["idealized_insolation"]
idealized_clouds = parsed_args["idealized_clouds"]
vert_diff = parsed_args["vert_diff"]
surface_scheme = parsed_args["surface_scheme"]
coupled = parsed_args["coupled"]
hyperdiff = parsed_args["hyperdiff"]
disable_qt_hyperdiffusion = parsed_args["disable_qt_hyperdiffusion"]
turbconv = parsed_args["turbconv"]
case_name = parsed_args["turbconv_case"]
κ₄ = parsed_args["kappa_4"]
rayleigh_sponge = parsed_args["rayleigh_sponge"]
viscous_sponge = parsed_args["viscous_sponge"]
zd_rayleigh = parsed_args["zd_rayleigh"]
α_rayleigh_uₕ = parsed_args["alpha_rayleigh_uh"]
α_rayleigh_w = parsed_args["alpha_rayleigh_w"]
zd_viscous = parsed_args["zd_viscous"]
κ₂_sponge = parsed_args["kappa_2_sponge"]

@assert idealized_insolation in (true, false)
@assert idealized_clouds in (true, false)
@assert vert_diff in (true, false)
@assert surface_scheme in ("bulk", "monin_obukhov")
@assert hyperdiff in (true, false)
@assert parsed_args["config"] in ("sphere", "column")
@assert rayleigh_sponge in (true, false)
@assert viscous_sponge in (true, false)

include(joinpath("..", "RRTMGPInterface.jl"))
import .RRTMGPInterface as RRTMGPI

include("types.jl")

import ClimaAtmos as CA
import ClimaAtmos.TurbulenceConvection as TC
include("TurbulenceConvectionUtils.jl")
import .TurbulenceConvectionUtils as TCU
namelist = if turbconv == "edmf"
    nl = TCU.NameList.default_namelist(case_name)
    nl["set_src_seed"] = true
    nl
else
    nothing
end

include("parameter_set.jl")
# TODO: unify parsed_args and namelist
params = create_parameter_set(FT, parsed_args, namelist)

model_spec = get_model_spec(FT, parsed_args, namelist)
numerics = get_numerics(parsed_args)
simulation = get_simulation(FT, parsed_args)

diffuse_momentum = vert_diff && !(model_spec.forcing_type isa HeldSuarezForcing)

# TODO: use import istead of using
using Colors
using OrdinaryDiffEq
using PrettyTables
using DiffEqCallbacks
using JLD2
using ClimaCore.DataLayouts
using NCDatasets
using ClimaCore

import Random
Random.seed!(1234)

!isnothing(model_spec.radiation_model) && include("radiation_utilities.jl")

jacobi_flags(::TotalEnergy) =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
jacobi_flags(::InternalEnergy) =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
jacobi_flags(::PotentialTemperature) =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

# TODO: flip order so that NamedTuple() is fallback.
function additional_cache(Y, params, model_spec, dt; use_tempest_mode = false)
    FT = typeof(dt)
    (; microphysics_model, forcing_type, radiation_model, turbconv_model) =
        model_spec

    default_remaining_tendency! = if model_spec.anelastic_dycore
        (Yₜ, Y, p, t) -> nothing
    else
        if :ρe_tot in propertynames(Y.c) && enable_threading()
            default_remaining_tendency_special!
        else
            default_remaining_tendency_generic!
        end
    end

    return merge(
        hyperdiffusion_cache(
            Y,
            FT;
            κ₄ = FT(κ₄),
            use_tempest_mode,
            disable_qt_hyperdiffusion,
        ),
        rayleigh_sponge ?
        rayleigh_sponge_cache(
            Y,
            dt;
            zd_rayleigh = FT(zd_rayleigh),
            α_rayleigh_uₕ = FT(α_rayleigh_uₕ),
            α_rayleigh_w = FT(α_rayleigh_w),
        ) : NamedTuple(),
        viscous_sponge ?
        viscous_sponge_cache(
            Y;
            zd_viscous = FT(zd_viscous),
            κ₂ = FT(κ₂_sponge),
        ) : NamedTuple(),
        microphysics_cache(Y, microphysics_model),
        forcing_cache(Y, forcing_type),
        isnothing(radiation_model) ? NamedTuple() :
        rrtmgp_model_cache(
            Y,
            params,
            radiation_model;
            idealized_insolation,
            model_spec.idealized_h2o,
            idealized_clouds,
        ),
        vert_diff ?
        vertical_diffusion_boundary_layer_cache(
            Y;
            surface_scheme,
            diffuse_momentum,
            coupled,
        ) : NamedTuple(),
        (;
            tendency_knobs = (;
                hs_forcing = forcing_type isa HeldSuarezForcing,
                microphy_0M = microphysics_model isa Microphysics0Moment,
                rad_flux = !isnothing(radiation_model),
                vert_diff,
                rayleigh_sponge,
                viscous_sponge,
                hyperdiff,
                has_turbconv = !isnothing(turbconv_model),
            )
        ),
        (; Δt = dt),
        (; default_remaining_tendency!),
        !isnothing(turbconv_model) ?
        (; edmf_cache = TCU.get_edmf_cache(Y, namelist, params, parsed_args)) :
        NamedTuple(),
        (; apply_moisture_filter = parsed_args["apply_moisture_filter"]),
    )
end

function additional_tendency!(Yₜ, Y, p, t)
    (; rad_flux, vert_diff, hs_forcing) = p.tendency_knobs
    (; microphy_0M, hyperdiff, has_turbconv) = p.tendency_knobs
    (; rayleigh_sponge, viscous_sponge) = p.tendency_knobs
    hyperdiff && hyperdiffusion_tendency!(Yₜ, Y, p, t)
    rayleigh_sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    viscous_sponge && viscous_sponge_tendency!(Yₜ, Y, p, t)
    hs_forcing && held_suarez_tendency!(Yₜ, Y, p, t)
    vert_diff && vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    microphy_0M && zero_moment_microphysics_tendency!(Yₜ, Y, p, t)
    rad_flux && rrtmgp_model_tendency!(Yₜ, Y, p, t)
    has_turbconv && TCU.sgs_flux_tendency!(Yₜ, Y, p, t)
end

################################################################################

using Logging
if simulation.is_distributed
    using ClimaComms
    if ENV["CLIMACORE_DISTRIBUTED"] == "MPI"
        using ClimaCommsMPI
        const comms_ctx = ClimaCommsMPI.MPICommsContext()
    else
        error("ENV[\"CLIMACORE_DISTRIBUTED\"] only supports the \"MPI\" option")
    end
    const pid, nprocs = ClimaComms.init(comms_ctx)
    logger_stream = ClimaComms.iamroot(comms_ctx) ? stderr : devnull
    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    @info "Setting up distributed run on $nprocs \
        processor$(nprocs == 1 ? "" : "s")"
else
    const comms_ctx = nothing
    using TerminalLoggers: TerminalLogger
    prev_logger = global_logger(TerminalLogger())
end
atexit() do
    global_logger(prev_logger)
end
using OrdinaryDiffEq
using DiffEqCallbacks
using JLD2

import ClimaCore
if parsed_args["trunc_stack_traces"]
    ClimaCore.Fields.truncate_printing_field_types() = true
end

include(joinpath("sphere", "baroclinic_wave_utilities.jl"))


import ClimaCore: enable_threading
const enable_clima_core_threading = parsed_args["enable_threading"]
enable_threading() = enable_clima_core_threading


if simulation.restart
    (Y, t_start) = get_state_restart(simulation)
    spaces = get_spaces_restart(Y, parsed_args, FT)
else
    spaces = get_spaces(parsed_args, params, comms_ctx)
    (Y, t_start) =
        get_state_fresh_start(parsed_args, spaces, params, model_spec)
end

p = get_cache(Y, params, spaces, model_spec, numerics, simulation)
if parsed_args["turbconv"] == "edmf"
    TCU.init_tc!(Y, p, params, namelist)
end

# Print tendencies:
@info "Output directory: `$(simulation.output_dir)`"
for key in keys(p.tendency_knobs)
    @info "`$(key)`:$(getproperty(p.tendency_knobs, key))"
end

ode_config = ode_configuration(Y, parsed_args, model_spec)

include("callbacks.jl")

callback = get_callbacks(parsed_args, simulation, model_spec, params)
tspan = (t_start, simulation.t_end)
@info "tspan = `$tspan`"
integrator = get_integrator(parsed_args, Y, p, tspan, ode_config, callback)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end
@info "Running job:`$(simulation.job_id)`"
if simulation.is_distributed
    OrdinaryDiffEq.step!(integrator)
    ClimaComms.barrier(comms_ctx)
    walltime = @elapsed sol = OrdinaryDiffEq.solve!(integrator)
    ClimaComms.barrier(comms_ctx)
else
    sol = @timev OrdinaryDiffEq.solve!(integrator)
end

@assert last(sol.t) == simulation.t_end

verify_callbacks(sol.t)

if simulation.is_distributed
    export_scaling_file(sol, simulation.output_dir, walltime, comms_ctx, nprocs)
end

import JSON
using Test
import OrderedCollections
using ClimaCoreTempestRemap
using ClimaCorePlots, Plots
include(joinpath(@__DIR__, "define_post_processing.jl"))
if !simulation.is_distributed && parsed_args["post_process"]
    ENV["GKSwstype"] = "nul" # avoid displaying plots
    if is_baro_wave(parsed_args)
        paperplots_baro_wave(
            model_spec,
            sol,
            simulation.output_dir,
            p,
            FT(90),
            FT(180),
        )
    elseif is_column_radiative_equilibrium(parsed_args)
        custom_postprocessing(sol, simulation.output_dir)
    elseif is_column_edmf(parsed_args)
        postprocessing_edmf(sol, simulation.output_dir, fps)
    elseif model_spec.forcing_type isa HeldSuarezForcing
        paperplots_held_suarez(
            model_spec,
            sol,
            simulation.output_dir,
            p,
            FT(90),
            FT(180),
        )
    else
        postprocessing(sol, simulation.output_dir, fps)
    end
end

if parsed_args["regression_test"]
    # Test results against main branch
    include(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "regression_tests",
            "regression_tests.jl",
        ),
    )
    perform_regression_tests(
        simulation.job_id,
        sol.u[end],
        all_best_mse,
        simulation.output_dir,
    )
end
