if !(@isdefined parsed_args)
    include("cli_options.jl")
    (s, parsed_args) = parse_commandline()
end

include("comms.jl")
if startswith(parsed_args["ode_algo"], "ODE.") # TODO: use Preferences.jl instead:
    include("../ordinary_diff_eq_bug_fixes.jl")
end
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
@assert hyperdiff in (true, false)
@assert parsed_args["config"] in ("sphere", "column", "box")
@assert rayleigh_sponge in (true, false)
@assert viscous_sponge in (true, false)

import ClimaAtmos.RRTMGPInterface as RRTMGPI

include("types.jl")

import ClimaAtmos as CA
import ClimaAtmos.TurbulenceConvection as TC
include("TurbulenceConvectionUtils.jl")
import .TurbulenceConvectionUtils as TCU
namelist = if turbconv == "edmf"
    nl = TCU.NameList.default_namelist(case_name)
    nl
else
    nothing
end

include("parameter_set.jl")
# TODO: unify parsed_args and namelist
params = create_parameter_set(FT, parsed_args, namelist)

atmos = get_atmos(FT, parsed_args, namelist)
@info "AtmosModel: \n$(summary(atmos))"
numerics = get_numerics(parsed_args)
simulation = get_simulation(FT, parsed_args)

diffuse_momentum =
    vert_diff &&
    !(atmos.forcing_type isa HeldSuarezForcing) &&
    !isnothing(atmos.surface_scheme)

# TODO: use import istead of using
using Colors
using OrdinaryDiffEq
using PrettyTables
using DiffEqCallbacks
using JLD2
using ClimaCore.DataLayouts
using NCDatasets
using ClimaCore
using ClimaTimeSteppers

import Random
Random.seed!(1234)

isnothing(atmos.radiation_mode) || include("radiation_utilities.jl")

jacobi_flags(::TotalEnergy) =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
jacobi_flags(::InternalEnergy) =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
jacobi_flags(::PotentialTemperature) =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

# TODO: flip order so that NamedTuple() is fallback.
function additional_cache(
    Y,
    parsed_args,
    params,
    atmos,
    dt;
    use_tempest_mode = false,
)
    FT = typeof(dt)
    (;
        precip_model,
        forcing_type,
        radiation_mode,
        turbconv_model,
        precip_model,
    ) = atmos

    thermo_dispatcher = CA.ThermoDispatcher(atmos)
    compressibility_model = atmos.compressibility_model

    radiation_cache = if radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        CA.radiation_model_cache(
            Y,
            params,
            radiation_mode;
            idealized_insolation,
            idealized_clouds,
            thermo_dispatcher,
            data_loader,
            ᶜinterp,
        )
    else
        CA.radiation_model_cache(Y, params, radiation_mode)
    end

    return merge(
        CA.hyperdiffusion_cache(
            Y,
            FT;
            κ₄ = FT(κ₄),
            use_tempest_mode,
            disable_qt_hyperdiffusion,
        ),
        rayleigh_sponge ?
        CA.rayleigh_sponge_cache(
            Y,
            dt;
            zd_rayleigh = FT(zd_rayleigh),
            α_rayleigh_uₕ = FT(α_rayleigh_uₕ),
            α_rayleigh_w = FT(α_rayleigh_w),
        ) : NamedTuple(),
        viscous_sponge ?
        CA.viscous_sponge_cache(
            Y;
            zd_viscous = FT(zd_viscous),
            κ₂ = FT(κ₂_sponge),
        ) : NamedTuple(),
        CA.precipitation_cache(Y, precip_model),
        CA.subsidence_cache(Y, atmos.subsidence),
        CA.large_scale_advection_cache(Y, atmos.ls_adv),
        CA.edmf_coriolis_cache(Y, atmos.edmf_coriolis),
        CA.forcing_cache(Y, forcing_type),
        radiation_cache,
        vert_diff ?
        CA.vertical_diffusion_boundary_layer_cache(
            Y,
            FT;
            atmos.surface_scheme,
            C_E = FT(parsed_args["C_E"]),
            diffuse_momentum,
            coupled,
        ) : NamedTuple(),
        atmos.non_orographic_gravity_wave ?
        CA.gravity_wave_cache(atmos.model_config, Y, FT) : NamedTuple(),
        (;
            tendency_knobs = (;
                vert_diff,
                rayleigh_sponge,
                viscous_sponge,
                hyperdiff,
                non_orographic_gravity_wave = atmos.non_orographic_gravity_wave,
            )
        ),
        (; thermo_dispatcher),
        (; Δt = dt),
        (; compressibility_model),
        TCU.turbconv_cache(
            Y,
            turbconv_model,
            precip_model,
            namelist,
            params,
            parsed_args,
        ),
        (; apply_moisture_filter = parsed_args["apply_moisture_filter"]),
    )
end

function additional_tendency!(Yₜ, Y, p, t)
    (; viscous_sponge, hyperdiff) = p.tendency_knobs
    hyperdiff && CA.hyperdiffusion_tendency!(Yₜ, Y, p, t)
    viscous_sponge && CA.viscous_sponge_tendency!(Yₜ, Y, p, t)

    # Vertical tendencies
    Fields.bycolumn(axes(Y.c)) do colidx
        (; vert_diff) = p.tendency_knobs
        (; rayleigh_sponge) = p.tendency_knobs
        rayleigh_sponge && CA.rayleigh_sponge_tendency!(Yₜ, Y, p, t, colidx)
        CA.forcing_tendency!(Yₜ, Y, p, t, colidx, p.forcing_type)
        CA.subsidence_tendency!(Yₜ, Y, p, t, colidx, p.subsidence)
        CA.edmf_coriolis_tendency!(Yₜ, Y, p, t, colidx, p.edmf_coriolis)
        CA.large_scale_advection_tendency!(Yₜ, Y, p, t, colidx, p.ls_adv)
        if vert_diff
            (; coupled) = p
            !coupled && CA.get_surface_fluxes!(Y, p, colidx)
            CA.vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, colidx)
        end
        CA.precipitation_tendency!(Yₜ, Y, p, t, colidx, p.precip_model)
        CA.radiation_tendency!(Yₜ, Y, p, t, colidx, p.radiation_model)
        TCU.sgs_flux_tendency!(Yₜ, Y, p, t, colidx, p.turbconv_model)
    end
    # TODO: make bycolumn-able
    (; non_orographic_gravity_wave) = p.tendency_knobs
    non_orographic_gravity_wave && CA.gravity_wave_tendency!(Yₜ, Y, p, t)
end

################################################################################

using OrdinaryDiffEq
using DiffEqCallbacks
using JLD2

import ClimaCore
if parsed_args["trunc_stack_traces"]
    ClimaCore.Fields.truncate_printing_field_types() = true
end

using Statistics: mean
import SurfaceFluxes as SF
using CloudMicrophysics
const CCG = ClimaCore.Geometry
import ClimaAtmos.TurbulenceConvection as TC
import ClimaCore.Operators as CCO
const CM = CloudMicrophysics
import ClimaAtmos.Parameters as CAP

include("staggered_nonhydrostatic_model.jl")
include("initial_conditions.jl")

import ClimaCore: enable_threading
const enable_clima_core_threading = parsed_args["enable_threading"]
enable_threading() = enable_clima_core_threading

@time "Allocating Y" if simulation.restart
    (Y, t_start) = get_state_restart(comms_ctx)
    spaces = get_spaces_restart(Y)
else
    spaces = get_spaces(parsed_args, params, comms_ctx)
    (Y, t_start) = get_state_fresh_start(parsed_args, spaces, params, atmos)
end

p = get_cache(Y, parsed_args, params, spaces, atmos, numerics, simulation)
if parsed_args["turbconv"] == "edmf"
    @time "init_tc!" TCU.init_tc!(Y, p, params)
end

# Print tendencies:
# @info "Model composition" p.atmos...
@info "Tendencies" p.tendency_knobs...

@time "ode_configuration" ode_config = ode_configuration(Y, parsed_args, atmos)

include("callbacks.jl")

@time "get_callbacks" callback =
    get_callbacks(parsed_args, simulation, atmos, params)
tspan = (t_start, simulation.t_end)
@time "args_integrator" integrator_args, integrator_kwargs =
    args_integrator(parsed_args, Y, p, tspan, ode_config, callback)

if haskey(ENV, "CI_PERF_SKIP_INIT") # for performance analysis
    throw(:exit_profile_init)
end

@time "get_integrator" integrator =
    get_integrator(integrator_args, integrator_kwargs)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

@info "Running" job_id = simulation.job_id output_dir = simulation.output_dir tspan

if simulation.is_distributed
    OrdinaryDiffEq.step!(integrator)
    # GC.enable(false) # disabling GC causes a memory leak
    GC.gc()
    ClimaComms.barrier(comms_ctx)
    if ClimaComms.iamroot(comms_ctx)
        @timev begin
            walltime = @elapsed sol = OrdinaryDiffEq.solve!(integrator)
        end
    else
        walltime = @elapsed sol = OrdinaryDiffEq.solve!(integrator)
    end
    ClimaComms.barrier(comms_ctx)
    GC.enable(true)
else
    sol = @timev OrdinaryDiffEq.solve!(integrator)
end

@assert last(sol.t) == simulation.t_end

verify_callbacks(sol.t)

if simulation.is_distributed
    export_scaling_file(
        sol,
        simulation.output_dir,
        walltime,
        comms_ctx,
        ClimaComms.nprocs(comms_ctx),
    )
end

import JSON
using Test
import OrderedCollections
using ClimaCoreTempestRemap
using ClimaCorePlots, Plots
include(
    joinpath(pkgdir(ClimaAtmos), "post_processing", "post_processing_funcs.jl"),
)
if !simulation.is_distributed && parsed_args["post_process"]
    ENV["GKSwstype"] = "nul" # avoid displaying plots
    if is_baro_wave(parsed_args)
        paperplots_baro_wave(atmos, sol, simulation.output_dir, p, 90, 180)
    elseif is_solid_body(parsed_args)
        postprocessing(sol, simulation.output_dir, fps)
    elseif is_box(parsed_args)
        postprocessing_box(sol, simulation.output_dir)
    elseif atmos.forcing_type isa CA.HeldSuarezForcing
        paperplots_held_suarez(atmos, sol, simulation.output_dir, p, 90, 180)
    end
end

if parsed_args["debugging_tc"]
    include(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "regression_tests",
            "self_reference_or_path.jl",
        ),
    )
    include(
        joinpath(
            pkgdir(ClimaAtmos),
            "post_processing",
            "define_tc_quicklook_profiles.jl",
        ),
    )

    main_branch_root = get_main_branch_buildkite_path()
    quicklook_reference_job_id =
        parse_arg(parsed_args, "quicklook_reference_job_id", simulation.job_id)
    main_branch_data_path =
        joinpath(main_branch_root, quicklook_reference_job_id)

    day = floor(Int, simulation.t_end / (60 * 60 * 24))
    sec = floor(Int, simulation.t_end % (60 * 60 * 24))

    zip_file = "hdf5files"
    unzip_main(main_branch_data_path, zip_file)

    plot_tc_contours(simulation.output_dir; main_branch_data_path)
    plot_tc_profiles(
        simulation.output_dir;
        hdf5_filename = "day$day.$sec.hdf5",
        main_branch_data_path,
    )
    if atmos.model_config isa CA.SingleColumnModel
        zip_and_cleanup_output(simulation.output_dir, zip_file)
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
