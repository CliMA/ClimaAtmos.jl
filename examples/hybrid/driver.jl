import ClimaAtmos as CA

include("cli_options.jl")
(s, parsed_args_defaults) = parse_commandline()
if !(@isdefined parsed_args)
    parsed_args = parsed_args_defaults
end

include("nvtx.jl")

parse_arg(pa, key, default) = isnothing(pa[key]) ? default : pa[key]

const FT = parsed_args["FLOAT_TYPE"] == "Float64" ? Float64 : Float32

include("parameter_set.jl")
params, parsed_args = create_parameter_set(FT, parsed_args, cli_defaults(s))

include("comms.jl")
if startswith(parsed_args["ode_algo"], "ODE.") # TODO: use Preferences.jl instead:
    include("../ordinary_diff_eq_bug_fixes.jl")
end
fps = parsed_args["fps"]
idealized_insolation = parsed_args["idealized_insolation"]
idealized_clouds = parsed_args["idealized_clouds"]

@assert idealized_insolation in (true, false)
@assert idealized_clouds in (true, false)
@assert parsed_args["config"] in ("sphere", "column", "box", "plane")

import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaAtmos.InitialConditions as ICs

include(joinpath(pkgdir(CA), "artifacts", "artifact_funcs.jl"))

import ClimaAtmos.TurbulenceConvection as TC
include("TurbulenceConvectionUtils.jl")
import .TurbulenceConvectionUtils as TCU

atmos = CA.get_atmos(FT, parsed_args, params.turbconv_params)
@info "AtmosModel: \n$(summary(atmos))"
numerics = CA.get_numerics(parsed_args)
include("get_simulation_and_args_integrator.jl")
simulation = get_simulation(FT, parsed_args)
initial_condition = CA.get_initial_condition(parsed_args)

# TODO: use import instead of using
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

# TODO: flip order so that NamedTuple() is fallback.
function additional_cache(Y, default_cache, parsed_args, params, atmos, dt)
    (; precip_model, forcing_type, radiation_mode, turbconv_model) = atmos

    radiation_cache = if radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        CA.radiation_model_cache(
            Y,
            default_cache,
            params,
            radiation_mode;
            idealized_insolation,
            idealized_clouds,
            data_loader = CA.rrtmgp_data_loader,
        )
    else
        CA.radiation_model_cache(Y, params, radiation_mode)
    end

    return merge(
        CA.hyperdiffusion_cache(atmos.hyperdiff, Y),
        CA.rayleigh_sponge_cache(atmos.rayleigh_sponge, Y),
        CA.viscous_sponge_cache(atmos.viscous_sponge, Y),
        CA.precipitation_cache(Y, precip_model),
        CA.subsidence_cache(Y, atmos.subsidence),
        CA.large_scale_advection_cache(Y, atmos.ls_adv),
        CA.edmf_coriolis_cache(Y, atmos.edmf_coriolis),
        CA.forcing_cache(Y, forcing_type),
        radiation_cache,
        CA.vertical_diffusion_boundary_layer_cache(Y, atmos),
        CA.non_orographic_gravity_wave_cache(
            atmos.non_orographic_gravity_wave,
            atmos.model_config,
            Y,
        ),
        CA.orographic_gravity_wave_cache(
            atmos.orographic_gravity_wave,
            TOPO_DIR,
            Y,
            comms_ctx,
        ),
        (; Δt = dt),
        TCU.turbconv_cache(
            Y,
            turbconv_model,
            atmos,
            params,
            parsed_args,
            initial_condition,
        ),
    )
end

function additional_tendency!(Yₜ, Y, p, t)
    CA.hyperdiffusion_tendency!(Yₜ, Y, p, t)
    CA.viscous_sponge_tendency!(Yₜ, Y, p, t, p.atmos.viscous_sponge)

    # Vertical tendencies
    Fields.bycolumn(axes(Y.c)) do colidx
        CA.rayleigh_sponge_tendency!(
            Yₜ,
            Y,
            p,
            t,
            colidx,
            p.atmos.rayleigh_sponge,
        )
        CA.forcing_tendency!(Yₜ, Y, p, t, colidx, p.forcing_type)
        CA.subsidence_tendency!(Yₜ, Y, p, t, colidx, p.subsidence)
        CA.edmf_coriolis_tendency!(Yₜ, Y, p, t, colidx, p.edmf_coriolis)
        CA.large_scale_advection_tendency!(Yₜ, Y, p, t, colidx, p.ls_adv)

        (; vert_diff) = p.atmos
        if p.atmos.coupling isa CA.Decoupled
            CA.get_surface_fluxes!(Y, p, t, colidx, vert_diff)
        end
        CA.vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            colidx,
            vert_diff,
        )

        CA.radiation_tendency!(Yₜ, Y, p, t, colidx, p.radiation_model)
        TCU.explicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, p.turbconv_model)
        CA.precipitation_tendency!(Yₜ, Y, p, t, colidx, p.precip_model)
    end
    # TODO: make bycolumn-able
    CA.non_orographic_gravity_wave_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.non_orographic_gravity_wave,
    )
    CA.orographic_gravity_wave_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.orographic_gravity_wave,
    )
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

include("get_cache_and_tendency.jl")

import ClimaCore: enable_threading
const enable_clima_core_threading = parsed_args["enable_threading"]
enable_threading() = enable_clima_core_threading

@time "Allocating Y" if simulation.restart
    (Y, t_start) = CA.get_state_restart(comms_ctx)
    spaces = CA.get_spaces_restart(Y)
else
    spaces = CA.get_spaces(parsed_args, params, comms_ctx)
    Y = ICs.atmos_state(
        initial_condition(params),
        atmos,
        spaces.center_space,
        spaces.face_space,
    )
    t_start = FT(0)
end

# prepare topographic data if it runs with topography
if parsed_args["orographic_gravity_wave"] == true
    const TOPO_DIR = joinpath(@__DIR__, "topo_data/")
    if !isdir(TOPO_DIR)
        mkdir(TOPO_DIR)
    end
    include("orographic_gravity_wave_helper.jl")
    if !isfile(joinpath(TOPO_DIR, "topo_info.hdf5")) &
       ClimaComms.iamroot(comms_ctx)
        include(joinpath(pkgdir(CA), "artifacts", "artifact_funcs.jl"))
        # download topo data
        datafile_rll = joinpath(topo_res_path(), "topo_drag.res.nc")
        @show datafile_rll
        get_topo_info(Y, TOPO_DIR, datafile_rll, comms_ctx)
    end
else
    const TOPO_DIR = nothing
end

@time "Allocating cache (p)" begin
    p = get_cache(
        Y,
        parsed_args,
        params,
        spaces,
        atmos,
        numerics,
        simulation,
        comms_ctx,
    )
end

if parsed_args["discrete_hydrostatic_balance"]
    CA.set_discrete_hydrostatic_balanced_state!(Y, p)
end

@time "ode_configuration" ode_algo = CA.ode_configuration(Y, parsed_args, atmos)

include("get_callbacks.jl")

@time "get_callbacks" callback =
    get_callbacks(parsed_args, simulation, atmos, params)
tspan = (t_start, simulation.t_end)
@time "args_integrator" integrator_args, integrator_kwargs =
    args_integrator(parsed_args, Y, p, tspan, ode_algo, callback)

if haskey(ENV, "CI_PERF_SKIP_INIT") # for performance analysis
    throw(:exit_profile_init)
end

@time "get_integrator" integrator =
    CA.get_integrator(integrator_args, integrator_kwargs)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

@info "Running" job_id = simulation.job_id output_dir = simulation.output_dir tspan

struct SimulationResults{S, RT, WT}
    sol::S
    ret_code::RT
    walltime::WT
end
function perform_solve!(integrator, simulation, comms_ctx)
    try
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
            return SimulationResults(sol, :success, walltime)
        else
            sol = @timev OrdinaryDiffEq.solve!(integrator)
            return SimulationResults(sol, :success, nothing)
        end
    catch ret_code
        @error "ClimaAtmos simulation crashed. Stacktrace for failed simulation" exception =
            (ret_code, catch_backtrace())
        return SimulationResults(nothing, :simulation_crashed, nothing)
    end
end

sol_res = perform_solve!(integrator, simulation, comms_ctx)

import JSON
using Test
import OrderedCollections
using ClimaCoreTempestRemap
using ClimaCorePlots, Plots
using ClimaCoreMakie, CairoMakie
include(joinpath(pkgdir(CA), "post_processing", "contours_and_profiles.jl"))
include(joinpath(pkgdir(CA), "post_processing", "post_processing_funcs.jl"))
include(
    joinpath(pkgdir(CA), "post_processing", "define_tc_quicklook_profiles.jl"),
)

reference_job_id = parse_arg(parsed_args, "reference_job_id", simulation.job_id)

is_edmfx = atmos.turbconv_model isa CA.EDMFX
if is_edmfx && parsed_args["post_process"]
    contours_and_profiles(simulation.output_dir, reference_job_id)
    zip_and_cleanup_output(simulation.output_dir, "hdf5files.zip")
end

if parsed_args["debugging_tc"] && !is_edmfx
    include(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "regression_tests",
            "self_reference_or_path.jl",
        ),
    )

    main_branch_root = get_main_branch_buildkite_path()
    main_branch_data_path = joinpath(main_branch_root, reference_job_id)

    day = floor(Int, simulation.t_end / (60 * 60 * 24))
    sec = floor(Int, simulation.t_end % (60 * 60 * 24))

    zip_file = "hdf5files.zip"
    mktempdir(
        simulation.output_dir;
        prefix = "temp_unzip_path_",
    ) do temp_main_branch_path
        # Unzip files to temp directory, to avoid collisions with other jobs
        unzip_file_in_path(
            main_branch_data_path,
            zip_file,
            temp_main_branch_path,
        )
        # hdf5 files from the main branch are in `temp_dir`

        plot_tc_contours(
            simulation.output_dir;
            main_branch_data_path = temp_main_branch_path,
        )
        plot_tc_profiles(
            simulation.output_dir;
            hdf5_filename = "day$day.$sec.hdf5",
            main_branch_data_path = temp_main_branch_path,
        )
    end
    if atmos.model_config isa CA.SingleColumnModel
        zip_and_cleanup_output(simulation.output_dir, zip_file)
    end
end

if sol_res.ret_code == :simulation_crashed
    error(
        "The ClimaAtmos simulation has crashed. See the stack trace for details.",
    )
end
# Simulation did not crash
(; sol, walltime) = sol_res
@assert last(sol.t) == simulation.t_end
CA.verify_callbacks(sol.t)

if simulation.is_distributed
    CA.export_scaling_file(
        sol,
        simulation.output_dir,
        walltime,
        comms_ctx,
        ClimaComms.nprocs(comms_ctx),
    )
end

if !simulation.is_distributed &&
   parsed_args["post_process"] &&
   !is_edmfx &&
   !(atmos.model_config isa CA.SphericalModel)
    ENV["GKSwstype"] = "nul" # avoid displaying plots
    if CA.is_column_without_edmf(parsed_args)
        custom_postprocessing(sol, simulation.output_dir, p)
    elseif CA.is_column_edmf(parsed_args)
        postprocessing_edmf(sol, simulation.output_dir, fps)
    elseif CA.is_solid_body(parsed_args)
        postprocessing(sol, simulation.output_dir, fps)
    elseif atmos.model_config isa CA.BoxModel
        postprocessing_box(sol, simulation.output_dir)
    elseif atmos.model_config isa CA.PlaneModel
        postprocessing_plane(sol, simulation.output_dir, p)
    else
        error("Uncaught case")
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

if parsed_args["check_conservation"]
    @test sum(sol.u[1].c.ρ) ≈ sum(sol.u[end].c.ρ) rtol = 25 * eps(FT)
    @test sum(sol.u[1].c.ρe_tot) +
          (p.net_energy_flux_sfc[][] - p.net_energy_flux_toa[][]) ≈
          sum(sol.u[end].c.ρe_tot) rtol = 30 * eps(FT)
end
