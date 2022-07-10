include("cli_options.jl")
if !(@isdefined parsed_args)
    (s, parsed_args) = parse_commandline()
end

include("classify_case.jl")
include("utilities.jl")
include("nvtx.jl")

parse_arg(pa, key, default) = isnothing(pa[key]) ? default : pa[key]

const FT = parsed_args["FLOAT_TYPE"] == "Float64" ? Float64 : Float32

fps = parsed_args["fps"]
idealized_h2o = parsed_args["idealized_h2o"]
idealized_insolation = parsed_args["idealized_insolation"]
idealized_clouds = parsed_args["idealized_clouds"]
vert_diff = parsed_args["vert_diff"]
coupled = parsed_args["coupled"]
hyperdiff = parsed_args["hyperdiff"]
disable_qt_hyperdiffusion = parsed_args["disable_qt_hyperdiffusion"]
turbconv = parsed_args["turbconv"]
case_name = parsed_args["turbconv_case"]
h_elem = parsed_args["h_elem"]
z_elem = Int(parsed_args["z_elem"])
z_max = FT(parsed_args["z_max"])
dz_bottom = FT(parsed_args["dz_bottom"])
dz_top = FT(parsed_args["dz_top"])
κ₄ = parsed_args["kappa_4"]
rayleigh_sponge = parsed_args["rayleigh_sponge"]
viscous_sponge = parsed_args["viscous_sponge"]
zd_rayleigh = parsed_args["zd_rayleigh"]
zd_viscous = parsed_args["zd_viscous"]
κ₂_sponge = parsed_args["kappa_2_sponge"]
t_end = FT(time_to_seconds(parsed_args["t_end"]))
dt = FT(time_to_seconds(parsed_args["dt"]))
dt_save_to_sol = time_to_seconds(parsed_args["dt_save_to_sol"])
dt_save_to_disk = time_to_seconds(parsed_args["dt_save_to_disk"])

@assert idealized_insolation in (true, false)
@assert idealized_h2o in (true, false)
@assert idealized_clouds in (true, false)
@assert vert_diff in (true, false)
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
simulation = get_simulation(s, parsed_args)

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
jacobian_flags = jacobi_flags(model_spec.energy_form)
max_newton_iters = 2 # only required by ODE algorithms that use Newton's method
newton_κ = Inf # similar to a reltol for Newton's method (default is 0.01)
show_progress_bar = isinteractive()
additional_solver_kwargs = () # e.g., abstol and reltol
test_implicit_solver = false # makes solver extremely slow when set to `true`

# TODO: flip order so that NamedTuple() is fallback.
function additional_cache(Y, params, model_spec, dt; use_tempest_mode = false)
    FT = typeof(dt)
    (; microphysics_model, forcing_type, radiation_model, turbconv_model) =
        model_spec
    return merge(
        hyperdiffusion_cache(
            Y,
            FT;
            κ₄ = FT(κ₄),
            use_tempest_mode,
            disable_qt_hyperdiffusion,
        ),
        rayleigh_sponge ?
        rayleigh_sponge_cache(Y, dt; zd_rayleigh = FT(zd_rayleigh)) :
        NamedTuple(),
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
            idealized_h2o,
            idealized_clouds,
        ),
        vert_diff ?
        vertical_diffusion_boundary_layer_cache(Y; diffuse_momentum, coupled) :
        NamedTuple(),
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
        (; enable_default_remaining_tendency = isnothing(turbconv_model)),
        !isnothing(turbconv_model) ?
        (; edmf_cache = TCU.get_edmf_cache(Y, namelist, params)) : NamedTuple(),
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
    atexit() do
        logger_stream = ClimaComms.iamroot(comms_ctx) ? stderr : devnull
        global_logger(global_logger(ConsoleLogger(logger_stream, Logging.Info)))
    end
    @info "Setting up distributed run on $nprocs \
        processor$(nprocs == 1 ? "" : "s")"
else
    const comms_ctx = nothing
    using TerminalLoggers: TerminalLogger
    atexit() do
        global_logger(global_logger(TerminalLogger()))
    end
end
using OrdinaryDiffEq
using DiffEqCallbacks
using JLD2

parsed_args["trunc_stack_traces"] && include("truncate_stack_traces.jl")
include("../implicit_solver_debugging_tools.jl")
include("../ordinary_diff_eq_bug_fixes.jl")
include("../common_spaces.jl")

include(joinpath("sphere", "baroclinic_wave_utilities.jl"))

condition_every_iter(u, t, integrator) = true

function affect_filter!(Y::Fields.FieldVector)
    @. Y.c.ρq_tot = max(Y.c.ρq_tot, 0)
    return nothing
end

function affect_filter!(integrator)
    (; apply_moisture_filter) = integrator.p
    affect_filter!(integrator.u)
    # We're lying to OrdinaryDiffEq.jl, in order to avoid
    # paying for an additional tendency call, which is required
    # to support supplying a continuous representation of the solution.
    OrdinaryDiffEq.u_modified!(integrator, false)
end

callback_filters = OrdinaryDiffEq.DiscreteCallback(
    condition_every_iter,
    affect_filter!;
    save_positions = (false, false),
)

additional_callbacks = if !isnothing(model_spec.radiation_model)
    # TODO: better if-else criteria?
    dt_rad = parsed_args["config"] == "column" ? dt : FT(6 * 60 * 60)
    (
        PeriodicCallback(
            rrtmgp_model_callback!,
            dt_rad; # update RRTMGPModel every dt_rad
            initial_affect = true, # run callback at t = 0
            save_positions = (false, false), # do not save Y before and after callback
        ),
    )
else
    ()
end
if model_spec.moisture_model isa EquilMoistModel &&
   parsed_args["apply_moisture_filter"]
    additional_callbacks = (additional_callbacks..., callback_filters)
end

import ClimaCore: enable_threading
const enable_clima_core_threading = parsed_args["enable_threading"]
enable_threading() = enable_clima_core_threading

center_space, face_space = if parsed_args["config"] == "sphere"
    quad = Spaces.Quadratures.GLL{5}()
    horizontal_mesh = baroclinic_wave_mesh(; params, h_elem = h_elem)
    h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx)
    z_stretch = if parsed_args["z_stretch"]
        Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
    else
        Meshes.Uniform()
    end
    make_hybrid_spaces(h_space, z_max, z_elem, z_stretch)
elseif parsed_args["config"] == "column" # single column
    Δx = FT(1) # Note: This value shouldn't matter, since we only have 1 column.
    quad = Spaces.Quadratures.GL{1}()
    horizontal_mesh = periodic_rectangle_mesh(;
        x_max = Δx,
        y_max = Δx,
        x_elem = 1,
        y_elem = 1,
    )
    h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx)
    z_stretch = if parsed_args["z_stretch"]
        Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
    else
        Meshes.Uniform()
    end
    make_hybrid_spaces(h_space, z_max, z_elem, z_stretch)
end

if haskey(ENV, "RESTART_FILE")
    restart_file_name = ENV["RESTART_FILE"]
    if simulation.is_distributed
        restart_file_name =
            split(restart_file_name, ".jld2")[1] * "_pid$pid.jld2"
    end
    restart_data = jldopen(restart_file_name)
    t_start = restart_data["t"]
    Y = restart_data["Y"]
    close(restart_data)
    ᶜlocal_geometry = Fields.local_geometry_field(Y.c)
    ᶠlocal_geometry = Fields.local_geometry_field(Y.f)
    # TODO:   quad, horizontal_mesh, z_stretch,
    #         z_max, z_elem should be taken from Y.
    #         when restarting
else
    t_start = FT(0)
    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
    ᶠlocal_geometry = Fields.local_geometry_field(face_space)

    center_initial_condition = if is_baro_wave(parsed_args)
        center_initial_condition_baroclinic_wave
    elseif parsed_args["config"] == "sphere"
        center_initial_condition_sphere
    elseif parsed_args["config"] == "column"
        center_initial_condition_column
    end

    Y = init_state(
        center_initial_condition,
        face_initial_condition,
        center_space,
        face_space,
        params,
        model_spec,
    )
end

p = get_cache(Y, params, model_spec, numerics, simulation, dt)
if parsed_args["turbconv"] == "edmf"
    TCU.init_tc!(Y, p, params, namelist)
end

# Print tendencies:
for key in keys(p.tendency_knobs)
    @info "`$(key)`:$(getproperty(p.tendency_knobs, key))"
end

ode_algorithm = getproperty(OrdinaryDiffEq, Symbol(parsed_args["ode_algo"]))

ode_algorithm_type =
    ode_algorithm isa Function ? typeof(ode_algorithm()) : ode_algorithm
if ode_algorithm_type <: Union{
    OrdinaryDiffEq.OrdinaryDiffEqImplicitAlgorithm,
    OrdinaryDiffEq.OrdinaryDiffEqAdaptiveImplicitAlgorithm,
}
    use_transform = !(ode_algorithm_type in (Rosenbrock23, Rosenbrock32))
    W = SchurComplementW(Y, use_transform, jacobian_flags, test_implicit_solver)
    if :ρe_tot in propertynames(Y.c) &&
       W.flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :no_∂ᶜp∂ᶜK &&
       W.flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
        Wfact! = Wfact_special!
    else
        Wfact! = Wfact_generic!
    end
    jac_kwargs =
        use_transform ? (; jac_prototype = W, Wfact_t = Wfact!) :
        (; jac_prototype = W, Wfact = Wfact!)

    alg_kwargs = (; linsolve = linsolve!)
    if ode_algorithm_type <: Union{
        OrdinaryDiffEq.OrdinaryDiffEqNewtonAlgorithm,
        OrdinaryDiffEq.OrdinaryDiffEqNewtonAdaptiveAlgorithm,
    }
        alg_kwargs = (;
            alg_kwargs...,
            nlsolve = NLNewton(; κ = newton_κ, max_iter = max_newton_iters),
        )
    end
else
    jac_kwargs = alg_kwargs = ()
end

include("callbacks.jl")

dss_callback = FunctionCallingCallback(func_start = true) do Y, t, integrator
    p = integrator.p
    @nvtx "dss callback" color = colorant"yellow" begin
        Spaces.weighted_dss_start!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_start!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_internal!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_internal!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_ghost!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_ghost!(Y.f, p.ghost_buffer.f)
    end
end
save_to_disk_callback = if dt_save_to_disk == Inf
    nothing
else
    PeriodicCallback(save_to_disk_func, dt_save_to_disk; initial_affect = true)
end
callback =
    CallbackSet(dss_callback, save_to_disk_callback, additional_callbacks...)

problem = if parsed_args["split_ode"]
    SplitODEProblem(
        ODEFunction(
            implicit_tendency!;
            jac_kwargs...,
            tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= FT(0)),
        ),
        remaining_tendency!,
        Y,
        (t_start, t_end),
        p,
    )
else
    OrdinaryDiffEq.ODEProblem(remaining_tendency!, Y, (t_start, t_end), p)
end
integrator = OrdinaryDiffEq.init(
    problem,
    ode_algorithm(; alg_kwargs...);
    saveat = dt_save_to_sol == Inf ? [] : dt_save_to_sol,
    callback = callback,
    dt = dt,
    adaptive = false,
    progress = show_progress_bar,
    progress_steps = isinteractive() ? 1 : 1000,
    additional_solver_kwargs...,
)

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

if simulation.is_distributed # replace sol.u on the root processor with the global sol.u
    if ClimaComms.iamroot(comms_ctx)
        global_h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx)
        global_center_space, global_face_space =
            make_hybrid_spaces(global_h_space, z_max, z_elem, z_stretch)
        global_Y_c_type = Fields.Field{
            typeof(Fields.field_values(Y.c)),
            typeof(global_center_space),
        }
        global_Y_f_type = Fields.Field{
            typeof(Fields.field_values(Y.f)),
            typeof(global_face_space),
        }
        global_Y_type = Fields.FieldVector{
            FT,
            NamedTuple{(:c, :f), Tuple{global_Y_c_type, global_Y_f_type}},
        }
        global_sol_u = similar(sol.u, global_Y_type)
    end
    for i in 1:length(sol.u)
        global_Y_c =
            DataLayouts.gather(comms_ctx, Fields.field_values(sol.u[i].c))
        global_Y_f =
            DataLayouts.gather(comms_ctx, Fields.field_values(sol.u[i].f))
        if ClimaComms.iamroot(comms_ctx)
            global_sol_u[i] = Fields.FieldVector(
                c = Fields.Field(global_Y_c, global_center_space),
                f = Fields.Field(global_Y_f, global_face_space),
            )
        end
    end
    if ClimaComms.iamroot(comms_ctx)
        sol = DiffEqBase.sensitivity_solution(sol, global_sol_u, sol.t)
        println("walltime = $walltime (seconds)")
        scaling_file = joinpath(
            simulation.output_dir,
            "scaling_data_$(nprocs)_processes.jld2",
        )
        println("writing performance data to $scaling_file")
        jldsave(scaling_file; nprocs, walltime)
    end
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
    elseif model_spec.forcing_type isa HeldSuarezForcing &&
           t_end >= (3600 * 24 * 400)
        paperplots_held_suarez(sol, simulation.output_dir, p, FT(90), FT(180))
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
