include("cli_options.jl")
if !(@isdefined parsed_args)
    (s, parsed_args) = parse_commandline()
end

include("classify_case.jl")
const FT = parsed_args["FLOAT_TYPE"] == "Float64" ? Float64 : Float32

fps = parsed_args["fps"]
idealized_h2o = parsed_args["idealized_h2o"]
vert_diff = parsed_args["vert_diff"]
hyperdiff = parsed_args["hyperdiff"]

@assert idealized_h2o in (true, false)
@assert vert_diff in (true, false)
@assert hyperdiff in (true, false)
@assert parsed_args["config"] in ("sphere", "column")

include("types.jl")

moisture_model() = moisture_model(parsed_args)
energy_form() = energy_form(parsed_args)
radiation_model() = radiation_model(parsed_args)
microphysics_model() = microphysics_model(parsed_args)
forcing_type() = forcing_type(parsed_args)

using OrdinaryDiffEq
using PrettyTables
using DiffEqCallbacks
using JLD2
using ClimaCorePlots, Plots
using ClimaCore.DataLayouts
using NCDatasets
using ClimaCoreTempestRemap
using ClimaCore

import Random
Random.seed!(1234)
include(joinpath("..", "RRTMGPInterface.jl"))
import .RRTMGPInterface
RRTMGPI = RRTMGPInterface

!isnothing(radiation_model()) && include("radiation_utilities.jl")

parse_arg(pa, key, default) = isnothing(pa[key]) ? default : pa[key]

function time_to_seconds(s::String)
    factor = Dict(
        "secs" => 1,
        "mins" => 60,
        "hours" => 60 * 60,
        "days" => 60 * 60 * 24,
    )
    s == "Inf" && return Inf
    if count(occursin.(keys(factor), Ref(s))) != 1
        error(
            "Bad format for flag $s. Examples: [`10secs`, `20mins`, `30hours`, `40days`]",
        )
    end
    for match in keys(factor)
        occursin(match, s) || continue
        return parse(Float64, first(split(s, match))) * factor[match]
    end
    error("Uncaught case in computing time from given string.")
end

upwinding_mode() = Symbol(parse_arg(parsed_args, "upwinding", "third_order"))
@assert upwinding_mode() in (:none, :first_order, :third_order)

# Test-specific definitions (may be overwritten in each test case file)
# TODO: Allow some of these to be environment variables or command line arguments
t_end = FT(time_to_seconds(parsed_args["t_end"]))
dt = FT(time_to_seconds(parsed_args["dt"]))
dt_save_to_sol = time_to_seconds(parsed_args["dt_save_to_sol"])
dt_save_to_disk = time_to_seconds(parsed_args["dt_save_to_disk"])
jacobi_flags(::TotalEnergy) =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
jacobi_flags(::InternalEnergy) =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
jacobi_flags(::PotentialTemperature) =
    (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
jacobian_flags = jacobi_flags(energy_form())
max_newton_iters = 10 # only required by ODE algorithms that use Newton's method
show_progress_bar = isinteractive()
additional_solver_kwargs = () # e.g., abstol and reltol
test_implicit_solver = false # makes solver extremely slow when set to `true`

const sponge = false

# TODO: flip order so that NamedTuple() is fallback.
additional_cache(Y, params, dt; use_tempest_mode = false) = merge(
    hyperdiffusion_cache(Y; κ₄ = FT(2e17), use_tempest_mode),
    sponge ? rayleigh_sponge_cache(Y, dt) : NamedTuple(),
    microphysics_cache(Y, microphysics_model()),
    forcing_cache(Y, forcing_type()),
    isnothing(radiation_model()) ? NamedTuple() :
    rrtmgp_model_cache(Y, params, radiation_model(); idealized_h2o),
    vert_diff ? vertical_diffusion_boundary_layer_cache(Y) : NamedTuple(),
    (;
        tendency_knobs = (;
            hs_forcing = forcing_type() isa HeldSuarezForcing,
            microphy_0M = microphysics_model() isa Microphysics0Moment,
            rad_flux = !isnothing(radiation_model()),
            vert_diff,
            hyperdiff,
        )
    ),
)

additional_tendency!(Yₜ, Y, p, t) = begin
    (; rad_flux, vert_diff, hs_forcing) = p.tendency_knobs
    (; microphy_0M, hyperdiff) = p.tendency_knobs
    hyperdiff && hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    hs_forcing && held_suarez_tendency!(Yₜ, Y, p, t)
    vert_diff && vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    microphy_0M && zero_moment_microphysics_tendency!(Yₜ, Y, p, t)
    rad_flux && rrtmgp_model_tendency!(Yₜ, Y, p, t)
end

################################################################################
is_distributed = haskey(ENV, "CLIMACORE_DISTRIBUTED")

using Logging
if is_distributed
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

parsed_args["trunc_stack_traces"] && include("truncate_stack_traces.jl")
include("../implicit_solver_debugging_tools.jl")
include("../ordinary_diff_eq_bug_fixes.jl")
include("../common_spaces.jl")

include(joinpath("sphere", "baroclinic_wave_utilities.jl"))

# Variables required for driver.jl (modify as needed)
params = if is_column_radiative_equilibrium(parsed_args)
    EarthParameterSet()
else
    BaroclinicWaveParameterSet((; dt))
end
ode_algorithm = OrdinaryDiffEq.Rosenbrock23

additional_callbacks = if !isnothing(radiation_model())
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

import ClimaCore: enable_threading
enable_threading() = parsed_args["enable_threading"]

# TODO: When is_distributed is true, automatically compute the maximum number of
# bytes required to store an element from Y.c or Y.f (or, really, from any Field
# on which gather() or weighted_dss!() will get called). One option is to make a
# non-distributed space, extract the local_geometry type, and find the sizes of
# the output types of center_initial_condition() and face_initial_condition()
# for that local_geometry type. This is rather inefficient, though, so for now
# we will just hardcode the value of 4.
max_field_element_size = 4 # ρ = 1 byte, 𝔼 = 1 byte, uₕ = 2 bytes

center_space, face_space = if parsed_args["config"] == "sphere"
    quad = Spaces.Quadratures.GLL{5}()
    horizontal_mesh = baroclinic_wave_mesh(; params, h_elem = 4)
    h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx)
    z_stretch = Meshes.GeneralizedExponentialStretching(FT(500), FT(5000))
    z_max = FT(30e3)
    z_elem = 10
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
    z_max = FT(70e3)
    z_elem = 70
    z_stretch = Meshes.GeneralizedExponentialStretching(FT(100), FT(10000))
    make_hybrid_spaces(h_space, z_max, z_elem, z_stretch)
end

if haskey(ENV, "RESTART_FILE")
    restart_file_name = ENV["RESTART_FILE"]
    if is_distributed
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

    center_initial_condition = if parsed_args["config"] == "sphere"
        center_initial_condition_sphere
    elseif parsed_args["config"] == "column"
        center_initial_condition_column
    end

    Y = Fields.FieldVector(
        c = center_initial_condition.(
            ᶜlocal_geometry,
            params,
            Ref(energy_form()),
            Ref(moisture_model()),
        ),
        f = face_initial_condition.(ᶠlocal_geometry, params),
    )
end
p = get_cache(Y, params, upwinding_mode(), dt)

# Print tendencies:
for key in keys(p.tendency_knobs)
    @info "`$(key)`:$(getproperty(p.tendency_knobs, key))"
end

if ode_algorithm <: Union{
    OrdinaryDiffEq.OrdinaryDiffEqImplicitAlgorithm,
    OrdinaryDiffEq.OrdinaryDiffEqAdaptiveImplicitAlgorithm,
}
    use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
    W = SchurComplementW(Y, use_transform, jacobian_flags, test_implicit_solver)
    jac_kwargs =
        use_transform ? (; jac_prototype = W, Wfact_t = Wfact!) :
        (; jac_prototype = W, Wfact = Wfact!)

    alg_kwargs = (; linsolve = linsolve!)
    if ode_algorithm <: Union{
        OrdinaryDiffEq.OrdinaryDiffEqNewtonAlgorithm,
        OrdinaryDiffEq.OrdinaryDiffEqNewtonAdaptiveAlgorithm,
    }
        alg_kwargs =
            (; alg_kwargs..., nlsolve = NLNewton(; max_iter = max_newton_iters))
    end
else
    jac_kwargs = alg_kwargs = ()
end

job_id = if isnothing(parsed_args["job_id"])
    job_id_from_parsed_args(s, parsed_args)
else
    parsed_args["job_id"]
end
output_dir = parse_arg(parsed_args, "output_dir", job_id)
@info "Output directory: `$output_dir`"
mkpath(output_dir)

function make_save_to_disk_func(output_dir, is_distributed)
    function save_to_disk_func(integrator)
        day = floor(Int, integrator.t / (60 * 60 * 24))
        sec = Int(mod(integrator.t, 3600 * 24))
        @info "Saving prognostic variables to JLD2 file on day $day second $sec"
        suffix = is_distributed ? "_pid$pid.jld2" : ".jld2"
        output_file = joinpath(output_dir, "day$day.$sec$suffix")
        jldsave(output_file; t = integrator.t, Y = integrator.u)
        return nothing
    end
    return save_to_disk_func
end

save_to_disk_func = make_save_to_disk_func(output_dir, is_distributed)

dss_callback = FunctionCallingCallback(func_start = true) do Y, t, integrator
    p = integrator.p
    Spaces.weighted_dss!(Y.c, p.ghost_buffer.c)
    Spaces.weighted_dss!(Y.f, p.ghost_buffer.f)
end
save_to_disk_callback = if dt_save_to_disk == Inf
    nothing
else
    PeriodicCallback(save_to_disk_func, dt_save_to_disk; initial_affect = true)
end
callback =
    CallbackSet(dss_callback, save_to_disk_callback, additional_callbacks...)

problem = SplitODEProblem(
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

@info "Running job:`$job_id`"
sol = @timev OrdinaryDiffEq.solve!(integrator)

if is_distributed # replace sol.u on the root processor with the global sol.u
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
    end
end

import JSON
using Test
import OrderedCollections
include(joinpath(@__DIR__, "define_post_processing.jl"))
if !is_distributed
    ENV["GKSwstype"] = "nul" # avoid displaying plots
    if is_baro_wave(parsed_args)
        paperplots_baro_wave(sol, output_dir, p, FT(90), FT(180))
    elseif is_column_radiative_equilibrium(parsed_args)
        custom_postprocessing(sol, output_dir)
    elseif forcing_type() isa HeldSuarezForcing && t_end >= (3600 * 24 * 400)
        paperplots_held_suarez(sol, output_dir, p, FT(90), FT(180))
    else
        postprocessing(sol, output_dir, fps)
    end
end

if !is_distributed || ClimaComms.iamroot(comms_ctx)
    include(joinpath(@__DIR__, "..", "..", "post_processing", "mse_tables.jl"))

    if parsed_args["regression_test"]

        Y_last = sol.u[end]
        # This is helpful for starting up new tables
        @info "Job-specific MSE table format:"
        println("all_best_mse[\"$job_id\"] = OrderedCollections.OrderedDict()")
        for prop_chain in Fields.property_chains(Y_last)
            println("all_best_mse[\"$job_id\"][$prop_chain] = 0.0")
        end

        # Extract best mse for this job:
        best_mse = all_best_mse[job_id]

        include(
            joinpath(@__DIR__, "..", "..", "post_processing", "compute_mse.jl"),
        )

        ds_filename_computed = joinpath(output_dir, "prog_state.nc")

        function process_name(s::AbstractString)
            # "c_ρ", "c_ρe", "c_uₕ_1", "c_uₕ_2", "f_w_1"
            s = replace(s, "components_data_" => "")
            s = replace(s, "ₕ" => "_h")
            s = replace(s, "ρ" => "rho")
            return s
        end
        varname(pc::Tuple) = process_name(join(pc, "_"))

        export_nc(Y_last; nc_filename = ds_filename_computed, varname)
        computed_mse = regression_test(;
            job_id,
            reference_mse = best_mse,
            ds_filename_computed,
            varname,
        )

        computed_mse_filename = joinpath(job_id, "computed_mse.json")

        open(computed_mse_filename, "w") do io
            JSON.print(io, computed_mse)
        end
        NCRegressionTests.test_mse(computed_mse, best_mse)
    end

end
