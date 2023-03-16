using Dates: DateTime, @dateformat_str
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import OrdinaryDiffEq as ODE

function get_simulation(::Type{FT}, parsed_args) where {FT}

    job_id = if isnothing(parsed_args["job_id"])
        (s, default_parsed_args) = parse_commandline()
        job_id_from_parsed_args(s, parsed_args)
    else
        parsed_args["job_id"]
    end
    default_output = haskey(ENV, "CI") ? job_id : joinpath("output", job_id)
    output_dir = parse_arg(parsed_args, "output_dir", default_output)
    mkpath(output_dir)

    sim = (;
        is_distributed = haskey(ENV, "CLIMACORE_DISTRIBUTED"),
        is_debugging_tc = parsed_args["debugging_tc"],
        output_dir,
        restart = haskey(ENV, "RESTART_FILE"),
        job_id,
        dt = FT(CA.time_to_seconds(parsed_args["dt"])),
        start_date = DateTime(parsed_args["start_date"], dateformat"yyyymmdd"),
        t_end = FT(CA.time_to_seconds(parsed_args["t_end"])),
    )
    n_steps = floor(Int, sim.t_end / sim.dt)
    @info(
        "Time info:",
        dt = parsed_args["dt"],
        t_end = parsed_args["t_end"],
        floor_n_steps = n_steps,
    )

    return sim
end

function args_integrator(parsed_args, Y, p, tspan, ode_algo, callback)
    (; atmos, simulation) = p
    (; dt) = simulation
    dt_save_to_sol = CA.time_to_seconds(parsed_args["dt_save_to_sol"])

    @time "Define ode function" func = if parsed_args["split_ode"]
        implicit_func = ODE.ODEFunction(
            implicit_tendency!;
            CA.jac_kwargs(ode_algo, Y, atmos.energy_form)...,
            tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= 0),
        )
        if CA.is_cts_algo(ode_algo)
            CTS.ClimaODEFunction(;
                T_lim! = CA.horizontal_limiter_tendency!,
                T_exp! = remaining_tendency!,
                T_imp! = implicit_func,
                # Can we just pass implicit_tendency! and jac_prototype etc.?
                lim! = CA.limiters_func!,
                CA.dss!,
            )
        else
            ODE.SplitFunction(implicit_func, remaining_tendency!)
        end
    else
        remaining_tendency! # should be total_tendency!
    end
    problem = ODE.ODEProblem(func, Y, tspan, p)
    saveat = if dt_save_to_sol == Inf
        tspan[2]
    elseif tspan[2] % dt_save_to_sol == 0
        dt_save_to_sol
    else
        [tspan[1]:dt_save_to_sol:tspan[2]..., tspan[2]]
    end # ensure that tspan[2] is always saved
    args = (problem, ode_algo)
    kwargs =
        (; saveat, callback, dt, CA.additional_integrator_kwargs(ode_algo)...)
    return (args, kwargs)
end
