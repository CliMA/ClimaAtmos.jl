import ClimaAtmos.RRTMGPInterface as RRTMGPI
import LinearAlgebra
import ClimaCore.Fields
import OrdinaryDiffEq as ODE
import DiffEqCallbacks as DEQ

function get_callbacks(parsed_args, simulation, atmos, params)
    FT = eltype(params)
    (; dt) = simulation

    tc_callbacks =
        CA.call_every_n_steps(turb_conv_affect_filter!; skip_first = true)
    flux_accumulation_callback = CA.call_every_n_steps(
        CA.flux_accumulation!;
        skip_first = true,
        call_at_end = true,
    )

    additional_callbacks =
        if atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode
            # TODO: better if-else criteria?
            dt_rad = if parsed_args["config"] == "column"
                dt
            else
                FT(CA.time_to_seconds(parsed_args["dt_rad"]))
            end
            (CA.call_every_dt(CA.rrtmgp_model_callback!, dt_rad),)
        else
            ()
        end

    if p.atmos.turbconv_model isa TC.EDMFModel
        additional_callbacks = (additional_callbacks..., tc_callbacks)
    end

    if parsed_args["check_conservation"]
        additional_callbacks =
            (flux_accumulation_callback, additional_callbacks...)
    end

    dt_save_to_disk = CA.time_to_seconds(parsed_args["dt_save_to_disk"])
    dt_save_restart = CA.time_to_seconds(parsed_args["dt_save_restart"])

    dss_cb = if startswith(parsed_args["ode_algo"], "ODE.")
        CA.call_every_n_steps(CA.dss_callback)
    else
        nothing
    end
    save_to_disk_callback = if dt_save_to_disk == Inf
        nothing
    elseif simulation.restart
        CA.call_every_dt(CA.save_to_disk_func, dt_save_to_disk; skip_first = true)
    else
        CA.call_every_dt(CA.save_to_disk_func, dt_save_to_disk)
    end

    save_restart_callback = if dt_save_restart == Inf
        nothing
    else
        CA.call_every_dt(CA.save_restart_func, dt_save_restart)
    end

    gc_callback = if simulation.is_distributed
        CA.call_every_n_steps(
            CA.gc_func,
            parse(Int, get(ENV, "CLIMAATMOS_GC_NSTEPS", "1000")),
            skip_first = true,
        )
    else
        nothing
    end

    return ODE.CallbackSet(
        dss_cb,
        save_to_disk_callback,
        save_restart_callback,
        gc_callback,
        additional_callbacks...,
    )
end

function turb_conv_affect_filter!(integrator)
    p = integrator.p
    (; edmf_cache) = p
    (; edmf, param_set, surf_params) = edmf_cache
    t = integrator.t
    Y = integrator.u
    tc_params = CAP.turbconv_params(param_set)

    CA.set_precomputed_quantities!(Y, p, t) # sets ᶜts for set_edmf_surface_bc
    Fields.bycolumn(axes(Y.c)) do colidx
        state = TC.tc_column_state(Y, p, nothing, colidx)
        grid = TC.Grid(state)
        surf = CA.get_surface(
            p.atmos.model_config,
            surf_params,
            grid,
            state,
            t,
            tc_params,
        )
        TC.affect_filter!(edmf, grid, state, tc_params, surf, t)
    end

    # We're lying to OrdinaryDiffEq.jl, in order to avoid
    # paying for an additional `∑tendencies!` call, which is required
    # to support supplying a continuous representation of the
    # solution.
    ODE.u_modified!(integrator, false)
end
