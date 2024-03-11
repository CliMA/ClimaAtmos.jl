function get_callbacks(config, sim_info, atmos, params, Y, p, t_start)
    (; parsed_args, comms_ctx) = config
    FT = eltype(params)
    (; dt, output_dir) = sim_info

    callbacks = ()
    if parsed_args["log_progress"] && !sim_info.restart
        @info "Progress logging enabled."
        callbacks = (
            callbacks...,
            call_every_n_steps(
                (integrator) -> print_walltime_estimate(integrator);
                skip_first = true,
            ),
        )
    end
    callbacks = (
        callbacks...,
        call_every_n_steps(
            terminate!;
            skip_first = true,
            condition = (u, t, integrator) ->
                maybe_graceful_exit(integrator),
        ),
    )

    dt_save_state_to_disk =
        time_to_seconds(parsed_args["dt_save_state_to_disk"])
    if !(dt_save_state_to_disk == Inf)
        callbacks = (
            callbacks...,
            call_every_dt(
                (integrator) ->
                    save_state_to_disk_func(integrator, output_dir),
                dt_save_state_to_disk;
                skip_first = sim_info.restart,
            ),
        )
    end

    if is_distributed(comms_ctx)
        callbacks = (
            callbacks...,
            call_every_n_steps(
                gc_func,
                parse(Int, get(ENV, "CLIMAATMOS_GC_NSTEPS", "1000")),
                skip_first = true,
            ),
        )
    end

    if parsed_args["check_conservation"]
        callbacks = (
            callbacks...,
            call_every_n_steps(
                flux_accumulation!;
                skip_first = true,
                call_at_end = true,
            ),
        )
    end

    dt_cf = FT(time_to_seconds(parsed_args["dt_cloud_fraction"]))
    callbacks =
        (callbacks..., call_every_dt(cloud_fraction_model_callback!, dt_cf))

    if atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        # TODO: better if-else criteria?
        dt_rad = if parsed_args["config"] == "column"
            dt
        else
            FT(time_to_seconds(parsed_args["dt_rad"]))
        end
        callbacks =
            (callbacks..., call_every_dt(rrtmgp_model_callback!, dt_rad))
    end

    return callbacks
end
