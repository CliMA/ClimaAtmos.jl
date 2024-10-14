function get_diagnostics(parsed_args, atmos_model, Y, p, dt, t_start)

    FT = Spaces.undertype(axes(Y.c))

    # We either get the diagnostics section in the YAML file, or we return an empty list
    # (which will result in an empty list being created by the map below)
    yaml_diagnostics = get(parsed_args, "diagnostics", [])

    # ALLOWED_REDUCTIONS is the collection of reductions we support. The keys are the
    # strings that have to be provided in the YAML file. The values are tuples with the
    # function that has to be passed to reduction_time_func and the one that has to passed
    # to pre_output_hook!

    # We make "nothing" a string so that we can accept also the word "nothing", in addition
    # to the absence of the value
    #
    # NOTE: Everything has to be lowercase in ALLOWED_REDUCTIONS (so that we can match
    # "max" and "Max")
    ALLOWED_REDUCTIONS = Dict(
        "nothing" => (nothing, nothing), # nothing is: just dump the variable
        "max" => (max, nothing),
        "min" => (min, nothing),
        "average" => ((+), CAD.average_pre_output_hook!),
    )

    dict_writer = CAD.DictWriter()
    hdf5_writer = CAD.HDF5Writer(p.output_dir)

    if !isnothing(parsed_args["netcdf_interpolation_num_points"])
        num_netcdf_points =
            tuple(parsed_args["netcdf_interpolation_num_points"]...)
    else
        # TODO: Once https://github.com/CliMA/ClimaCore.jl/pull/1567 is merged,
        # dispatch over the Grid type
        num_netcdf_points = (180, 90, 50)
    end

    z_sampling_method =
        parsed_args["netcdf_output_at_levels"] ? CAD.LevelsMethod() :
        CAD.FakePressureLevelsMethod()

    netcdf_writer = CAD.NetCDFWriter(
        axes(Y.c),
        p.output_dir,
        num_points = num_netcdf_points;
        z_sampling_method,
        sync_schedule = CAD.EveryStepSchedule(),
    )

    writers = (; dict_writer, hdf5_writer, netcdf_writer)

    # The default writer is NetCDF
    ALLOWED_WRITERS = Dict(
        "nothing" => netcdf_writer,
        "dict" => dict_writer,
        "h5" => hdf5_writer,
        "hdf5" => hdf5_writer,
        "nc" => netcdf_writer,
        "netcdf" => netcdf_writer,
    )

    diagnostics_ragged = map(yaml_diagnostics) do yaml_diag
        short_names = yaml_diag["short_name"]
        output_name = get(yaml_diag, "output_name", nothing)

        if short_names isa Vector
            isnothing(output_name) || error(
                "Diagnostics: cannot have multiple short_names while specifying output_name",
            )
        else
            short_names = [short_names]
        end

        map(short_names) do short_name
            # Return "nothing" if "reduction_time" is not in the YAML block
            #
            # We also normalize everything to lowercase, so that can accept "max" but
            # also "Max"
            reduction_time_yaml =
                lowercase(get(yaml_diag, "reduction_time", "nothing"))

            if !haskey(ALLOWED_REDUCTIONS, reduction_time_yaml)
                error("reduction $reduction_time_yaml not implemented")
            else
                reduction_time_func, pre_output_hook! =
                    ALLOWED_REDUCTIONS[reduction_time_yaml]
            end

            writer_ext = lowercase(get(yaml_diag, "writer", "nothing"))

            if !haskey(ALLOWED_WRITERS, writer_ext)
                error("writer $writer_ext not implemented")
            else
                writer = ALLOWED_WRITERS[writer_ext]
            end

            haskey(yaml_diag, "period") ||
                error("period keyword required for diagnostics")

            period_str = yaml_diag["period"]

            if occursin("months", period_str)
                months = match(r"^(\d+)months$", period_str)
                isnothing(months) && error(
                    "$(period_str) has to be of the form <NUM>months, e.g. 2months for 2 months",
                )
                period_dates = Dates.Month(parse(Int, first(months)))
                output_schedule = CAD.EveryCalendarDtSchedule(
                    period_dates;
                    reference_date = p.start_date,
                )
                compute_schedule = CAD.EveryCalendarDtSchedule(
                    period_dates;
                    reference_date = p.start_date,
                )
            else
                period_seconds = FT(time_to_seconds(period_str))
                output_schedule = CAD.EveryDtSchedule(period_seconds)
                compute_schedule = CAD.EveryDtSchedule(period_seconds)
            end

            if isnothing(output_name)
                output_short_name = CAD.descriptive_short_name(
                    CAD.get_diagnostic_variable(short_name),
                    output_schedule,
                    reduction_time_func,
                    pre_output_hook!,
                )
            end

            if isnothing(reduction_time_func)
                compute_every = compute_schedule
            else
                compute_every = CAD.EveryStepSchedule()
            end

            CAD.ScheduledDiagnostic(
                variable = CAD.get_diagnostic_variable(short_name),
                output_schedule_func = output_schedule,
                compute_schedule_func = compute_every,
                reduction_time_func = reduction_time_func,
                pre_output_hook! = pre_output_hook!,
                output_writer = writer,
                output_short_name = output_short_name,
            )
        end
    end

    # Flatten the array of arrays of diagnostics
    diagnostics = vcat(diagnostics_ragged...)

    if parsed_args["output_default_diagnostics"]
        diagnostics = [
            CAD.default_diagnostics(
                atmos_model,
                time_to_seconds(parsed_args["t_end"]) - t_start,
                p.start_date;
                output_writer = netcdf_writer,
            )...,
            diagnostics...,
        ]
    end
    if parsed_args["output_default_diagnostics"] && (
        parsed_args["use_exact_jacobian"] ||
        parsed_args["debug_approximate_jacobian"]
    )
        period_seconds =
            min(
                max(
                    1,
                    parsed_args["n_steps_update_exact_jacobian"],
                    fld(p.t_end / 9, p.dt), # Save no more than 10 matrices.
                ),
                fld(p.t_end, p.dt), # Save at least 2 matrices.
            ) * p.dt
        schedule = CAD.EveryDtSchedule(period_seconds)
        exact_jacobian_diagnostic = CAD.ScheduledDiagnostic(;
            variable = CAD.get_diagnostic_variable("ejac1"),
            output_schedule_func = schedule,
            compute_schedule_func = deepcopy(schedule),
            output_writer = dict_writer,
        )
        diagnostics = [diagnostics..., exact_jacobian_diagnostic]
        if parsed_args["debug_approximate_jacobian"]
            approx_jacobian_diagnostic = CAD.ScheduledDiagnostic(;
                variable = CAD.get_diagnostic_variable("ajac1"),
                output_schedule_func = deepcopy(schedule),
                compute_schedule_func = deepcopy(schedule),
                output_writer = dict_writer,
            )
            approx_jacobian_error_diagnostic = CAD.ScheduledDiagnostic(;
                variable = CAD.get_diagnostic_variable("ajacerr1"),
                output_schedule_func = deepcopy(schedule),
                compute_schedule_func = deepcopy(schedule),
                output_writer = dict_writer,
            )
            diagnostics = [
                diagnostics...,
                approx_jacobian_diagnostic,
                approx_jacobian_error_diagnostic,
            ]
        end
    end

    for writer in writers
        writer_str = nameof(typeof(writer))
        diags_with_writer =
            filter((x) -> getproperty(x, :output_writer) == writer, diagnostics)
        diags_outputs = [
            getproperty(diag, :output_short_name) for diag in diags_with_writer
        ]
        @info "$writer_str: $diags_outputs"
    end

    return diagnostics, writers
end

function get_callbacks(config, sim_info, atmos, params, Y, p, t_start)
    (; parsed_args, comms_ctx) = config
    FT = eltype(params)
    (; dt, output_dir) = sim_info

    callbacks = ()
    if parsed_args["log_progress"]
        @info "Progress logging enabled."
        callbacks = (
            callbacks...,
            call_every_n_steps(
                (integrator) -> print_walltime_estimate(integrator);
                skip_first = true,
            ),
        )
    end
    check_nan_every = parsed_args["check_nan_every"]
    if check_nan_every > 0
        @info "Checking NaNs in the state every $(check_nan_every) steps"
        callbacks = (
            callbacks...,
            call_every_n_steps(
                (integrator) -> check_nans(integrator),
                check_nan_every,
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

    # Save dt_save_state_to_disk as a Dates.Period object. This is used to check
    # if it is an integer multiple of other frequencies.
    dt_save_state_to_disk_dates = Dates.today() # Value will be overwritten
    if occursin("months", parsed_args["dt_save_state_to_disk"])
        months = match(r"^(\d+)months$", parsed_args["dt_save_state_to_disk"])
        isnothing(months) && error(
            "$(period_str) has to be of the form <NUM>months, e.g. 2months for 2 months",
        )
        dt_save_state_to_disk_dates = Dates.Month(parse(Int, first(months)))
        schedule = CAD.EveryCalendarDtSchedule(
            dt_save_state_to_disk_dates;
            reference_date = p.start_date,
            date_last = p.start_date + Dates.Second(t_start),
        )
        cond = let schedule = schedule
            (u, t, integrator) -> schedule(integrator)
        end
        affect! = let output_dir = output_dir
            (integrator) -> save_state_to_disk_func(integrator, output_dir)
        end
        callbacks = (callbacks..., SciMLBase.DiscreteCallback(cond, affect!))
    else
        dt_save_state_to_disk =
            time_to_seconds(parsed_args["dt_save_state_to_disk"])
        if !(dt_save_state_to_disk == Inf)
            # We use Millisecond to support fractional seconds, eg. 0.1
            dt_save_state_to_disk_dates =
                Dates.Millisecond(dt_save_state_to_disk)
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

    if !parsed_args["call_cloud_diagnostics_per_stage"]
        dt_cf = FT(time_to_seconds(parsed_args["dt_cloud_fraction"]))
        callbacks =
            (callbacks..., call_every_dt(cloud_fraction_model_callback!, dt_cf))
    end

    if (
        parsed_args["use_exact_jacobian"] ||
        parsed_args["debug_approximate_jacobian"]
    ) && parsed_args["n_steps_update_exact_jacobian"] != 0
        callbacks = (
            callbacks...,
            call_every_n_steps(
                update_exact_jacobian!,
                parsed_args["n_steps_update_exact_jacobian"],
            ),
        )
    end

    if atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        dt_rad = FT(time_to_seconds(parsed_args["dt_rad"]))
        # We use Millisecond to support fractional seconds, eg. 0.1
        dt_rad_ms = Dates.Millisecond(dt_rad)
        if parsed_args["dt_save_state_to_disk"] != "Inf" &&
           !CA.isdivisible(dt_save_state_to_disk_dates, dt_rad_ms)
            @warn "Radiation period ($(dt_rad_ms)) is not an even divisor of the checkpoint frequency ($dt_save_state_to_disk_dates)"
            @warn "This simulation will not be reproducible when restarted"
        end

        callbacks =
            (callbacks..., call_every_dt(rrtmgp_model_callback!, dt_rad))
    end

    return callbacks
end
