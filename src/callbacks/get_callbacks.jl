function get_diagnostics(parsed_args, atmos_model, Y, p, sim_info, output_dir)

    (; dt, t_start, start_date) = sim_info

    FT = Spaces.undertype(axes(Y.c))
    context = ClimaComms.context(axes(Y.c))

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
    hdf5_writer = CAD.HDF5Writer(output_dir)

    if !isnothing(parsed_args["netcdf_interpolation_num_points"])
        num_netcdf_points =
            tuple(parsed_args["netcdf_interpolation_num_points"]...)
    else
        # From the given space, calculate the number of diagnostic grid points if not
        # specified by the user
        num_netcdf_points = default_netcdf_points(axes(Y.c), parsed_args)
    end

    z_sampling_method =
        parsed_args["netcdf_output_at_levels"] ? CAD.LevelsMethod() :
        CAD.FakePressureLevelsMethod()

    # The start_date keyword was added in v0.2.9. For prior versions, the diagnostics will
    # not contain the date
    maybe_add_start_date =
        pkgversion(CAD.ClimaDiagnostics) >= v"0.2.9" ? (; start_date) : (;)

    netcdf_writer = CAD.NetCDFWriter(
        axes(Y.c),
        output_dir,
        num_points = num_netcdf_points;
        z_sampling_method,
        sync_schedule = CAD.EveryStepSchedule(),
        maybe_add_start_date...,
    )
    writers = (dict_writer, hdf5_writer, netcdf_writer)

    # The default writer is netcdf
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
            else
                period_seconds = FT(time_to_seconds(period_str))
                period_dates =
                    CA.promote_period.(Dates.Second(period_seconds))
            end

            output_schedule = CAD.EveryCalendarDtSchedule(
                period_dates;
                reference_date = start_date,
            )
            compute_schedule = CAD.EveryCalendarDtSchedule(
                period_dates;
                reference_date = start_date,
            )

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
                sim_info.dt isa ITime ?
                ITime(time_to_seconds(parsed_args["t_end"])) - t_start :
                FT(time_to_seconds(parsed_args["t_end"]) - t_start),
                start_date;
                output_writer = netcdf_writer,
            )...,
            diagnostics...,
        ]
    end
    diagnostics = collect(diagnostics)

    periods_reductions = Set()
    for diag in diagnostics
        isa_reduction = !isnothing(diag.reduction_time_func)
        isa_reduction || continue

        if diag.output_schedule_func isa CAD.EveryDtSchedule
            period = Dates.Second(diag.output_schedule_func.dt)
        elseif diag.output_schedule_func isa CAD.EveryCalendarDtSchedule
            period = diag.output_schedule_func.dt
        else
            continue
        end

        push!(periods_reductions, period)
    end

    periods_str = join(CA.promote_period.(periods_reductions), ", ")
    @info "Saving accumulated diagnostics to disk with frequency: $(periods_str)"

    for writer in writers
        writer_str = nameof(typeof(writer))
        diags_with_writer =
            filter((x) -> getproperty(x, :output_writer) == writer, diagnostics)
        diags_outputs = [
            getproperty(diag, :output_short_name) for diag in diags_with_writer
        ]
        @info "$writer_str: $diags_outputs"
    end

    return diagnostics, writers, periods_reductions
end

function checkpoint_frequency_from_parsed_args(dt_save_state_to_disk::String)
    if occursin("months", dt_save_state_to_disk)
        months = match(r"^(\d+)months$", dt_save_state_to_disk)
        isnothing(months) && error(
            "$(period_str) has to be of the form <NUM>months, e.g. 2months for 2 months",
        )
        return Dates.Month(parse(Int, first(months)))
    else
        dt_save_state_to_disk = time_to_seconds(dt_save_state_to_disk)
        if !(dt_save_state_to_disk == Inf)
            # We use Millisecond to support fractional seconds, eg. 0.1
            return Dates.Millisecond(1000dt_save_state_to_disk)
        else
            return Inf
        end
    end
end


function get_callbacks(config, sim_info, atmos, params, Y, p)
    (; parsed_args, comms_ctx) = config
    FT = eltype(params)
    (; dt, output_dir, start_date, t_start, t_end) = sim_info

    callbacks = ()
    if parsed_args["log_progress"]
        @info "Progress logging enabled"
        walltime_info = WallTimeInfo()
        tot_steps = ceil(Int, (t_end - t_start) / dt)
        five_percent_steps = ceil(Int, 0.05 * tot_steps)
        cond = let schedule = CappedGeometricSeriesSchedule(five_percent_steps)
            (u, t, integrator) -> schedule(integrator)
        end
        affect! = let wt = walltime_info
            (integrator) -> report_walltime(wt, integrator)
        end
        callbacks = (callbacks..., SciMLBase.DiscreteCallback(cond, affect!))
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
            condition = let output_dir = output_dir
                (u, t, integrator) ->
                    maybe_graceful_exit(output_dir, integrator)
            end,
        ),
    )

    # Save dt_save_state_to_disk as a Dates.Period object. This is used to check
    # if it is an integer multiple of other frequencies.
    dt_save_state_to_disk_dates = checkpoint_frequency_from_parsed_args(
        parsed_args["dt_save_state_to_disk"],
    )
    if dt_save_state_to_disk_dates != Inf
        schedule = CAD.EveryCalendarDtSchedule(
            dt_save_state_to_disk_dates;
            reference_date = start_date,
            date_last = t_start isa ITime ?
                        ClimaUtilities.TimeManager.date(t_start) :
                        start_date + Dates.Second(t_start),
        )
        cond = let schedule = schedule
            (u, t, integrator) -> schedule(integrator)
        end
        affect! = let output_dir = output_dir
            (integrator) -> save_state_to_disk_func(integrator, output_dir)
        end
        callbacks = (callbacks..., SciMLBase.DiscreteCallback(cond, affect!))
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

    if parsed_args["external_forcing"] in
       ["ReanalysisTimeVarying", "ReanalysisMonthlyAveragedDiurnal"] &&
       parsed_args["config"] == "column"
        callbacks = (
            callbacks...,
            call_every_n_steps(
                external_driven_single_column!;
                call_at_end = true,
            ),
        )
    end

    if !parsed_args["call_cloud_diagnostics_per_stage"]
        dt_cf =
            dt isa ITime ?
            ITime(time_to_seconds(parsed_args["dt_cloud_fraction"])) :
            FT(time_to_seconds(parsed_args["dt_cloud_fraction"]))
        dt_cf, _, _, _ = promote(dt_cf, t_start, dt, t_end)
        callbacks =
            (callbacks..., call_every_dt(cloud_fraction_model_callback!, dt_cf))
    end

    if atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        dt_rad =
            dt isa ITime ? ITime(time_to_seconds(parsed_args["dt_rad"])) :
            FT(time_to_seconds(parsed_args["dt_rad"]))
        dt_rad, _, _, _ = promote(dt_rad, t_start, dt, t_end)
        # We use Millisecond to support fractional seconds, eg. 0.1
        dt_rad_ms = Dates.Millisecond(1_000 * float(dt_rad))
        if parsed_args["dt_save_state_to_disk"] != "Inf" &&
           !CA.isdivisible(dt_save_state_to_disk_dates, dt_rad_ms)
            @warn "Radiation period ($(dt_rad_ms)) is not an even divisor of the checkpoint frequency ($dt_save_state_to_disk_dates)"
            @warn "This simulation will not be reproducible when restarted"
        end

        callbacks =
            (callbacks..., call_every_dt(rrtmgp_model_callback!, dt_rad))
    end

    if atmos.non_orographic_gravity_wave isa NonOrographicGravityWave
        dt_nogw =
            dt isa ITime ? ITime(time_to_seconds(parsed_args["dt_nogw"])) :
            FT(time_to_seconds(parsed_args["dt_nogw"]))
        dt_nogw, _, _, _ = promote(dt_nogw, t_start, dt, sim_info.t_end)
        # We use Millisecond to support fractional seconds, eg. 0.1
        dt_nogw_ms = Dates.Millisecond(1_000 * float(dt_nogw))
        if parsed_args["dt_save_state_to_disk"] != "Inf" &&
           !CA.isdivisible(dt_save_state_to_disk_dates, dt_nogw_ms)
            @warn "Non-orographic gravity wave period ($(dt_nogw_ms)) is not an even divisor of the checkpoint frequency ($dt_save_state_to_disk_dates)"
            @warn "This simulation will not be reproducible when restarted"
        end

        callbacks = (callbacks..., call_every_dt(nogw_model_callback!, dt_nogw))
    end

    if atmos.orographic_gravity_wave isa OrographicGravityWave
        dt_ogw =
            dt isa ITime ? ITime(time_to_seconds(parsed_args["dt_ogw"])) :
            FT(time_to_seconds(parsed_args["dt_ogw"]))
        dt_ogw, _, _, _ = promote(dt_ogw, t_start, dt, sim_info.t_end)
        # We use Millisecond to support fractional seconds, eg. 0.1
        dt_ogw_ms = Dates.Millisecond(1_000 * float(dt_ogw))
        if parsed_args["dt_save_state_to_disk"] != "Inf" &&
           !CA.isdivisible(dt_save_state_to_disk_dates, dt_ogw_ms)
            @warn "Orographic gravity wave period ($(dt_ogw_ms)) is not an even divisor of the checkpoint frequency ($dt_save_state_to_disk_dates)"
            @warn "This simulation will not be reproducible when restarted"
        end

        callbacks = (callbacks..., call_every_dt(ogw_model_callback!, dt_ogw))
    end

    return callbacks
end
