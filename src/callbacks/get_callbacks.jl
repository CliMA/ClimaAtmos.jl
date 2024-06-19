import ClimaCore.Utilities: half

function get_diagnostics(parsed_args, atmos_model, Y, p, t_start, dt)

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
    writers = (hdf5_writer, netcdf_writer)

    # The default writer is HDF5
    ALLOWED_WRITERS = Dict(
        "nothing" => netcdf_writer,
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

            period_seconds = FT(time_to_seconds(yaml_diag["period"]))

            if isnothing(output_name)
                output_short_name = CAD.descriptive_short_name(
                    CAD.get_diagnostic_variable(short_name),
                    CAD.EveryDtSchedule(period_seconds; t_start),
                    reduction_time_func,
                    pre_output_hook!,
                )
            end

            if isnothing(reduction_time_func)
                compute_every = CAD.EveryDtSchedule(period_seconds; t_start)
            else
                compute_every = CAD.EveryStepSchedule()
            end

            CAD.ScheduledDiagnostic(
                variable = CAD.get_diagnostic_variable(short_name),
                output_schedule_func = CAD.EveryDtSchedule(
                    period_seconds;
                    t_start,
                ),
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
                t_start,
                time_to_seconds(parsed_args["t_end"]);
                output_writer = netcdf_writer,
            )...,
            diagnostics...,
        ]
    end
    diagnostics = collect(diagnostics)

    for writer in writers
        writer_str = nameof(typeof(writer))
        diags_with_writer =
            filter((x) -> getproperty(x, :output_writer) == writer, diagnostics)
        diags_outputs = [
            getproperty(diag, :output_short_name) for diag in diags_with_writer
        ]
        @info "$writer_str: $diags_outputs"
    end

    function WhenRSUTIsNaN(integrator)
        nlevels = Spaces.nlevels(axes(integrator.u.c))
        return any(
            isnan,
            parent(
                Fields.level(
                    Fields.array2field(
                        integrator.p.radiation.radiation_model.face_sw_flux_up,
                        axes(integrator.u.f),
                    ),
                    nlevels + half,
                ),
            ),
        )
    end

    short_names_debug = ["ta", "hus", "wa"]

    if !isnothing(p.atmos.radiation_mode)
        debug_diagnostics = [
            CAD.ScheduledDiagnostic(
                variable = CAD.get_diagnostic_variable(short_name),
                compute_schedule_func = WhenRSUTIsNaN,
                output_schedule_func = WhenRSUTIsNaN,
                output_writer = netcdf_writer,
            ) for short_name in short_names_debug
        ]
    else
        debug_diagnostics = []
    end

    return [diagnostics..., debug_diagnostics...], writers
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

    if !parsed_args["call_cloud_diagnostics_per_stage"]
        dt_cf = FT(time_to_seconds(parsed_args["dt_cloud_fraction"]))
        callbacks =
            (callbacks..., call_every_dt(cloud_fraction_model_callback!, dt_cf))
    end

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
