function init_diagnostics!(
    diagnostics_iterations,
    diagnostic_storage,
    diagnostic_accumulators,
    diagnostic_counters,
    output_dir,
    Y,
    p,
    t;
    warn_allocations,
)
    for diag in diagnostics_iterations
        variable = diag.variable
        try
            # The first time we call compute! we use its return value. All
            # the subsequent times (in the callbacks), we will write the
            # result in place
            diagnostic_storage[diag] = variable.compute!(nothing, Y, p, t)
            diagnostic_counters[diag] = 1
            # If it is not a reduction, call the output writer as well
            if isnothing(diag.reduction_time_func)
                writer = diag.output_writer
                CAD.write_field!(
                    writer,
                    diagnostic_storage[diag],
                    diag,
                    Y,
                    p,
                    t,
                    output_dir,
                )
                if writer isa CAD.NetCDFWriter &&
                   ClimaComms.iamroot(ClimaComms.context(Y.c))
                    output_path = CAD.outpath_name(output_dir, diag)
                    NCDatasets.sync(writer.open_files[output_path])
                end
            else
                # Add to the accumulator

                # We use similar + .= instead of copy because CUDA 5.2 does
                # not supported nested wrappers with view(reshape(view))
                # objects. See discussion in
                # https://github.com/CliMA/ClimaAtmos.jl/pull/2579 and
                # https://github.com/JuliaGPU/Adapt.jl/issues/21
                diagnostic_accumulators[diag] =
                    similar(diagnostic_storage[diag])
                diagnostic_accumulators[diag] .= diagnostic_storage[diag]
            end
        catch e
            error("Could not compute diagnostic $(variable.long_name): $e")
        end
    end
    if warn_allocations
        for diag in diagnostics_iterations
            # We need to hoist these variables/functions to avoid measuring
            # allocations due to these variables/functions not being type-stable.
            dstorage = diagnostic_storage[diag]
            compute! = diag.variable.compute!
            # We write over the storage space we have already prepared (and filled) before
            allocs = @allocated compute!(dstorage, Y, p, t)
            if allocs > 10 * 1024
                @warn "Diagnostics $(diag.output_short_name) allocates $allocs bytes"
            end
        end
    end
end

function get_diagnostics(parsed_args, atmos_model, cspace)

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

    hdf5_writer = CAD.HDF5Writer()

    if !isnothing(parsed_args["netcdf_interpolation_num_points"])
        num_netcdf_points =
            tuple(parsed_args["netcdf_interpolation_num_points"]...)
    else
        # TODO: Once https://github.com/CliMA/ClimaCore.jl/pull/1567 is merged,
        # dispatch over the Grid type
        num_netcdf_points = (180, 90, 50)
    end

    netcdf_writer = CAD.NetCDFWriter(;
        cspace,
        num_points = num_netcdf_points,
        disable_vertical_interpolation = parsed_args["netcdf_output_at_levels"],
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

            period_seconds = time_to_seconds(yaml_diag["period"])

            if isnothing(output_name)
                output_short_name = CAD.descriptive_short_name(
                    CAD.get_diagnostic_variable(short_name),
                    period_seconds,
                    reduction_time_func,
                    pre_output_hook!,
                )
            end

            if isnothing(reduction_time_func)
                compute_every = period_seconds
            else
                compute_every = :timestep
            end

            CAD.ScheduledDiagnosticTime(
                variable = CAD.get_diagnostic_variable(short_name),
                output_every = period_seconds,
                compute_every = compute_every,
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

    # stochastic 
    if  parsed_args["edmfx_entr_model"] == "StochasticEntrainmentExponentialSolver"
        callbacks = (callbacks..., call_every_n_steps(set_stochastic_entrainment_with_exponential_solver!))
    end
    if parsed_args["edmfx_detr_model"] == "StochasticDetrainmentExponentialSolver"
        callbacks = (callbacks..., call_every_n_steps(set_stochastic_detrainment_with_exponential_solver!))
    end

    return callbacks
end
