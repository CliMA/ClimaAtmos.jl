"""
    ClimaCalibrate.observation_map(interface::PerfectAtmosModelInterface, iteration)

Transform the simulation outputs into the G ensemble matrix as expected by
`EnsembleKalmanProcesses`.
"""
function ClimaCalibrate.observation_map(interface::PerfectAtmosModelInterface, iteration)
    (; output_dir) = interface
    ekp = JLD2.load_object(ClimaCalibrate.ekp_path(output_dir, iteration))
    ensemble_size = EKP.get_N_ens(ekp)

    g_ens_builder = EnsembleBuilder.GEnsembleBuilder(ekp)
    for m in 1:ensemble_size
        try
            member_path =
                ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)
            diagnostics_path = joinpath(member_path, "output_active")
            @info "Processing member $m: $diagnostics_path"
            process_member_data!(g_ens_builder, m, diagnostics_path, interface)
        catch e
            @error "Error processing member $m, filling observation map entry with NaNs" exception =
                e
            EnsembleBuilder.fill_g_ens_col!(g_ens_builder, m, NaN)
        end
    end

    if EnsembleBuilder.is_complete(g_ens_builder)
        return EnsembleBuilder.get_g_ensemble(g_ens_builder)
    else
        @error "G ensemble matrix is not completed. You may find it useful to call `EnsembleBuilder.missing_short_names(g_ens_builder, 1) or display the GEnsembleBuilder object in the REPL"
    end
end

"""
    process_member_data!(g_ens_builder, m, diagnostics_path, diagnostic_dicts)

Load the relevant `OutputVar`s from the simulation outputs at `diagnostic_path`,
preprocess each `OutputVar`, and fill out the `m`th column of the G ensemble
matrix using `g_ens_builder`.

The variables to load as `OutputVar`s are determined by examining the
diagnostics created in the perfect model simulation.
"""
function process_member_data!(g_ens_builder, m, diagnostics_path, interface)
    (; output_dir) = interface

    # We assume that all the diagnostics created by a perfect model simulation
    # is used for the observational data
    perfect_model_output_dir =
        joinpath(output_dir, "perfect_model_simulation", "output_active")
    perfect_model_simdir = ClimaAnalysis.SimDir(perfect_model_output_dir)
    simdir = ClimaAnalysis.SimDir(diagnostics_path)

    # Load the simulation data as OutputVar
    vars = []
    for short_name in ClimaAnalysis.available_vars(perfect_model_simdir)
        for reduction in
            ClimaAnalysis.available_reductions(perfect_model_simdir; short_name)
            for period in
                ClimaAnalysis.available_periods(perfect_model_simdir; short_name, reduction)
                for coord_type in ClimaAnalysis.available_coord_types(
                    perfect_model_simdir;
                    short_name,
                    reduction,
                    period,
                )
                    push!(vars, get(simdir; short_name, reduction, period, coord_type))
                end
            end
        end
    end

    # This check should be used, because fill_g_ens_col! is not aware of the
    # meaning of the time dimension (e.g. seasonal averages vs monthly
    # averages). For example, without this check, if the simulation data contain
    # monthly averages and metadata track seasonal averages, then no error is
    # thrown, because all dates in metadata are in all the dates in var.
    seq_indices_checker = Checker.SequentialIndicesChecker()
    # Ensure that the proportion of positve values in the observational data and
    # simulation data are not different by 0.9
    sign_checker = ClimaCalibrate.Checker.SignChecker(0.9)
    checkers = (seq_indices_checker, sign_checker)

    for var in vars
        var = preprocess(var, interface)
        EnsembleBuilder.fill_g_ens_col!(
            g_ens_builder,
            m,
            var;
            checkers,
            verbose = true,
        )
    end
    return nothing
end
