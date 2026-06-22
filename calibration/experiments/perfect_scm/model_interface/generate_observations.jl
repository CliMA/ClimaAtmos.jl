"""
    make_observation_vec(
       interface::PerfectAtmosModelInterface,
       diagnostics_filepath,
       obs_vec_filename;
       overwrite = true,
    )

Make a vector of `EKP.Observation`s with data generated from a `ClimaAtmos`
simulation and save it as a `JLD2` object with the name `obs_vec_filename`.

If `overwrite` is `true`, then this always run. Otherwise, this will not run if
the file already exists.
"""
function make_observation_vec(
    interface::PerfectAtmosModelInterface,
    diagnostics_filepath,
    obs_vec_filename;
    overwrite = true,
)
    if !overwrite && isfile(obs_vec_filename)
        @info "Vector of EKP.Observations at ($obs_vec_filename) will not be overwritten"
        return nothing
    end
    obs_vec = create_ekp_observations(diagnostics_filepath, interface)
    JLD2.save_object(obs_vec_filename, obs_vec)
    return nothing
end

"""
    generate_perfect_model_data(interface::PerfectAtmosModelInterface; tomls = [])

Generate data by running a `ClimaAtmos` simulation with the configuration
at `interface.config`.

The output are saved in `perfect_model_simulation`.

For the `tomls` keyword argument, you can pass a vector of filepaths of TOML
parameter files to override the parameters of the Atmos simulation.
"""
function generate_perfect_model_data(interface::PerfectAtmosModelInterface; tomls = [])
    (; config, output_dir, diagnostic_dicts) = interface

    perfect_model_output_dir = joinpath(output_dir, "perfect_model_simulation")
    isdir(perfect_model_output_dir) || mkpath(perfect_model_output_dir)

    # Overwrite output directory and replace diagnostic_dicts
    config_dict = CA.load_yaml_file(config)
    config_dict["output_dir"] = perfect_model_output_dir
    # We use all the diagnostics found in output_active to determine which
    # diagnostics to use for the observations, so we will not output the default
    # diagnostics
    config_dict["output_default_diagnostics"] = false
    replace_diagnostic_dicts!(config_dict, diagnostic_dicts)

    # Don't save state to speed up creating the observations
    config_dict["dt_save_state_to_disk"] = "Inf"

    # Overwrite parameters
    for toml in tomls
        add_parameter_filepath!(config_dict, toml)
    end

    atmos_config = CA.AtmosConfig(config_dict)
    simulation = CA.get_simulation(atmos_config)
    CA.solve_atmos!(simulation)
    return simulation.output_dir
end

"""
    create_ekp_observations(
        diagnostics_filepath,
        interface::PerfectAtmosModelInterface
    )

Create an `EKP.Observation` using all diagnostics at `diagnostics_filepath`.

Each variable is processed by `preprocess`. See the documentation of that
function to see what kind of processing is done.

The covariance matrix for each variable is the
`1e-11 + (model_error_scale * Statistics.mean(var.data))^2` where the
`model_error_scale = 0.05`.
"""
function create_ekp_observations(
    diagnostics_filepath,
    interface::PerfectAtmosModelInterface,
)
    simdir = ClimaAnalysis.SimDir(diagnostics_filepath)

    # Load all data we want as OutputVar
    vars = []
    for short_name in ClimaAnalysis.available_vars(simdir)
        for reduction in ClimaAnalysis.available_reductions(simdir; short_name)
            for period in ClimaAnalysis.available_periods(simdir; short_name, reduction)
                for coord_type in ClimaAnalysis.available_coord_types(
                    simdir;
                    short_name,
                    reduction,
                    period,
                )
                    push!(vars, get(simdir; short_name, reduction, period, coord_type))
                end
            end
        end
    end

    # Create observations using all the data
    # This is where we would do preprocessing and minibatching before creating the
    # EKP.Observation
    obs_vec = map(vars) do var
        var = preprocess(var, interface)

        model_error_scale = 0.05
        scalar = 1e-11 + (model_error_scale * Statistics.mean(var.data))^2
        @info "Scalar for $(ClimaAnalysis.short_name(var)): $scalar"

        covar_estimator = ClimaCalibrate.ObservationRecipe.ScalarCovariance(;
            scalar,
            use_latitude_weights = false,
        )

        var_dates = ClimaAnalysis.dates(var)
        ClimaCalibrate.ObservationRecipe.observation(
            covar_estimator,
            var,
            first(var_dates),
            last(var_dates),
        )
    end

    # EKP will cycle through the observations again if all observations are
    # exhausted
    # Return a single EKP.Observation since minibatching is not supported
    # right now
    return [EKP.combine_observations(obs_vec)]
end
