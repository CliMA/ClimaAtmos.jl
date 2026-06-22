"""
    ClimaCalibrate.forward_model(interface::PerfectAtmosModelInterface, iter, member)

Run the forward model using the configuration at `interface.config`.

This is ran in parallel depending on the backend used.

Only the diagnostics for calibration will be produced and no checkpoints are
saved.
"""
function ClimaCalibrate.forward_model(interface::PerfectAtmosModelInterface, iter, member)
    Random.seed!(1234)
    (; config, output_dir, diagnostic_dicts) = interface

    config_dict = CA.load_yaml_file(config)

    # Override output directory
    member_output_dir =
        ClimaCalibrate.path_to_ensemble_member(output_dir, iter, member)
    config_dict["output_dir"] = member_output_dir

    # Only create diagnostics that are needed for calibration
    config_dict["output_default_diagnostics"] = false
    replace_diagnostic_dicts!(config_dict, diagnostic_dicts)

    # Don't save state to speed up forward model
    config_dict["dt_save_state_to_disk"] = "Inf"

    sampled_parameter_file = ClimaCalibrate.parameter_path(output_dir, iter, member)
    add_parameter_filepath!(config_dict, sampled_parameter_file)

    atmos_config = CA.AtmosConfig(config_dict)
    simulation = CA.get_simulation(atmos_config)
    CA.solve_atmos!(simulation)
    return nothing
end
