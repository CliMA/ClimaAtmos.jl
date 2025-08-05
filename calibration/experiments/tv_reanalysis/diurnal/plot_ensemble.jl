import EnsembleKalmanProcesses: TOMLInterface
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
import ClimaCalibrate: observation_map, ExperimentConfig
using ClimaAnalysis
import ClimaCalibrate
using Plots
using JLD2
using Statistics
using YAML
using Dates
import ClimaAtmos as CA 
using Glob

include("helper_funcs.jl")
include("observation_map.jl")


output_dir = "/central/groups/esm/jschmitt/calibrations/diurnal_diag_1/" # output directory


ylims = (0, 4000) # y limits for plotting (`z` coord)
iterations = nothing # iterations to plot (i.e., 0:2). default is all iterations
var_names =
    ("hus", "clw", "arup", "entr", "detr", "waup", "ta")
reduction = "inst"

config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))
const z_max = config_dict["z_max"]
const cal_var_names = config_dict["y_var_names"]
const const_noise_by_var = config_dict["const_noise_by_var"]
const n_vert_levels = config_dict["dims_per_var"]
model_config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "model_config.yml"))
model_config_dict["toml"] = ["scm_tomls/prognostic_prospct.toml"]

start_dates, latitudes, longitudes, convection_type, num_configs = get_era5_calibration_library()
atmos_config = CA.AtmosConfig(model_config_dict)


startdate = Dates.DateTime(config_dict["start_time"], "yyyymmdd")

obs_start = startdate + Dates.Second(config_dict["g_t_start_sec"])
obs_end = startdate + Dates.Second(config_dict["g_t_end_sec"])

# get/store LES obs and norm factors 
zc_model = get_z_grid(atmos_config, z_max = z_max)

# if isnothing(iterations)
#     iterations = get_iters_with_config(config_i, config_dict)
# end

xlims_dict = Dict("arup" => (-0.1, 0.4), "clw" => "auto")


function compute_plot_limits(data; margin_ratio = 0.5, fixed_margin = 1.0)

    min_val = minimum(data)
    max_val = maximum(data)

    data_range = max_val - min_val
    margin = data_range == 0 ? fixed_margin : margin_ratio * data_range
    limits = (min_val - margin, max_val + margin)

    return limits
end

for config_i in 1:num_configs
    iterations = get_iters_with_config(config_i, config_dict)
    for iteration in iterations
        @show "Iter: $iteration"
        for var_name in var_names

            data = ensemble_data(
                process_profile_variable,
                iteration,
                config_i,
                config_dict;
                var_name = var_name,
                reduction = reduction,
                output_dir = output_dir,
                z_max = z_max,
                n_vert_levels = n_vert_levels,
            )
            println("data size: ", size(data))
            # if var_name == "clw"
            #     data = log10.(data)
            # end
            
            # size(data) = [num vertical levels x ensemble size]
            eki_filepath = joinpath(
                ClimaCalibrate.path_to_iteration(output_dir, iteration),
                "eki_file.jld2",
            )
            eki = JLD2.load_object(eki_filepath)

            plot(legend = false)
            for i in 1:size(data, 2)

                y_var = data[:, i]
                Plots.plot!(y_var, zc_model)
            end
            if in(var_name, cal_var_names)
                forcing_file_path = CA.get_external_forcing_file_path(
                    Dict(
                        "start_date" => start_dates[config_i],
                        "site_latitude" => lats[config_i],
                        "site_longitude" => lons[config_i],
                    ),
                )
                y_truth = get_obs(
                    forcing_file_path,
                    [var_name],
                    obs_start, 
                    obs_end;
                    normalize = false,
                    norm_factors_dict = config_dict["norm_factors_by_var"],
                    z_scm = zc_model,
                    log_vars = [""], #["clw]
                )
                Plots.plot!(y_truth, zc_model, color = :black, label = "LES")
            end


            xlims = get(xlims_dict, var_name, nothing)

            if xlims == "auto"
                xlims = compute_plot_limits(y_truth)
            end

            Plots.plot!(
                xlabel = var_name,
                ylabel = "Height (z)",
                title = var_name,
                xlims = xlims,
                ylims = ylims,
            )

            plot_rel_dir = joinpath(
                output_dir,
                "plots",
                "ensemble_plots",
                "config_$(config_i)",
            )
            if !isdir(plot_rel_dir)
                mkpath(plot_rel_dir)
            end

            savefig(
                joinpath(
                    plot_rel_dir,
                    "ensemble_plot_$(var_name)_i_$(iteration).png",
                ),
            )
        end
    end
end
