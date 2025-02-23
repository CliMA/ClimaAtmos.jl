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
using CairoMakie
using Makie
@info "loaded packages ..."
include("helper_funcs.jl")
include("observation_map.jl")

output_dir = "/scratch/julian/calibrations/exp_5"
config_i = 1
ylims = (0, 4000)
iteration = nothing
integrated_vars = ["rsut", "rlut", "clwvi", "clivi", "pr"]

var_names = ("thetaa", "hus", "clw", "arup", "entr", "detr", "waup", 
                "tke", "turbentr", "ta", "rlut", "rsut", "clwvi", "clivi")

label_dict = Dict(
    "rsut" => "TOA Shortwave (W/m^2)",
    "rlut" => "TOA Longwave (W/m^2)",
    "clwvi" => "Cloud Water Path (kg/m^2)",
    "clivi" => "Cloud Ice Path (kg/m^2)",
    "pr" => "Precipitation (mm/s)",
)

reduction = "inst"

config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))
const z_max = config_dict["z_max"]
const cal_var_names = config_dict["y_var_names"]
const const_noise_by_var = config_dict["const_noise_by_var"]
const n_vert_levels = config_dict["dims_per_var"]
model_config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "model_config.yml"));


ref_paths, months, sites = get_era5_calibration_library()
atmos_config = CA.AtmosConfig(model_config_dict)

# get/store LES obs and norm factors 
zc_model = get_z_grid(atmos_config, z_max = z_max)

# if isnothing(iterations)
#     iterations = get_iters_with_config(config_i, config_dict)
# end

# xlims_dict = Dict("arup" => (-0.1, 0.4), "clw" => "auto")

# function compute_plot_limits(data; margin_ratio = 0.5, fixed_margin = 1.0)

#     min_val = minimum(data)
#     max_val = maximum(data)

#     data_range = max_val - min_val
#     margin = data_range == 0 ? fixed_margin : margin_ratio * data_range
#     limits = (min_val - margin, max_val + margin)

#     return limits
# end

integrated_simulation_data = [[ensemble_data(
    process_profile_variable,
    i, 
    1, 
    config_dict;
    var_name = var_name,
    reduction = reduction,
    output_dir = output_dir,
    z_max = z_max,
    n_vert_levels = n_vert_levels, 
) for i in 1:10] for var_name in integrated_vars];


dims_per_var = n_vert_levels
y_truth = get_obs(
    ref_paths[1],
    months[1],
    sites[1],
    integrated_vars;
    normalize = false, 
    z_scm = zc_model,
    log_vars = [""],
)
@info "plotting..."
function filtnan(x, i=1)
    x[:, .!isnan.(x[i, :])][:]
end
fig = Figure(size = (1000, 1300))
for (ind, name) in enumerate(integrated_vars)
    ax = Axis(fig[ind, 1], xlabel = "Iteration", ylabel = label_dict[name])
    for i in 1:10
        data = filtnan.(integrated_simulation_data[ind])[i]
        if name in ["pr"] # flip because era5 is directionally challenged 
            data = -data
        end
        if iszero(data) # violin plot is misleading when all entries are zero
            Makie.scatter!(ax, repeat([i-1], length(data)), data, color = :black)
        elseif i == 1 # add legend
            Makie.violin!(ax, repeat([i-1], length(data)) , data, show_median = true, color = (:blue, 0.5), label = "Ensemble")
        else
            Makie.violin!(ax, repeat([i-1], length(data)) , data, show_median = true, color = (:blue, 0.5))
        end
    end
    Makie.hlines!(ax, [y_truth[ind]], color = :red, label = "ERA5 Observation")
    if ind == 1
        axislegend(ax, position = :rb)
    end
end

# save figure to simulation plot directory
save(joinpath(output_dir, "plots", "integrated_simulation_violin.png"), fig)

