using Plots
using LinearAlgebra
using YAML
import ClimaAtmos as CA
using Statistics

include("helper_funcs.jl")
include("get_les_metadata.jl")


var_i = "cli" #"clw"

experiment_dir = dirname(Base.active_project())
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
const model_config = experiment_config["model_config"]
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)
const z_cal_grid = experiment_config["z_cal_grid"]
const const_noise_by_var = experiment_config["const_noise_by_var"]

function get_local_les_calibration_library()
    # les_library = get_shallow_LES_library()
    les_library = get_LES_library()


    ref_paths = []
    for model in keys(les_library)
        for month in keys(les_library[model])
            cfsite_numbers = Tuple(les_library[model][month]["cfsite_numbers"])
            les_kwargs = (
                forcing_model = model,
                month = parse(Int, month),
                experiment = "amip",
            )
            append!(
                ref_paths,
                [
                    get_stats_path(
                        get_cfsite_les_dir(
                            parse(Int, cfsite_number[1]);
                            les_kwargs...,
                        ),
                    ) for cfsite_number in cfsite_numbers
                ],
            )

        end
    end


    return ref_paths
end



ref_paths = get_local_les_calibration_library()



y_obs_all = Float64[]

for ref_path in ref_paths

    try
        cfsite_number, _, _, _ = parse_les_path(ref_path)
        forcing_type = get_cfsite_type(cfsite_number)
        zc_model = get_cal_z_grid(atmos_config, z_cal_grid, forcing_type)

        ti = experiment_config["y_t_start_sec"]
        ti = isa(ti, AbstractFloat) ? ti : ti[forcing_type]
        tf = experiment_config["y_t_end_sec"]
        tf = isa(tf, AbstractFloat) ? tf : tf[forcing_type]

        y_obs, Σ_obs, norm_vec_obs = get_obs(
            ref_path,
            [var_i],
            zc_model;
            ti = ti,
            tf = tf,
            norm_factors_dict = nothing,
            z_score_norm = false,
            Σ_const = const_noise_by_var,
            Σ_scaling = "const",
        )
        if (var_i == "clw" || var_i == "cli") && any(<(0), y_obs)
            negative_values = filter(<(0), y_obs)
            @info "Skipping $ref_path due to negative values: $negative_values"
        else
            append!(y_obs_all, y_obs)
        end
    catch e
        @info "Error: $e"
    end
end


# clw/cli
log_y_obs_all = log10.(y_obs_all .+ 1e-12)
log_mean, log_std = mean(log_y_obs_all), std(log_y_obs_all)
log_y_obs_all_norm = (log_y_obs_all .- log_mean) ./ log_std
histogram(
    log_y_obs_all_norm,
    bins = 30,
    xlabel = "Value",
    ylabel = "Frequency (log scale)",
    title = var_i,
)


## hus
# variable_mean, variable_std = mean(y_obs_all), std(y_obs_all)
# y_obs_all_norm = (y_obs_all .- variable_mean) ./ variable_std
# histogram(
#     y_obs_all_norm,
#     bins = 30,
#     xlabel = "Value",
#     ylabel = "Frequency (log scale)",
#     title = var_i,
# )

# @show variable_mean, variable_std




# # histogram(log_y_obs_all, bins=30, yscale=:log10, xlabel="Value", ylabel="Frequency (log scale)", title=var_i)

# # λ = 0.1  # Choose an appropriate λ, often determined empirically
# # box_cox_transformed = (y_obs_all .^ λ .- 1) ./ λ
# # histogram(box_cox_transformed, bins=30, xlabel="Value", ylabel="Frequency (log scale)", title=var_i)
