import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML
using Plots

include("helper_funcs.jl")
include("get_les_metadata.jl")

plot_vars = ["thetaa", "hus", "cli", "clw"]

experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))


# unpack experiment_config vars into scope 
for (key, value) in experiment_config
    @eval const $(Symbol(key)) = $value
end

const prior = CAL.get_prior(joinpath(experiment_dir, prior_path))

# load configs and directories 
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)


CFSITE_TYPES = Dict(
    "shallow" => (collect(4:15)..., collect(17:23)...),
    "deep" =>
        (collect(30:33)..., collect(66:70)..., 82, 92, 94, 96, 99, 100),
)


function get_les_calibration_library()
    les_library = get_shallow_LES_library()
    cfsite_numbers = [
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        30,
        31,
        32,
        33,
        66,
        67,
        68,
        69,
        70,
        82,
        92,
        94,
        96,
        99,
        100,
    ]

    les_kwargs = (forcing_model = "HadGEM2-A", month = 7, experiment = "amip")
    ref_paths = [
        get_stats_path(get_cfsite_les_dir(cfsite_number; les_kwargs...)) for
        cfsite_number in cfsite_numbers
    ]
    return (ref_paths, cfsite_numbers)
end

### get LES obs (Y) and norm factors
ref_paths, _ = get_les_calibration_library()

for plot_var in plot_vars
    # Create directory for each plot_var under prof_plots
    plot_dir = joinpath("./prof_plots", plot_var)
    if !isdir(plot_dir)
        mkdir(plot_dir)  # Create the directory if it doesn't exist
    end

    obs_vec = []
    for ref_path in ref_paths
        cfsite_number, _, _, _ = parse_les_path(ref_path)
        forcing_type = get_cfsite_type(cfsite_number)
        zc_model = get_cal_z_grid(atmos_config, z_cal_grid, forcing_type)

        ti = experiment_config["y_t_start_sec"]
        ti = isa(ti, AbstractFloat) ? ti : ti[forcing_type]
        tf = experiment_config["y_t_end_sec"]
        tf = isa(tf, AbstractFloat) ? tf : tf[forcing_type]

        y_obs, Σ_obs, norm_vec_obs = get_obs(
            ref_path,
            [plot_var],
            zc_model;
            ti = ti,
            tf = tf,
            norm_factors_dict = norm_factors_by_var,
            z_score_norm = false,
            log_vars = [],
            Σ_const = const_noise_by_var,
            Σ_scaling = "const",
        )

        # Plot and save the figure
        plot(y_obs, zc_model)
        savefig(joinpath(plot_dir, "les_obs_$(cfsite_number).png"))

        push!(
            obs_vec,
            EKP.Observation(
                Dict(
                    "samples" => y_obs,
                    "covariances" => Σ_obs,
                    "names" => split(ref_path, "/")[end],
                ),
            ),
        )
    end
end

series_names = [ref_paths[i] for i in 1:length(ref_paths)]
