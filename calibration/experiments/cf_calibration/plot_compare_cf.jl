using ClimaCalibrate
import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML
import TOML
using Distributions
using Distributed
using Random
using Flux
using Logging
using Glob
using YAML
using Printf

import JLD2
using LinearAlgebra
using BSON

include("helper_funcs.jl")
include("observation_map.jl")
include("get_les_metadata.jl")
# include("nn_helpers.jl")
include("nn_helpers.jl")


experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))

# Set a deterministic RNG seed for reproducible minibatching and any other randomness.
# Can be overridden by specifying `rng_seed` in `experiment_config.yml`.
const RNG_SEED = get(experiment_config, "rng_seed", 1234)
Random.seed!(RNG_SEED)
@info "Random seed set" RNG_SEED


# unpack experiment_config vars into scope
for (key, value) in experiment_config
    @eval const $(Symbol(key)) = $value
end

# load configs and directories
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)

### get LES obs (Y) and norm factors
ref_paths, _ = get_les_calibration_library()
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
        experiment_config["y_var_names"],
        zc_model;
        ti = ti,
        tf = tf,
        norm_factors_dict = norm_factors_by_var,
        z_score_norm = true,
        log_vars = log_vars,
        Σ_const = const_noise_by_var,
        Σ_scaling = "const",
    )

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

series_names = [ref_paths[i] for i in 1:length(ref_paths)]


# site names 
site_names = [split(k.names[1], "_")[1][13:end] for k in obs_vec]

# hardcode dictionary 
site_dict = Dict(
    "11" => "8"
)


"""
    find_simulations_by_site(base_dir::String, site_number::String)

Search recursively inside `base_dir` for simulation folders ending at `output_0000`.
Each simulation folder contains a `.yml` config file with a field `cfsite_number`.

Return a dictionary: iteration_number => Vector{String of paths}.
"""
function find_simulations_by_site(base_dir::String, site_number::String)
    # Find all directories ending with "output_0000"
    sim_paths = filter(isdir, glob("iteration_*/member_*/config_*/output_0000", base))
    matches = []

    for path in sim_paths
        # Each simulation has one config YAML file in the parent directory (config_X)
        cfg = YAML.load_file(joinpath(path, ".yml"))

        # e.g. "11"
        cfsite = get(cfg, "cfsite_number", nothing)

        # Skip non-matching site numbers
        if cfsite == "site$site_number"
            push!(matches, path)
        end
    end

    return matches
end

base = "/resnick/groups/esm/jschmitt/climaatmos_scm_calibrations/output_cf_ml_v1/exp2_calphys"

matches = find_simulations_by_site(base, "11")

glob("iteration_*/member_*/config_*/output_0000", base)