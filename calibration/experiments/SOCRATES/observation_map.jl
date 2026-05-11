import ClimaAnalysis: SimDir, average_time, get, slice, window
import ClimaCalibrate as CAL
import ClimaCalibrate: observation_map, path_to_ensemble_member
import EnsembleKalmanProcesses as EKP
import Interpolations
import JLD2
import YAML

include(joinpath(@__DIR__, "../gcm_driven_scm/helper_funcs.jl"))

const EXPERIMENT_CONFIG_PATH = joinpath(@__DIR__, "experiment_config.yml")

# Maps ClimaAtmos diagnostic short_name → variable name stored in the preprocessed
# Atlas LES reference files (written by preprocess_reference.jl via ATLAS_TO_CLIMA_VARS).
# Variables absent from this map fall back to using the short_name directly as the
# reference-file key, which works when both sides use the same name (e.g. "cli", "ta").
const CLIMADIAGNOSTICS_TO_LES_NAME_MAP = Dict(
    "thetaa" => "thetal",   # ClimaAtmos potential T   → LES liquid-ice pot T
    "hus"    => "hus",      # total specific humidity (after wt_to_qt scaling)
    "clw"    => "clw",      # cloud liquid water (g→kg scaling applied in preprocess)
    "ta"     => "ta",       # air temperature (TABS, K — same name in reference file)
    "cli"    => "cli",      # cloud ice (QCI, g→kg scaling applied in preprocess)
    "husra"  => "prw",      # rain water sp. hum. (ClimaAtmos) → "prw" (QR in Atlas)
    "hussn"  => "snw",      # snow sp. hum. (ClimaAtmos)       → "snw" (QS in Atlas)
)

function resolve_z_model_by_case!(cfg::Dict)
    if haskey(cfg, "z_model_by_case")
        z_by_case = [Float64.(z_case) for z_case in cfg["z_model_by_case"]]
        cfg["z_model_by_case"] = z_by_case
        return z_by_case
    end

    metadata_path = joinpath(cfg["output_dir"], "obs_metadata.jld2")
    if !isfile(metadata_path)
        error("Missing z_model_by_case and metadata file not found at $metadata_path")
    end

    metadata = JLD2.load(metadata_path)
    z_by_case = if haskey(metadata, "z_model_by_case")
        [Float64.(z_case) for z_case in metadata["z_model_by_case"]]
    elseif haskey(metadata, "z_model")
        z_model = Float64.(metadata["z_model"])
        [copy(z_model) for _ in cfg["cases"]]
    else
        error("Could not infer z_model_by_case from $metadata_path")
    end

    cfg["z_model_by_case"] = z_by_case
    return z_by_case
end

function resolve_dims_per_var!(cfg::Dict)
    if haskey(cfg, "dims_per_var")
        return Int(cfg["dims_per_var"])
    end

    metadata_path = joinpath(cfg["output_dir"], "obs_metadata.jld2")
    if !isfile(metadata_path)
        error("Missing dims_per_var and metadata file not found at $metadata_path")
    end

    metadata = JLD2.load(metadata_path)
    if haskey(metadata, "dims_per_var")
        dims = Int(metadata["dims_per_var"])
    elseif haskey(metadata, "z_model")
        dims = length(metadata["z_model"])
    else
        error("Could not infer dims_per_var from $metadata_path")
    end

    cfg["dims_per_var"] = dims
    return dims
end

function resolve_z_model!(cfg::Dict)
    if haskey(cfg, "z_model")
        return Float64.(cfg["z_model"])
    end

    metadata_path = joinpath(cfg["output_dir"], "obs_metadata.jld2")
    if !isfile(metadata_path)
        error("Missing z_model and metadata file not found at $metadata_path")
    end

    metadata = JLD2.load(metadata_path)
    haskey(metadata, "z_model") ||
        error("Could not infer z_model from $metadata_path")

    z_model = Float64.(metadata["z_model"])
    cfg["z_model"] = z_model
    return z_model
end

function interpolate_profile_to_target_grid(y_src::AbstractVector, z_src::AbstractVector, z_target::AbstractVector)
    length(y_src) == length(z_src) ||
        throw(
            DimensionMismatch(
                "Profile interpolation requires matching source lengths, got y=$(length(y_src)) and z=$(length(z_src)).",
            ),
        )

    # Ensure monotonic nodes for interpolation.
    perm = sortperm(z_src)
    z_sorted = Float64.(z_src[perm])
    y_sorted = Float64.(y_src[perm])

    itp = Interpolations.LinearInterpolation(
        (z_sorted,),
        y_sorted;
        extrapolation_bc = Interpolations.Line(),
    )
    return Float64.(itp(z_target))
end

function resolve_member_simdir(member_path::String, case_idx::Int)
    config_path = joinpath(member_path, "config_$(case_idx)")
    active_output = joinpath(config_path, "output_active")
    if ispath(active_output)
        return SimDir(active_output)
    end

    # Fallback for older runs without output_active symlink.
    return SimDir(joinpath(config_path, "output_0000"))
end

function process_profile_variable(
    simdir,
    y_name;
    reduction,
    t_start,
    t_end,
    z_target = nothing,
    z_max = nothing,
    pooled_norms_dict = nothing,
)
    var_i = get(simdir; short_name = y_name, reduction)

    if !isnothing(z_max)
        z_window = filter(x -> x <= z_max, var_i.dims["z"])
        var_i = window(var_i, "z", right = maximum(z_window))
    end

    sim_t_end = var_i.dims["time"][end]
    if sim_t_end < 0.95 * t_end
        throw(ErrorException("Simulation ended too early for case output."))
    end

    var_i_ave = average_time(window(var_i, "time", left = t_start, right = sim_t_end))
    y_var_i = slice(var_i_ave, x = 1, y = 1).data

    if !isnothing(z_target)
        z_src = var_i_ave.dims["z"]
        y_var_i = interpolate_profile_to_target_grid(vec(Array(y_var_i)), vec(Array(z_src)), z_target)
    end

    # Normalize by pooled characteristic scale (mean of nonzero elements from reference).
    # Matches CalibrateEDMF pooled_nonzero_mean_to_value normalization.
    if !isnothing(pooled_norms_dict) && haskey(pooled_norms_dict, y_name)
        y_var_i = y_var_i ./ pooled_norms_dict[y_name]
    end

    return y_var_i
end

function process_observation_variable(
    filename::String,
    y_name::String,
    z_scm::Vector;
    t_start,
    t_end,
    pooled_norms_dict = nothing,
)
    # Map ClimaAtmos diagnostic name to LES variable name
    les_var_name = get(CLIMADIAGNOSTICS_TO_LES_NAME_MAP, y_name, y_name)
    
    # Use helper_funcs to read and interpolate the LES variable
    var_data = fetch_interpolate_transform(les_var_name, filename, z_scm)
    
    # Handle time averaging if data has time dimension
    if ndims(var_data) == 2
        # Has time dimension: average over time window
        t = nc_fetch(filename, "time")
        ti_index = argmin(abs.(t .- t_start))
        tf_index = argmin(abs.(t .- t_end))
        y_var = mean(var_data[:, ti_index:tf_index], dims=2)[:]
    else
        # No time dimension (e.g., single profile)
        y_var = vec(var_data)
    end
    
    # Normalize by pooled characteristic scale (mean of nonzero elements from reference).
    # Matches CalibrateEDMF pooled_nonzero_mean_to_value normalization.
    if !isnothing(pooled_norms_dict) && haskey(pooled_norms_dict, y_name)
        y_var = y_var ./ pooled_norms_dict[y_name]
    end
    
    return y_var
end

function process_member_data(member_path, eki, config_dict::Dict)
    case_indices = EKP.get_current_minibatch(eki)
    y_names = config_dict["y_var_names"]
    z_model_by_case = resolve_z_model_by_case!(config_dict)

    g = Float64[]
    for case_idx in case_indices
        case = config_dict["cases"][case_idx]
        z_target = z_model_by_case[case_idx]
        g_ti = get(case, "g_t_start_sec", config_dict["g_t_start_sec"])
        g_tf = get(case, "g_t_end_sec", config_dict["g_t_end_sec"])

        simdir = resolve_member_simdir(member_path, case_idx)
        for y_name in y_names
            try
                y_var_i = process_profile_variable(
                    simdir,
                    y_name;
                    reduction = config_dict["reduction"],
                    t_start = g_ti,
                    t_end = g_tf,
                    z_target = z_target,
                    z_max = config_dict["z_max"],
                    pooled_norms_dict = get(config_dict, "pooled_norms_by_var", nothing),
                )
                y_vec = vec(Array(y_var_i))
                length(y_vec) == length(z_target) || throw(
                    DimensionMismatch(
                        "Expected $(length(z_target)) vertical points for y_name=$(y_name), case_idx=$(case_idx); got $(length(y_vec)).",
                    ),
                )
                append!(g, y_vec)
            catch err
                @warn "Error processing model output for y_name=$y_name, case_idx=$case_idx: $err"
                append!(g, fill(NaN, length(z_target)))
            end
        end
    end

    return g
end

function observation_map(iteration; config_dict = nothing)
    cfg = isnothing(config_dict) ? YAML.load_file(EXPERIMENT_CONFIG_PATH) : config_dict

    iter_path = CAL.path_to_iteration(cfg["output_dir"], iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))

    # Load pooled normalization factors computed during observation generation.
    metadata_path = joinpath(cfg["output_dir"], "obs_metadata.jld2")
    if isfile(metadata_path)
        metadata = JLD2.load(metadata_path)
        if haskey(metadata, "pooled_norms_by_var")
            cfg = copy(cfg)
            cfg["pooled_norms_by_var"] = metadata["pooled_norms_by_var"]
        end
    end

    # Use the EKP observation vector length as the source of truth for expected
    # observation-map output size for this iteration/minibatch.
    full_dim = length(EKP.get_obs(eki))
    G_ensemble = Array{Float64}(undef, full_dim, cfg["ensemble_size"])

    for member in 1:cfg["ensemble_size"]
        member_path = path_to_ensemble_member(cfg["output_dir"], iteration, member)
        try
            g_member = process_member_data(member_path, eki, cfg)
            length(g_member) == full_dim || throw(
                DimensionMismatch(
                    "observation_map length mismatch for member $(member): expected $(full_dim), got $(length(g_member)).",
                ),
            )
            G_ensemble[:, member] .= g_member
        catch err
            @info "Error during observation map for ensemble member $(member)." err
            G_ensemble[:, member] .= NaN
        end
    end

    return G_ensemble
end