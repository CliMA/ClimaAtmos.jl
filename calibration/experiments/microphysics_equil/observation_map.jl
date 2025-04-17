### to go in here:
# - observation_map function



"""
version from the TUTORIAL
"""
const days = 86_400
function CAL.observation_map(iteration)
    single_member_dims = (1,)
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

    for m in 1:ensemble_size
        member_path = CAL.path_to_ensemble_member(output_dir, iteration, m)
        simdir_path = joinpath(member_path, "output_active")
        if isdir(simdir_path)
            simdir = SimDir(simdir_path)
            G_ensemble[:, m] .= process_member_data(simdir)
        else
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

function process_member_data(simdir::SimDir)
    isempty(simdir.vars) && return NaN
    rsut =
        get(simdir; short_name = "rsut", reduction = "average", period = "30d")
    return slice(average_xy(rsut); time = 30days).data
end

"""
comparing w the model ground truth. I think Is can probably just use this simple version for now?
"""
function observation_map(iteration)
    model_output = "model_ustar_array.jld2"

    dims = 1
    G_ensemble = Array{Float64}(undef, dims..., ensemble_size)
    for m in 1:ensemble_size
        member_path = path_to_ensemble_member(output_dir, iteration, m)

        try
            ustar = JLD2.load_object(joinpath(member_path, model_output))
            G_ensemble[:, m] = process_member_data(ustar)
        catch e
            @info "An error occured in the observation map for member $m"
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

function process_member_data(ustar; output_variance = false)

    profile_mean = nanmean(ustar)
    observation = Float64[profile_mean]
    if !(output_variance)
        return observation
    else
        variance = Matrix{Float64}(undef, 1, 1)
        variance[1] = nanvar(ustar)
        return (; observation, variance)
    end
end
nanmean(x) = mean(filter(!isnan, x))
nanvar(x) = var(filter(!isnan, x))
