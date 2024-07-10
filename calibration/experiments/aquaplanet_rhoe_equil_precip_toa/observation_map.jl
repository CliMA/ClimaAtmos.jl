import ClimaCalibrate: observation_map, path_to_ensemble_member
import ClimaAnalysis: SimDir, get, slice, average_lat, average_lon, window
import Statistics: var

function observation_map(iteration)
    ensemble_size = 12
    output_dir = joinpath("output", "aquaplanet_rhoe_equil_precip_toa")
    single_member_dims = (47,)
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

    for m in 1:ensemble_size
        member_path =
            path_to_ensemble_member(output_dir, iteration, m)
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

const meters = 1.0
const days = 86400.0
const time = 360days
const period = "60d"
# Cut off first 120 day to get equilibrium, take second level slice
function process_member_data(simdir::SimDir)
    isempty(simdir) && return NaN

    observations = Vector{Float64}(undef, 47)
    noise = zeros(Float64, 47, 47)

    rlut = average_lat(average_lon(get(simdir; short_name = "rlut", period)))
    rsut = average_lat(average_lon(get(simdir; short_name = "rsut", period)))
    observations[1] = slice(rlut, time = 360days).data[1]
    observations[2] = slice(rsut, time = 360days).data[1]

    pr = average_lon(get(simdir; short_name = "pr", period))
    southern_hemisphere_pr = window(pr, "lat", left = -100, right = 0)
    observations[3:47] .= slice(southern_hemisphere_pr, time = 360days).data

    noise[1,1] = var(rlut.data)
    noise[2,2] = var(rsut.data)

    southern_hemisphere_pr_variance = var(southern_hemisphere_pr.data,dims=1)
    foreach(3:47) do i
        noise[i,i] = southern_hemisphere_pr_variance[i-2]
    end
    return observations
end
