import ClimaCalibrate: observation_map, path_to_ensemble_member
import ClimaAnalysis: SimDir, get, slice, average_lat, average_lon, window, times
import Statistics: var, quantile

function observation_map(iteration)
    output_dir = joinpath("output", "aquaplanet_rhoe_equil_precip_toa")
    single_member_dims = (3,)
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

const days = 86400.0
const time = 360days
const period = "60d"
# simdir = SimDir("calibration/experiments/aquaplanet_rhoe_equil_precip_toa/generate_observations/")
function process_member_data(simdir::SimDir)
    isempty(simdir) && return NaN

    observations = Vector{Float64}(undef, 3)
    noise = zeros(Float64, 3, 3)
    for (i, short_name) in enumerate(("rsut", "rlut"))
        output_var = average_lat(average_lon(get(simdir; short_name, period)))
        observations[i] = slice(output_var; time).data[1]
        noise[i,i] = var(output_var.data)
    end

    precip = average_lon(get(simdir; short_name = "pr", period))
    precip = window(precip, "lat", left = -100, right = 0)
    precip_slice = slice(precip; time)
    # Take the bottom percentile because precipitation is negative
    observations[3] = quantile(precip_slice.data, 0.01)
    percentiles = map(times(precip)) do time
        precip_slice = slice(precip; time)
        quantile(precip_slice.data, 0.01)
    end
    noise[3,3] = var(percentiles)
    return observations
end
