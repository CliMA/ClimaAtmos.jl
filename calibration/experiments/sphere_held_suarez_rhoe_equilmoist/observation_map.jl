import EnsembleKalmanProcesses: TOMLInterface
import ClimaCalibrate: observation_map, ExperimentConfig
import ClimaAnalysis: SimDir, get, slice, average_lat, average_lon

function observation_map(iteration)
    ensemble_size = 10
    output_dir = joinpath("output", "sphere_held_suarez_rhoe_equilmoist")
    single_member_dims = (1,)
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

    for m in 1:ensemble_size
        member_path =
            TOMLInterface.path_to_ensemble_member(output_dir, iteration, m)
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
# Cut off first 120 day to get equilibrium, take second level slice
function process_member_data(simdir::SimDir)
    isempty(simdir.vars) && return NaN
    ta = get(simdir; short_name = "ta", reduction = "average", period = "60d")
    zonal_avg_temp_observations =
        slice(average_lat(average_lon(ta)), z = 242meters)
    return slice(zonal_avg_temp_observations, time = 240days).data
end
