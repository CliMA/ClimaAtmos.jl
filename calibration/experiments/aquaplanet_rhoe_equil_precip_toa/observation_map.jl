import EnsembleKalmanProcesses: TOMLInterface
import ClimaCalibrate: observation_map, ExperimentConfig
import ClimaAnalysis: SimDir, get, slice, average_lat, average_lon

function observation_map(iteration)
    ensemble_size = 10
    output_dir = joinpath("output", "aquaplanet_rhoe_equil_precip_toa")
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
    isempty(simdir) && return NaN

    observations = Vector{Float64}(undef, 3)
    noise = Matrix{Float64}(undef, 3, 1)

    short_names = ("rsut", "rlut", "pr")
    for (i, short_name) in enumerate(short_names)
        output_var = average_lat(average_lon(get(simdir; short_name, period)))
        observations[i] = slice(output_var, time = 360days).data[1]
        noise[i] = var(output_var.data)
    end

    return observations
end
