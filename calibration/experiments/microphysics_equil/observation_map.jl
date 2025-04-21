### to go in here:
# - observation_map function
using Distributed
import ClimaCalibrate as CAL
import ClimaAnalysis: SimDir, get, slice, average_xy
using ClimaUtilities.ClimaArtifacts
import EnsembleKalmanProcesses: I, ParameterDistributions.constrained_gaussian

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
