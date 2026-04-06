import ClimaCalibrate as CAL
import ClimaAnalysis: SimDir, get, slice, average_lat, average_lon
import JLD2

function _observation_template(expc)
    if haskey(expc, "_observation_template")
        return expc["_observation_template"]
    end
    experiment_dir = dirname(Base.active_project())
    return JLD2.load_object(joinpath(experiment_dir, expc["observations_path"]))
end

function CAL.observation_map(iteration::Integer)
    expc = experiment_config()
    out_root = expc["output_dir"]
    ensemble_size = expc["ensemble_size"]
    y_ref = _observation_template(expc)
    n_y = length(y_ref)
    G = Array{Float64}(undef, n_y, ensemble_size)
    for m in 1:ensemble_size
        mp = CAL.path_to_ensemble_member(out_root, iteration, m)
        sp = joinpath(mp, "output_active")
        if isdir(sp)
            G[:, m] .= process_member_column(sp, y_ref)
        else
            G[:, m] .= NaN
        end
    end
    return G
end

"""
    process_member_column(output_active_path, template_y)

Build observation vector: time-mean `thetaa` profile (environment column) subsampled to length(template_y).
"""
function process_member_column(output_active_path::AbstractString, template_y::AbstractVector)
    !isdir(output_active_path) && return fill(NaN, length(template_y))
    simdir = SimDir(output_active_path)
    isempty(simdir.vars) && return fill(NaN, length(template_y))
    theta = get(
        simdir;
        short_name = "thetaa",
        reduction = "average",
        period = "10mins",
    )
    tcoord = ClimaAnalysis.times(theta)
    t_last = tcoord[end]
    th_t = slice(theta, time = t_last)
    prof_xy = average_lat(average_lon(th_t))
    v = vec(prof_xy.data)
    n = length(template_y)
    if length(v) >= n
        return Float64.(v[1:n])
    else
        out = zeros(Float64, n)
        out[1:length(v)] .= Float64.(v)
        return out
    end
end
