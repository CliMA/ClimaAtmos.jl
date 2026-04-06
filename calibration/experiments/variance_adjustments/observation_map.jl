include(joinpath(@__DIR__, "experiment_common.jl"))

import ClimaCalibrate as CAL
import ClimaAnalysis
import ClimaAnalysis: SimDir, get, slice, average_xy
import JLD2
import YAML

"""Match `period` to a key in `SimDir` (YAML `10mins` → often `10m`; if only one period exists, use it)."""
function _resolve_simdir_period(simdir, short_name, reduction, requested::AbstractString)
    avail = collect(ClimaAnalysis.available_periods(simdir; short_name, reduction))
    requested in avail && return requested
    length(avail) == 1 && return first(avail)
    error(
        "Period $(repr(requested)) not available for $(repr(short_name)) and reduction $(repr(reduction)). " *
        "Available: $(repr(avail)). Use a SimDir period string (e.g. \"10m\" for YAML `10mins`).",
    )
end

function _observation_template(expc)
    if haskey(expc, "_observation_template")
        return expc["_observation_template"]
    end
    experiment_dir = dirname(Base.active_project())
    return JLD2.load_object(va_observations_abs_path(experiment_dir, expc))
end

function CAL.observation_map(iteration::Integer)
    expc = experiment_config()
    experiment_dir = dirname(Base.active_project())
    out_rel = expc["output_dir"]
    out_root = isabspath(out_rel) ? String(out_rel) : joinpath(experiment_dir, out_rel)
    ensemble_size = expc["ensemble_size"]
    y_ref = _observation_template(expc)
    n_y = length(y_ref)
    G = Array{Float64}(undef, n_y, ensemble_size)
    field_specs = va_field_specs(expc)
    for m in 1:ensemble_size
        mp = CAL.path_to_ensemble_member(out_root, iteration, m)
        sp = joinpath(mp, "output_active")
        if isdir(sp)
            G[:, m] .= process_member_column(sp, y_ref, field_specs)
        else
            G[:, m] .= NaN
        end
    end
    return G
end

"""
    process_member_column(output_active_path, template_y, field_specs)

Build observation vector: for each diagnostic spec, take the last time slice, mean over
horizontal dims (`average_xy`: column `SimDir` uses `x`/`y`, not lon/lat), vertical profile
as a vector, then concatenate (each block length `length(template_y) ÷ n_fields`).
"""
function process_member_column(
    output_active_path::AbstractString,
    template_y::AbstractVector,
    field_specs = nothing,
)
    n_expected = length(template_y)
    if isnothing(field_specs)
        experiment_dir = dirname(Base.active_project())
        expc = va_load_experiment_config(experiment_dir)
        field_specs = va_field_specs(expc)
    end
    n_fields = length(field_specs)
    n_fields > 0 || return fill(NaN, n_expected)
    z_each = n_expected ÷ n_fields
    z_each * n_fields == n_expected || return fill(NaN, n_expected)

    !isdir(output_active_path) && return fill(NaN, n_expected)
    simdir = SimDir(output_active_path)
    isempty(simdir.vars) && return fill(NaN, n_expected)

    out = Vector{Float64}(undef, n_expected)
    o = 1
    for spec in field_specs
        period = _resolve_simdir_period(
            simdir,
            spec["short_name"],
            spec["reduction"],
            spec["period"],
        )
        theta = get(
            simdir;
            short_name = spec["short_name"],
            reduction = spec["reduction"],
            period,
        )
        tcoord = ClimaAnalysis.times(theta)
        t_last = tcoord[end]
        th_t = slice(theta, time = t_last)
        prof_h = average_xy(th_t)
        v = vec(prof_h.data)
        n = z_each
        if length(v) >= n
            out[o:(o + n - 1)] .= Float64.(v[1:n])
        else
            out[o:(o + n - 1)] .= 0
            out[o:(o + length(v) - 1)] .= Float64.(v)
        end
        o += n
    end
    return out
end
