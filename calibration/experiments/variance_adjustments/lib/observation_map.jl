include(joinpath(@__DIR__, "experiment_common.jl"))

import ClimaCalibrate as CAL
import ClimaAnalysis
import ClimaAnalysis: SimDir, get, slice, average_xy
import EnsembleKalmanProcesses as EKP
import JLD2
import LinearAlgebra: dot
import YAML

const VA_COMPOSITE_SHORT_NAME_CLW_PLUS_CLI = "clw_plus_cli"

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

"""Vertical profile (last time, `average_xy`) as `Vector{Float64}`; length may differ from `z_elem`."""
function _column_profile_last_time(
    simdir,
    short_name::AbstractString,
    reduction::AbstractString,
    period_requested::AbstractString,
)
    period = _resolve_simdir_period(simdir, short_name, reduction, period_requested)
    theta = get(
        simdir;
        short_name = String(short_name),
        reduction = String(reduction),
        period,
    )
    tcoord = ClimaAnalysis.times(theta)
    isempty(tcoord) && return Float64[]
    th_t = slice(theta, time = tcoord[end])
    prof_h = average_xy(th_t)
    return Float64.(vec(prof_h.data))
end

function _fill_profile_block!(out, o::Integer, v::AbstractVector, n::Integer)
    if length(v) >= n
        out[o:(o + n - 1)] .= v[1:n]
    else
        out[o:(o + n - 1)] .= 0
        out[o:(o + length(v) - 1)] .= v
    end
    return nothing
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

Composite `short_name` `clw_plus_cli` sums `clw` and `cli` profiles (cloud condensate only; excludes
precipitating species).
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
        sn = spec["short_name"]
        reduction = spec["reduction"]
        period_req = spec["period"]
        n = z_each
        if sn == VA_COMPOSITE_SHORT_NAME_CLW_PLUS_CLI
            v_l = _column_profile_last_time(simdir, "clw", reduction, period_req)
            v_i = _column_profile_last_time(simdir, "cli", reduction, period_req)
            m = min(length(v_l), length(v_i))
            v = m == 0 ? Float64[] : (v_l[1:m] .+ v_i[1:m])
        else
            period = _resolve_simdir_period(simdir, sn, reduction, period_req)
            theta = get(
                simdir;
                short_name = sn,
                reduction = reduction,
                period,
            )
            tcoord = ClimaAnalysis.times(theta)
            v = if isempty(tcoord)
                Float64[]
            else
                th_t = slice(theta, time = tcoord[end])
                prof_h = average_xy(th_t)
                Float64.(vec(prof_h.data))
            end
        end
        _fill_profile_block!(out, o, v, n)
        o += n
    end
    return out
end

"""
    va_eki_best_member_by_obs_loss(experiment_dir, config_relp, iteration) -> Int

Return the ensemble member index **m** that minimizes **Mahalanobis** squared error to **`y`**:
``(g_m - y)' Σ^{-1} (g_m - y)`` with **`Σ`** from `va_build_noise_matrix` (same likelihood as calibration).

**Data source:** `iteration_*/eki_file.jld2` only. The forward map **`G`** is read with
`EnsembleKalmanProcesses.get_g_final(eki)` — the same stored ensemble predictions EKP used.

**Why not `error_metrics["loss"]`?**  
`ClimaCalibrate` saves **scalar** diagnostics per iteration (e.g. `"loss" => [51.2]` — one number per
**EKI iteration**, not `ensemble_size` values). That summarizes the **whole ensemble** (or the step), so it
cannot pick “member 3 vs member 5”. Per-member ranking needs **columns of `G`**, not the aggregate `loss`.

**Why not `get_error(eki)`?**  
That API is also **one value per stored EKI iteration** (see `plot_losses.jl`), not a vector over members.

If **`eki_file.jld2` is missing**, **`get_g_final` fails**, or **`G`** has the wrong shape vs **`length(y)`**, this
function **errors** — there is no silent SimDir fallback. Fix the calibration output or pass a fixed
`--eki-member=k`.
"""
function va_eki_best_member_by_obs_loss(
    experiment_dir::AbstractString,
    config_relp::AbstractString,
    iteration::Integer,
)::Int
    expc = va_load_experiment_config(experiment_dir, config_relp)
    y = JLD2.load_object(va_observations_abs_path(experiment_dir, expc))
    Σ = va_build_noise_matrix(y, expc, experiment_dir)
    out = expc["output_dir"]
    root = isabspath(out) ? String(out) : joinpath(experiment_dir, out)

    eki_path = joinpath(CAL.path_to_iteration(root, iteration), "eki_file.jld2")
    isfile(eki_path) || error(
        "Missing EKI file at $eki_path — cannot select best member. Run calibration for $(repr(config_relp)) first.",
    )
    eki = JLD2.load_object(eki_path)
    G = EKP.get_g_final(eki; return_array = true)
    G isa AbstractMatrix || error("get_g_final did not return a matrix; got $(typeof(G)) at $eki_path")
    size(G, 1) == length(y) || error(
        "Observation length $(length(y)) does not match G rows $(size(G, 1)) in $eki_path — regenerate observations or check EKI.",
    )
    size(G, 2) >= 1 || error("Empty G in $eki_path")

    best_m = 1
    best_score = Inf
    for m in 1:size(G, 2)
        g = view(G, :, m)
        any(isnan, g) && continue
        δ = g .- y
        score = dot(δ, Σ \ δ)
        if score < best_score
            best_score = score
            best_m = m
        end
    end
    isfinite(best_score) || error(
        "All columns of G were NaN in $eki_path — cannot select best member.",
    )
    @info "EKI member selected (min Mahalanobis to observations)" config_relp iteration eki_path member = best_m loss =
        best_score
    return best_m
end

"""
    va_resolve_eki_member_index(experiment_dir, config_relp, iteration, eki_member) -> Int

If **`eki_member === nothing`**, return [`va_eki_best_member_by_obs_loss`](@ref); otherwise return
**`eki_member`** (fixed index).
"""
function va_resolve_eki_member_index(
    experiment_dir::AbstractString,
    config_relp::AbstractString,
    iteration::Integer,
    eki_member::Union{Nothing, Int},
)::Int
    eki_member === nothing && return va_eki_best_member_by_obs_loss(experiment_dir, config_relp, iteration)
    return Int(eki_member)
end
