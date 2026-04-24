# Naive vs separately calibrated varfix-on: one figure set per `(varfix_off_yaml, varfix_on_yaml)` pair.
# Requires `plot_profiles.jl` (`va_plot_all_case_diagnostic_profiles`, `va_load_experiment_config`, …) and
# [`lib/calibration_sweep_configs.jl`](../../lib/calibration_sweep_configs.jl) (`va_naive_vs_calibrated_varfix_on_yaml_pairs`).
#
const _VA_ROOT_NAIVEPLOT = joinpath(@__DIR__, "..", "..") |> abspath
include(joinpath(_VA_ROOT_NAIVEPLOT, "lib", "experiment_common.jl"))
include(joinpath(_VA_ROOT_NAIVEPLOT, "lib", "observation_map.jl"))

import ClimaCalibrate as CAL

function _va_member_output_active(
    experiment_dir::AbstractString,
    config_relp::AbstractString,
    iteration::Integer,
    member::Integer,
)
    expc = va_load_experiment_config(experiment_dir, config_relp)
    out = expc["output_dir"]
    root = isabspath(out) ? String(out) : joinpath(experiment_dir, out)
    return joinpath(CAL.path_to_ensemble_member(root, iteration, member), "output_active")
end

"""
    va_plot_naive_vs_calibrated_varfix_on_profiles!(experiment_dir, varfix_off_yaml, varfix_on_yaml; eki_member) -> Vector{String}

Overlay **reference** (varfix-off baseline), **EKI member varfix off**, **naive varfix on** (same parameters as
varfix-off member), and **EKI member varfix on** (separate calibration). Writes under
`analysis/figures/<CASE>_N<n>_naive_vs_calibrated_varfix_on/profiles/`.
"""
function va_plot_naive_vs_calibrated_varfix_on_profiles!(
    experiment_dir::AbstractString,
    varfix_off_yaml::AbstractString,
    varfix_on_yaml::AbstractString;
    eki_member::Union{Nothing, Int} = nothing,
)
    off_ex = va_load_experiment_config(experiment_dir, varfix_off_yaml)
    on_ex = va_load_experiment_config(experiment_dir, varfix_on_yaml)
    off_case = va_load_merged_case_yaml_dict(experiment_dir, off_ex["model_config_path"])
    on_case = va_load_merged_case_yaml_dict(experiment_dir, on_ex["model_config_path"])
    string(off_ex["case_name"]) == string(on_ex["case_name"]) ||
        error("naive vs calibrated pair: case_name mismatch $(varfix_off_yaml) vs $(varfix_on_yaml)")
    Int(off_ex["quadrature_order"]) == Int(on_ex["quadrature_order"]) ||
        error("naive vs calibrated pair: quadrature_order mismatch")
    va_varfix_tag(off_ex, off_case) == "varfix_on" &&
        error("varfix_off_yaml must have varfix off (base sgs_distribution, not a *_vertical_profile* name)")
    va_varfix_tag(on_ex, on_case) == "varfix_off" &&
        error("varfix_on_yaml must use a vertical-profile SGS name (`*_vertical_profile*`) / varfix on")

    iter_off = va_latest_eki_iteration_number(experiment_dir, varfix_off_yaml)
    iter_on = va_latest_eki_iteration_number(experiment_dir, varfix_on_yaml)

    mem_off = va_resolve_eki_member_index(experiment_dir, varfix_off_yaml, iter_off, eki_member)
    mem_on = va_resolve_eki_member_index(experiment_dir, varfix_on_yaml, iter_on, eki_member)

    ref = va_reference_output_active(experiment_dir, varfix_off_yaml)
    m_off = _va_member_output_active(experiment_dir, varfix_off_yaml, iter_off, mem_off)
    naive = va_naive_forward_output_active(experiment_dir, varfix_off_yaml)
    m_on = _va_member_output_active(experiment_dir, varfix_on_yaml, iter_on, mem_on)

    paths = String[]
    labels = String[]
    colors = Any[]
    linestyles = Any[]
    linewidths = Float64[]

    function offer!(p, lab, col, ls, lw)
        !isdir(p) && (@warn "Skip missing output_active (naive vs calibrated plot)" label = lab path = p; return)
        push!(paths, p)
        push!(labels, lab)
        push!(colors, col)
        push!(linestyles, ls)
        push!(linewidths, lw)
    end

    offer!(ref, "reference (varfix-off baseline)", :black, :solid, 2.35)
    offer!(
        m_off,
        "EKI varfix off (iter=$(iter_off), m=$(mem_off))",
        :steelblue,
        :solid,
        1.65,
    )
    offer!(naive, "naive varfix on (varfix-off params)", :orangered, :dash, 1.85)
    offer!(
        m_on,
        "EKI varfix on (iter=$(iter_on), m=$(mem_on))",
        :seagreen,
        :solid,
        1.65,
    )

    if length(paths) < 2
        @warn "Too few runs for naive vs calibrated varfix-on profiles" varfix_off_yaml varfix_on_yaml
        return String[]
    end

    case = replace(string(off_ex["case_name"]), r"[^\w\.\-]+" => "_")
    n = Int(off_ex["quadrature_order"])
    outdir = joinpath(
        experiment_dir,
        "analysis",
        "figures",
        "$(case)_N$(n)_naive_vs_calibrated_varfix_on",
        "profiles",
    )
    mkpath(outdir)
    model_rel = string(off_ex["model_config_path"])
    return va_plot_all_case_diagnostic_profiles(
        paths;
        experiment_dir,
        experiment_config = nothing,
        outdir,
        path_labels = labels,
        model_config_rel = model_rel,
        path_colors = colors,
        path_linestyles = linestyles,
        path_linewidths = linewidths,
        profile_title = "Naive vs calibrated varfix-on — ref=black; varfix-off EKI=blue; naive vf_on=red dash; calibrated vf_on=green",
    )
end

function va_plot_all_naive_vs_calibrated_varfix_on_profiles!(experiment_dir::AbstractString)
    written = String[]
    for (off_yml, on_yml) in va_naive_vs_calibrated_varfix_on_yaml_pairs(experiment_dir)
        @info "Naive vs calibrated varfix-on profiles" varfix_off = off_yml varfix_on = on_yml
        append!(
            written,
            va_plot_naive_vs_calibrated_varfix_on_profiles!(
                experiment_dir,
                off_yml,
                on_yml,
            ),
        )
    end
    return written
end
