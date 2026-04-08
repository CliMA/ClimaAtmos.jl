# One entry point for default post-run figures (losses, parameters, profiles).
#
# **REPL** (experiment project active):
#   include("analysis/plotting/run_post_analysis.jl")
#   va_run_post_analysis!()
#   va_run_post_analysis!(profile_paths = [va_reference_output_active(pwd()), "simulation_output/.../member_001/output_active"])
#
# **CLI** (from `calibration/experiments/variance_adjustments`):
#   julia --project=. analysis/plotting/run_post_analysis.jl [--figures-dir=DIR] [--experiment-config=REL.yml] [output_active path1 [path2 ...]]
#
# One slice (default figure dir, losses + parameters + case diagnostics):
#   julia --project=. analysis/plotting/run_post_analysis.jl --experiment-config=config/experiment_config.yml
#
# All slices in `va_calibration_sweep_configs()` (same as full study’s figure pass):
#   julia --project=. scripts/run_full_study.jl --figures-only
#
# Figures default to `analysis/figures/<CASE>_N<n>_<varfix>_<calibration_mode>/` so different slices do not overwrite.
#
_va_analysis_is_main() =
    !isempty(Base.PROGRAM_FILE) && abspath(Base.PROGRAM_FILE) == abspath(@__FILE__)

const _VA_PLOTTING_DIR = @__DIR__
const _VA_ANALYSIS_DIR = joinpath(_VA_PLOTTING_DIR, "..") |> abspath
const _VA_EXPERIMENT_DIR = joinpath(_VA_ANALYSIS_DIR, "..") |> abspath
const _VA_FIGURES_DIR = joinpath(_VA_ANALYSIS_DIR, "figures")

if _va_analysis_is_main()
    import Pkg
    Pkg.activate(_VA_EXPERIMENT_DIR)
end

include(joinpath(_VA_PLOTTING_DIR, "plot_profiles.jl"))
include(joinpath(_VA_PLOTTING_DIR, "plot_losses.jl"))
include(joinpath(_VA_PLOTTING_DIR, "plot_parameters.jl"))

function _va_parse_post_analysis_cli_args(args::Vector{String})
    figure_root = nothing
    experiment_config = nothing
    paths = String[]
    for a in args
        if startswith(a, "--figures-dir=")
            figure_root = String(split(a, '=', limit = 2)[2])
        elseif startswith(a, "--experiment-config=")
            experiment_config = String(split(a, '=', limit = 2)[2])
        else
            push!(paths, a)
        end
    end
    return (; figure_root, experiment_config, paths)
end

"""
    va_run_post_analysis!(; experiment_dir, experiment_config, figure_root, profile_paths, ...)

Generate standard figures after calibration. Resolves the experiment YAML via [`va_experiment_config_path`](@ref)
(`VA_EXPERIMENT_CONFIG` or optional `experiment_config` relative path).

`figure_root` defaults to [`va_default_figure_dir`](@ref) so each slice writes under its own subfolder of
`analysis/figures/`.

- `do_observation_profiles`: if `true`, writes `profiles_eki_observation_stack.png` under `figure_root`.
- `do_case_diagnostic_profiles`: one PNG per case diagnostic → `figure_root/profiles/profile_<name>.png`.
"""
function va_run_post_analysis!(;
    experiment_dir::AbstractString = _VA_EXPERIMENT_DIR,
    experiment_config::Union{Nothing, AbstractString} = nothing,
    figure_root::Union{Nothing, AbstractString} = nothing,
    profile_paths::AbstractVector{<:AbstractString} = String[],
    do_losses::Bool = true,
    do_parameters::Bool = true,
    do_observation_profiles::Bool = false,
    do_case_diagnostic_profiles::Bool = true,
)
    expc = va_load_experiment_config(experiment_dir, experiment_config)
    root = something(figure_root, va_default_figure_dir(experiment_dir, expc))
    mkpath(root)
    mkpath(joinpath(root, "profiles"))
    obs_png = nothing
    case_pngs = String[]
    loss_png = nothing
    param_png = nothing
    if do_observation_profiles
        obs_png = va_plot_profiles(
            profile_paths;
            experiment_dir,
            experiment_config,
            outpng = joinpath(root, "profiles_eki_observation_stack.png"),
        )
    end
    if do_case_diagnostic_profiles
        append!(
            case_pngs,
            va_plot_all_case_diagnostic_profiles(
                profile_paths;
                experiment_dir,
                experiment_config,
                outdir = joinpath(root, "profiles"),
            ),
        )
    end
    if do_losses
        loss_png = va_plot_losses(;
            experiment_dir,
            experiment_config,
            outpng = joinpath(root, "losses.png"),
        )
    end
    if do_parameters
        param_png = va_plot_parameters(;
            experiment_dir,
            experiment_config,
            outpng = joinpath(root, "parameters.png"),
        )
    end
    @info "Post-analysis finished" figure_root = root
    return (; obs_png, case_pngs, loss_png, param_png, figure_root = root)
end

function _va_run_post_analysis_cli()
    opts = _va_parse_post_analysis_cli_args(collect(String, ARGS))
    va_run_post_analysis!(;
        profile_paths = opts.paths,
        experiment_dir = _VA_EXPERIMENT_DIR,
        experiment_config = opts.experiment_config,
        figure_root = opts.figure_root,
    )
end

if _va_analysis_is_main()
    _va_run_post_analysis_cli()
end
