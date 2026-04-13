# CLI for forward-sweep comparison plots (see `plot_forward_sweep_body.jl`).
# Writes per-case profile grids, **`forward_sweep_clw_plus_cli_summary.png`**, and scalar figures.
#
# From `calibration/experiments/variance_adjustments`:
#   julia --project=. analysis/plotting/plot_forward_sweep.jl [--eki-calibrated-forward|--baseline-scm-forward] [--baseline-only] [--registry=REL.yml] [--ladder-*]
#
_va_pfw_is_main() =
    !isempty(Base.PROGRAM_FILE) && abspath(Base.PROGRAM_FILE) == abspath(@__FILE__)

const _VA_PFW_PLOT_DIR = @__DIR__
const _VA_PFW_ANALYSIS_DIR = joinpath(_VA_PFW_PLOT_DIR, "..") |> abspath
const _VA_PFW_EXPERIMENT_DIR = joinpath(_VA_PFW_ANALYSIS_DIR, "..") |> abspath

if _va_pfw_is_main()
    import Pkg
    Pkg.activate(_VA_PFW_EXPERIMENT_DIR)
end

include(joinpath(_VA_PFW_EXPERIMENT_DIR, "lib", "experiment_common.jl"))
include(joinpath(_VA_PFW_EXPERIMENT_DIR, "scripts", "resolution_ladder.jl"))
include(joinpath(_VA_PFW_EXPERIMENT_DIR, "lib", "forward_sweep_grid.jl"))
include(joinpath(_VA_PFW_PLOT_DIR, "plot_profiles.jl"))
include(joinpath(_VA_PFW_PLOT_DIR, "plot_forward_sweep_body.jl"))

function _va_parse_forward_sweep_plot_cli(argv::Vector{String})::ForwardSweepConfig
    cfg = ForwardSweepConfig(; resolution_ladder = true)
    for a in argv
        if a == "--baseline-scm-forward"
            cfg.forward_parameters = VA_FORWARD_PARAM_BASELINE_SCM
        elseif a == "--eki-calibrated-forward"
            cfg.forward_parameters = VA_FORWARD_PARAM_EKI_CALIBRATED
        elseif a == "--baseline-only" || a == "--no-resolution-ladder"
            cfg.resolution_ladder = false
        elseif a == "--resolution-ladder"
            cfg.resolution_ladder = true
        elseif startswith(a, "--registry=")
            cfg.registry_path = String(split(a, '=', limit = 2)[2])
        elseif startswith(a, "--ladder-n-tiers=")
            p = cfg.ladder
            cfg.ladder = VALadderParams(
                parse(Int, split(a, '=', limit = 2)[2]),
                p.coarsen_ratio,
                p.z_elem_min,
                p.min_dz_factor,
            )
        elseif startswith(a, "--ladder-coarsen-ratio=")
            p = cfg.ladder
            cfg.ladder = VALadderParams(
                p.n_tiers,
                parse(Float64, split(a, '=', limit = 2)[2]),
                p.z_elem_min,
                p.min_dz_factor,
            )
        elseif startswith(a, "--ladder-z-elem-min=")
            p = cfg.ladder
            cfg.ladder = VALadderParams(
                p.n_tiers,
                p.coarsen_ratio,
                parse(Int, split(a, '=', limit = 2)[2]),
                p.min_dz_factor,
            )
        elseif startswith(a, "--ladder-min-dz-factor=")
            p = cfg.ladder
            cfg.ladder = VALadderParams(
                p.n_tiers,
                p.coarsen_ratio,
                p.z_elem_min,
                parse(Float64, split(a, '=', limit = 2)[2]),
            )
        elseif startswith(a, "--figures-dir=")
            continue
        elseif a == "--help" || a == "-h"
            println("""
Usage: julia --project=. analysis/plotting/plot_forward_sweep.jl [options]

  --eki-calibrated-forward | --baseline-scm-forward   Match forward_eki vs forward_only outputs
  --baseline-only | --resolution-ladder   Match the forward sweep grid
  --registry=REL.yml            default when omitted: registries/forward_sweep_cases.yml
  --ladder-n-tiers=N  --ladder-coarsen-ratio=R  --ladder-z-elem-min=N  --ladder-min-dz-factor=F
  --figures-dir=DIR   optional; default folder depends on EKI vs baseline mode

Also writes forward_sweep_clw_plus_cli_summary.png in that figures folder.
""")
            exit(0)
        else
            error("Unknown argument: $(repr(a)). Try --help.")
        end
    end
    return cfg
end

function _va_forward_sweep_plot_figure_root(argv::Vector{String})::Union{Nothing, String}
    for a in argv
        if startswith(a, "--figures-dir=")
            return String(split(a, '=', limit = 2)[2])
        end
    end
    return nothing
end

function _va_run_forward_sweep_plot_cli()
    argv = collect(String, ARGS)
    cfg = _va_parse_forward_sweep_plot_cli(argv)
    root = _va_forward_sweep_plot_figure_root(argv)
    profs = va_plot_forward_sweep_comparisons!(
        _VA_PFW_EXPERIMENT_DIR,
        cfg;
        figure_root = root,
    )
    summary = va_plot_forward_sweep_clw_plus_cli_summary!(
        _VA_PFW_EXPERIMENT_DIR,
        cfg;
        figure_root = root,
    )
    scal = va_plot_forward_sweep_scalars_vs_nquad!(
        _VA_PFW_EXPERIMENT_DIR,
        cfg;
        figure_root = root,
    )
    return (; profiles = profs, clw_cli_summary = summary, scalars = scal)
end

if _va_pfw_is_main()
    _va_run_forward_sweep_plot_cli()
end
