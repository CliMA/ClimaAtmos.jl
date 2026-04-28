# Plot EKI loss vs iteration from saved artifacts.
#
# **REPL**:
#   include("analysis/plotting/plot_losses.jl")
#   va_plot_losses()   # latest eki_file.jld2 from experiment_config output_dir
#   va_plot_losses("simulation_output/.../eki/iteration_001/eki_file.jld2")
#
# **CLI**: julia --project=. analysis/plotting/plot_losses.jl [eki_file.jld2]
#
# Uses `EnsembleKalmanProcesses.get_error(eki)` (stored loss per assimilation step). This is **not**
# the same as iterating `get_g(eki, i)` as a vector — `get_g(eki)` is a vector of **per-iteration**
# forecast matrices, so the old ``‖g‖₂`` loop only had length `N_iterations`.
#
_va_analysis_is_main() =
    !isempty(Base.PROGRAM_FILE) && abspath(Base.PROGRAM_FILE) == abspath(@__FILE__)

include(joinpath(@__DIR__, "plot_paths.jl"))

if _va_analysis_is_main()
    import Pkg
    Pkg.activate(va_variance_adjustments_plot_paths().experiment)
end

include(joinpath(va_variance_adjustments_plot_paths().experiment, "lib", "experiment_common.jl"))
import CairoMakie as M
import JLD2
import EnsembleKalmanProcesses as EKP

"""
    va_plot_losses([eki_jld2_path]; experiment_dir, outpng) -> Union{Nothing,String}

Plot `EnsembleKalmanProcesses.get_error(eki)` vs iteration index (one point per stored EKI step).
If `eki_jld2_path` is `nothing`, uses [`va_latest_eki_jld2_path`](@ref)`(experiment_dir, experiment_config)`.

`outpng` defaults to `analysis/figures/losses.png`.
"""
function va_plot_losses(
    eki_jld2_path::Union{Nothing, AbstractString} = nothing;
    experiment_dir::AbstractString = va_variance_adjustments_plot_paths().experiment,
    experiment_config::Union{Nothing, AbstractString} = nothing,
    outpng::AbstractString = joinpath(va_variance_adjustments_plot_paths().figures, "losses.png"),
)
    mkpath(dirname(outpng))
    path = something(eki_jld2_path, va_latest_eki_jld2_path(experiment_dir, experiment_config))
    if path === nothing || !isfile(path)
        @warn "EKI file not found; pass a path or run calibration" path
        return nothing
    end
    eki = JLD2.load_object(path)
    errs = collect(EKP.get_error(eki))
    n = length(errs)
    xs = 0:(n - 1)
    fig = M.Figure()
    ax = M.Axis(fig[1, 1]; xlabel = "iteration", ylabel = "EKP loss (get_error)")
    M.lines!(ax, xs, errs)
    M.scatter!(ax, xs, errs)
    M.save(outpng, fig)
    @info "Saved" outpng eki = path n_points = n
    return outpng
end

function _va_plot_losses_cli()
    path = length(ARGS) >= 1 ? ARGS[1] : nothing
    va_plot_losses(path; experiment_dir = va_variance_adjustments_plot_paths().experiment)
end

if _va_analysis_is_main()
    _va_plot_losses_cli()
end
