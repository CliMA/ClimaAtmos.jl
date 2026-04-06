# Constrained-parameter trajectories from EKI (one subplot per parameter, lines = ensemble members).
#
# **REPL**:
#   include("analysis/plotting/plot_parameters.jl")
#   va_plot_parameters()   # latest eki + experiment_config prior_path
#   va_plot_parameters("path/to/eki_file.jld2", "prior.toml")
#
# **CLI**: julia --project=. analysis/plotting/plot_parameters.jl [eki_file.jld2] [prior.toml]
#
# `outpng` defaults to `analysis/figures/parameters.png`.
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

include(joinpath(_VA_EXPERIMENT_DIR, "experiment_common.jl"))
import CairoMakie as M
import JLD2
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate as CAL
import Statistics: mean

function _va_short_title(s::AbstractString, n::Int = 42)
    length(s) <= n && return s
    return first(s, n) * "…"
end

"""
    va_plot_parameters([eki_jld2_path], [prior_path]; experiment_dir, outpng) -> Union{Nothing,String}

For each scalar parameter row in `EKP.get_ϕ(prior, eki)`, plot constrained values vs iteration:
thin colored lines = ensemble members, thick black = ensemble mean. Subplot titles use prior names
(`prior.name`).

`outpng` defaults to `analysis/figures/parameters.png`.
"""
function va_plot_parameters(
    eki_jld2_path::Union{Nothing, AbstractString} = nothing,
    prior_path::Union{Nothing, AbstractString} = nothing;
    experiment_dir::AbstractString = _VA_EXPERIMENT_DIR,
    experiment_config::Union{Nothing, AbstractString} = nothing,
    outpng::AbstractString = joinpath(_VA_FIGURES_DIR, "parameters.png"),
)
    mkpath(dirname(outpng))
    eki_path = something(eki_jld2_path, va_latest_eki_jld2_path(experiment_dir, experiment_config))
    prior_p = something(prior_path, va_prior_abs_path(experiment_dir, experiment_config))
    if eki_path === nothing || !isfile(eki_path)
        @warn "Missing eki file" eki_path
        return nothing
    end
    !isfile(prior_p) && error("Prior not found: $(repr(prior_p))")
    eki = JLD2.load_object(eki_path)
    prior = CAL.get_prior(prior_p)
    phi = EKP.get_ϕ(prior, eki)
    isempty(phi) && error("get_ϕ returned no iterations")
    n_p, N_ens = size(phi[1])
    names = prior.name
    length(names) == n_p || error(
        "Prior has $(length(names)) names but ϕ has $n_p rows; cannot label subplots.",
    )
    n_it = length(phi)
    xs = collect(0:(n_it - 1))

    row_h = 160
    fig = M.Figure(size = (520, row_h * n_p + 40))
    for p in 1:n_p
        ax = M.Axis(
            fig[p, 1];
            xlabel = p == n_p ? "iteration" : "",
            ylabel = "ϕ (constrained)",
            title = _va_short_title(names[p]),
        )
        for m in 1:N_ens
            ys = [Float64(phi[it][p, m]) for it in 1:n_it]
            M.lines!(ax, xs, ys; color = (:steelblue, 0.35), linewidth = 1)
        end
        μ = [mean(phi[it][p, :]) for it in 1:n_it]
        M.lines!(ax, xs, μ; color = :black, linewidth = 2.5)
    end
    M.save(outpng, fig)
    @info "Saved" outpng eki = eki_path prior = prior_p
    return outpng
end

function _va_plot_parameters_cli()
    eki_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    prior_path = length(ARGS) >= 2 ? ARGS[2] : nothing
    va_plot_parameters(eki_path, prior_path; experiment_dir = _VA_EXPERIMENT_DIR)
end

if _va_analysis_is_main()
    _va_plot_parameters_cli()
end
