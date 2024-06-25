import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
using Distributions
import JLD2
import Statistics: mean
import YAML
import TOML
import CairoMakie: Makie

function convergence_plot(eki, prior; theta_star = 65., output = joinpath("output", "sphere_held_suarez_rhoe_equilmoist"))
    u_vec = EKP.get_u(eki)

    error_vec = Float64[]
    spread_vec = Float64[]
    for ensemble in u_vec
        ensemble_error = 0
        ensemble_spread = 0
        ensemble_mean = mean(ensemble)
        for i in ensemble
            ensemble_error += abs(i - theta_star)^2
            ensemble_spread += abs(i - ensemble_mean)^2

        end
        ensemble_error /= length(ensemble)
        ensemble_spread /= length(ensemble)

        push!(error_vec, ensemble_error)
        push!(spread_vec, ensemble_spread)
    end

    phi_vec = transform_unconstrained_to_constrained(prior, u_vec)
    u_series = [getindex.(u_vec, i) for i in 1:10]
    phi_series = [getindex.(phi_vec, i) for i in 1:10]

    f = Makie.Figure(title = "Convergence Plot", resolution = (800, 800))

    ax = Makie.Axis(
        f[1, 1],
        xlabel = "Iteration",
        ylabel = "Error",
        xticks = 0:50,
    )
    Makie.lines!(ax, 0.0:(length(error_vec) - 1), error_vec)

    ax = Makie.Axis(
        f[1, 2],
        xlabel = "Iteration",
        ylabel = "Spread",
        xticks = 0:50,
    )
    Makie.lines!(ax, 0.0:(length(spread_vec) - 1), spread_vec)

    ax = Makie.Axis(
        f[2, 1],
        xlabel = "Iteration",
        ylabel = "Unconstrained Parameters",
        xticks = 0:50,
    )
    Makie.lines!.(ax, tuple(0.0:(length(u_series[1]) - 1)), u_series)

    ax = Makie.Axis(
        f[2, 2],
        xlabel = "Iteration",
        ylabel = "Constrained Parameters",
        xticks = 0:50,
    )
    Makie.lines!.(ax, tuple(0.0:(length(phi_series[1]) - 1)), phi_series)
    Makie.hlines!(ax, [65.0], color = :red, linestyle = :dash)
    Makie.save(joinpath(output, "convergence.png"), f)
end
