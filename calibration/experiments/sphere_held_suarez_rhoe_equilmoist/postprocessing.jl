import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
using Distributions
import Statistics: mean
import CairoMakie: Makie
import ClimaCalibrate

function convergence_plot(
    eki::EKP.EnsembleKalmanProcess,
    prior::ParameterDistribution;
    theta_star = 65.0,
    output = joinpath("output", "sphere_held_suarez_rhoe_equilmoist"),
)
    u_vec = EKP.get_u(eki)
    meanabsdiff²(x, e) = sum(i -> abs(i - x)^2, e) / length(e)
    error_vec = map(ensemble -> meanabsdiff²(theta_star, ensemble), u_vec)
    spread_vec = map(ensemble -> meanabsdiff²(mean(ensemble), ensemble), u_vec)
    phi_vec = transform_unconstrained_to_constrained(prior, u_vec)
    u_series = eachcol(reduce(vcat, u_vec))
    phi_series = eachcol(reduce(vcat, phi_vec))

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
    Makie.hlines!(ax, [theta_star], color = :red, linestyle = :dash)
    Makie.save(joinpath(output, "convergence.png"), f)
end

function scatter_plot(
    eki,
    output = joinpath("output", "sphere_held_suarez_rhoe_equilmoist"),
)
    # Define figure with explicit size
    f = Makie.Figure(resolution = (800, 600))
    a = Makie.Axis(
        f[1, 1],
        title = "60-Day Zonal Avg Temp at 242m versus Unconstrained Equator-Pole Temp Gradient",
        ylabel = "Parameter Value",
        xlabel = "Temperature (K)",
    )

    g = vec.(EKP.get_g(eki; return_array = true))
    u = vec.(EKP.get_u(eki; return_array = true))

    for (gg, uu) in zip(g, u)
        Makie.scatter!(a, gg, uu)
    end

    # Save the figure
    Makie.save(joinpath(output, "scatter.png"), f)
end

# Uncomment for easy plotting
# import JLD2
# iteration = 2
# output_dir = joinpath("output", "sphere_held_suarez_rhoe_equilmoist")
# eki_filepath = joinpath(ClimaCalibrate.path_to_iteration(output_dir, iteration), "eki_file.jld2")
# eki = JLD2.load_object(eki_filepath)
# prior_path = joinpath("calibration", "experiments", "sphere_held_suarez_rhoe_equilmoist", "prior.toml")
# prior = ClimaCalibrate.get_prior(prior_path)
