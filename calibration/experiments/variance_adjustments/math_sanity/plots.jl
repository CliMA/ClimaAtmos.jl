# Requires `defaults.jl`, `scatter_series.jl`, and `moment_recovery.jl` loaded first.

using CairoMakie
using LinearAlgebra
using Printf
using Random
using Statistics

function mathsanity_plot_quadrature_scatter(
    path::AbstractString;
    μ_q = nothing,
    μ_T = nothing,
    σ_q = nothing,
    σ_T = nothing,
    ρ = nothing,
    N_quad = nothing,
    n_mc = nothing,
    rng::AbstractRNG = Random.Xoshiro(42),
    clamped::Bool = false,
    lim_z = nothing,
)
    d = mathsanity_default_scatter_knobs()
    μ_q = something(μ_q, d.μ_q)
    μ_T = something(μ_T, d.μ_T)
    σ_q = something(σ_q, d.σ_q)
    σ_T = something(σ_T, d.σ_T)
    ρ = something(ρ, d.ρ)
    N_quad = something(N_quad, d.N_quad)
    n_mc = something(n_mc, d.n_mc)
    lim_z = something(lim_z, d.lim_z)
    FT = typeof(μ_q)

    dat = mathsanity_standardized_mc_quad_data(
        FT(μ_q),
        FT(μ_T),
        FT(σ_q),
        FT(σ_T),
        FT(ρ);
        N_quad = N_quad,
        n_mc = n_mc,
        clamped = clamped,
        rng = rng,
        lim_z = FT(lim_z),
    )

    fig = Figure(;
        size = (1320, 640),
        figure_padding = (28, 28, 24, 24),
        fontsize = 15,
    )
    subt =
        clamped ?
        "GaussianSGS with physical clamps — red ellipses = unclamped analytic Σ_z (reference only)" :
        "Unclamped linear map — standardized axes (ellipse = analytic Σ_z)"
    Label(fig[1, 1:3], subt; fontsize = 16, tellwidth = false, halign = :center)
    phys = @sprintf "Physical: μ_q=%.4f σ_q=%.4f | μ_T=%.2f σ_T=%.3f | ρ=%.2f" μ_q σ_q μ_T σ_T ρ
    znote =
        "Standardized z: blue = GH nodes, gray = MC (no coral here). " *
        "Physical (q,T) figures: blue GH = turbulent (σ_q,σ_T,ρ); purple GH = (σ_tot,ρ_eff) when Δz changes Σ; coral MC; black column."
    Label(fig[2, 1:3], phys * "\n" * znote; fontsize = 10, color = (:black, 0.62), tellwidth = false, halign = :center)

    ax1 = Axis(
        fig[3, 1];
        aspect = DataAspect(),
        title = "Gauss–Hermite nodes (marker area ∝ weight, markersize ∝ √ω; tails are integration nodes)",
        xlabel = "(q - μ_q) / σ_q",
        ylabel = "(T - μ_T) / σ_T",
        xlabelsize = 15,
        ylabelsize = 15,
        titlesize = 15,
    )
    scatter!(ax1, dat.zq_quad, dat.zT_quad; markersize = dat.ms, color = (:dodgerblue, 0.88), strokewidth = 0)
    lines!(ax1, dat.xe1, dat.ye1; color = :red, linewidth = 3, label = "1σ analytic Σ_z")
    lines!(ax1, dat.xe2, dat.ye2; color = :red, linestyle = :dash, linewidth = 2.5, label = "2σ analytic Σ_z")
    xlims!(ax1, dat.xlo, dat.xhi)
    ylims!(ax1, dat.ylo, dat.yhi)

    ax2 = Axis(
        fig[3, 2];
        aspect = DataAspect(),
        title = clamped ? "Monte Carlo (same χ, same clamps as quadrature)" :
            "Monte Carlo (same bivariate Gaussian)",
        xlabel = "(q - μ_q) / σ_q",
        ylabel = "(T - μ_T) / σ_T",
        xlabelsize = 15,
        ylabelsize = 15,
        titlesize = 15,
    )
    scatter!(ax2, dat.zq_mc, dat.zT_mc; markersize = 5, color = (:gray, 0.35))
    lines!(ax2, dat.xe1, dat.ye1; color = :red, linewidth = 3)
    lines!(ax2, dat.xe2, dat.ye2; color = :red, linestyle = :dash, linewidth = 2.5)
    xlims!(ax2, dat.xlo, dat.xhi)
    ylims!(ax2, dat.ylo, dat.yhi)

    Legend(
        fig[3, 3],
        ax1;
        valign = :center,
        margin = (12, 12, 12, 12),
        framevisible = true,
        labelsize = 13,
        titlesize = 13,
    )
    colsize!(fig.layout, 3, Fixed(200))
    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 10)

    mkpath(dirname(path))
    save(path, fig; px_per_unit = 2)
    return fig
end

function mathsanity_plot_sigma_ratio_scan(path::AbstractString; N_quad = nothing, μ_q = nothing, μ_T = nothing, σ_q = nothing, σ_T = nothing, ρ = nothing)
    d = mathsanity_default_scatter_knobs()
    μ_q = something(μ_q, d.μ_q)
    μ_T = something(μ_T, d.μ_T)
    σ_q = something(σ_q, d.σ_q)
    σ_T = something(σ_T, d.σ_T)
    ρ = something(ρ, d.ρ)

    FT = eltype(μ_q)

    N_quad = something(N_quad, d.N_quad)
    ratios = range(FT(0.4), FT(1.6); length = 80)
    rows = mathsanity_sigma_ratio_scan(FT(μ_q), FT(μ_T), FT(σ_q), FT(σ_T), FT(ρ), N_quad, collect(ratios))

    rs = [r[1] for r in rows]
    ferr = [r[4] for r in rows]
    serr = [r[5] for r in rows]

    fig = Figure(; size = (880, 520), figure_padding = (40, 40, 28, 28), fontsize = 16)
    Label(
        fig[1, 1],
        "GH tensor product (N=$N_quad): linear map ⇒ mean relative error is 0 for all σ ratios (not plotted)";
        fontsize = 13,
        color = (:black, 0.55),
        tellwidth = false,
        halign = :left,
    )
    ax = Axis(
        fig[2, 1];
        xlabel = "σ_used / σ_true (q and T scaled together)",
        ylabel = "‖Σ̂ − Σ‖_2 / ‖Σ‖_2   (covariance only)",
        title = "Covariance error vs mis-scaled subgrid σ",
        titlesize = 17,
        xlabelsize = 15,
        ylabelsize = 15,
        xgridvisible = true,
        ygridvisible = true,
        xgridstyle = :dot,
        ygridstyle = :dot,
    )
    lines!(ax, rs, serr; color = (:orangered, 0.85), linewidth = 4)
    scatter!(ax, rs, serr; color = :orangered, markersize = 7, strokewidth = 1, strokecolor = :white)
    vlines!(ax, [1.0]; color = (:black, 0.35), linestyle = :dashdot, linewidth = 2.5)
    ymax = max(1.05, maximum(serr) * 1.06)
    ylims!(ax, 0.0, ymax)
    xlims!(ax, first(rs), last(rs))
    mferr = maximum(abs, ferr)
    if mferr > 1e-12
        text!(
            ax,
            0.02,
            0.98;
            text = @sprintf("max mean rel. err = %.1e", mferr),
            align = (:left, :top),
            space = :relative,
            fontsize = 11,
            color = (:steelblue, 0.9),
        )
    end
    mkpath(dirname(path))
    save(path, fig; px_per_unit = 2)
    return fig
end
