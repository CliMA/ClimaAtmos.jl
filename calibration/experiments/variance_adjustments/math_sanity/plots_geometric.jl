# Requires `defaults.jl` and `geometric_moments.jl` loaded first.
# (Mosaic drivers in `grid_plots.jl` pair turbulent `σ` with `z_q,z_T` axes; these figures stay in raw SI.)

using CairoMakie

"""Bar-style breakdown of turbulent vs `(1/12)Δz²(∂·/∂z)²` contributions + effective ρ."""
function mathsanity_plot_geometric_breakdown(
    path::AbstractString;
    qq = nothing,
    TT = nothing,
    ρ_param = nothing,
    dz = nothing,
    dq_dz = nothing,
    dtheta_dz = nothing,
    dT_dθ = nothing,
)
    k = mathsanity_default_geometric_knobs()
    FT = typeof(k.qq)
    qq = FT(something(qq, k.qq))
    TT = FT(something(TT, k.TT))
    ρp = FT(something(ρ_param, k.ρ_param))
    dz = FT(something(dz, k.dz))
    dq_dz = FT(something(dq_dz, k.dq_dz))
    dtheta_dz = FT(something(dtheta_dz, k.dtheta_dz))
    dT_dθ = FT(something(dT_dθ, k.dT_dθ))
    var_q, var_T, ρ_eff, ge = mathsanity_sgs_quad_moments_with_geometry(
        qq,
        TT,
        ρp,
        dz,
        dq_dz,
        dtheta_dz,
        dT_dθ,
    )

    fig = Figure(; size = (960, 520), fontsize = 15, figure_padding = 36)
    Label(
        fig[1, :],
        "Subcell geometry (scalar column): same formulas as materialize_sgs_quadrature_moments!";
        fontsize = 16,
        tellwidth = false,
    )
    sub =
        "Δz = $(dz) m  |  ρ_param = $(ρp)  →  ρ_eff = $(round(ρ_eff; digits=4))  |  toy ∂q/∂z, ∂θ/∂z, ∂T/∂θ"
    Label(fig[2, :], sub; fontsize = 11, color = (:black, 0.6), tellwidth = false)

    ax1 = Axis(
        fig[3, 1];
        title = "Specific humidity variance (q′q′)",
        ylabel = "variance (SI units)",
        xticks = ([1, 2, 3], ["q′q′ turb", "+ Δ_geom", "= total"]),
        xlabelsize = 13,
    )
    ys = [qq, ge.Δvar_q, var_q]
    barplot!(ax1, 1:3, ys; color = [:steelblue, :coral, :seagreen], width = 0.65)
    ax1.xticklabelrotation = deg2rad(30)

    ax2 = Axis(
        fig[3, 2];
        title = "Temperature variance (T′T′)",
        ylabel = "variance (K²)",
        xticks = ([1, 2, 3], ["T′T′ turb", "+ Δ_geom", "= total"]),
        xlabelsize = 13,
    )
    ys2 = [TT, ge.Δvar_T, var_T]
    barplot!(ax2, 1:3, ys2; color = [:steelblue, :coral, :seagreen], width = 0.65)
    ax2.xticklabelrotation = deg2rad(30)

    mkpath(dirname(path))
    save(path, fig; px_per_unit = 2)
    return fig
end

"""Curves vs `Δz` for total variances and `ρ_eff` at default column geometry (same scalar formulas as materialize)."""
function mathsanity_plot_geometric_breakdown_dz_scan(
    path::AbstractString;
    qq = nothing,
    TT = nothing,
    ρ_param = nothing,
    dz_list = mathsanity_default_dz_scan_list(),
    dq_dz = nothing,
    dtheta_dz = nothing,
    dT_dθ = nothing,
)
    k = mathsanity_default_geometric_knobs()
    FT = typeof(k.qq)
    qq = FT(something(qq, k.qq))
    TT = FT(something(TT, k.TT))
    ρp = FT(something(ρ_param, k.ρ_param))
    dq_dz = FT(something(dq_dz, k.dq_dz))
    dtheta_dz = FT(something(dtheta_dz, k.dtheta_dz))
    dT_dθ = FT(something(dT_dθ, k.dT_dθ))
    dzv = collect(dz_list)

    var_q = Vector{FT}(undef, length(dzv))
    var_T = Vector{FT}(undef, length(dzv))
    ρ_eff = Vector{FT}(undef, length(dzv))
    for (kdz, dz) in enumerate(dzv)
        vq, vT, ρe, _ = mathsanity_sgs_quad_moments_with_geometry(qq, TT, ρp, FT(dz), dq_dz, dtheta_dz, dT_dθ)
        var_q[kdz] = vq
        var_T[kdz] = vT
        ρ_eff[kdz] = ρe
    end

    fig = Figure(; size = (980, 520), fontsize = 14, figure_padding = 32)
    Label(
        fig[1, :],
        "Geometry vs layer thickness Δz (turb fixed; subcell (1/12)Δz² gradient terms grow with Δz)";
        fontsize = 16,
        tellwidth = false,
    )

    ax1 = Axis(fig[2, 1]; xlabel = "Δz (m)", ylabel = "variance", title = "q and T total variance vs Δz")
    lines!(ax1, dzv, var_q; label = "var_q total", color = :steelblue, linewidth = 2.5)
    lines!(ax1, dzv, var_T; label = "var_T total", color = :coral, linewidth = 2.5)
    axislegend(ax1; position = :lt)

    ax2 = Axis(fig[2, 2]; xlabel = "Δz (m)", ylabel = "ρ_eff", title = "Effective correlation vs Δz")
    lines!(ax2, dzv, ρ_eff; color = :seagreen, linewidth = 2.5)

    sub =
        "q′q′ turb = $(qq)  |  T′T′ turb = $(TT)  |  ρ_param = $(ρp)  |  ∂q/∂z, ∂θ/∂z, ∂T/∂θ from defaults"
    Label(fig[3, :], sub; fontsize = 11, color = (:black, 0.55), tellwidth = false)

    mkpath(dirname(path))
    save(path, fig; px_per_unit = 2)
    return fig
end
