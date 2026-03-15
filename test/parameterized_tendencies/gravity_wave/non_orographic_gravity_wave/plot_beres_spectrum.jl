#=
Standalone script to visualize the Beres (2004) convective gravity wave
source spectrum using the actual ClimaAtmos implementation.

Reproduces idealized cases from Beres et al. (2004, JAS):
  - Figure 1: Momentum flux vs phase speed for varying heating depth h
  - Figure 2: Momentum flux vs phase speed for varying mean wind U
  - Figure 3: Momentum flux vs phase speed for varying heating rate Q0
  - Figure 4: Beres spectrum vs AD Gaussian spectrum comparison
  - Figure 5: Momentum flux vs phase speed for varying σ_x

Usage (from ClimaAtmos.jl root):
  julia --project -e 'include("test/parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/plot_beres_spectrum.jl")'
=#

using CairoMakie

# Import the actual wave_source and BeresSourceParams from ClimaAtmos
import ClimaAtmos as CA

const BeresSourceParams = CA.BeresSourceParams
const wave_source = CA.wave_source

# --- AD Gaussian wave_source for comparison ---

function wave_source_ad(
    c::NTuple{nc, FT},
    u_source::FT,
    Bw::FT,
    Bn::FT,
    cw::FT,
    cn::FT,
    c0::FT,
    flag::FT,
    gw_ncval::Val{nc},
) where {nc, FT}
    ntuple(
        n ->
            sign(c[n] - u_source) * (
                Bw * exp(
                    -log(FT(2)) *
                    ((c[n] * flag + (c[n] - u_source) * (1 - flag) - c0) / cw)^2,
                ) +
                Bn * exp(
                    -log(FT(2)) *
                    ((c[n] * flag + (c[n] - u_source) * (1 - flag) - c0) / cn)^2,
                )
            ),
        Val(nc),
    )
end

# --- Setup ---

FT = Float64
dc = FT(4.0)  
cmax = FT(100.0)
nc = Int(2 * cmax / dc + 1)
c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))
c_vec = collect(c)

N_source = FT(0.01)

beres_default = BeresSourceParams{FT}(;
    Q0_threshold = FT(1.157e-4),
    beres_scale_factor = FT(1.0),
    σ_x = FT(4000.0),
    ν_min = FT(2π / (120 * 60)),
    ν_max = FT(2π / (10 * 60)),
    n_ν = 9,
)

output_dir = joinpath(@__DIR__, "beres_plots")
mkpath(output_dir)

# ============================================================
# Figure 1: Varying heating depth h (U=0, Q0=10 K/day)
# ============================================================
let
    fig = Figure(; size = (800, 500), fontsize = 16)
    ax = Axis(
        fig[1, 1];
        xlabel = "Phase speed c (m/s)",
        ylabel = "Momentum flux density",
        title = "Beres spectrum: varying heating depth h (U=0, Q0=10 K/day)",
    )
    Q0 = FT(10.0 / 86400.0)
    u_heat = FT(0.0)
    for (h_km, color) in [(5, :blue), (8, :green), (10, :orange), (15, :red)]
        h = FT(h_km * 1000.0)
        B = wave_source(c, u_heat, Q0, h, N_source, beres_default, Val(nc))
        lines!(ax, c_vec, collect(B); label = "h = $(h_km) km", color)
    end
    axislegend(ax; position = :rt)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 0.5)
    save(joinpath(output_dir, "beres_varying_h.png"), fig)
    println("Saved: beres_varying_h.png")
end

# ============================================================
# Figure 2: Varying mean wind U (h=10km, Q0=10 K/day)
# ============================================================
let
    fig = Figure(; size = (800, 500), fontsize = 16)
    ax = Axis(
        fig[1, 1];
        xlabel = "Phase speed c (m/s)",
        ylabel = "Momentum flux density",
        title = "Beres spectrum: varying mean wind U (h=10km, Q0=10 K/day)",
    )
    h = FT(10000.0)
    Q0 = FT(10.0 / 86400.0)
    for (U, color) in [(0, :blue), (5, :green), (10, :orange), (20, :red)]
        B = wave_source(c, FT(U), Q0, h, N_source, beres_default, Val(nc))
        lines!(ax, c_vec, collect(B); label = "U = $(U) m/s", color)
        # Vertical line at c=U marking the critical layer
        if U > 0
            vlines!(ax, [Float64(U)]; color, linestyle = :dot, linewidth = 1)
        end
    end
    axislegend(ax; position = :rt)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 0.5)
    save(joinpath(output_dir, "beres_varying_U.png"), fig)
    println("Saved: beres_varying_U.png")
end

# ============================================================
# Figure 3: Varying heating rate Q0 (h=10km, U=0)
# ============================================================
let
    fig = Figure(; size = (800, 500), fontsize = 16)
    ax = Axis(
        fig[1, 1];
        xlabel = "Phase speed c (m/s)",
        ylabel = "Momentum flux density",
        title = "Beres spectrum: varying Q0 (h=10km, U=0)",
    )
    h = FT(10000.0)
    u_heat = FT(0.0)
    for (Q0_Kday, color) in [(1, :blue), (5, :green), (10, :orange), (20, :red)]
        Q0 = FT(Q0_Kday / 86400.0)
        B = wave_source(c, u_heat, Q0, h, N_source, beres_default, Val(nc))
        lines!(ax, c_vec, collect(B); label = "Q0 = $(Q0_Kday) K/day", color)
    end
    axislegend(ax; position = :rt)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 0.5)
    save(joinpath(output_dir, "beres_varying_Q0.png"), fig)
    println("Saved: beres_varying_Q0.png")
end

# ============================================================
# Figure 4: Beres vs AD Gaussian comparison
# ============================================================
let
    fig = Figure(; size = (800, 500), fontsize = 16)
    ax = Axis(
        fig[1, 1];
        xlabel = "Phase speed c (m/s)",
        ylabel = "Momentum flux density",
        title = "Beres vs AD Gaussian spectrum (tropical, U=0)",
    )
    Q0 = FT(10.0 / 86400.0)
    h = FT(10000.0)
    u_heat = FT(0.0)
    B_beres = wave_source(c, u_heat, Q0, h, N_source, beres_default, Val(nc))
    Bw = FT(0.4); Bn = FT(0.0); cw = FT(35.0); cn = FT(2.0); c0 = FT(0.0); flag = FT(0.0)
    B_ad = wave_source_ad(c, u_heat, Bw, Bn, cw, cn, c0, flag, Val(nc))
    lines!(ax, c_vec, collect(B_beres); label = "Beres (Q0=10 K/day, h=10km)", color = :blue)
    lines!(ax, c_vec, collect(B_ad); label = "AD Gaussian (Bw=0.4, cw=35)", color = :red, linestyle = :dash)
    axislegend(ax; position = :rt)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 0.5)
    save(joinpath(output_dir, "beres_vs_ad.png"), fig)
    println("Saved: beres_vs_ad.png")
end

# ============================================================
# Figure 5: Varying horizontal scale σ_x
# ============================================================
let
    fig = Figure(; size = (800, 500), fontsize = 16)
    ax = Axis(
        fig[1, 1];
        xlabel = "Phase speed c (m/s)",
        ylabel = "Momentum flux density",
        title = "Beres spectrum: varying σ_x (h=10km, Q0=10 K/day, U=0)",
    )
    Q0 = FT(10.0 / 86400.0)
    h = FT(10000.0)
    u_heat = FT(0.0)
    for (σ_km, color) in [(2, :blue), (4, :green), (6, :orange), (10, :red)]
        beres_σ = BeresSourceParams{FT}(;
            beres_default.Q0_threshold, beres_default.beres_scale_factor,
            σ_x = FT(σ_km * 1000.0),
            beres_default.ν_min, beres_default.ν_max, beres_default.n_ν,
        )
        B = wave_source(c, u_heat, Q0, h, N_source, beres_σ, Val(nc))
        lines!(ax, c_vec, collect(B); label = "σ_x = $(σ_km) km", color)
    end
    axislegend(ax; position = :rt)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 0.5)
    save(joinpath(output_dir, "beres_varying_sigma_x.png"), fig)
    println("Saved: beres_varying_sigma_x.png")
end

# ============================================================
# Figure 6: Smoothing comparison for U=20 m/s
# Option 1: higher n_ν (21 vs 9)
# Option 2: h-averaging (n_h_avg=20, Δh_frac=0.1)
# ============================================================
let
    fig = Figure(; size = (900, 500), fontsize = 16)
    ax = Axis(
        fig[1, 1];
        xlabel = "Phase speed c (m/s)",
        ylabel = "Momentum flux density",
        title = "Smoothing comparison: U=20 m/s, h=10km, Q0=10 K/day",
    )
    h = FT(10000.0)
    Q0 = FT(10.0 / 86400.0)
    U = FT(20.0)

    # Baseline: n_ν=9, no h-averaging
    B_base = wave_source(c, U, Q0, h, N_source, beres_default, Val(nc))
    lines!(ax, c_vec, collect(B_base);
        label = "n_ν=9, no avg (baseline)", color = :red, linewidth = 1)

    # Option 1: higher n_ν = 21
    beres_nv21 = BeresSourceParams{FT}(;
        beres_default.Q0_threshold, beres_default.beres_scale_factor,
        beres_default.σ_x, beres_default.ν_min, beres_default.ν_max,
        n_ν = 21,
    )
    B_nv21 = wave_source(c, U, Q0, h, N_source, beres_nv21, Val(nc))
    lines!(ax, c_vec, collect(B_nv21);
        label = "n_ν=21, no avg", color = :blue, linewidth = 1.5)

    # Option 2: h-averaging (n_h_avg=20, Δh_frac=0.1)
    beres_havg = BeresSourceParams{FT}(;
        beres_default.Q0_threshold, beres_default.beres_scale_factor,
        beres_default.σ_x, beres_default.ν_min, beres_default.ν_max,
        beres_default.n_ν,
        n_h_avg = 20, Δh_frac = FT(0.1),
    )
    B_havg = wave_source(c, U, Q0, h, N_source, beres_havg, Val(nc))
    lines!(ax, c_vec, collect(B_havg);
        label = "n_ν=9, h-avg (n=20, ±10%)", color = :green, linewidth = 1.5, linestyle = :dash)

    # Option 1+2: both
    beres_both = BeresSourceParams{FT}(;
        beres_default.Q0_threshold, beres_default.beres_scale_factor,
        beres_default.σ_x, beres_default.ν_min, beres_default.ν_max,
        n_ν = 21, n_h_avg = 20, Δh_frac = FT(0.1),
    )
    B_both = wave_source(c, U, Q0, h, N_source, beres_both, Val(nc))
    lines!(ax, c_vec, collect(B_both);
        label = "n_ν=21, h-avg (n=20, ±10%)", color = :purple, linewidth = 2, linestyle = :dash)

    axislegend(ax; position = :lt)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 0.5)
    save(joinpath(output_dir, "beres_smoothing_comparison.png"), fig)
    println("Saved: beres_smoothing_comparison.png")
end

println("\nAll plots saved to: $(output_dir)")
