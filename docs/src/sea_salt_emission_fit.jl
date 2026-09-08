#=
Renders the aerosols.md figure assets/gong_ln3_modes.png (gitignored; rebuilt by
docs/make.jl): the ClimaParams-stored 3-mode lognormal fit of the Gong (2003)
sea salt emission spectrum, its modes, and the stored per-bin number emission,
against the exact Gong spectrum. Everything plotted comes from ClimaParams; the
spectra live in SeaSaltFitSkill, which the fit-skill unit test also exercises.

Run from the docs environment, e.g.
`julia --project=docs docs/src/sea_salt_emission_fit.jl`. Requires ClimaParams
>= 1.1.6, where the ssa_* entries live.
=#

import ClimaParams as CP
using CairoMakie

include(
    joinpath(
        @__DIR__,
        "..",
        "..",
        "test",
        "parameterized_tendencies",
        "aerosols",
        "sea_salt_fit_skill.jl",
    ),
)
import .SeaSaltFitSkill as FS

#####
##### Parameters
#####

sp = FS.stored_params(CP.create_toml_dict(Float64))
(; r_hat_edges, r80_per_dry, r_ref) = sp
stored_modes = sp.modes
# Per-bin number flux scales [m⁻² s⁻¹] at u_10 = u_ref
# (ssa_gong_logfit_bin_0M_flux), stored alongside the modes.
number_scales = sp.bin_number_scales
n_bins = length(r_hat_edges) - 1

#####
##### Figure: modes, their sum, and the Gong spectrum, as dF/dln r̂
#####

u_10 = 10.0  # m/s; the wind power law u_10^3.41 only scales the amplitude
wind_factor = u_10^3.41
r_plot = exp10.(range(log10(0.015), log10(10), 400))

edges_μm = r_hat_edges .* (r_ref * 1e6)  # r̂ is numerically μm for r_ref = 1 μm
fig = Figure(size = (900, 560))
ax = Axis(
    fig[1, 1],
    xscale = log10,
    yscale = log10,
    xticks = (collect(edges_μm), string.(round.(edges_μm; sigdigits = 3))),
    xlabel = "dry radius (μm)",
    ylabel = "dF/dln r̂ (m⁻² s⁻¹)",
    title = "3-mode lognormal fit of the Gong (2003) spectrum (u₁₀ = $u_10 m/s)",
)
# Stored MERRA-2 bin number emissions, drawn as the bin-mean dF/dln r̂
# (k_i⁽⁰⁾ / Δln r̂) so the bar heights are directly comparable to the curves.
y_floor = 1e-2  # bar base; matches the axis lower limit
bin_bars = map(1:n_bins) do i
    lo, hi = r_hat_edges[i], r_hat_edges[i + 1]
    height = wind_factor * number_scales[i] / log(hi / lo)
    Rect2(lo, y_floor, hi - lo, height - y_floor)
end
poly!(
    ax,
    bin_bars,
    color = (:steelblue, 0.3),
    strokecolor = :steelblue,
    strokewidth = 1,
    label = "bin emission",
)
for (i, bar) in enumerate(bin_bars)
    text!(
        ax,
        sqrt(bar.origin[1] * (bar.origin[1] + bar.widths[1])),
        2 * y_floor;
        text = "bin $i",
        align = (:center, :bottom),
        color = :steelblue,
        fontsize = 13,
    )
end
for (i, (F, r_mode, σg)) in enumerate(stored_modes)
    lines!(
        ax,
        r_plot,
        [
            wind_factor * F * exp(-log(r / r_mode)^2 / (2 * log(σg)^2)) for
            r in r_plot
        ],
        label = "mode $i (r = $r_mode)",
    )
end
lines!(
    ax,
    r_plot,
    [wind_factor * r * FS.lognormal_spectrum(r, stored_modes) for r in r_plot],
    color = :black,
    linewidth = 3,
    label = "sum of 3 modes",
)
lines!(
    ax,
    r_plot,
    [wind_factor * r * FS.gong_spectrum(r, r80_per_dry) for r in r_plot],
    color = :gray,
    linestyle = :dot,
    linewidth = 3,
    label = "Gong (2003), Θ = 30",
)
ylims!(ax, 1e-2, 1e6)
axislegend(ax, position = :rt)
figure_path = joinpath(@__DIR__, "assets", "gong_ln3_modes.png")
save(figure_path, fig)
println("Saved ", figure_path)
