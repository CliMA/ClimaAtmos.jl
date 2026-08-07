import ClimaParams as CP
using CairoMakie

#####
##### Parameters
#####

params = CP.get_parameter_values(
    CP.create_toml_dict(Float64),
    [
        "ssa_gong_logfit_mode1",
        "ssa_gong_logfit_mode2",
        "ssa_gong_logfit_mode3",
        "ssa_size_bin_divisions",
        "ssa_r_ref",
        "ssa_r80_per_dry",
        "seasalt_aerosol_density",
    ],
)
stored_modes = (
    Tuple(params.ssa_gong_logfit_mode1),
    Tuple(params.ssa_gong_logfit_mode2),
    Tuple(params.ssa_gong_logfit_mode3),
)
r_ref = params.ssa_r_ref
ρ_dry = params.seasalt_aerosol_density
r80_per_dry = params.ssa_r80_per_dry
r_hat_edges = params.ssa_size_bin_divisions ./ r_ref


#####
##### Spectra
#####

# Gong (2003) eq. (2) number spectrum at u_10 = 1 m/s and Θ = 30, defined at
# the 80% RH radius, converted to the dimensionless dry radius
# r̂ = r_dry / r_ref via r̂₈₀ = χ r̂: dF/dr̂ = χ (dF/dr̂₈₀)(χ r̂).
function gong2003_spectrum(r_hat_dry; theta = 30.0, chi = r80_per_dry)
    r_hat = chi * r_hat_dry
    A = 4.7 * (1 + theta * r_hat)^(-0.017 * r_hat^(-1.44))
    B = (0.433 - log10(r_hat)) / 0.433
    return chi *
           1.373 *
           r_hat^(-A) *
           (1 + 0.057 * r_hat^3.45) *
           10^(1.607 * exp(-B^2))
end

# 3-mode lognormal composed from distributions parameterized by (F, r, σg)
# with scale factor F [m⁻² s⁻¹], modal radius r [m], and geometric standard deviation σg [-].
lognormal_spectrum(r_hat, modes) = sum(modes) do (F, r_mode, σg)
    F / r_hat * exp(-log(r_hat / r_mode)^2 / (2 * log(σg)^2))
end

#####
##### Fit: minimize the max relative error on a log-spaced radius grid
#####

# Dependency-free Nelder-Mead (reflection/expansion/contraction/shrink).
function nelder_mead(f, x0; iters = 20_000, init_step = 0.1)
    n = length(x0)
    simplex = [copy(x0)]
    for i in 1:n
        x = copy(x0)
        x[i] += init_step
        push!(simplex, x)
    end
    fvals = f.(simplex)
    for _ in 1:iters
        order = sortperm(fvals)
        simplex, fvals = simplex[order], fvals[order]
        centroid = sum(simplex[1:n]) / n
        xr = 2 * centroid - simplex[end]
        fr = f(xr)
        if fr < fvals[1]
            xe = 3 * centroid - 2 * simplex[end]
            fe = f(xe)
            simplex[end], fvals[end] = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < fvals[n]
            simplex[end], fvals[end] = xr, fr
        else
            xc = (centroid + simplex[end]) / 2
            fc = f(xc)
            if fc < fvals[end]
                simplex[end], fvals[end] = xc, fc
            else
                for i in 2:(n + 1)
                    simplex[i] = (simplex[i] + simplex[1]) / 2
                    fvals[i] = f(simplex[i])
                end
            end
        end
    end
    i_best = argmin(fvals)
    return simplex[i_best], fvals[i_best]
end

# Unconstrained fit variables: (log F, log r, log(σg - 1)) per mode.
to_modes(x) = ntuple(i -> (exp(x[3i - 2]), exp(x[3i - 1]), 1 + exp(x[3i])), 3)
from_modes(modes) =
    collect(Iterators.flatten((log(F), log(r), log(σ - 1)) for (F, r, σ) in modes))

r_grid = exp10.(range(log10(0.015), log10(10), 200))
target = gong2003_spectrum.(r_grid)
max_rel_err(modes) =
    maximum(abs(lognormal_spectrum(r, modes) / t - 1) for (r, t) in zip(r_grid, target))

guess = ((0.2, 0.05, 15.0), (60.0, 0.1, 1.8), (6.0, 0.8, 1.8))
x_fit, err_fit = nelder_mead(x -> max_rel_err(to_modes(x)), from_modes(guess))
fit_modes = to_modes(x_fit)
println("Fitted modes (F [m⁻² s⁻¹], r [r_ref], σg): ", fit_modes)
println("Max relative error vs Gong (2003): ", round(err_fit; sigdigits = 3))

#####
##### Per-bin flux scales from the ClimaParams (rounded) modes
#####

# ∫ f(x) dx over [lo, hi] by composite Simpson in log(x); the spectrum
# spans decades in radius, so log spacing converges much faster than linear.
function bin_log_simpson(f, lo, hi; N = 4096)
    x0, h = log(lo), (log(hi) - log(lo)) / N
    s = f(lo) * lo + f(hi) * hi
    for j in 1:(N - 1)
        x = exp(x0 + j * h)
        s += (isodd(j) ? 4 : 2) * f(x) * x
    end
    return s * h / 3
end

# moment = 0: number flux scale [m⁻² s⁻¹]; moment = 3 with the 4π/3 ρ_dry
# prefactor: dry-mass flux scale [kg m⁻² s⁻¹]. Both at u_10 = u_ref.
bin_moment(i, moment) = bin_log_simpson(
    r_hat -> lognormal_spectrum(r_hat, stored_modes) * (r_hat * r_ref)^moment,
    r_hat_edges[i],
    r_hat_edges[i + 1],
)
n_bins = length(r_hat_edges) - 1
number_scales = [bin_moment(i, 0) for i in 1:n_bins]
mass_scales = [4π / 3 * ρ_dry * bin_moment(i, 3) for i in 1:n_bins]
println("ssa_gong_logfit_bin_0M_flux: ", round.(number_scales; sigdigits = 4))
println("ssa_gong_logfit_bin_3M_flux: ", round.(mass_scales; sigdigits = 4))

#####
##### Fit skill vs the exact Gong spectrum (numbers quoted in aerosols.md)
#####

# The same module feeds the `@eval` block in aerosols.md at docs build time.
include(joinpath(@__DIR__, "sea_salt_fit_skill.jl"))
skill = SeaSaltFitSkill.skill(stored_modes, r_hat_edges, r80_per_dry)
println(
    "max pointwise deviation vs Gong: ",
    SeaSaltFitSkill.pct(skill.max_pointwise), "%",
)
println(
    "MERRA-2 bin number-flux deviations [%]: ",
    SeaSaltFitSkill.pct.(skill.number),
)
println(
    "MERRA-2 bin dry-mass-flux deviations [%]: ",
    SeaSaltFitSkill.pct.(skill.mass),
)

#####
##### Figure: modes, their sum, and the Gong spectrum, as dF/dln r̂
#####

u_10 = 10.0  # m/s; the wind power law u_10^3.41 only scales the amplitude
wind_factor = u_10^3.41
r_plot = exp10.(range(log10(0.015), log10(10), 400))

fig = Figure(size = (900, 560))
ax = Axis(
    fig[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = "dry radius r̂ = r_dry/r_ref (r_ref = $(round(r_ref * 1e6; sigdigits = 3)) μm)",
    ylabel = "dF/dln r̂ (m⁻² s⁻¹)",
    title = "3-mode lognormal fit of the Gong (2003) spectrum (u₁₀ = $u_10 m/s)",
)
for (i, (F, r_mode, σg)) in enumerate(stored_modes)
    lines!(
        ax,
        r_plot,
        [wind_factor * F * exp(-log(r / r_mode)^2 / (2 * log(σg)^2)) for r in r_plot],
        label = "mode $i (r = $r_mode)",
    )
end
lines!(
    ax,
    r_plot,
    [wind_factor * r * lognormal_spectrum(r, stored_modes) for r in r_plot],
    color = :black,
    linewidth = 3,
    label = "sum of 3 modes",
)
lines!(
    ax,
    r_plot,
    [wind_factor * r * gong2003_spectrum(r) for r in r_plot],
    color = :gray,
    linestyle = :dot,
    linewidth = 3,
    label = "Gong (2003), Θ = 30",
)
ylims!(ax, 1e-2, 1e6)
axislegend(ax, position = :lb)
save(joinpath(@__DIR__, "assets", "gong_ln3_modes.png"), fig)
println("Saved assets/gong_ln3_modes.png")
