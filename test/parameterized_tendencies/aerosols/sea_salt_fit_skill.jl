module SeaSaltFitSkill

import ClimaParams as CP
import Markdown

function gong_spectrum(r_hat_dry, chi; theta = 30.0)
    r_hat = chi * r_hat_dry
    A = 4.7 * (1 + theta * r_hat)^(-0.017 * r_hat^(-1.44))
    B = (0.433 - log10(r_hat)) / 0.433
    return chi *
           1.373 *
           r_hat^(-A) *
           (1 + 0.057 * r_hat^3.45) *
           10^(1.607 * exp(-B^2))
end

lognormal_spectrum(r_hat, modes) = sum(
    ((F, r, σg),) -> (F / r_hat) * exp(-log(r_hat / r)^2 / (2 * log(σg)^2)),
    modes,
)

function bin_log_simpson(f, lo, hi; N = 4096)
    x0, h = log(lo), (log(hi) - log(lo)) / N
    s = f(lo) * lo + f(hi) * hi
    for j in 1:(N - 1)
        x = exp(x0 + j * h)
        s += (isodd(j) ? 4 : 2) * f(x) * x
    end
    return s * h / 3
end

function stored_params(toml_dict = CP.create_toml_dict(Float64))
    params = CP.get_parameter_values(
        toml_dict,
        [
            "ssa_gong_logfit_mode1",
            "ssa_gong_logfit_mode2",
            "ssa_gong_logfit_mode3",
            "ssa_gong_logfit_bin_0M_flux",
            "ssa_gong_logfit_bin_3M_flux",
            "ssa_size_bin_divisions",
            "ssa_r_ref",
            "ssa_r80_per_dry",
            "seasalt_aerosol_density",
        ],
    )
    return (;
        modes = (
            Tuple(params.ssa_gong_logfit_mode1),
            Tuple(params.ssa_gong_logfit_mode2),
            Tuple(params.ssa_gong_logfit_mode3),
        ),
        r_hat_edges = params.ssa_size_bin_divisions ./ params.ssa_r_ref,
        r80_per_dry = params.ssa_r80_per_dry,
        bin_number_scales = params.ssa_gong_logfit_bin_0M_flux,
        bin_mass_scales = params.ssa_gong_logfit_bin_3M_flux,
        r_ref = params.ssa_r_ref,
        ρ_dry = params.seasalt_aerosol_density,
    )
end

"""
    skill(sp)

Deviations of the stored fit `sp` (see [`stored_params`](@ref)) from the exact
Gong spectrum in dry-radius coordinates: max pointwise relative error over
r̂ ∈ [0.015, 10], and per-bin relative errors of the number and dry-mass fluxes.
"""
function skill(sp)
    (; modes, r_hat_edges, r80_per_dry) = sp
    r_dense = exp10.(range(log10(0.015), log10(10), 600))
    max_pointwise = maximum(
        abs(lognormal_spectrum(r, modes) / gong_spectrum(r, r80_per_dry) - 1)
        for r in r_dense
    )
    moment_errors(moment) = map(1:(length(r_hat_edges) - 1)) do i
        fit = bin_log_simpson(
            r -> lognormal_spectrum(r, modes) * r^moment,
            r_hat_edges[i],
            r_hat_edges[i + 1],
        )
        ref = bin_log_simpson(
            r -> gong_spectrum(r, r80_per_dry) * r^moment,
            r_hat_edges[i],
            r_hat_edges[i + 1],
        )
        return fit / ref - 1
    end
    return (; max_pointwise, number = moment_errors(0), mass = moment_errors(3))
end

pct(x) = round(100 * x; digits = 1)

markdown(s = skill(stored_params())) = Markdown.parse(
    "The fit's maximum pointwise deviation from the Gong spectrum is \
     $(pct(s.max_pointwise))%; over the $(length(s.number)) MERRA-2 bins, \
     its number fluxes deviate by at most $(pct(maximum(abs, s.number)))% \
     and its dry-mass fluxes by at most $(pct(maximum(abs, s.mass)))% \
     (recomputed from the ClimaParams-stored modes at docs build time).",
)

end # module
