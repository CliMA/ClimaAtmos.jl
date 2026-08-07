# Skill of the stored 3-mode lognormal fit against the exact Gong (2003)
# spectrum. The `@eval` block in `aerosols.md` includes this file at docs
# build time, so the quoted numbers regenerate from the ClimaParams-stored
# modes and can never go stale; `sea_salt_emission_fit.jl` includes it too
# and prints the same numbers alongside the flux scales.
module SeaSaltFitSkill

import ClimaParams as CP
import Markdown

# Gong (2003) dimensionless number spectrum dF/dr̂ at u_10 = 1 m/s, Θ = 30.
function gong_spectrum(r_hat; theta = 30.0)
    A = 4.7 * (1 + theta * r_hat)^(-0.017 * r_hat^(-1.44))
    B = (0.433 - log10(r_hat)) / 0.433
    return 1.373 *
           r_hat^(-A) *
           (1 + 0.057 * r_hat^3.45) *
           10^(1.607 * exp(-B^2))
end

lognormal_spectrum(r_hat, modes) = sum(
    ((F, r, σg),) -> (F / r_hat) * exp(-log(r_hat / r)^2 / (2 * log(σg)^2)),
    modes,
)

# Composite Simpson in log(x); the spectrum spans decades in radius.
function bin_log_simpson(f, lo, hi; N = 4096)
    x0, h = log(lo), (log(hi) - log(lo)) / N
    s = f(lo) * lo + f(hi) * hi
    for j in 1:(N - 1)
        x = exp(x0 + j * h)
        s += (isodd(j) ? 4 : 2) * f(x) * x
    end
    return s * h / 3
end

function stored_params()
    params = CP.get_parameter_values(
        CP.create_toml_dict(Float64),
        [
            "ssa_gong_logfit_mode1",
            "ssa_gong_logfit_mode2",
            "ssa_gong_logfit_mode3",
            "ssa_size_bin_divisions",
            "ssa_r_ref",
        ],
    )
    modes = (
        Tuple(params.ssa_gong_logfit_mode1),
        Tuple(params.ssa_gong_logfit_mode2),
        Tuple(params.ssa_gong_logfit_mode3),
    )
    return modes, params.ssa_size_bin_divisions ./ params.ssa_r_ref
end

"""
    skill(modes, r_hat_edges)

Deviations of the lognormal `modes` from the exact Gong spectrum: max
pointwise relative error over r̂ ∈ [0.03, 20], and per-bin relative errors
of the number (0th-moment) and dry-mass (3rd-moment) fluxes over the
`r_hat_edges` bins. Constant moment prefactors cancel in the ratios.
"""
function skill(modes, r_hat_edges)
    r_dense = exp10.(range(log10(0.03), log10(20), 600))
    max_pointwise = maximum(
        abs(lognormal_spectrum(r, modes) / gong_spectrum(r) - 1) for
        r in r_dense
    )
    moment_errors(moment) = map(1:(length(r_hat_edges) - 1)) do i
        fit = bin_log_simpson(
            r -> lognormal_spectrum(r, modes) * r^moment,
            r_hat_edges[i],
            r_hat_edges[i + 1],
        )
        ref = bin_log_simpson(
            r -> gong_spectrum(r) * r^moment,
            r_hat_edges[i],
            r_hat_edges[i + 1],
        )
        return fit / ref - 1
    end
    return (; max_pointwise, number = moment_errors(0), mass = moment_errors(3))
end

pct(x) = round(100 * x; digits = 1)

markdown(s = skill(stored_params()...)) = Markdown.parse(
    "The fit's maximum pointwise deviation from the Gong spectrum is \
     $(pct(s.max_pointwise))%; over the $(length(s.number)) MERRA-2 bins, \
     its number fluxes deviate by at most $(pct(maximum(abs, s.number)))% \
     and its dry-mass fluxes by at most $(pct(maximum(abs, s.mass)))% \
     (recomputed from the ClimaParams-stored modes at docs build time).",
)

end # module
