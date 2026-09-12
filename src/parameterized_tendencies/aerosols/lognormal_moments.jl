import SpecialFunctions: erf, erfcx

"""
    lognormal_mode_spectrum(r, modes)

Size distribution `dF/dr` of a mixture of lognormal `modes`, each a tuple
`(F, r_mode, σg)` of amplitude, mode radius, and geometric standard deviation.
"""
lognormal_mode_spectrum(r, modes) = sum(modes) do (F, r_mode, σg)
    F / r * exp(-log(r / r_mode)^2 / (2 * log(σg)^2))
end

"""
    lognormal_bin_moment(k, r_lo, r_hi, modes)

The `k`-th radius moment `∫_{r_lo}^{r_hi} rᵏ (dF/dr) dr` of the lognormal
mixture `modes` (see [`lognormal_mode_spectrum`](@ref)), in closed form.

Evaluated using `erf`, branched to use `erfcx` when bins don't include 
the lognomal mean.
"""
lognormal_bin_moment(k, r_lo, r_hi, modes) = sum(modes) do (F, r_mode, σg)
    s, μ = log(σg), log(r_mode)
    ℓ(r) = k * log(r) - (log(r) - μ)^2 / (2 * s^2)
    x(r) = (log(r) - μ - k * s^2) / (s * sqrt(2))
    x_lo, x_hi = x(r_lo), x(r_hi)
    ∫ = if x_lo ≥ 0       # center below the bin
        exp(ℓ(r_lo)) * erfcx(x_lo) - exp(ℓ(r_hi)) * erfcx(x_hi)
    elseif x_hi ≤ 0   # center above the bin
        exp(ℓ(r_hi)) * erfcx(-x_hi) - exp(ℓ(r_lo)) * erfcx(-x_lo)
    else              # center inside the bin: no cancellation
        exp(k * μ + k^2 * s^2 / 2) * (erf(x_hi) - erf(x_lo))
    end
    F * s * sqrt(π / 2) * ∫
end

_spectrum_moment(bin_moments, k) = bin_moments[k + 1]
