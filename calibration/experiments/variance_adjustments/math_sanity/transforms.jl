# Requires `gauss_hermite_tables.jl` included first (defines `mathsanity_z_from_χ`).

# Maps matching ClimaAtmos `get_physical_point(::GaussianSGS, ...)` (before physical clamps).

function mathsanity_gaussian_sgs_unclamped(χ1, χ2, μ_q, μ_T, σ_q, σ_T, ρ)
    sqrt2 = sqrt(2)
    z_q = mathsanity_z_from_χ(χ1)
    z_T = ρ * mathsanity_z_from_χ(χ1) + sqrt(max(0.0, 1.0 - ρ^2)) * mathsanity_z_from_χ(χ2)
    q = μ_q + σ_q * z_q
    T = μ_T + σ_T * z_T
    return q, T
end

"""Clamped version (same as ClimaAtmos GaussianSGS path)."""
function mathsanity_gaussian_sgs_clamped(χ1, χ2, μ_q, μ_T, σ_q, σ_T, ρ, T_min, q_max)
    sqrt2 = sqrt(2)
    ϵ = 1e-14
    q_hat = clamp(μ_q + sqrt2 * σ_q * χ1, 0.0, q_max)
    χ1_eff = (q_hat - μ_q) / (sqrt2 * max(σ_q, ϵ))
    σ_c = sqrt(max(1.0 - ρ^2, 0.0)) * σ_T
    μ_c = μ_T + sqrt2 * ρ * σ_T * χ1_eff
    T_hat = max(T_min, μ_c + sqrt2 * σ_c * χ2)
    return q_hat, T_hat
end
