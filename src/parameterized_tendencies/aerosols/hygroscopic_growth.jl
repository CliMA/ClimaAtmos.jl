# Hygroscopic growth for prognostic sea salt.
#
# Consumers (settling, dry deposition) evaluate the per-bin growth factor
# in-kernel with `sslt_growth_factor` from the relative humidity of each
# subdomain's own state, so nothing is cached. Subdomain values are composed
# as fluxes, never by averaging ξ.

"""
    sslt_kelvin_coefficient(r_dry, a, σ_w, ρ_w, R_v)

Temperature-independent part `C = (2σ_w / (ρ_w R_v a r_dry))^(3/2)` of the
Lewis (2008) Kelvin shift `(ξ_σ0 / a)^(3/2) = C · T^(-3/2)`, where
`ξ_σ0 = 2σ_w / (ρ_w R_v T r_dry)` is the surface-tension radius ratio.
Evaluated once per bin on the host.
"""
function sslt_kelvin_coefficient(r_dry, a, σ_w, ρ_w, R_v)
    x = 2 * σ_w / (ρ_w * R_v * a * r_dry)
    return x * sqrt(x)
end

"""
    sslt_kelvin_shift(C, T)

Kelvin shift `ε = C · T^(-3/2)` of the saturation deficit in Lewis Eq. 34.
"""
sslt_kelvin_shift(C, T) = C / (T * sqrt(T))

"""
    sslt_kappa_kohler_growth_factor(RH, ap)

κ-Köhler growth factor `(1 + κ a_w / (1 - a_w))^(1/3)`, `a_w = clamp(RH, 0, rh_cap)`,
without Kelvin term.
"""
function sslt_kappa_kohler_growth_factor(RH, ap)
    a_w = clamp(RH, zero(RH), ap.rh_cap)
    return cbrt(1 + ap.seasalt_kappa * a_w / (1 - a_w))
end

"""
    sslt_lewis33_growth_factor(RH, ap)

Lewis (2008, Eq. 33) bulk NaCl fit `a (b + 1 / (1 - a_w))^(1/3)`,
`a_w = clamp(RH, 0, rh_cap)`, `a = 1.08`, `b = 1.10`.
"""
function sslt_lewis33_growth_factor(RH, ap)
    a_w = clamp(RH, zero(RH), ap.rh_cap)
    return ap.lewis_a * cbrt(ap.lewis_b + 1 / (1 - a_w))
end

"""
    sslt_lewis34_growth_factor(RH, ε, ap)

Lewis (2008, Eq. 34): Eq. 33 with the dry-size-dependent Kelvin shift `ε`
([`sslt_kelvin_shift`](@ref)), `a (b + 1 / (1 - h + ε))^(1/3)`,
`h = clamp(RH, 0, 1)`; finite at RH = 1 without a cap.
"""
function sslt_lewis34_growth_factor(RH, ε, ap)
    h = clamp(RH, zero(RH), one(RH))
    return ap.lewis_a * cbrt(ap.lewis_b + 1 / (1 - h + ε))
end

"""
    sslt_growth_factor(RH, ε, ap)

Growth factor `ξ = r_wet / r_dry` used by the size consumers:
[`sslt_lewis34_growth_factor`](@ref), and `1` below the efflorescence RH.
"""
function sslt_growth_factor(RH, ε, ap)
    ξ = sslt_lewis34_growth_factor(RH, ε, ap)
    return ifelse(RH < ap.rh_effl, one(ξ), ξ)
end
