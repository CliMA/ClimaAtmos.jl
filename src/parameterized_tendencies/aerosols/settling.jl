# Gravitational settling physics for prognostic sea salt: slip-corrected
# Stokes velocity of a bin at its wet mass-weighted settling radius.

"""
    air_dynamic_viscosity(T, ap)

Dynamic viscosity of air μ(T) (Pa s) from Sutherland's law, with the reference
viscosity/temperature and Sutherland constant from ClimaParams
(`μ_air_ref`, `T_μ_ref`, `S_μ`).
"""
function air_dynamic_viscosity(T, ap)
    (; μ_air_ref, T_μ_ref, S_μ) = ap
    x = T / T_μ_ref
    return μ_air_ref * x * sqrt(x) * (T_μ_ref + S_μ) / (T + S_μ)
end

"""
    cunningham_slip_correction(Kn, ap)

Cunningham slip-correction factor `Cc(Kn) = 1 + Kn(A + B·exp(-C/Kn))` with
coefficients `ap.cunningham_C = [A, B, C]`. `Cc → 1` for the coarse bins
(continuum regime) and grows for fine bins where the particle size approaches
the mean free path.
"""
function cunningham_slip_correction(Kn, ap)
    C = ap.cunningham_C
    return 1 + Kn * (C[1] + C[2] * exp(-C[3] / Kn))
end

"""
    sslt_settling_velocity(r_wet, ρ_wet, ρ_air, T, R_d, grav, ap)

Slip-corrected Stokes terminal velocity of a wet sea salt particle (positive
downward, m s⁻¹): `v_g = (2/9)·(ρ_wet - ρ_air)·g·r_wet²·Cc/μ`, with the mean
free path `λ = μ/(0.499·ρ_air·v̄)`, `v̄ = √(8 R_d T/π)`, `Kn = λ/r_wet`, and
`Cc` from [`cunningham_slip_correction`](@ref).

`r_wet` is the bin's wet settling radius `ξ · √(⟨r_dry⁵⟩/⟨r_dry³⟩)`
([`sslt_settling_radii`](@ref)).
"""
function sslt_settling_velocity(r_wet, ρ_wet, ρ_air, T, R_d, grav, ap)
    FT = typeof(r_wet)
    μ = air_dynamic_viscosity(T, ap)
    v̄ = sqrt(8 * R_d * T / FT(π))
    λ = μ / (FT(0.499) * ρ_air * v̄)
    C_c = cunningham_slip_correction(λ / r_wet, ap)
    v_g = FT(2 / 9) * (ρ_wet - ρ_air) * grav * r_wet^2 * C_c / μ
    return max(v_g, zero(FT))
end

"""
    sslt_wet_density(ρ_s, ρ_w, ξ)

Volume-weighted wet-particle density for a dry salt core (density `ρ_s`)
coated with condensed water (density `ρ_w`):
`ρ_wet = (ρ_s + (ξ³ - 1) · ρ_w) / ξ³`, tending to `ρ_s` as `ξ → 1` and to
`ρ_w` as `ξ → ∞`.
"""
function sslt_wet_density(ρ_s, ρ_w, ξ)
    ξ3 = ξ^3
    return (ρ_s + (ξ3 - 1) * ρ_w) / ξ3
end

"""
    sslt_bin_settling_velocity(RH, T, r_dry, C_kelvin, ρ_s, ρ_air, R_d, grav, ap)

Settling velocity of one bin at its dry settling radius `r_dry`: the growth
factor is evaluated once and feeds both the wet radius and the wet density.
"""
function sslt_bin_settling_velocity(RH, T, r_dry, C_kelvin, ρ_s, ρ_air, R_d, grav, ap)
    ξ = sslt_growth_factor(RH, sslt_kelvin_shift(C_kelvin, T), ap)
    ρ_wet = sslt_wet_density(ρ_s, ap.ρ_water, ξ)
    return sslt_settling_velocity(r_dry * ξ, ρ_wet, ρ_air, T, R_d, grav, ap)
end
