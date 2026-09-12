###
### Settling Velocities
###

"""
    settling_velocity(r_wet, ρ_wet, ρ_air, T, R_d, grav, ap)

Slip-corrected Stokes terminal velocity of a wet aerosol (positive
downward, m s⁻¹).

`r_wet` is the bin's wet settling radius `ξ · √(⟨r_dry⁵⟩/⟨r_dry³⟩)`
([`sslt_settling_radii`](@ref), built on [`mass_settling_radius`](@ref)).
"""
function settling_velocity(r_wet, ρ_wet, ρ_air, T, R_d, grav, ap)
    FT = typeof(r_wet)
    μ = air_dynamic_viscosity(T, ap)
    v̄ = sqrt(8 * R_d * T / FT(π))
    λ = μ / (FT(0.499) * ρ_air * v̄)
    C_c = cunningham_slip_correction(λ / r_wet, ap.cunningham_C)
    v_g = FT(2 / 9) * (ρ_wet - ρ_air) * grav * r_wet^2 * C_c / μ
    return max(v_g, zero(FT))
end

"""
    bin_settling_velocity(RH, T, r_dry, C_kelvin, ρ_s, ρ_air, R_d, grav, ap)

Settling velocity of one bin at its dry settling radius `r_dry`: the growth
factor is evaluated once and feeds both the wet radius and the wet density.
"""
function bin_settling_velocity(RH, T, r_dry, C_kelvin, ρ_s, ρ_air, R_d, grav, ap)
    ξ = growth_factor(RH, kelvin_shift(C_kelvin, T), ap)
    r_wet = ξ * r_dry
    ρ_wet = wet_density(ρ_s, ap.ρ_water, ξ)
    return settling_velocity(r_wet, ρ_wet, ρ_air, T, R_d, grav, ap)
end

###
### Helpers
###

"""
    mass_settling_radius(bin_moments)

Mass-weighted settling radius `r_ref · √(M̂₅/M̂₃)` of a size bin
"""
mass_settling_radius(bin_moments, ap) =
    ap.r_ref * sqrt(_spectrum_moment(bin_moments, 5) / _spectrum_moment(bin_moments, 3))

"""
    cunningham_slip_correction(Kn, C)

Cunningham slip-correction factor [TODO: cite]
"""
cunningham_slip_correction(Kn, C) = 1 + Kn * (C[1] + C[2] * exp(-C[3] / Kn))

"""
    wet_density(ρ_s, ρ_w, ξ)

Volume-weighted wet-particle density for fully dissolved aerosol solution.
"""
wet_density(ρ_s, ρ_w, ξ) =  (ρ_s + (ξ^3 - 1) * ρ_w) / ξ^3
