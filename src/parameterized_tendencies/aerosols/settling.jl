###
### Settling Velocities
###

"""
    settling_velocity(r_dry, ξ, ρ_wet, ρ_air, μ, λ, grav, ap)

Slip-corrected Stokes terminal velocity of a wet aerosol (positive
downward, m s⁻¹), at the wet settling radius `r_wet = ξ · r_dry`, with the
air viscosity `μ` and mean free path `λ` from [`_aerosol_air_state`](@ref).

`r_dry` is the bin's dry settling radius `√(⟨r_dry⁵⟩/⟨r_dry³⟩)`
([`sslt_settling_radii`](@ref), built on [`mass_settling_radius`](@ref)).
"""
function settling_velocity(r_dry, ξ, ρ_wet, ρ_air, μ, λ, grav, ap)
    FT = typeof(r_dry)
    r_wet = r_dry * ξ
    C_c = cunningham_slip_correction(λ / r_wet, ap.cunningham_C)
    v_g = FT(2 / 9) * (ρ_wet - ρ_air) * grav * r_wet^2 * C_c / μ
    return max(v_g, zero(FT))
end

"""
    bin_settling_velocity(air, T, r_dry, C_kelvin, ρ_s, ρ_air, grav, ap)

Settling velocity of one bin at its dry settling radius `r_dry`, given the
cell's bin-independent air state `air = (; RH, μ, λ)` (relative humidity plus
[`_aerosol_air_state`](@ref)): the growth factor is evaluated once and feeds
both the wet radius and the wet density.
"""
function bin_settling_velocity(air, T, r_dry, C_kelvin, ρ_s, ρ_air, grav, ap)
    ξ = sslt_growth_factor(air.RH, sslt_kelvin_shift(C_kelvin, T), ap)
    ρ_wet = wet_density(ρ_s, ap.ρ_water, ξ)
    return settling_velocity(r_dry, ξ, ρ_wet, ρ_air, air.μ, air.λ, grav, ap)
end

###
### Helpers
###

"""
    mass_settling_radius(bin_moments, ap)

Mass-weighted settling radius `r_ref · √(M̂₅/M̂₃)` of a size bin
"""
mass_settling_radius(bin_moments, ap) =
    ap.aerosol_r_ref *
    sqrt(spectrum_moment(bin_moments, 5) / spectrum_moment(bin_moments, 3))

"""
    air_dynamic_viscosity(T, ap)

Dynamic viscosity of air μ(T) (Pa s), Seinfeld & Pandis (2006) Eq. 9.7:
`μ = μ_ref · (T/T_ref)^(3/2) · (T_ref + S)/(T + S)` with
`μ_ref = 1.8325e-5 Pa s`, `T_ref = 296.16 K`, `S = 120 K` (Sutherland form;
the constants come from ClimaParams as `μ_air_ref`, `T_μ_ref`, `S_μ`).
"""
function air_dynamic_viscosity(T, ap)
    (; μ_air_ref, T_μ_ref, S_μ) = ap
    x = T / T_μ_ref
    return μ_air_ref * x * sqrt(x) * (T_μ_ref + S_μ) / (T + S_μ)
end

"""
    cunningham_slip_correction(Kn, C)

Cunningham slip-correction factor of Zhang et al. (2001, Eq. 3).
"""
cunningham_slip_correction(Kn, C) = 1 + Kn * (C[1] + C[2] * exp(-C[3] / Kn))

"""
    wet_density(ρ_s, ρ_w, ξ)

Volume-weighted wet-particle density for fully dissolved aerosol solution.
"""
wet_density(ρ_s, ρ_w, ξ) = (ρ_s + (ξ^3 - 1) * ρ_w) / ξ^3
