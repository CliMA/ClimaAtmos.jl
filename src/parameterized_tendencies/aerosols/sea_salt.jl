import SurfaceFluxes as SF
import SurfaceFluxes.Parameters as SFP
import SurfaceFluxes.UniversalFunctions as UF

# Unwrap Val{names} stored in atmos.prognostic_aerosols
_aerosol_names(::Val{names}) where {names} = names

# BIN DEFNITIONS
const SEA_SALT_BIN_BOUNDS = (
    (0.03, 0.1 ),   # μm, SSLT01
    (0.1,  0.5 ),   # μm, SSLT02
    (0.5,  1.5 ),   # μm, SSLT03
    (1.5,  5.0 ),   # μm, SSLT04
    (5.0,  10.0),   # μm, SSLT05
)


"""
    monin_obukhov_wind_extrapolated(z_target, z_anchor, u_anchor, L, uf_params, κ, z₀)

Extrapolate wind speed at `z_target` (m) by anchoring to a known model-level wind
`u_anchor` at height `z_anchor` (m) via the MOST profile ratio.
"""
function monin_obukhov_wind_extrapolated(z_target, z_anchor, u_anchor, L, uf_params, κ, z₀)
    FT = typeof(u_anchor)
    ζ_target = ifelse(iszero(L), FT(0), z_target / L)
    ζ_anchor = ifelse(iszero(L), FT(0), z_anchor / L)
    F_target = UF.dimensionless_profile(uf_params, z_target, ζ_target, z₀, UF.MomentumTransport())
    F_anchor = UF.dimensionless_profile(uf_params, z_anchor, ζ_anchor, z₀, UF.MomentumTransport())
    return max(u_anchor * F_target / F_anchor, FT(0))
end


"""
    _gong2003_r_integrand(r, theta)

The radius-dependent part of the Gong (2003) integrand with u_10 and SST
factored out:

    dF/dr = 1.373 · u_10^3.41 · SST_factor(SST) · _gong2003_r_integrand(r, theta)

This factorization allows the r-integral to be precomputed once per bin.
"""
function _gong2003_r_integrand(r, theta)
    A = 4.7 * (1 + theta * r)^(-0.017 * r^(-1.44))
    B = (0.433 - log10(r)) / 0.433
    return 1.373 * r^(-A) * (1 + 0.057 * r^3.45) * 10^(1.607 * exp(-B^2))
end

function _precompute_bin_integral(r_lo, r_hi, N = 512)
    theta = 30.0
    dr = (r_hi - r_lo) / N
    s = (_gong2003_r_integrand(r_lo, theta) + _gong2003_r_integrand(r_hi, theta)) / 2
    for i in 1:(N - 1)
        s += _gong2003_r_integrand(r_lo + i * dr, theta)
    end
    return s * dr
end

const SEA_SALT_BIN_R_INTEGRALS = ntuple(
    i -> _precompute_bin_integral(SEA_SALT_BIN_BOUNDS[i]...),
    Val(5),
)

"""
    sea_salt_emission_flux(u_10, T_sfc, bin_index)

Compute the upward sea salt number flux (particles m⁻² s⁻¹) for the bin
given by `bin_index` (1–5) using Gong (2003).
"""
function sea_salt_emission_flux(u_10, T_sfc, bin_index; SST_adj = false)
    FT = typeof(u_10)
    bin_integral = FT(SEA_SALT_BIN_R_INTEGRALS[bin_index])
    number_flux = bin_integral * abs(u_10) ^ FT(3.41)
    if SST_adj
        SST_factor = FT(0.3) + FT(0.1) * T_sfc - FT(0.0076) * T_sfc^2 + FT(0.00021) * T_sfc^3
        number_flux *= SST_factor
    end
    return number_flux
end


"""
    set_sea_salt_emission_flux!(Y, p)

Compute per-bin and total sea salt surface emission fluxes (kg m⁻² s⁻¹) and
store them in `p.tracers`, along with wind diagnostic fields.

Called during `set_explicit_precomputed_quantities!` — after surface conditions
(ustar, L) are available but before the DiagnosticEDMF column-march — so
the fluxes are ready to serve as updraft surface BCs for the EDMF scheme.

`sea_salt_emission_tendency!` reads the cached values rather than recomputing.
"""
function set_sea_salt_emission_flux!(Y, p)
    aerosol_names = _aerosol_names(p.atmos.prognostic_aerosols)
    isempty(aerosol_names) && return

    FT = eltype(Y)
    (; sfc_conditions) = p.precomputed
    surface_fluxes_params = CAP.surface_fluxes_params(p.params)
    uf_params = SFP.uf_params(surface_fluxes_params)
    κ = SFP.von_karman_const(surface_fluxes_params)

    z_c1_p  = parent(Fields.level(Fields.coordinate_field(axes(Y.c)).z, 1))
    u_z1_p  = parent(norm.(Fields.level(Y.c.uₕ, 1)))
    ustar_p = parent(sfc_conditions.ustar)
    L_p     = parent(sfc_conditions.obukhov_length)
    roughness_spec = SF.COARE3RoughnessParams{FT}()
    z0_eff = p.scratch.temp_field_level
    z₀_p   = parent(z0_eff)
    z₀_p  .= SF.momentum_roughness.(roughness_spec, ustar_p, surface_fluxes_params, nothing)

    u_10 = p.scratch.ᶠtemp_field_level
    parent(u_10) .= monin_obukhov_wind_extrapolated.(FT(10), z_c1_p, u_z1_p, L_p, uf_params, κ, z₀_p)
    # @. p.tracers.sea_salt_u10_sfc = abs(u_10)

    # Compare method for z1 winds to ground truth
    # z_c2_p  = parent(Fields.level(Fields.coordinate_field(axes(Y.c)).z, 2))
    # u_z2_p  = parent(norm.(Fields.level(Y.c.uₕ, 2)))
    # parent(p.tracers.sea_salt_u_z1_ext_sfc) .=
    #     monin_obukhov_wind_extrapolated.(z_c1_p, z_c2_p, u_z2_p, L_p, uf_params, κ, z₀_p)
    # parent(p.tracers.sea_salt_u_actual_lowest_sfc) .= u_z1_p
    
    T_sfc = sfc_conditions.T_sfc
    ocean_fraction = p.ocean_fraction
    aero_params = p.params.prescribed_aerosol_params
    bin_radii = (
        aero_params.SSLT01_radius, 
        aero_params.SSLT02_radius, 
        aero_params.SSLT03_radius,
        aero_params.SSLT04_radius, 
        aero_params.SSLT05_radius,
    )
    mass_per_particle = ntuple(
        i -> FT(4 / 3 * π * bin_radii[i]^3 * aero_params.seasalt_density),
        Val(5),
    )

    for (bin_index, name) in enumerate(aerosol_names)
        bin_flux_cache = getproperty(p.tracers.prognostic_aerosols_field, name)
        @. bin_flux_cache = sea_salt_emission_flux(u_10, T_sfc, bin_index) * mass_per_particle[bin_index] * ocean_fraction
    end
end

"""
    sea_salt_emission_tendency!(Yₜ, Y, p, t)

Apply surface emission tendencies for prognostic sea salt bins (ρSSLT01 …).

Reads per-bin fluxes from `p.tracers.prognostic_aerosols_field`, which
are pre-computed by `set_sea_salt_emission_flux!` during the precomputed-
quantities stage. The flux is applied as a bottom boundary condition using
`boundary_tendency_scalar`, which localises it to the lowest model layer.
"""
function sea_salt_emission_tendency!(Yₜ, Y, p, t)
    aerosol_names = _aerosol_names(p.atmos.prognostic_aerosols)
    isempty(aerosol_names) && return

    for (bin_index, name) in enumerate(aerosol_names)
        ρχ_name = Symbol(:ρ, name)
        ᶜρχ  = getproperty(Y.c, ρχ_name)
        ᶜρχₜ = getproperty(Yₜ.c, ρχ_name)
        ᶜχ   = @. lazy(specific(ᶜρχ, Y.c.ρ))

        bin_flux_cache = getproperty(p.tracers.prognostic_aerosols_field, name)
        sfc_flux = p.scratch.sfc_temp_C3
        @. sfc_flux = C3(bin_flux_cache)

        btt = boundary_tendency_scalar(ᶜχ, sfc_flux)
        @. ᶜρχₜ -= btt
    end
end

"""
    sea_salt_deposition_tendency!(Yₜ, Y, p, t)

Apply deposition tendencies to all prognostic sea salt bins.

Currently implemented as a simple exponential decay representing net
deposition (dry + wet) without resolving individual processes:

    d(ρSSLTxx)/dt = -λ * ρSSLTxx

where `λ = log(2) / half_life`.

TODO: replace with explicit dry deposition (function of near-surface wind
speed, particle size, and surface layer stability) and wet deposition
(function of precipitation and cloud liquid water from p.precomputed).
TODO: add size-dependent gravitational (Stokes) settling as an explicit
downward tendency for coarse bins (SSLT04–SSLT05, r > 1.5 μm), whose
settling velocities (mm s⁻¹ to cm s⁻¹) dominate over turbulent diffusion.
Fine bins (SSLT01–SSLT02) are genuinely passive and do not need settling.
TODO: make half-lives bin-specific and load from ClimaParams.
"""
function sea_salt_deposition_tendency!(Yₜ, Y, p, t)
    aerosol_names = _aerosol_names(p.atmos.prognostic_aerosols)
    isempty(aerosol_names) && return

    FT = eltype(Y)

    # TODO: make bin-specific and load from ClimaParams
    half_life = FT(0.55 * 86400)  # 1 day placeholder, in seconds
    λ = FT(log(2)) / half_life

    for name in aerosol_names
        ρχ_name = Symbol(:ρ, name)
        ᶜρχ = getproperty(Y.c, ρχ_name)
        ᶜρχₜ = getproperty(Yₜ.c, ρχ_name)
        @. ᶜρχₜ -= λ * ᶜρχ
    end
end
