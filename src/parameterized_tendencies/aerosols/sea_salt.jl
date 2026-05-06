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
    monin_obukhov_wind_at_height(z_target, ustar, L, buoyancy_flux, uf_params, κ, z₀;
                                  gustiness_coeff = nothing, zi = nothing)

Reconstruct mean wind speed at height `z_target` (m) from Monin-Obukhov similarity
theory. `buoyancy_flux` and the keyword args `gustiness_coeff`/`zi` are only used for
the optional Beljaars (1995) free-convection gustiness correction; pass `buoyancy_flux = 0`
and omit the keyword args to get the plain MOST profile.
"""
function monin_obukhov_wind_at_height(z_target, ustar, L, 
                                       uf_params, κ, z₀; 
                                       buoyancy_flux = nothing,
                                       gustiness_coeff = nothing, 
                                       zi = nothing)
    FT = typeof(ustar)

    # MOST profile, clamped to match SurfaceFluxes.jl internal bounds
    ζ = ifelse(iszero(L), FT(0), clamp(z_target / L, FT(-100), FT(100)))
    F_m = UF.dimensionless_profile(uf_params, z_target, ζ, z₀, UF.MomentumTransport())
    u_MOST = max(ustar / κ * F_m, FT(0))

    # Beljaars (1995) free-convection gustiness floor
    if !isnothing(gustiness_coeff) && !isnothing(zi)
        w_star = cbrt(max(buoyancy_flux * zi, FT(0)))
        u_gust = gustiness_coeff * w_star
        return sqrt(u_MOST^2 + u_gust^2)
    end

    return u_MOST
end


"""
    monin_obukhov_wind_extrapolated(z_target, z_anchor, u_anchor, L, buoyancy_flux, uf_params, κ, z₀;
                                               gustiness_coeff = nothing, zi = nothing)

Extrapolate wind speed at `z_target` (m) by anchoring to a known model-level wind
`u_anchor` at height `z_anchor` (m) via the MOST profile ratio, with optional
Beljaars (1995) free-convection gustiness correction applied.
"""
function monin_obukhov_wind_extrapolated(z_target, z_anchor, u_anchor, L, buoyancy_flux, uf_params, κ, z₀;
                                                    gustiness_coeff = nothing, zi = nothing)
    FT = typeof(u_anchor)

    # MOST extrapolated profile
    ζ_target = ifelse(iszero(L), FT(0), clamp(z_target / L, FT(-100), FT(100)))
    ζ_anchor = ifelse(iszero(L), FT(0), clamp(z_anchor / L, FT(-100), FT(100)))
    F_target = UF.dimensionless_profile(uf_params, z_target, ζ_target, z₀, UF.MomentumTransport())
    F_anchor = UF.dimensionless_profile(uf_params, z_anchor, ζ_anchor, z₀, UF.MomentumTransport())
    u_MOST = max(u_anchor * F_target / F_anchor, FT(0))

    # Beljaars (1995) free-convection gustiness floor
    if !isnothing(gustiness_coeff) && !isnothing(zi)
        w_star = cbrt(max(buoyancy_flux * zi, FT(0)))
        u_gust = gustiness_coeff * w_star
        return sqrt(u_MOST^2 + u_gust^2)
    end

    return u_MOST
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

The r-integral is precomputed once per bin at module load time
(`SEA_SALT_BIN_R_INTEGRALS`). At runtime this reduces to:

    F = bin_integral · u_10^3.41 · SST_factor(T_sfc)

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
    sea_salt_emission_tendency!(Yₜ, Y, p, t)

Apply surface emission tendencies for prognostic sea salt bins (ρSSLT01 …).

Only emits over ocean grid cells (weighted by `p.ocean_fraction`, which is
set by the coupler before the first timestep and defaults to 1 everywhere
in uncoupled runs).
Reads 2.5 m wind speed from the lowest model layer and SST from
`p.precomputed.sfc_conditions.T_sfc`.

The flux from `sea_salt_emission_flux` is in kg m⁻² s⁻¹ and is applied
as a bottom boundary condition using `boundary_tendency_scalar`, which
localises it to the lowest model layer.
"""
function sea_salt_emission_tendency!(Yₜ, Y, p, t)
    aerosol_names = _aerosol_names(p.atmos.prognostic_aerosols)
    isempty(aerosol_names) && return

    FT = eltype(Y)
    (; sfc_conditions) = p.precomputed
    surface_fluxes_params = CAP.surface_fluxes_params(p.params)
    uf_params = SFP.uf_params(surface_fluxes_params)
    κ = SFP.von_karman_const(surface_fluxes_params)
    z₀ = SFP.z0m_fixed(surface_fluxes_params)

    # Center-level geometry and horizontal winds.
    # Both are on center-level-n space, which differs from the face-surface space
    # of sfc_conditions fields; use parent-level arrays to bypass the space check.
    z_c1_p = parent(Fields.level(Fields.coordinate_field(axes(Y.c)).z, 1))
    z_c2_p = parent(Fields.level(Fields.coordinate_field(axes(Y.c)).z, 2))
    u_z1_p = parent(norm.(Fields.level(Y.c.uₕ, 1)))
    u_z2_p = parent(norm.(Fields.level(Y.c.uₕ, 2)))
    ustar_p = parent(sfc_conditions.ustar)
    L_p     = parent(sfc_conditions.obukhov_length)

    u_10 = p.scratch.ᶠtemp_field_level   # will hold u_10_ext for the emission loop

    # Surface-only MOST at 10 m → diagnostic u10_mo
    @. u_10 = monin_obukhov_wind_at_height(
        FT(10), sfc_conditions.ustar, sfc_conditions.obukhov_length, uf_params, κ, z₀,
    )
    @. p.tracers.sea_salt_u10_mo_sfc = abs(u_10)

    # Extrapolated u_10: anchor = z₁ model wind → used for emission flux and diagnostic u10_ext
    parent(u_10) .= monin_obukhov_wind_extrapolated.(FT(10), z_c1_p, u_z1_p, L_p, uf_params, z₀)

    # z₁_ext: anchor = z₂ model wind, target = z₁ → diagnostic z1_ext
    parent(p.tracers.sea_salt_u_z1_ext_sfc) .=
        monin_obukhov_wind_extrapolated.(z_c1_p, z_c2_p, u_z2_p, L_p, uf_params, z₀)

    # z₁_mo: surface-only MOST at z₁ → diagnostic z1_mo (reuses existing field)
    parent(p.tracers.sea_salt_u_mo_lowest_sfc) .=
        monin_obukhov_wind_at_height.(z_c1_p, ustar_p, L_p, uf_params, κ, z₀)

    # Actual model wind at z₁ → ground truth for comparison (reuses existing field)
    parent(p.tracers.sea_salt_u_actual_lowest_sfc) .= u_z1_p

    ᶜz = Fields.coordinate_field(axes(Y.c)).z
    zi = p.scratch.ᶠtemp_field_level
    get_pbl_z!(zi, p.precomputed.ᶜp, p.precomputed.ᶜT, ᶜz, p.params.grav, p.params.cp_d)
    buoyancy_flux = sfc_conditions.buoyancy_flux
    gustiness_coeff = FT(0.5)
    parent(p.tracers.sea_salt_u_z1_ext_gust_sfc) .= monin_obukhov_wind_extrapolated.(z_c1_p, 
                                                                                     z_c2_p, 
                                                                                     u_z2_p, 
                                                                                     L_p, 
                                                                                     buoyancy_flux, 
                                                                                     uf_params, 
                                                                                     κ, 
                                                                                     z₀; 
                                                                                     gustiness_coeff, 
                                                                                     zi = zi)


    @. p.tracers.sea_salt_emission_flux_sfc = 0
    @. p.tracers.sea_salt_u10_sfc = abs(u_10)   # stores u10_ext (extrapolated, used for flux)

    T_sfc = sfc_conditions.T_sfc
    ocean_fraction = p.ocean_fraction
    aero_params = p.params.prescribed_aerosol_params
    bin_radii = (
        aero_params.SSLT01_radius, aero_params.SSLT02_radius, aero_params.SSLT03_radius,
        aero_params.SSLT04_radius, aero_params.SSLT05_radius,
    )
    # Precompute mass per particle for each bin once — constant, independent of
    # grid cell state, so no reason to recompute inside the broadcast.
    mass_per_particle = ntuple(
        i -> FT(4 / 3 * π * bin_radii[i]^3 * aero_params.seasalt_density),
        Val(5),
    )

    for (bin_index, name) in enumerate(aerosol_names)
        ρχ_name = Symbol(:ρ, name)
        ᶜρχ = getproperty(Y.c, ρχ_name)
        ᶜρχₜ = getproperty(Yₜ.c, ρχ_name)
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))

        m_p = mass_per_particle[bin_index]
        sfc_flux = p.scratch.sfc_temp_C3
        bin_flux_cache = getproperty(p.tracers.sea_salt_emission_flux_bins_sfc, name)
        @. bin_flux_cache = sea_salt_emission_flux(u_10, T_sfc, bin_index) * m_p * ocean_fraction
        @. p.tracers.sea_salt_emission_flux_sfc += bin_flux_cache
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
