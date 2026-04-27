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
    monin_obukhov_wind_at_height(z_target, ustar, L, uf_params, κ, z₀)

Reconstruct mean wind speed at height `z_target` (m) from Monin-Obukhov
similarity theory, given friction velocity `ustar`, Obukhov length `L`,
universal function params `uf_params`, von Kármán constant `κ`, and
roughness length `z₀`.

Pure scalar function — GPU-compatible, no allocations.
"""
function monin_obukhov_wind_at_height(z_target, ustar, L, uf_params, κ, z₀)
    ψ = UF.psi(uf_params, z_target / L, UF.MomentumTransport())
    u_z = ustar / κ * (log(z_target / z₀) - ψ)
    return u_z
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

# Precompute ∫_{r_lo}^{r_hi} _gong2003_r_integrand(r, 30) dr for each bin
# using a high-accuracy 512-point trapezoidal rule at Float64 precision.
# This runs once at module load time, not per timestep or grid cell.
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

TODO: add SST-dependent theta correction (currently fixed at theta = 30).
TODO: apply land-sea mask upstream so this is only called over ocean.
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
    sea_salt_emission_tendency_debug!(Yₜ, Y, p, t)

Constant-emission version of `sea_salt_emission_tendency!` for debugging.
Applies a uniform flux of `1e-10 kg m⁻² s⁻¹` per bin at all ocean grid cells,
skipping all Monin-Obukhov and Gong (2003) computations. Swap this in place of
`sea_salt_emission_tendency!` in `surface_flux.jl` to isolate tracer transport
issues from emission parameterization issues.
"""
function sea_salt_emission_tendency_debug!(Yₜ, Y, p, t)
    aerosol_names = _aerosol_names(p.atmos.prognostic_aerosols)
    isempty(aerosol_names) && return

    FT = eltype(Y)
    # Rough global-mean sea salt emission: ~5000 Tg/yr total, 5 bins,
    # ~3.6e14 m² ocean area → ~1e-10 kg m⁻² s⁻¹ per bin.
    const_flux = FT(1e-10)
    ocean_fraction = p.ocean_fraction

    for name in aerosol_names
        ρχ_name = Symbol(:ρ, name)
        ᶜρχ = getproperty(Y.c, ρχ_name)
        ᶜρχₜ = getproperty(Yₜ.c, ρχ_name)
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))

        sfc_flux = p.scratch.sfc_temp_C3
        @. sfc_flux = C3(const_flux * ocean_fraction)

        btt = boundary_tendency_scalar(ᶜχ, sfc_flux)
        @. ᶜρχₜ -= btt
    end
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

    u_10 = p.scratch.ᶠtemp_field_level
    @. u_10 = monin_obukhov_wind_at_height(
        FT(10),
        sfc_conditions.ustar,
        sfc_conditions.obukhov_length,
        uf_params,
        κ,
        z₀,
    )
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

    @. p.tracers.sea_salt_emission_flux_sfc = 0

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
