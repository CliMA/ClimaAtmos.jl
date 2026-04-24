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
    gong2003_dF_dr(r, u_10, theta)

Gong (2003) sea salt number emission spectrum (particles m⁻² s⁻¹ μm⁻¹) at
dry radius `r` (μm), 10 m wind speed `u_10` (m s⁻¹), and temperature-
dependent parameter `theta`.

Reference: Gong, S. L. (2003), A parameterization of sea-salt aerosol source
function for sub- and super-micron particles, Global Biogeochem. Cycles,
17(4), 1097, doi:10.1029/2003GB002079.
"""
function gong2003_dF_dr(r, u_10, ϴ, SST; SST_adj=true)
    A = 4.7 * (1 + ϴ * r)^(-0.017 * r^(-1.44))
    B = (0.433 - log10(r)) / 0.433

    dF_dr = 1.373 * u_10^3.41 * r^(-A) * (1 + 0.057 * r^3.45) * 10^(1.607 * exp(-B^2))

    if SST_adj
        SST_factor = 0.3 + 0.1 * SST - 0.0076 * SST^2 + 0.00021 * SST^3
        dF_dr *= SST_factor
    end

    return dF_dr
end

"""
    integrate_bin_gong2003(r_lo, r_hi, u_10, theta, SST, ::Val{N}) where {N}

Integrate `gong2003_dF_dr` over the radius interval [r_lo, r_hi] (μm) using
an N-point composite trapezoidal rule. Returns particle number flux
(particles m⁻² s⁻¹) for the bin.

Using `Val{N}` keeps N a compile-time constant so this is GPU-compatible.
The number of quadrature points is set by the caller via e.g. `Val(32)`.
"""
function integrate_bin_gong2003(r_lo, r_hi, u_10, theta, SST, ::Val{N}) where {N}
    dr = (r_hi - r_lo) / N
    # trapezoidal: 0.5 * (f(r0) + f(rN)) + sum f(r1..r_{N-1})
    s = (gong2003_dF_dr(r_lo, u_10, theta, SST) + gong2003_dF_dr(r_hi, u_10, theta, SST)) / 2
    s += sum(ntuple(i -> gong2003_dF_dr(r_lo + i * dr, u_10, theta, SST), Val(N - 1)))
    return s * dr
end

"""
    sea_salt_emission_flux(u_10, T_sfc, bin_index)

Compute the upward sea salt number flux (particles m⁻² s⁻¹) for the bin
given by `bin_index` (1–5, corresponding to SSLT01–SSLT05), using the
Gong (2003) parameterization integrated over the bin's radius range.

`u_10` is 10 m wind speed (m s⁻¹), `T_sfc` is sea surface temperature (K).

TODO: convert number flux → mass flux (kg m⁻² s⁻¹) using assumed particle
density and bin mean radius.
TODO: add SST-dependent theta correction (currently fixed at theta = 30).
TODO: apply land-sea mask upstream so this is only called over ocean.
"""
function sea_salt_emission_flux(u_10, T_sfc, bin_index)
    theta = typeof(u_10)(30) # default value from Gong (2003)
    r_lo, r_hi = typeof(u_10).(SEA_SALT_BIN_BOUNDS[bin_index])
    return integrate_bin_gong2003(r_lo, r_hi, u_10, theta, T_sfc, Val(32))
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
        @. ᶜρχₜ += btt
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

    for (bin_index, name) in enumerate(aerosol_names)
        ρχ_name = Symbol(:ρ, name)
        ᶜρχ = getproperty(Y.c, ρχ_name)
        ᶜρχₜ = getproperty(Yₜ.c, ρχ_name)
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))

        sfc_flux = p.scratch.sfc_temp_C3
        @. sfc_flux = C3(sea_salt_emission_flux(u_10, T_sfc, bin_index) * ocean_fraction)

        btt = boundary_tendency_scalar(ᶜχ, sfc_flux)
        @. ᶜρχₜ += btt
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
