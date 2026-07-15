import SurfaceFluxes as SF
import SurfaceFluxes.Parameters as SFP
import SurfaceFluxes.UniversalFunctions as UF

# BIN DEFNITIONS
const SEA_SALT_BIN_BOUNDS = (
    (0.03, 0.1),   # μm, SSLT01
    (0.1, 0.5),   # μm, SSLT02
    (0.5, 1.5),   # μm, SSLT03
    (1.5, 5.0),   # μm, SSLT04
    (5.0, 10.0),   # μm, SSLT05
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
    F_target =
        UF.dimensionless_profile(uf_params, z_target, ζ_target, z₀, UF.MomentumTransport())
    F_anchor =
        UF.dimensionless_profile(uf_params, z_anchor, ζ_anchor, z₀, UF.MomentumTransport())
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
    sea_salt_mean_particle_volume(r_dry, σ)

Number-weighted mean particle volume of a lognormal mode with number-median
dry radius `r_dry` and geometric standard deviation `σ` (its third radial
moment)
"""
function sea_salt_mean_particle_volume(r_dry, σ)
    FT = typeof(r_dry)
    return FT(4 / 3) * FT(π) * r_dry^3 * exp(FT(9 / 2) * log(σ)^2)
end

"""
    sea_salt_emission_flux(u_10, T_sfc, bin_index)

Compute the upward sea salt number flux (particles m⁻² s⁻¹) for the bin
given by `bin_index` (1–5) using Gong (2003).
"""
function sea_salt_emission_flux(u_10, T_sfc, bin_index; SST_adj = false)
    FT = typeof(u_10)
    bin_integral = FT(SEA_SALT_BIN_R_INTEGRALS[bin_index])
    number_flux = bin_integral * abs(u_10)^FT(3.41)
    if SST_adj
        SST_factor =
            FT(0.3) + FT(0.1) * T_sfc - FT(0.0076) * T_sfc^2 + FT(0.00021) * T_sfc^3
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
    interactive_aerosol_names = _aerosol_names(p.atmos.interactive_aerosols)
    isempty(interactive_aerosol_names) && return

    FT = eltype(Y)
    (; sfc_conditions) = p.precomputed
    surface_fluxes_params = CAP.surface_fluxes_params(p.params)
    uf_params = SFP.uf_params(surface_fluxes_params)
    κ = SFP.von_karman_const(surface_fluxes_params)

    z_c1_p = parent(Fields.level(Fields.coordinate_field(axes(Y.c)).z, 1))
    u_z1_p = parent(norm.(Fields.level(Y.c.uₕ, 1)))
    ustar_p = parent(sfc_conditions.ustar)
    L_p = parent(sfc_conditions.obukhov_length)
    roughness_spec = SF.COARE3RoughnessParams{FT}()
    z0_eff = p.scratch.temp_field_level
    z₀_p = parent(z0_eff)
    z₀_p .= SF.momentum_roughness.(roughness_spec, ustar_p, surface_fluxes_params, nothing)

    u_10 = p.scratch.ᶠtemp_field_level
    parent(u_10) .=
        monin_obukhov_wind_extrapolated.(FT(10), z_c1_p, u_z1_p, L_p, uf_params, κ, z₀_p)

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
    # Mass per particle for the number→mass conversion. Treats each bin as a
    # lognormal mode (median radius `SSLTxx_radius`, width `seasalt_std`), so it
    # uses the lognormal mean particle volume — the SAME convention as the
    # number↔mass bridge in `bins_to_aerosol_distribution`, keeping emitted mass
    # and activated number consistent.
    σ = FT(aero_params.seasalt_std)
    ρ_s = FT(aero_params.seasalt_density)
    mass_per_particle = ntuple(
        i -> sea_salt_mean_particle_volume(FT(bin_radii[i]), σ) * ρ_s,
        Val(5),
    )

    for (bin_index, name) in enumerate(interactive_aerosol_names)
        bin_flux_cache = getproperty(p.tracers.interactive_aerosols_field, name)
        @. bin_flux_cache =
            sea_salt_emission_flux(u_10, T_sfc, bin_index) * mass_per_particle[bin_index] *
            ocean_fraction
    end
end

"""
    sea_salt_emission_tendency!(Yₜ, Y, p, t)

Apply surface emission tendencies for prognostic sea salt bins (ρSSLT01 …).

Reads per-bin fluxes from `p.tracers.interactive_aerosols_field`, which
are pre-computed by `set_sea_salt_emission_flux!` during the precomputed-
quantities stage. The flux is applied as a bottom boundary condition using
`boundary_tendency_scalar`, which localises it to the lowest model layer.
"""
function sea_salt_emission_tendency!(Yₜ, Y, p, t)
    interactive_aerosol_names = _aerosol_names(p.atmos.interactive_aerosols)
    isempty(interactive_aerosol_names) && return

    for (bin_index, name) in enumerate(interactive_aerosol_names)
        ρχ_name = Symbol(:ρ, name)
        ᶜρχ = getproperty(Y.c, ρχ_name)
        ᶜρχₜ = getproperty(Yₜ.c, ρχ_name)
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))

        bin_flux_cache = getproperty(p.tracers.interactive_aerosols_field, name)
        sfc_flux = p.scratch.sfc_temp_C3
        @. sfc_flux = C3(bin_flux_cache)

        btt = boundary_tendency_scalar(ᶜχ, sfc_flux)
        @. ᶜρχₜ -= btt
    end
end

"""
    sea_salt_settling_tendency!(Yₜ, Y, p, t)

Apply gravitational settling to all prognostic sea-salt bins as an explicit
downward vertical advection with the per-bin, slip-corrected terminal velocity:

    ∂(ρSSLTxx)/∂t -= ∇·(ρ · w_settle · χ)  (downward, free outflow at surface)

The settling speed is derived from the cached wet radius `p.precomputed.ᶜsslt_r_wet`
(wet density is a cheap inline function of it) and materialized into a scratch
field — Courant-capped for explicit stability — before being used in the
`ᶠright_bias`/`ᶜprecipdivᵥ` stencil, so the stencil kernel stays small (as precip
does with its precomputed terminal velocity). The free-outflow bottom boundary
means this term also **deposits** the gravitational flux `V_g · ρSSLTxx` at the
surface — the gravitational contribution to dry deposition — so no separate
`V_g` surface flux is added (see `sea_salt_dry_deposition_tendency!`, which
carries only the turbulent part).

Applied in the explicit tendency, consistent with the grid-mean vertical
advection of passive tracers (also explicit). Grid-mean only: updraft (`sgsʲs`)
sea-salt copies are not settled here (deferred; see the subdomain-sedimentation
TODO).
"""
function sea_salt_settling_tendency!(Yₜ, Y, p, t)
    interactive_aerosol_names = _aerosol_names(p.atmos.interactive_aerosols)
    isempty(interactive_aerosol_names) && return

    FT = eltype(Y)
    (; ᶜT, ᶜsslt_r_wet) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    aero_params = p.params.prescribed_aerosol_params
    grav = FT(CAP.grav(p.params))
    R_d = FT(TD.Parameters.R_d(thermo_params))
    ρ_s = FT(aero_params.seasalt_density)
    ρ_w = FT(SEA_SALT_WATER_DENSITY)
    σ = FT(aero_params.seasalt_std)
    mass_weight = exp(2 * log(σ)^2)
    bin_radii = (
        aero_params.SSLT01_radius,
        aero_params.SSLT02_radius,
        aero_params.SSLT03_radius,
        aero_params.SSLT04_radius,
        aero_params.SSLT05_radius,
    )
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    ᶜΔz = Fields.Δz_field(axes(Y.c))
    dt = FT(p.dt)
    courant_max = FT(SEA_SALT_SETTLING_COURANT_MAX)

    for (bin_index, name) in enumerate(interactive_aerosol_names)
        ρχ_name = Symbol(:ρ, name)
        ᶜρχ = getproperty(Y.c, ρχ_name)
        ᶜρχₜ = getproperty(Yₜ.c, ρχ_name)
        r_dry = FT(bin_radii[bin_index])
        r_wet = getproperty(ᶜsslt_r_wet, name)

        # Materialize the Courant-capped settling speed into scratch (wet density
        # is a cheap inline function of r_wet; the growth factor is r_wet/r_dry).
        # ᶜtemp_scalar is written and consumed within this iteration.
        ᶜw = p.scratch.ᶜtemp_scalar
        @. ᶜw = min(
            sea_salt_settling_velocity(
                r_wet,
                sea_salt_wet_density(ρ_s, ρ_w, r_wet / r_dry),
                Y.c.ρ, ᶜT, R_d, grav, mass_weight,
            ),
            courant_max * ᶜΔz / dt,
        )
        @. ᶜρχₜ -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜw)) * specific(ᶜρχ, Y.c.ρ),
            ),
        )
    end
    return nothing
end

"""
    sea_salt_dry_deposition_tendency!(Yₜ, Y, p, t)

Apply the **turbulent** part of dry deposition of prognostic sea salt as a
surface-flux sink:

    ρ_flux_SSLTxx|_surface = − V_{d,turb} · ρSSLTxx|_{level 1}

with `V_{d,turb} = 1 / (R_a + R_s)` from
[`sea_salt_dry_deposition_velocity`](@ref) (MOST aerodynamic resistance `R_a`

  - Zhang et al. 2001 surface resistance `R_s`). The gravitational settling
    contribution is deposited separately by `sea_salt_settling_tendency!`'s
    free-outflow bottom boundary, so the two do not double count (their sum is
    `V_g + 1/(R_a+R_s)` times the surface concentration, i.e. the full deposition
    velocity).

Applied everywhere like `sea_salt_emission_tendency!` but as a sink. Every
surface currently uses the ocean/water land-use category (TODO: per-land-use
parameters from the coupler). The deposition velocity is Courant-capped at the
lowest cell so the explicit surface sink cannot over-deplete it in one step.
"""
function sea_salt_dry_deposition_tendency!(Yₜ, Y, p, t)
    interactive_aerosol_names = _aerosol_names(p.atmos.interactive_aerosols)
    isempty(interactive_aerosol_names) && return

    FT = eltype(Y)
    (; sfc_conditions, ᶜT, ᶜsslt_r_wet) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    aero_params = p.params.prescribed_aerosol_params
    R_d = FT(TD.Parameters.R_d(thermo_params))
    grav = FT(CAP.grav(p.params))
    ρ_s = FT(aero_params.seasalt_density)
    ρ_w = FT(SEA_SALT_WATER_DENSITY)
    σ = FT(aero_params.seasalt_std)
    mass_weight = exp(2 * log(σ)^2)
    bin_radii = (
        aero_params.SSLT01_radius,
        aero_params.SSLT02_radius,
        aero_params.SSLT03_radius,
        aero_params.SSLT04_radius,
        aero_params.SSLT05_radius,
    )
    surface_fluxes_params = CAP.surface_fluxes_params(p.params)
    uf_params = SFP.uf_params(surface_fluxes_params)
    κ = SFP.von_karman_const(surface_fluxes_params)
    roughness_spec = SF.COARE3RoughnessParams{FT}()
    dt = FT(p.dt)
    courant_max = FT(SEA_SALT_SETTLING_COURANT_MAX)

    # Surface-level parent arrays (bridge the center-level-1 vs face-half
    # spaces, as in set_sea_salt_emission_flux!).
    z1_p = parent(Fields.level(Fields.coordinate_field(axes(Y.c)).z, 1))
    ρ1_p = parent(Fields.level(Y.c.ρ, 1))
    T1_p = parent(Fields.level(ᶜT, 1))
    Δz1_p = parent(Fields.level(Fields.Δz_field(axes(Y.c)), 1))
    ustar_p = parent(sfc_conditions.ustar)
    L_p = parent(sfc_conditions.obukhov_length)
    z₀ = p.scratch.temp_field_level
    z₀_p = parent(z₀)
    z₀_p .= SF.momentum_roughness.(roughness_spec, ustar_p, surface_fluxes_params, nothing)

    for (bin_index, name) in enumerate(interactive_aerosol_names)
        ρχ_name = Symbol(:ρ, name)
        ᶜρχ = getproperty(Y.c, ρχ_name)
        ᶜρχₜ = getproperty(Yₜ.c, ρχ_name)
        r_dry = FT(bin_radii[bin_index])
        r_wet1_p = parent(Fields.level(getproperty(ᶜsslt_r_wet, name), 1))
        ρχ1_p = parent(Fields.level(ᶜρχ, 1))

        # Surface settling velocity for the Zhang Stokes number, computed inline
        # from the cached surface wet radius (uncapped — the Courant cap is a
        # numerical device for the explicit settling advection and must not
        # enter the deposition physics).
        vg1 = p.scratch.temp_field_level_2
        parent(vg1) .=
            sea_salt_settling_velocity.(
                r_wet1_p,
                sea_salt_wet_density.(ρ_s, ρ_w, r_wet1_p ./ r_dry),
                ρ1_p, T1_p, R_d, grav, mass_weight,
            )
        vg1_p = parent(vg1)

        # V_{d,turb} · ρSSLTxx|₁, computed on parent arrays into a face-half
        # scalar scratch, then wrapped as a downward (sink) C3 surface flux.
        # V_{d,turb} is Courant-capped (≤ courant_max·Δz₁/dt) so the explicit
        # surface sink cannot remove more than the lowest cell holds in a step.
        dep_flux = p.scratch.ᶠtemp_field_level
        parent(dep_flux) .=
            min.(
                sea_salt_dry_deposition_velocity.(
                    vg1_p, r_wet1_p, ρ1_p, T1_p, z1_p, L_p, z₀_p, ustar_p,
                    Ref(uf_params), FT(κ), R_d,
                ),
                courant_max .* Δz1_p ./ dt,
            ) .* ρχ1_p

        sfc_flux = p.scratch.sfc_temp_C3
        @. sfc_flux = C3(-(dep_flux))

        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        btt = boundary_tendency_scalar(ᶜχ, sfc_flux)
        @. ᶜρχₜ -= btt
    end
    return nothing
end
