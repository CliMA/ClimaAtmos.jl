import SpecialFunctions: erf, erfcx
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import ..Parameters as CAP


@generated sslt_state_names(::PrognosticSeaSalt{names}) where {names} =
    :($(map(n -> MatrixFields.FieldName(Symbol(:ρ, n)), names)))

"""
    wind_at_height(z, ustar, obukhov_length, sfp)

MOST point wind speed [m/s] at height `z` over water, recovered
via `SurfaceFluxes.compute_profile_value` with COARE3 roughness.
Fed by ClimaCoupler into [`set_sslt_surface_fluxes!`](@ref)).
"""
function wind_at_height(z::FT, ustar::FT, obukhov_length::FT, sfp) where {FT}
    roughness = SF.COARE3RoughnessParams{FT}()
    u = SF.compute_profile_value(
        sfp,
        safe_obukhov_length(obukhov_length),
        SF.momentum_roughness(roughness, ustar, sfp, nothing),
        z,
        ustar,
        zero(ustar),
        UF.MomentumTransport(),
        SF.PointValueScheme(),
    )
    return max(u, zero(u))
end

# Keep the MOST profile finite near neutral stratification.
safe_obukhov_length(L) =
    ifelse(L < zero(L), min(L, -eps(typeof(L))), max(L, eps(typeof(L))))



sslt_lognormal_modes(ap) =
    (Tuple(ap.gong_mode1), Tuple(ap.gong_mode2), Tuple(ap.gong_mode3))

_ssa_mode_spectrum(r̂, modes) = sum(modes) do (F, r_mode, σg)
    F / r̂ * exp(-log(r̂ / r_mode)^2 / (2 * log(σg)^2))
end

# k-th radius moment ∫ r̂ᵏ (dF/dr̂) dr̂ of the fitted spectrum over one bin, in
# closed form. Completing the square in x = ln r̂ turns each lognormal mode
# into a Gaussian centered at μ + ks² (s = ln σg, μ = ln r_mode), so the
# partial moment is
#   F · s√(π/2) · exp(kμ + k²s²/2) · [erf(x_hi) − erf(x_lo)],
#   x = (ln r̂ − μ − ks²) / (s√2),
# the same erf-of-shifted-lognormal form CloudMicrophysics uses for
# activated aerosol mass. For high k the shifted center lies far outside the
# bin, both erfs round to ±1 and the difference cancels, so the tail cases
# use erfc(x) = erfcx(x)·exp(-x²): the prefactor absorbs exp(-x²) and each
# edge contributes exp(ℓ(r̂ₑ))·erfcx(±xₑ) with ℓ the log-integrand at the edge,
# which stays finite and accurate however far the center is.
_ssa_bin_moment(k, r̂_lo, r̂_hi, modes) = sum(modes) do (F, r_mode, σg)
    s, μ = log(σg), log(r_mode)
    ℓ(r̂) = k * log(r̂) - (log(r̂) - μ)^2 / (2 * s^2)   # ln[r̂ᵏ · r̂ dF/dr̂ / F]
    x(r̂) = (log(r̂) - μ - k * s^2) / (s * sqrt(2))
    x_lo, x_hi = x(r̂_lo), x(r̂_hi)
    ∫ = if x_lo ≥ 0       # center below the bin
        exp(ℓ(r̂_lo)) * erfcx(x_lo) - exp(ℓ(r̂_hi)) * erfcx(x_hi)
    elseif x_hi ≤ 0   # center above the bin
        exp(ℓ(r̂_hi)) * erfcx(-x_hi) - exp(ℓ(r̂_lo)) * erfcx(-x_lo)
    else              # center inside the bin: no cancellation
        exp(k * μ + k^2 * s^2 / 2) * (erf(x_hi) - erf(x_lo))
    end
    F * s * sqrt(π / 2) * ∫
end

# Highest radius moment of the emitted spectrum kept in the cache. Orders 0
# (number), 3 (mass) and 5 (Stokes mass flux) are consumed today; 6 is the
# radar-reflectivity / mass-squared order, cheap to carry alongside.
const SSLT_MAX_MOMENT = 6

"""
    sslt_bin_moments(params, FT)

Per-bin radius moments `M̂ₖ = ∫ r̂ᵏ (dF/dr̂) dr̂`, `k = 0:$(SSLT_MAX_MOMENT)`, of
the fitted Gong (2003) emission spectrum over each dry size bin, as a tuple of
`NTuple{$(SSLT_MAX_MOMENT + 1), FT}` (one per bin, index `k + 1`). The moments
are in the dimensionless dry radius `r̂ = r_dry / ssa_r_ref`, so the dimensional
`k`-th moment is `ssa_r_ref^k · M̂ₖ`; `M̂₀` is the bin's number flux and
`(4π/3) ρ_s ssa_r_ref³ M̂₃` its dry mass flux, both at the reference wind.
Evaluated once at cache construction ([`sslt_bin_moment`](@ref) reads them).
"""
function sslt_bin_moments(params, ::Type{FT}) where {FT}
    ap = CAP.prognostic_aerosol_params(params)
    modes = sslt_lognormal_modes(ap)
    r̂_edges = Tuple(ap.ssa_bin_edges ./ ap.ssa_r_ref)
    return ntuple(Val(length(r̂_edges) - 1)) do i
        ntuple(Val(SSLT_MAX_MOMENT + 1)) do kp1
            FT(_ssa_bin_moment(kp1 - 1, r̂_edges[i], r̂_edges[i + 1], modes))
        end
    end
end

"""
    sslt_bin_moment(bin_moments, k)

The `k`-th dimensionless radius moment `M̂ₖ` of one bin from its
[`sslt_bin_moments`](@ref) entry.
"""
sslt_bin_moment(bin_moments, k) = bin_moments[k + 1]

"""
    sslt_settling_radii(bin_moments, params)

Per-bin dry mass-weighted settling radius `√(⟨r⁵⟩/⟨r³⟩)` (m): the single dry
radius whose Stokes speed (∝ r²) carries the bin's mass settling flux, from
the bin's cached spectrum moments ([`sslt_bin_moments`](@ref)). Settling and
dry deposition scale these dry radii by the growth factor ξ(RH).
"""
function sslt_settling_radii(bin_moments, params)
    r_ref = CAP.prognostic_aerosol_params(params).ssa_r_ref
    return map(bin_moments) do m
        r_ref * sqrt(sslt_bin_moment(m, 5) / sslt_bin_moment(m, 3))
    end
end

"""
    sslt_kelvin_coefficients(dry_radii, params)

Per-bin [`sslt_kelvin_coefficient`](@ref) at the dry settling radii.
"""
function sslt_kelvin_coefficients(dry_radii, params)
    (; lewis_a, σ_w, ρ_water) = CAP.prognostic_aerosol_params(params)
    R_v = CAP.R_v(params)
    return map(r -> sslt_kelvin_coefficient(r, lewis_a, σ_w, ρ_water, R_v), dry_radii)
end


#####
##### Tendencies
#####

"""
    set_sslt_surface_fluxes!(Y, p, bin_fluxes)

Called by Coupler to compute per-bin upward emission mass fluxes `bin_fluxes`:
a tuple of scalar surface Fields, positive up, ocean area-weighted [kg m⁻² s⁻¹].
Stored into `p.tracers.sslt_sfc_fluxes` and used by [`aerosol_emission_tendency!`](@ref).
"""
set_sslt_surface_fluxes!(Y, p, u₁₀_ocean, ocean_fraction) =
    set_sslt_surface_fluxes!(Y, p, u₁₀_ocean, ocean_fraction, p.atmos.seasalt)
set_sslt_surface_fluxes!(Y, p, u₁₀_ocean, ocean_fraction, ::Nothing) = nothing
function set_sslt_surface_fluxes!(
    Y,
    p,
    u₁₀_ocean,
    ocean_fraction,
    ::PrognosticSeaSalt,
)
    (; bin_mass_flux, ssa_u_ref, gong_wind_exp) =
        CAP.prognostic_aerosol_params(p.params)
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(Y.f), Fields.half)
    for (sfc_flux, mass_flux_scale) in
        zip(values(p.tracers.sslt_sfc_fluxes), bin_mass_flux)
        @. sfc_flux = C3(
            ocean_fraction *
            mass_flux_scale *
            (u₁₀_ocean / ssa_u_ref)^gong_wind_exp *
            unit_basis_vector_data(C3, sfc_local_geometry),
        )
    end
    return nothing
end

"""
    aerosol_emission_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)

Apply the per-bin emission fluxes cached in `p.tracers.sslt_sfc_fluxes`
(see [`set_sslt_surface_fluxes!`](@ref)) as bottom boundary conditions on
the grid-mean `Y.c.ρ<bin>` tracers, using [`boundary_tendency_scalar`](@ref),
and mirror the specific tendency onto each updraft tracer.
"""
function aerosol_emission_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)
    n_updrafts = n_mass_flux_subdomains(p.atmos.turbconv_model)
    fluxes = p.tracers.sslt_sfc_fluxes

    MatrixFields.unrolled_foreach(sslt_state_names(sslt)) do ρχ_name
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        btt = boundary_tendency_scalar(ᶜχ, fluxes[MatrixFields.extract_first(ρχ_name)])
        @. ᶜρχₜ -= btt

        for j in 1:n_updrafts
            ᶜχʲₜ = MatrixFields.get_field(
                Yₜ.c.sgsʲs.:($j),
                specific_tracer_name(ρχ_name),
            )
            @. ᶜχʲₜ -= specific(btt, p.precomputed.ᶜρʲs.:($$j))
        end
    end
    return nothing
end

"""
    aerosol_settling_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)

Explicit gravitational settling of the prognostic sea salt bins — a downward
vertical advection of each bin tracer at the terminal velocity of its
mass-weighted settling radius:

    ∂(ρχ)/∂t -= ∇·(ρ · w_settle · χ)   (free outflow at the surface)

The velocity is [`sslt_settling_velocity`](@ref) at the bin's wet settling
radius ([`sslt_settling_radii`](@ref) times the growth factor),
Courant-capped (`settling_courant_max`) for explicit stability; it is
materialized into scratch so the `ᶠright_bias`/`ᶜprecipdivᵥ` stencil kernel
stays small, as precipitation does. The free-outflow bottom boundary deposits
the gravitational flux `V_g · ρχ` at the surface — the gravitational part of
dry deposition (the turbulent part is a forthcoming surface-flux sink).

The treatment mirrors `set_precipitation_velocities!` and the updraft
sedimentation of the microphysics species in
`edmfx_sgs_vertical_advection_tendency!`:

  - the environment velocity `w⁰` is evaluated on the environment state and
    each updraft velocity `wʲ` on its draft state;
  - the grid-mean tracer settles at the mass-weighted velocity
    `w = (ρa⁰χ⁰w⁰ + Σⱼ ρaʲχʲwʲ) / (ρa⁰χ⁰ + Σⱼ ρaʲχʲ)`, so the grid-scale flux
    equals the sum of the subdomain fluxes (the velocities, not the growth
    factors, are averaged: the Stokes speed is convex in ξ);
  - each updraft tracer receives its within-updraft flux convergence with the
    lateral (detrainment) correction of `updraft_sedimentation!`, using the
    environment flux density `ρ⁰w⁰χ⁰` directly (no reconstruction from the
    grid mean is needed because `w⁰` is known), so the path is valid for any
    number of updrafts.
"""
function aerosol_settling_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)
    turbconv_model = p.atmos.turbconv_model
    FT = eltype(Y)
    ap = CAP.prognostic_aerosol_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; ᶜTʲs, ᶜρʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
    grav = FT(CAP.grav(p.params))
    R_d = FT(CAP.R_d(p.params))
    ρ_s = CAP.prescribed_aerosol_params(p.params).seasalt_density
    α_lat = CAP.sedimentation_lateral_coeff(p.params)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜΔz = Fields.Δz_field(Y.c)
    dt = float(p.dt)
    n = n_mass_flux_subdomains(turbconv_model)

    # Bin-independent fields, hoisted: environment density and area, and the
    # face density of the grid-mean flux. `updraft_sedimentation!` overwrites
    # `ᶠtemp_scalar`..`ᶠtemp_scalar_3`, so the face density lives in `_4`.
    ᶜρ⁰ = p.scratch.ᶜtemp_scalar_5
    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜρa⁰ = @. lazy(max(zero(Y.c.ρ), ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)))
    ᶠρ = p.scratch.ᶠtemp_scalar_4
    @. ᶠρ = ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    # Per-bin scratch, written and consumed within one bin iteration.
    ᶜw⁰ = p.scratch.ᶜtemp_scalar
    ᶜρaχw = p.scratch.ᶜtemp_scalar_2   # Σ ρa·χ·w over subdomains, then w_gs
    ᶜρaχ = p.scratch.ᶜtemp_scalar_3    # Σ ρa·χ over subdomains
    ᶜwʲ = p.scratch.ᶜtemp_scalar_4
    ᶜvtt = p.scratch.ᶜtemp_scalar_6

    (; sslt_settling_radii, sslt_kelvin_coeffs) = p.tracers
    ρχ_names = sslt_state_names(sslt)
    bins = ntuple(Val(length(ρχ_names))) do i
        (ρχ_names[i], sslt_settling_radii[i], sslt_kelvin_coeffs[i])
    end
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, r_settle, C_kelvin)
        χ_name = specific_tracer_name(ρχ_name)
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)

        # Environment: velocity from the environment growth factor on the
        # environment state; mass and mass flux seed the grid-mean average.
        ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
        @. ᶜw⁰ = min(
            sslt_bin_settling_velocity(
                TD.relative_humidity(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
                ᶜT⁰,
                r_settle,
                C_kelvin,
                ρ_s,
                ᶜρ⁰,
                R_d,
                grav,
                (ap,),
            ),
            ap.settling_courant_max * ᶜΔz / dt,
        )
        @. ᶜρaχ = ᶜρa⁰ * max(zero(FT), ᶜχ⁰)
        @. ᶜρaχw = ᶜρaχ * ᶜw⁰

        for j in 1:n
            ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
            ᶜρʲ = ᶜρʲs.:($j)
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)

            @. ᶜwʲ = min(
                sslt_bin_settling_velocity(
                    TD.relative_humidity(
                        thp,
                        ᶜTʲs.:($$j),
                        ᶜp,
                        ᶜq_tot_nonnegʲs.:($$j),
                        ᶜq_liqʲs.:($$j),
                        ᶜq_iceʲs.:($$j),
                    ),
                    ᶜTʲs.:($$j),
                    r_settle,
                    C_kelvin,
                    ρ_s,
                    ᶜρʲ,
                    R_d,
                    grav,
                    (ap,),
                ),
                ap.settling_courant_max * ᶜΔz / dt,
            )
            ᶜρaʲχʲ = @. lazy(max(zero(FT), ᶜρaʲ) * max(zero(FT), ᶜχʲ))
            @. ᶜρaχ += ᶜρaʲχʲ
            @. ᶜρaχw += ᶜρaʲχʲ * ᶜwʲ

            # Within-updraft flux convergence with the lateral correction
            # of the microphysics species; the environment flux density
            # ρ⁰w⁰χ⁰ is known directly from the environment velocity.
            ᶜa = @. lazy(draft_area(ᶜρaʲ, ᶜρʲ))
            ᶜρ⁰w⁰χ⁰ = @. lazy(ᶜρ⁰ * ᶜw⁰ * ᶜχ⁰)
            updraft_sedimentation!(
                ᶜvtt,
                p,
                ᶜρʲ,
                ᶜwʲ,
                ᶜa,
                ᶜχʲ,
                ᶠJ,
                ᶜρ⁰w⁰χ⁰,
                α_lat,
            )
            @. ᶜχʲₜ +=
                specific(FT(1), ᶜρaʲ, FT(0), ᶜρʲ, turbconv_model) * ᶜvtt
        end

        # Grid mean: mass-weighted velocity so the grid-scale flux is the
        # sum of the subdomain fluxes; where the bin carries no mass the
        # environment velocity stands in (the tendency is zero anyway).
        @. ᶜρaχw = ifelse(
            ᶜρaχ > ϵ_numerics(FT),
            max(ᶜρaχw / ᶜρaχ, zero(FT)),
            ᶜw⁰,
        )
        @. ᶜρχₜ -= ᶜprecipdivᵥ(
            ᶠρ * ᶠright_bias(
                Geometry.WVector(-(ᶜρaχw)) * specific(ᶜρχ, Y.c.ρ),
            ),
        )
    end
    return nothing
end

"""
    aerosol_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)

Exponential decay of the grid-mean and updraft sea salt tracers with the
`ssa_residence` timescale (0.55 days, from AeroCom III). A uniform-rate
placeholder that over-deposits small bins and under-deposits large ones;
forthcoming branches replace it with the accumulated dry and wet
deposition sinks.
"""
function aerosol_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)
    (; turbconv_model) = p.atmos
    ap = CAP.prognostic_aerosol_params(p.params)

    λ = inv(ap.τ_ssa)
    n_updrafts = n_mass_flux_subdomains(turbconv_model)

    MatrixFields.unrolled_foreach(sslt_state_names(sslt)) do ρχ_name
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
        @. ᶜρχₜ -= λ * ᶜρχ

        for j in 1:n_updrafts
            χ_name = specific_tracer_name(ρχ_name)
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
            @. ᶜχʲₜ -= λ * ᶜχʲ
        end
    end
    return nothing
end
