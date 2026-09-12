import SurfaceFluxes as SF
import SurfaceFluxes.Parameters as SFP
import SurfaceFluxes.UniversalFunctions as UF
import ..Parameters as CAP



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

# Floor |L| away from zero, where ζ = z/L and the MOST profile diverge.
safe_obukhov_length(L) =
    ifelse(L < zero(L), min(L, -eps(typeof(L))), max(L, eps(typeof(L))))


"""
    sslt_lognormal_modes(ap)

Three lognormal modes `(F, r_mode, σg)` fit to Gong (2003) emission.
"""
sslt_lognormal_modes(ap) =
    (Tuple(ap.gong_mode1), Tuple(ap.gong_mode2), Tuple(ap.gong_mode3))

"""
    sslt_bin_moments(params, max_moment, FT)

Per-bin spectrum moments of orders `k = 0:max_moment`, read back with
[`spectrum_moment`](@ref).
"""
function sslt_bin_moments(params, max_moment, ::Type{FT}) where {FT}
    ap = CAP.prognostic_aerosol_params(params)
    modes = sslt_lognormal_modes(ap)
    r̂_edges = Tuple(ap.ssa_bin_edges ./ ap.aerosol_r_ref)
    return ntuple(Val(length(r̂_edges) - 1)) do i
        ntuple(Val(max_moment + 1)) do kp1
            FT(lognormal_bin_moment(kp1 - 1, r̂_edges[i], r̂_edges[i + 1], modes))
        end
    end
end

sslt_settling_radii(bin_moments, ap) = map(m -> mass_settling_radius(m, ap), bin_moments)

"""
    sslt_settling_bin_props(p, sslt)

Per-bin `(ρχ_name, r_settle, C_kelvin)` tuples for [`bin_settling_velocity`](@ref),
shared by gravitational settling and dry deposition so both evaluate the same
settling radius for a bin.
"""
function sslt_settling_bin_props(p, sslt::PrognosticSeaSalt)
    (; sslt_settling_radii, sslt_kelvin_coeffs) = p.tracers
    ρχ_names = aerosol_state_names(sslt)
    return ntuple(Val(length(ρχ_names))) do i
        (ρχ_names[i], sslt_settling_radii[i], sslt_kelvin_coeffs[i])
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
    set_sslt_surface_fluxes!(Y, p, u₁₀_ocean, ocean_fraction)

Called by ClimaCoupler with the MOST 10 m wind over the ocean portion of the
surface ([`wind_at_height`](@ref)) and the ocean area fraction, to compute the
per-bin upward emission mass fluxes [kg m⁻² s⁻¹]. Stored as `C3` surface Fields
in `p.tracers.sslt_sfc_fluxes` and used by [`aerosol_emission_tendency!`](@ref).
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
(written by ClimaCoupler via [`set_sslt_surface_fluxes!`](@ref)) via
[`aerosol_surface_flux_tendency!`](@ref).
"""
aerosol_emission_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt) =
    aerosol_surface_flux_tendency!(Yₜ, Y, p, sslt, p.tracers.sslt_sfc_fluxes)

"""
    aerosol_settling_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)

Downward vertical advection of aerosol tracers at terminal velocity of its
mass-weighted settling radius. Tendency formed from divergence:

    ∂(ρχ)/∂t -= ∇·(ρ · w_settle · χ)

Velocity given by [`settling_velocity`](@ref) at the bin's wet settling
radius ([`sslt_settling_radii`](@ref) times the growth factor), with the
bin-independent air state (RH, viscosity, mean free path) of each subdomain
hoisted out of the bin loop ([`_aerosol_air_state`](@ref)), Courant-capped
(`settling_courant_max`); it is materialized into scratch so the
`ᶠright_bias`/`ᶜprecipdivᵥ` stencil kernel stays small, as precipitation does.
The free-outflow bottom boundary deposits the gravitational flux `V_g · ρχ` at
the surface — the gravitational part of dry deposition — so
[`aerosol_dry_deposition_tendency!`](@ref) carries only the turbulent part and
nothing is double counted.

Mirrors: `set_precipitation_velocities!`, `edmfx_sgs_vertical_advection_tendency!`:

  - the grid-mean tracer settles at the mass-weighted velocity
    `w = (ρa⁰χ⁰w⁰ + Σⱼ ρaʲχʲwʲ) / (ρa⁰χ⁰ + Σⱼ ρaʲχʲ)`, so the grid-scale flux
    equals the sum of the subdomain fluxes;
  - each updraft tracer receives its within-updraft flux convergence with the
    lateral (detrainment) correction of `updraft_sedimentation!`, using the
    environment flux density `ρ⁰w⁰χ⁰`
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


    ᶜρ⁰ = p.scratch.ᶜtemp_scalar_5
    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜρa⁰ = @. lazy(max(zero(Y.c.ρ), ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)))
    ᶠρ = p.scratch.ᶠtemp_scalar_4
    @. ᶠρ = ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    # Bin-independent air state of each subdomain (RH, μ, λ), once per stage.
    (; sslt_air_state⁰, sslt_air_stateʲs) = p.tracers
    ᶜair⁰ = sslt_air_state⁰
    @. ᶜair⁰ = _aerosol_air_state(
        thp,
        ᶜT⁰,
        ᶜp,
        ᶜq_tot_nonneg⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
        ᶜρ⁰,
        R_d,
        (ap,),
    )
    for j in 1:n
        ᶜairʲ = sslt_air_stateʲs[j]
        @. ᶜairʲ = _aerosol_air_state(
            thp,
            ᶜTʲs.:($$j),
            ᶜp,
            ᶜq_tot_nonnegʲs.:($$j),
            ᶜq_liqʲs.:($$j),
            ᶜq_iceʲs.:($$j),
            ᶜρʲs.:($$j),
            R_d,
            (ap,),
        )
    end
    # Per-bin scratch, written and consumed within one bin iteration.
    ᶜw⁰ = p.scratch.ᶜtemp_scalar
    ᶜρaχw = p.scratch.ᶜtemp_scalar_2
    ᶜρaχ = p.scratch.ᶜtemp_scalar_3
    ᶜwʲ = p.scratch.ᶜtemp_scalar_4
    ᶜvtt = p.scratch.ᶜtemp_scalar_6

    bins = sslt_settling_bin_props(p, sslt)
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, r_settle, C_kelvin)
        χ_name = specific_tracer_name(ρχ_name)
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)

        # Environment
        ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
        @. ᶜw⁰ = min(
            bin_settling_velocity(
                ᶜair⁰,
                ᶜT⁰,
                r_settle,
                C_kelvin,
                ρ_s,
                ᶜρ⁰,
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
                bin_settling_velocity(
                    sslt_air_stateʲs[$j],
                    ᶜTʲs.:($$j),
                    r_settle,
                    C_kelvin,
                    ρ_s,
                    ᶜρʲ,
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

        # Mass-weighted grid mean
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
    set_sslt_dry_deposition_fluxes!(Y, p, sslt_model)

Write each bin's turbulent dry-deposition mass flux,
`ρ_flux|_sfc = -V_d,turb · ρχ|₁` (downward, so negative), into the surface
fields `p.tracers.sslt_drydep_fluxes`, from which
[`aerosol_dry_deposition_tendency!`](@ref) builds the tracers' bottom boundary
conditions. `V_d,turb = 1/(R_a + R_s)` from [`sslt_dry_deposition_velocity`](@ref)
(aerodynamic resistance plus surface resistance), evaluated on the grid-mean lowest-level state
at the bin's wet settling radius. `V_d,turb` is Courant-capped so the
explicit sink cannot over-deplete the lowest cell in one step (a numerical
device, not deposition physics — the settling speed that feeds the
deposition Stokes number is uncapped). Surface and level-1 fields live on different
spaces, so each flux is assembled in one fused broadcast over their data
values, as `update_surface_conditions!` does. Reads
`p.precomputed.sfc_conditions`, so it must run after the surface conditions
are updated.
"""
set_sslt_dry_deposition_fluxes!(Y, p, ::Nothing) = nothing
function set_sslt_dry_deposition_fluxes!(Y, p, sslt::PrognosticSeaSalt)
    FT = eltype(Y)
    ap = CAP.prognostic_aerosol_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    (; sfc_conditions, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    sfp = CAP.surface_fluxes_params(p.params)
    uf_params = SFP.uf_params(sfp)
    κ_vk = SFP.von_karman_const(sfp)
    R_d = FT(CAP.R_d(p.params))
    grav = FT(CAP.grav(p.params))
    ρ_s = CAP.prescribed_aerosol_params(p.params).seasalt_density
    roughness_spec = SF.COARE3RoughnessParams{FT}()
    dt = float(p.dt)
    fluxes = p.tracers.sslt_drydep_fluxes
    velocities = p.tracers.sslt_drydep_velocities

    level1(f) = Fields.field_values(Fields.level(f, 1))
    z_sfc_values =
        Fields.field_values(Fields.level(Fields.coordinate_field(Y.f).z, Fields.half))
    sfc_lg_values =
        Fields.field_values(Fields.level(Fields.local_geometry_field(Y.f), Fields.half))
    ustar_values = Fields.field_values(sfc_conditions.ustar)
    L_values = Fields.field_values(sfc_conditions.obukhov_length)
    z1_values = level1(Fields.coordinate_field(Y.c).z)
    Δz1_values = level1(Fields.Δz_field(Y.c))
    ρ1_values = level1(Y.c.ρ)
    p1_values = level1(ᶜp)
    T1_values = level1(ᶜT)
    q_tot1_values = level1(ᶜq_tot_nonneg)
    q_liq1_values = level1(ᶜq_liq)
    q_ice1_values = level1(ᶜq_ice)

    bins = sslt_settling_bin_props(p, sslt)
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, r_settle, C_kelvin)
        bin = MatrixFields.extract_first(ρχ_name)
        ρχ1_values = level1(MatrixFields.get_field(Y.c, ρχ_name))
        sfc_flux_values = Fields.field_values(fluxes[bin])
        V_d_values = Fields.field_values(velocities[bin])
        @. V_d_values = min(
            sslt_bin_dry_deposition_velocity(
                _aerosol_air_state(
                    thp,
                    T1_values,
                    p1_values,
                    q_tot1_values,
                    q_liq1_values,
                    q_ice1_values,
                    ρ1_values,
                    R_d,
                    (ap,),
                ),
                T1_values,
                r_settle,
                C_kelvin,
                ρ_s,
                ρ1_values,
                z1_values - z_sfc_values,
                L_values,
                SF.momentum_roughness(
                    roughness_spec,
                    ustar_values,
                    sfp,
                    nothing,
                ),
                ustar_values,
                uf_params,
                κ_vk,
                grav,
                (ap,),
            ),
            ap.settling_courant_max * Δz1_values / dt,
        )
        @. sfc_flux_values = C3(
            -V_d_values *
            max(zero(FT), ρχ1_values) *
            unit_basis_vector_data(C3, sfc_lg_values),
        )
    end
    return nothing
end

"""
    aerosol_dry_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)

Apply the per-bin turbulent dry-deposition fluxes cached by
[`set_sslt_dry_deposition_fluxes!`](@ref) via
[`aerosol_surface_flux_tendency!`](@ref), the same bottom-boundary treatment
(grid-mean tendency mirrored onto each updraft) as the emission source.
"""
aerosol_dry_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt) =
    aerosol_surface_flux_tendency!(
        Yₜ,
        Y,
        p,
        sslt,
        p.tracers.sslt_drydep_fluxes,
    )
