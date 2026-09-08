import SurfaceFluxes as SF
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

# Keep the MOST profile finite near neutral stratification.
safe_obukhov_length(L) =
    ifelse(L < zero(L), min(L, -eps(typeof(L))), max(L, eps(typeof(L))))

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
(written by ClimaCoupler via [`set_sslt_surface_fluxes!`](@ref)) via
[`aerosol_surface_flux_tendency!`](@ref).
"""
aerosol_emission_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt) =
    aerosol_surface_flux_tendency!(Yₜ, Y, p, sslt, p.tracers.sslt_sfc_fluxes)

"""
    aerosol_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)

Exponential decay of the grid-mean and updraft sea salt tracers with the
`ssa_residence` timescale (0.55 days, from AeroCom III). Uniform rate
over-deposits small bins and under-deposits large ones; separate wet and dry
deposition tendencies are forthcoming.
"""
function aerosol_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)
    (; turbconv_model) = p.atmos
    ap = CAP.prognostic_aerosol_params(p.params)

    λ = inv(ap.τ_ssa)
    n_updrafts = n_mass_flux_subdomains(turbconv_model)

    MatrixFields.unrolled_foreach(aerosol_state_names(sslt)) do ρχ_name
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
