import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF

"""
    wind_at_height(z, ustar, obukhov_length, sfp)

MOST point wind speed [m/s] at height `z` over water, recovered
via `SurfaceFluxes.compute_profile_value` with COARE3 roughness.
Used by ClimaCoupler to compute the sea salt emission flux.
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

"""
    uniform_surface_fluxes(sslt::PrognosticSeaSalt, params)

Gong (2003) per-bin mass fluxes at the reference wind speed over open ocean.
"""
uniform_surface_fluxes(::PrognosticSeaSalt, params) =
    CAP.prognostic_aerosol_params(params).bin_mass_flux

#####
##### Tendencies
#####

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

    unrolled_foreach(aerosol_state_names(sslt)) do ρχ_name
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
        @. ᶜρχₜ -= λ * ᶜρχ

        χ_name = specific_tracer_name(ρχ_name)
        for j in 1:n_updrafts
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
            @. ᶜχʲₜ -= λ * ᶜχʲ
        end
    end
    return nothing
end
