import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF

"""
    set_wind_at_height!(u_z, z, Y, p)

`SurfaceFluxes.compute_profile_value` wrapper that overwrites the surface
field `u_z` with the MOST point wind speed [m/s] at height z, recovered from
`p.precomputed.sfc_conditions` and clamped at zero. Returns `u_z`.
"""
function set_wind_at_height!(u_z, z, Y, p)
    FT = eltype(Y)
    sfp = CAP.surface_fluxes_params(p.params)
    roughness = SF.COARE3RoughnessParams{FT}()
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    @. u_z = max(
        SF.compute_profile_value(
            sfp,
            safe_obukhov_length(obukhov_length),
            SF.momentum_roughness(roughness, ustar, sfp, nothing),
            FT(z),
            ustar,
            zero(ustar),
            UF.MomentumTransport(),
            SF.PointValueScheme(),
        ),
        zero(ustar),
    )
    return u_z
end

# Keep the MOST profile finite near neutral stratification.
safe_obukhov_length(L) =
    ifelse(L < zero(L), min(L, -eps(typeof(L))), max(L, eps(typeof(L))))


#####
##### Tendencies
#####

"""
    set_sea_salt_surface_fluxes!(Y, p, seasalt_model)

Write each bin's upward sea salt mass flux [kg m⁻² s⁻¹] — the
[Gong2003](@cite) wind power law `(u_10 / ssa_u_ref)^gong_wind_exp` times
the bin's `ssa_gong_logfit_bin_3M_flux` scale (a mass-flux moment of the
emission size distribution precomputed offline; see
`docs/src/sea_salt_emission_fit.jl`) and the ocean fraction — into the surface
fields
`p.tracers.seasalt_sfc_fluxes`, from which
[`aerosol_emission_tendency!`](@ref) builds the tracers' bottom boundary
conditions. Reads `p.precomputed.sfc_conditions`, so it must run after those
are updated; writes zero flux while the Obukhov length field still holds the
pre-coupling `init_sfc_conditions_zero!` default everywhere.
"""
set_sea_salt_surface_fluxes!(Y, p, ::Union{Nothing, PrescribedSeaSalt}) =
    nothing
function set_sea_salt_surface_fluxes!(
    Y,
    p,
    ::PrognosticSeaSalt,
)
    FT = eltype(Y)
    ap = p.params.prognostic_aerosol_params
    (; obukhov_length) = p.precomputed.sfc_conditions
    ocean_fraction = p.ocean_fraction

    u_10 = set_wind_at_height!(p.scratch.ᶠtemp_field_level, FT(10.0), Y, p)
    # TODO: Likely a better way to gate against emission in the
    # first step before we get coupler values, but I haven't found it
    @. u_10 = ifelse(obukhov_length == FT(1e-4), FT(0), u_10)

    wind_factor = u_10
    @. wind_factor = (u_10 / ap.ssa_u_ref) ^ ap.gong_wind_exp

    for (bin_index, sfc_flux) in enumerate(p.tracers.seasalt_sfc_fluxes)
        mass_flux_scale = FT(ap.bin_mass_flux[bin_index])
        @. sfc_flux = C3(mass_flux_scale * wind_factor * ocean_fraction)
    end
    return nothing
end

"""
    aerosol_emission_tendency!(Yₜ, Y, p, t, seasalt::PrognosticSeaSalt)

Apply the per-bin emission fluxes cached in `p.tracers.seasalt_sfc_fluxes`
(see [`set_sea_salt_surface_fluxes!`](@ref)) as bottom boundary conditions on
the grid-mean `Y.c.ρ<bin>` tracers, using [`boundary_tendency_scalar`](@ref),
and mirror the specific tendency onto each updraft tracer.
"""
function aerosol_emission_tendency!(Yₜ, Y, p, t, seasalt::PrognosticSeaSalt)
    n_updrafts = n_mass_flux_subdomains(p.atmos.turbconv_model)

    for name in bin_names(seasalt)
        ρχ_name = Symbol(:ρ, name)
        ᶜρχ = getproperty(Y.c, ρχ_name)
        ᶜρχₜ = getproperty(Yₜ.c, ρχ_name)
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        btt = boundary_tendency_scalar(ᶜχ, p.tracers.seasalt_sfc_fluxes[ρχ_name])
        @. ᶜρχₜ -= btt

        for j in 1:n_updrafts
            ᶜχʲₜ = getproperty(Yₜ.c.sgsʲs.:($j), name)
            @. ᶜχʲₜ -= specific(btt, p.precomputed.ᶜρʲs.:($j))
        end
    end
    return nothing
end

"""
    aerosol_deposition_tendency!(Yₜ, Y, p, t, seasalt::PrognosticSeaSalt)

Exponential decay of the grid-mean and updraft sea salt tracers with the
`ssa_residence` timescale (0.55 days, from AeroCom III). Uniform rate
over-deposits small bins and under-deposits large ones; separate wet and dry
deposition tendencies are forthcoming.
"""
function aerosol_deposition_tendency!(Yₜ, Y, p, t, seasalt::PrognosticSeaSalt)
    (; turbconv_model) = p.atmos
    ap = p.params.prognostic_aerosol_params

    λ = inv(ap.τ_ssa)
    n_updrafts = n_mass_flux_subdomains(turbconv_model)

    for name in bin_names(seasalt)
        ᶜρχ = getproperty(Y.c, Symbol(:ρ, name))
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        @. ᶜρχₜ -= λ * ᶜρχ

        for j in 1:n_updrafts
            ᶜχʲ = getproperty(Y.c.sgsʲs.:($j), name)
            ᶜχʲₜ = getproperty(Yₜ.c.sgsʲs.:($j), name)
            @. ᶜχʲₜ -= λ * ᶜχʲ
        end
    end
    return nothing
end
