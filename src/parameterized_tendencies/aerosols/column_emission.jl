#####
##### Standalone (uncoupled) sea-salt emission driver
#####

"""
    sslt_column_u10(p)

Lazy surface Field of the MOST 10 m wind [m/s] diagnosed from the model's own
surface conditions (`ustar`, `obukhov_length`) via [`wind_at_height`](@ref).
This is the wind the standalone emission driver feeds to the Gong flux, so it
also serves the `u10ss` diagnostic; using the same `ustar` as dry deposition
keeps the emission and deposition winds consistent within a column.
"""
function sslt_column_u10(p)
    FT = eltype(p.params)
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    sfp = CAP.surface_fluxes_params(p.params)
    return @. lazy(wind_at_height(FT(10), ustar, obukhov_length, sfp))
end

"""
    set_sslt_column_surface_fluxes!(Y, p)

Standalone replacement for the ClimaCoupler call to
[`set_sslt_surface_fluxes!`](@ref): when `prognostic_aerosol_column_emission`
is on and sea salt is prognostic, set the per-bin emission fluxes from
[`sslt_column_u10`](@ref) over an all-ocean surface (ocean fraction 1).
Must run after `update_surface_conditions!` and before
[`aerosol_emission_tendency!`](@ref); it is called from
`set_explicit_precomputed_quantities!`. A no-op otherwise.
"""
set_sslt_column_surface_fluxes!(Y, p) =
    set_sslt_column_surface_fluxes!(Y, p, p.atmos.seasalt, p.atmos.column_emission)
set_sslt_column_surface_fluxes!(Y, p, _, _) = nothing
function set_sslt_column_surface_fluxes!(
    Y,
    p,
    sslt::PrognosticSeaSalt,
    column_emission::Bool,
)
    column_emission || return nothing
    FT = eltype(p.params)
    set_sslt_surface_fluxes!(Y, p, sslt_column_u10(p), FT(1), sslt)
    return nothing
end
